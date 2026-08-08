"""FastAPI routes for authenticated LeIsaac discovery and WebRTC signaling."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket

LOG = logging.getLogger(__name__)
_BACKHAUL_HEADER_SIZE = 9
_BACKHAUL_MAX_FRAME = 4 * 1024 * 1024
_WS_SESSION_COOKIES = {
    "control": "npa_leisaac_control_ws",
    "video": "npa_leisaac_video_ws",
}
_WS_SESSION_TTL_SECONDS = 120
_BACKHAUL_SUBPROTOCOL = "npa.leisaac.backhaul.v1"

try:  # agent VM: /opt/npa-agent is on sys.path
    from agent_backend.leisaac import (
        LEISAAC_CLIENT_MODULE_PATH,
        LEISAAC_CLIENT_JS_SHA256,
        LEISAAC_BUNDLE_RESET_PATH,
        LEISAAC_BUNDLES_PATH,
        LEISAAC_CONTROL_DATACHANNEL_PATH,
        LEISAAC_CONTROL_WS_PATH,
        LEISAAC_RECORDER_PATH,
        LEISAAC_SIGNAL_PORT,
        LEISAAC_VIEW_PATH,
        LEISAAC_VIDEO_DATACHANNEL_PATH,
        LEISAAC_VIDEO_WS_PATH,
        normalize_manifest,
        selected_run_id,
        status_payload,
        validate_health,
    )
    from agent_backend.leisaac_transport import (
        AsyncLatestByKey,
        CONTROL_SUBPROTOCOL,
        MAX_CONTROL_MESSAGE_BYTES,
        MAX_FRAME_BYTES,
        TransportMetrics,
        TransportProtocolError,
        VIDEO_SUBPROTOCOL,
        parse_control_message,
        parse_video_ack,
        stamp_verified_frame,
        unpack_frame,
    )
    from agent_backend.leisaac_datachannel import (
        VideoDataChannelError,
        parse_video_datachannel_offer,
    )
    from agent_backend.leisaac_episodes import (
        EpisodeStore,
        EpisodeStoreError,
        RangeNotSatisfiable,
        iter_s3_body,
        parse_http_range,
    )
    from agent_backend.leisaac_bundles import BundleError, BundleStore
except ImportError:  # repository tests
    from npa.agent_backend.leisaac import (
        LEISAAC_CLIENT_MODULE_PATH,
        LEISAAC_CLIENT_JS_SHA256,
        LEISAAC_BUNDLE_RESET_PATH,
        LEISAAC_BUNDLES_PATH,
        LEISAAC_CONTROL_DATACHANNEL_PATH,
        LEISAAC_CONTROL_WS_PATH,
        LEISAAC_RECORDER_PATH,
        LEISAAC_SIGNAL_PORT,
        LEISAAC_VIEW_PATH,
        LEISAAC_VIDEO_DATACHANNEL_PATH,
        LEISAAC_VIDEO_WS_PATH,
        normalize_manifest,
        selected_run_id,
        status_payload,
        validate_health,
    )
    from npa.agent_backend.leisaac_transport import (
        AsyncLatestByKey,
        CONTROL_SUBPROTOCOL,
        MAX_CONTROL_MESSAGE_BYTES,
        MAX_FRAME_BYTES,
        TransportMetrics,
        TransportProtocolError,
        VIDEO_SUBPROTOCOL,
        parse_control_message,
        parse_video_ack,
        stamp_verified_frame,
        unpack_frame,
    )
    from npa.agent_backend.leisaac_datachannel import (
        VideoDataChannelError,
        parse_video_datachannel_offer,
    )
    from npa.agent_backend.leisaac_episodes import (
        EpisodeStore,
        EpisodeStoreError,
        RangeNotSatisfiable,
        iter_s3_body,
        parse_http_range,
    )
    from npa.agent_backend.leisaac_bundles import BundleError, BundleStore


@dataclass
class LeIsaacDeps:
    """Dependencies supplied by the rendered agent backend."""

    load_state: Callable[[], dict]
    resolve_manifest: Callable[[str], dict | None]
    http_get: Callable[..., Any]
    response: Any
    websocket_connect: Callable[..., Any]
    http_post: Callable[..., Any] | None = None
    save_state: Callable[[dict], None] | None = None
    mutate_state: Callable[[Callable[[dict], Any]], Any] | None = None
    s3_client: Callable[[], tuple[Any, dict]] | None = None
    s3_buckets: Callable[[Any, dict], list[str]] | None = None


class _RedactedException(RuntimeError):
    """Safe surrogate that retains a traceback without untrusted message text."""


def _log_exception(level: int, event: str, exc: BaseException) -> None:
    """Log traceback and exception types without auth, query, or payload values."""

    causes: list[str] = []
    cause = exc.__cause__ or exc.__context__
    while cause is not None and len(causes) < 4:
        causes.append(type(cause).__name__)
        cause = cause.__cause__ or cause.__context__
    safe = _RedactedException(type(exc).__name__)
    LOG.log(
        level,
        "%s",
        event,
        extra={"exception_type": type(exc).__name__, "cause_types": causes},
        exc_info=(type(safe), safe, exc.__traceback__),
    )


def _resolve(deps: LeIsaacDeps, requested_run_id: str) -> tuple[dict | None, str]:
    run_id = selected_run_id(deps.load_state(), requested_run_id)
    if not run_id:
        return None, "Select a run that exposes a LeIsaac teleoperation session."
    try:
        raw = deps.resolve_manifest(run_id)
    except Exception as exc:  # storage failures are capability absence, not a UI 500
        _log_exception(logging.WARNING, "LeIsaac capability resolution failed", exc)
        return None, "LeIsaac capability discovery is unavailable."
    return normalize_manifest(raw, expected_run_id=run_id)


def _health(deps: LeIsaacDeps, manifest: dict) -> tuple[dict | None, str]:
    try:
        response = deps.http_get(
            f"{manifest['service_url']}/status",
            timeout=3.0,
            follow_redirects=False,
        )
        if int(response.status_code) != 200:
            return None, f"LeIsaac service health returned HTTP {response.status_code}"
        payload = response.json()
    except Exception as exc:
        _log_exception(logging.WARNING, "LeIsaac health request failed", exc)
        return None, "LeIsaac service is unreachable."
    return validate_health(manifest, payload)


def _same_https_origin(headers: Any) -> bool:
    """Validate that an nginx-forwarded browser request has the public HTTPS origin."""

    if str(headers.get("x-forwarded-proto") or "").lower() != "https":
        return False
    origin = str(headers.get("origin") or "")
    host = str(headers.get("host") or "").lower()
    try:
        parsed = urlparse(origin)
        forwarded_host = urlparse(f"//{host}")
        origin_host = str(parsed.hostname or "").lower()
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 0)
        host_name = str(forwarded_host.hostname or "").lower()
        host_port = forwarded_host.port or 443
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme == "https"
        and origin_host == host_name
        and origin_port == host_port
        and not parsed.username
        and not parsed.password
        and parsed.path in ("", "/")
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def _same_origin_session_request(headers: Any) -> bool:
    """Accept exact Origin or Chromium's same-origin Fetch Metadata + referrer."""

    if _same_https_origin(headers):
        return True
    if str(headers.get("sec-fetch-site") or "").lower() != "same-origin":
        return False
    forwarded = dict(headers)
    referer = urlparse(str(headers.get("referer") or ""))
    forwarded["origin"] = f"{referer.scheme}://{referer.netloc}"
    return _same_https_origin(forwarded)


def _same_origin_websocket(websocket: WebSocket, subprotocol: str) -> bool:
    """Validate the public HTTPS origin and exact NPA subprotocol."""

    protocols = {
        item.strip()
        for item in str(websocket.headers.get("sec-websocket-protocol") or "").split(
            ","
        )
        if item.strip()
    }
    return _same_https_origin(websocket.headers) and protocols == {subprotocol}


def _client_address(headers: Any, client: Any) -> str:
    """Return the nginx-attested public client address without trusting browser input."""

    del client  # nginx is the sole attestor; the ASGI peer is normally loopback.
    address = str(headers.get("x-real-ip") or "").strip()
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return ""
    return parsed.compressed if parsed.is_global else ""


def _mint_ws_session(
    secret: bytes,
    run_id: str,
    client_address: str,
    audience: str,
    *,
    now: int | None = None,
) -> str:
    """Mint a short-lived opaque-to-the-browser transport authorization."""

    issued_at = int(time.time() if now is None else now)
    payload = json.dumps(
        {
            "audience": audience,
            "client": client_address,
            "expires": issued_at + _WS_SESSION_TTL_SECONDS,
            "nonce": secrets.token_urlsafe(12),
            "run_id": run_id,
            "v": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    body = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    return f"{body}.{signature}"


def _valid_ws_session(
    secret: bytes,
    token: str,
    run_id: str,
    client_address: str,
    audience: str,
    *,
    now: int | None = None,
    consumed_nonces: dict[str, int] | None = None,
) -> bool:
    """Validate and optionally consume a run/address/audience credential."""

    try:
        body, signature = token.split(".", 1)
        expected = (
            base64.urlsafe_b64encode(
                hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        if not hmac.compare_digest(signature, expected):
            return False
        padding = "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(body + padding))
        current = int(time.time() if now is None else now)
        expires = int(payload.get("expires", 0))
    except (
        UnicodeEncodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        binascii.Error,
    ):
        return False
    nonce = payload.get("nonce")
    valid = (
        payload.get("v") == 1
        and payload.get("run_id") == run_id
        and payload.get("client") == client_address
        and payload.get("audience") == audience
        and isinstance(nonce, str)
        and 1 <= len(nonce) <= 64
        and current <= expires <= current + _WS_SESSION_TTL_SECONDS + 5
    )
    if not valid or consumed_nonces is None:
        return valid
    for stale in [key for key, expiry in consumed_nonces.items() if expiry < current]:
        consumed_nonces.pop(stale, None)
    replay_key = f"{audience}:{nonce}"
    if replay_key in consumed_nonces:
        return False
    consumed_nonces[replay_key] = expires
    return True


def _runtime_ws_uri(manifest: dict[str, Any], path: str) -> str:
    base = str(manifest["service_url"])
    if not base.startswith("http://"):
        raise ValueError("LeIsaac runtime service URL is not loopback HTTP")
    return "ws://" + base.removeprefix("http://").rstrip("/") + path


async def _relay_browser_to_upstream(browser: Any, upstream: Any) -> None:
    while True:
        message = await browser.receive()
        kind = message.get("type")
        if kind == "websocket.disconnect":
            return
        if message.get("bytes") is not None:
            await upstream.send(message["bytes"])
        elif message.get("text") is not None:
            await upstream.send(message["text"])


async def _relay_upstream_to_browser(browser: Any, upstream: Any) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await browser.send_bytes(message)
        else:
            await browser.send_text(str(message))


def register_leisaac_routes(app: Any, deps: LeIsaacDeps) -> None:
    """Register the LeIsaac capability, client-module, and signaling routes."""

    manifest_cache: dict[str, tuple[float, dict | None, str]] = {}
    manifest_cache_lock = threading.Lock()
    transport_metrics = TransportMetrics()
    ws_session_secret = secrets.token_bytes(32)
    consumed_ws_nonces: dict[str, int] = {}
    bundle_selection_lock = asyncio.Lock()
    bundle_restore_pending: dict[str, dict[str, str]] = {}
    bundle_restore_lock = threading.Lock()

    def mutate_state(mutation: Callable[[dict], Any]) -> Any:
        """Apply one state mutation atomically when the backend supports it."""

        if deps.mutate_state is not None:
            return deps.mutate_state(mutation)
        if deps.save_state is None:
            raise RuntimeError("LeIsaac state mutation is unavailable")
        state = deps.load_state()
        if not isinstance(state, dict):
            state = {}
        result = mutation(state)
        deps.save_state(state)
        return result

    def cached_resolve(run_id: str) -> tuple[dict | None, str]:
        """Reuse immutable live manifests while retaining short negative caching."""

        now = time.monotonic()
        selected = selected_run_id(deps.load_state(), run_id)
        if not selected:
            return None, "Select a run that exposes a LeIsaac teleoperation session."
        with manifest_cache_lock:
            cached = manifest_cache.get(selected)
            if cached is not None:
                cached_at, manifest, reason = cached
                if manifest is None and now - cached_at < 5.0:
                    return None, reason
                if manifest is not None:
                    raw_expiry = str(manifest.get("expires_at") or "").strip()
                    if not raw_expiry:
                        # Session manifests are write-once. An omitted expiry means
                        # service-lifecycle validity, so re-discovering the same
                        # immutable object on every five-second poll only adds S3
                        # latency and cannot produce a newer value for this run id.
                        return manifest, reason
                    try:
                        expiry = (
                            datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
                        )
                    except ValueError:
                        expiry = datetime.min.replace(tzinfo=timezone.utc)
                    if expiry is not None and expiry > datetime.now(timezone.utc):
                        return manifest, reason
                    manifest_cache.pop(selected, None)
        manifest, reason = _resolve(deps, selected)
        with manifest_cache_lock:
            if len(manifest_cache) >= 128 and selected not in manifest_cache:
                oldest = min(manifest_cache, key=lambda key: manifest_cache[key][0])
                manifest_cache.pop(oldest, None)
            manifest_cache[selected] = (now, manifest, reason)
        return manifest, reason

    def episode_store(run_id: str) -> EpisodeStore:
        manifest, reason = cached_resolve(run_id)
        if not manifest:
            raise EpisodeStoreError(reason, status_code=404)
        if deps.s3_client is None or deps.s3_buckets is None:
            raise EpisodeStoreError("episode storage is unavailable", status_code=503)
        try:
            s3, settings = deps.s3_client()
            buckets = deps.s3_buckets(s3, settings)
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac episode storage setup failed", exc)
            raise EpisodeStoreError(
                "episode storage is unavailable", status_code=503
            ) from exc
        return EpisodeStore(
            s3,
            str(manifest.get("dataset_uri") or ""),
            allowed_buckets=buckets,
            run_id=str(manifest["run_id"]),
        )

    def bundle_store(run_id: str) -> BundleStore:
        manifest, reason = cached_resolve(run_id)
        if not manifest:
            raise BundleError(reason, status_code=404)
        if deps.s3_client is None or deps.s3_buckets is None:
            raise BundleError("bundle storage is unavailable", status_code=503)
        try:
            s3, settings = deps.s3_client()
            buckets = deps.s3_buckets(s3, settings)
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac bundle storage setup failed", exc)
            raise BundleError("bundle storage is unavailable", status_code=503) from exc
        return BundleStore(
            s3,
            str(manifest.get("dataset_uri") or ""),
            allowed_buckets=buckets,
        )

    def selection_scope(manifest: dict[str, Any]) -> dict[str, str]:
        return {
            "run_id": str(manifest.get("run_id") or ""),
            "dataset_uri": str(manifest.get("dataset_uri") or "").rstrip("/"),
            "task": str(manifest.get("task") or ""),
            "task_registry_fingerprint": str(
                manifest.get("task_registry_fingerprint") or ""
            ),
        }

    def scoped_selection(
        state: dict[str, Any] | None, manifest: dict[str, Any] | None
    ) -> dict[str, dict[str, str]]:
        if not isinstance(state, dict) or not isinstance(manifest, dict):
            return {}
        leisaac = state.get("leisaac")
        if (
            not isinstance(leisaac, dict)
            or leisaac.get("bundle_selection_scope") != selection_scope(manifest)
        ):
            return {}
        raw = leisaac.get("bundle_selection")
        if not isinstance(raw, dict) or len(raw) > 3:
            return {}
        result: dict[str, dict[str, str]] = {}
        for kind, item in raw.items():
            if kind not in {"robot", "scene", "device"} or not isinstance(item, dict):
                return {}
            normalized = {
                "bundle_sha256": str(item.get("bundle_sha256") or ""),
                "name": str(item.get("name") or ""),
                "entrypoint": str(item.get("entrypoint") or ""),
            }
            if (
                not re.fullmatch(r"[a-f0-9]{64}", normalized["bundle_sha256"])
                or not normalized["name"]
                or not normalized["entrypoint"]
            ):
                return {}
            result[str(kind)] = normalized
        return result

    def _selection_digests(selection: dict[str, dict[str, str]]) -> dict[str, str]:
        return {
            kind: str(item["bundle_sha256"])
            for kind, item in sorted(selection.items())
        }

    def bundle_error(exc: BundleError) -> Any:
        return deps.response(
            content=json.dumps({"detail": exc.detail}),
            status_code=exc.status_code,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    def episode_error(exc: EpisodeStoreError) -> Any:
        headers = {"Cache-Control": "private, no-store"}
        if isinstance(exc, RangeNotSatisfiable):
            headers["Content-Range"] = f"bytes */{exc.size}"
            headers["Accept-Ranges"] = "bytes"
        return deps.response(
            content=json.dumps({"detail": exc.detail}),
            status_code=exc.status_code,
            media_type="application/json",
            headers=headers,
        )

    def episode_request_allowed(request: Request) -> bool:
        if str(request.headers.get("x-forwarded-proto") or "").lower() != "https":
            return False
        # nginx enforces the agent authentication.  Reject explicit cross-site
        # browser fetches while allowing video-element Range requests, which do
        # not consistently carry Origin across supported browsers.
        return str(request.headers.get("sec-fetch-site") or "").lower() != "cross-site"

    async def stream_object(
        store: EpisodeStore,
        ref: Any,
        *,
        range_header: str = "",
        media_type: str,
        filename: str = "",
    ) -> Any:
        try:
            head = await asyncio.to_thread(
                store.client.head_object, Bucket=store.bucket, Key=ref.key
            )
            size = int(head.get("ContentLength", -1))
        except Exception as exc:
            _log_exception(
                logging.WARNING, "LeIsaac episode metadata lookup failed", exc
            )
            return episode_error(
                EpisodeStoreError("episode artifact is unavailable", status_code=502)
            )
        if size < 0 or size != int(ref.size):
            return episode_error(
                EpisodeStoreError(
                    "episode artifact size does not match its immutable commit",
                    status_code=502,
                )
            )
        metadata_sha = str((head.get("Metadata") or {}).get("sha256") or "")
        if metadata_sha and metadata_sha != str(ref.sha256):
            return episode_error(
                EpisodeStoreError(
                    "episode artifact storage checksum does not match its commit",
                    status_code=502,
                )
            )
        try:
            byte_range = parse_http_range(range_header, size)
        except RangeNotSatisfiable as exc:
            return episode_error(exc)
        get_kwargs: dict[str, Any] = {"Bucket": store.bucket, "Key": ref.key}
        status_code = 200
        length = size
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "X-NPA-SHA256": str(ref.sha256),
            "X-NPA-Checksum-State": (
                "storage-metadata-match" if metadata_sha else "commit-declared"
            ),
        }
        if byte_range is not None:
            get_kwargs["Range"] = f"bytes={byte_range.start}-{byte_range.end}"
            status_code = 206
            length = byte_range.length
            headers["Content-Range"] = (
                f"bytes {byte_range.start}-{byte_range.end}/{byte_range.size}"
            )
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        headers["Content-Length"] = str(length)
        try:
            upstream = await asyncio.to_thread(store.client.get_object, **get_kwargs)
            body = upstream["Body"]
        except Exception as exc:
            _log_exception(
                logging.WARNING, "LeIsaac episode artifact download failed", exc
            )
            return episode_error(
                EpisodeStoreError("episode artifact is unavailable", status_code=502)
            )
        return StreamingResponse(
            iter_s3_body(body, length),
            status_code=status_code,
            media_type=media_type,
            headers=headers,
        )

    @app.get("/leisaac/status")
    def leisaac_status(request: Request, run_id: str = "") -> Any:
        manifest: dict[str, Any] | None = None
        health: dict[str, Any] | None = None
        if str(request.headers.get("x-forwarded-proto") or "").lower() != "https":
            payload = status_payload(
                None,
                reason="LeIsaac teleoperation requires the public HTTPS agent endpoint.",
            )
        else:
            manifest, reason = cached_resolve(run_id)
            if not manifest:
                payload = status_payload(None, reason=reason)
            else:
                health, reason = _health(deps, manifest)
                state = deps.load_state()
                desired_selection = scoped_selection(state, manifest)
                desired_digests = _selection_digests(desired_selection)
                actual_digests = _selection_digests(
                    health.get("selected_bundles", {}) if health else {}
                )
                restore_pending = bool(
                    health and desired_digests and actual_digests != desired_digests
                )
                if health and desired_digests and actual_digests != desired_digests:
                    with bundle_restore_lock:
                        pending = bundle_restore_pending.get(str(manifest["run_id"]))
                        if pending != desired_digests and deps.http_post is not None:
                            try:
                                upstream = deps.http_post(
                                    f"{manifest['service_url']}/bundles/apply",
                                    json={"selection": desired_digests},
                                    headers={
                                        "X-NPA-LeIsaac-Nonce": manifest[
                                            "session_nonce"
                                        ]
                                    },
                                    timeout=30.0,
                                    follow_redirects=False,
                                )
                            except Exception as exc:
                                _log_exception(
                                    logging.WARNING,
                                    "LeIsaac bundle restore failed",
                                    exc,
                                )
                            else:
                                if (
                                    upstream is not None
                                    and int(upstream.status_code) == 202
                                ):
                                    bundle_restore_pending[
                                        str(manifest["run_id"])
                                    ] = desired_digests
                elif health and actual_digests == desired_digests:
                    with bundle_restore_lock:
                        bundle_restore_pending.pop(str(manifest["run_id"]), None)
                if restore_pending:
                    payload = status_payload(
                        manifest,
                        reason="Restoring persisted checksum-verified custom bundles.",
                    )
                else:
                    payload = status_payload(manifest, health, reason=reason)
        if payload.get("available"):
            payload["agent_transport_metrics"] = transport_metrics.snapshot()
        state = deps.load_state()
        payload["bundle_selection"] = scoped_selection(state, manifest)
        payload["bundle_selection_scope"] = (
            selection_scope(manifest) if manifest else {}
        )
        return deps.response(
            content=json.dumps(payload),
            status_code=200,
            media_type="application/json",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get(LEISAAC_BUNDLES_PATH.removeprefix("/api"))
    async def leisaac_bundles(request: Request) -> Any:
        if not episode_request_allowed(request):
            return bundle_error(
                BundleError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        try:
            store = await asyncio.to_thread(
                bundle_store, str(request.query_params.get("run_id") or "")
            )
            result = await asyncio.to_thread(
                store.list, kind=str(request.query_params.get("kind") or "")
            )
        except BundleError as exc:
            return bundle_error(exc)
        return deps.response(
            content=json.dumps(result),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post(LEISAAC_BUNDLES_PATH.removeprefix("/api"))
    async def leisaac_bundle_upload(request: Request) -> Any:
        if (
            not episode_request_allowed(request)
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return bundle_error(
                BundleError(
                    "same-origin authenticated HTTPS upload is required",
                    status_code=403,
                )
            )
        try:
            length = int(request.headers.get("content-length") or "0")
        except ValueError:
            length = 0
        if not 1 <= length <= 18 * 1024 * 1024:
            return bundle_error(BundleError("bundle request size is invalid"))
        try:
            payload = await request.json()
            store = await asyncio.to_thread(
                bundle_store, str(request.query_params.get("run_id") or "")
            )
            result = await asyncio.to_thread(store.publish, payload)
        except ValueError:
            return bundle_error(BundleError("bundle request is not valid JSON"))
        except BundleError as exc:
            return bundle_error(exc)
        return deps.response(
            content=json.dumps(result),
            status_code=201,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    async def _leisaac_bundle_select_impl(request: Request) -> Any:
        if (
            not episode_request_allowed(request)
            or request.headers.get("x-npa-leisaac-control") != "1"
            or (deps.save_state is None and deps.mutate_state is None)
        ):
            return bundle_error(
                BundleError(
                    "same-origin authenticated selection is required", status_code=403
                )
            )
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {
                "kind",
                "bundle_sha256",
            }:
                raise BundleError("bundle selection fields are invalid")
            store = await asyncio.to_thread(
                bundle_store, str(request.query_params.get("run_id") or "")
            )
            manifest = await asyncio.to_thread(
                store.get, str(payload.get("bundle_sha256") or "")
            )
            if manifest.get("kind") != payload.get("kind"):
                raise BundleError("bundle selection kind does not match")
            capability, reason = await asyncio.to_thread(
                cached_resolve, str(request.query_params.get("run_id") or "")
            )
            if not capability:
                raise BundleError(reason, status_code=404)
            current_scope = selection_scope(capability)
            loaded_state = deps.load_state()
            state = (
                json.loads(json.dumps(loaded_state))
                if isinstance(loaded_state, dict)
                else {}
            )
            leisaac_state = state.setdefault("leisaac", {})
            if not isinstance(leisaac_state, dict):
                leisaac_state = {}
                state["leisaac"] = leisaac_state
            selection = (
                leisaac_state.setdefault("bundle_selection", {})
                if leisaac_state.get("bundle_selection_scope") == current_scope
                else {}
            )
            if not isinstance(selection, dict):
                selection = {}
            leisaac_state["bundle_selection"] = selection
            leisaac_state["bundle_selection_scope"] = current_scope
            selection[str(payload["kind"])] = {
                "bundle_sha256": manifest["bundle_sha256"],
                "name": manifest["name"],
                "entrypoint": manifest["entrypoint"],
            }
            # Agent state can outlive an operator-directed dataset-prefix change.
            # Do not let an unavailable bundle from the previous store poison the
            # cumulative selection sent to the current runtime. This loop is
            # inherently bounded to robot, scene, and device.
            for selected_kind, selected_item in list(selection.items()):
                if selected_kind == payload["kind"]:
                    continue
                try:
                    selected_manifest = await asyncio.to_thread(
                        store.get,
                        str(
                            selected_item.get("bundle_sha256")
                            if isinstance(selected_item, dict)
                            else ""
                        ),
                    )
                except BundleError:
                    del selection[selected_kind]
                    continue
                if selected_manifest.get("kind") != selected_kind:
                    del selection[selected_kind]
            if deps.http_post is None:
                raise BundleError("bundle application is unavailable", status_code=503)
            apply_payload = {
                "selection": {
                    kind: str(item["bundle_sha256"])
                    for kind, item in sorted(selection.items())
                    if isinstance(item, dict) and item.get("bundle_sha256")
                }
            }
            try:
                upstream = await asyncio.to_thread(
                    deps.http_post,
                    f"{capability['service_url']}/bundles/apply",
                    json=apply_payload,
                    headers={"X-NPA-LeIsaac-Nonce": capability["session_nonce"]},
                    timeout=30.0,
                    follow_redirects=False,
                )
            except Exception as exc:
                _log_exception(
                    logging.WARNING, "LeIsaac bundle application failed", exc
                )
                upstream = None
            if upstream is None:
                raise BundleError("bundle application is unavailable", status_code=503)
            upstream_status = int(upstream.status_code)
            if upstream_status != 202:
                try:
                    upstream_detail = str(upstream.json().get("detail") or "")
                except Exception as exc:
                    _log_exception(
                        logging.DEBUG,
                        "LeIsaac bundle rejection body was invalid",
                        exc,
                    )
                    upstream_detail = ""
                raise BundleError(
                    upstream_detail or "bundle application was rejected",
                    status_code=(
                        upstream_status if upstream_status in {400, 409} else 502
                    ),
                )
            selection_snapshot = json.loads(json.dumps(selection))

            def persist_selection(current_state: dict) -> None:
                current = current_state.setdefault("leisaac", {})
                if not isinstance(current, dict):
                    current = {}
                    current_state["leisaac"] = current
                current["bundle_selection"] = selection_snapshot
                current["bundle_selection_scope"] = current_scope

            await asyncio.to_thread(mutate_state, persist_selection)
        except (ValueError, BundleError) as exc:
            return bundle_error(
                exc
                if isinstance(exc, BundleError)
                else BundleError("bundle selection is invalid")
            )
        return deps.response(
            content=json.dumps(
                {
                    "selected": selection[str(payload["kind"])],
                    "restarting": True,
                }
            ),
            status_code=202,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post((LEISAAC_BUNDLES_PATH + "/select").removeprefix("/api"))
    async def leisaac_bundle_select(request: Request) -> Any:
        # Applying a bundle restarts the simulator. Serialize the cumulative
        # read/apply/persist transaction so two operator clicks cannot make the
        # runtime and persisted selection disagree.
        async with bundle_selection_lock:
            return await _leisaac_bundle_select_impl(request)

    @app.post(LEISAAC_BUNDLE_RESET_PATH.removeprefix("/api"))
    async def leisaac_bundle_reset(request: Request) -> Any:
        """Clear uploaded overrides and restart the exact task on real built-ins."""

        if (
            not episode_request_allowed(request)
            or request.headers.get("x-npa-leisaac-control") != "1"
            or (deps.save_state is None and deps.mutate_state is None)
        ):
            return bundle_error(
                BundleError(
                    "same-origin authenticated reset is required", status_code=403
                )
            )
        async with bundle_selection_lock:
            capability, reason = await asyncio.to_thread(
                cached_resolve, str(request.query_params.get("run_id") or "")
            )
            if not capability:
                return bundle_error(BundleError(reason, status_code=404))
            if deps.http_post is None:
                return bundle_error(
                    BundleError("bundle application is unavailable", status_code=503)
                )
            try:
                upstream = await asyncio.to_thread(
                    deps.http_post,
                    f"{capability['service_url']}/bundles/apply",
                    json={"selection": {}},
                    headers={
                        "X-NPA-LeIsaac-Nonce": capability["session_nonce"]
                    },
                    timeout=30.0,
                    follow_redirects=False,
                )
            except Exception as exc:
                _log_exception(logging.WARNING, "LeIsaac default reset failed", exc)
                upstream = None
            if upstream is None:
                return bundle_error(
                    BundleError("bundle application is unavailable", status_code=503)
                )
            upstream_status = int(upstream.status_code)
            if upstream_status != 202:
                try:
                    upstream_detail = str(upstream.json().get("detail") or "")
                except Exception:
                    upstream_detail = ""
                return bundle_error(
                    BundleError(
                        upstream_detail or "default reset was rejected",
                        status_code=(
                            upstream_status if upstream_status in {400, 409} else 502
                        ),
                    )
                )
            current_scope = selection_scope(capability)

            def persist_reset(current_state: dict) -> None:
                current = current_state.setdefault("leisaac", {})
                if not isinstance(current, dict):
                    current = {}
                    current_state["leisaac"] = current
                current["bundle_selection"] = {}
                current["bundle_selection_scope"] = current_scope

            await asyncio.to_thread(mutate_state, persist_reset)
            with bundle_restore_lock:
                bundle_restore_pending.pop(str(capability["run_id"]), None)
        return deps.response(
            content=json.dumps(
                {
                    "reset": True,
                    "selected_bundles": {},
                    "restarting": True,
                    "configuration": capability.get("configuration", {}),
                }
            ),
            status_code=202,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/leisaac/episodes/versions")
    async def leisaac_episode_versions(request: Request) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            payload = await asyncio.to_thread(
                store.list_versions,
                limit=query.get("limit", "20"),
                cursor=query.get("cursor", ""),
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        return deps.response(
            content=json.dumps(payload),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/leisaac/episodes")
    async def leisaac_episodes(request: Request) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            payload = await asyncio.to_thread(
                store.list_episodes,
                limit=query.get("limit", "20"),
                cursor=query.get("cursor", ""),
                version_id=str(query.get("version_id") or ""),
                filters={
                    name: query.get(name, "")
                    for name in (
                        "task",
                        "environment",
                        "outcome",
                        "robot",
                        "scene",
                        "device",
                        "date_from",
                        "date_to",
                    )
                },
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        return deps.response(
            content=json.dumps(payload),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/leisaac/episodes/{episode_id}")
    async def leisaac_episode_detail(request: Request, episode_id: str) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            payload = await asyncio.to_thread(
                store.detail,
                episode_id,
                version_id=str(query.get("version_id") or ""),
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        return deps.response(
            content=json.dumps(payload),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/leisaac/episodes/{episode_id}/timeline")
    async def leisaac_episode_timeline(request: Request, episode_id: str) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            payload = await asyncio.to_thread(
                store.timeline,
                episode_id,
                version_id=str(query.get("version_id") or ""),
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        return deps.response(
            content=json.dumps(payload),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get("/leisaac/episodes/{episode_id}/media/{camera_id}")
    async def leisaac_episode_media(
        request: Request, episode_id: str, camera_id: str
    ) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            ref = await asyncio.to_thread(
                store.media_ref,
                episode_id,
                camera_id,
                version_id=str(query.get("version_id") or ""),
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        return await stream_object(
            store,
            ref,
            range_header=str(request.headers.get("range") or ""),
            media_type="video/mp4",
        )

    @app.get("/leisaac/episodes/{episode_id}/download/{artifact_id}")
    async def leisaac_episode_download(
        request: Request, episode_id: str, artifact_id: str
    ) -> Any:
        if not episode_request_allowed(request):
            return episode_error(
                EpisodeStoreError(
                    "same-origin authenticated HTTPS is required", status_code=403
                )
            )
        query = request.query_params
        try:
            store = await asyncio.to_thread(
                episode_store, str(query.get("run_id") or "")
            )
            ref = await asyncio.to_thread(
                store.download_ref,
                episode_id,
                artifact_id,
                version_id=str(query.get("version_id") or ""),
            )
        except EpisodeStoreError as exc:
            return episode_error(exc)
        media_type = {
            "records": "application/x-ndjson",
            "metadata": "application/json",
        }.get(artifact_id, "application/octet-stream")
        suffix = (
            "jsonl"
            if artifact_id == "records"
            else "json"
            if artifact_id == "metadata"
            else "bin"
        )
        return await stream_object(
            store,
            ref,
            range_header=str(request.headers.get("range") or ""),
            media_type=media_type,
            filename=f"episode-{int(episode_id):06d}-{artifact_id}.{suffix}",
        )

    @app.post("/leisaac/select")
    async def leisaac_select(request: Request) -> Any:
        if (
            str(request.headers.get("x-forwarded-proto") or "").lower() != "https"
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "authenticated HTTPS capability selection is required"}
                ),
                status_code=403,
                media_type="application/json",
            )
        try:
            body = await request.json()
        except ValueError:
            body = None
        run_id = str(body.get("run_id") if isinstance(body, dict) else "")
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        health, reason = await asyncio.to_thread(_health, deps, manifest)
        if not health:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=503,
                media_type="application/json",
            )
        if deps.save_state is None and deps.mutate_state is None:
            return deps.response(
                content=json.dumps({"detail": "LeIsaac selection is unavailable"}),
                status_code=503,
                media_type="application/json",
            )

        def save_selection() -> None:
            def select_run(state: dict) -> None:
                # Periodic refresh must preserve the checksum-verified bundle
                # set written by a concurrent bundle restart.
                current = state.get("leisaac")
                leisaac_state = current if isinstance(current, dict) else {}
                leisaac_state["run_id"] = manifest["run_id"]
                state["leisaac"] = leisaac_state

            mutate_state(select_run)

        try:
            await asyncio.to_thread(save_selection)
        except Exception as exc:
            _log_exception(logging.ERROR, "LeIsaac selection state save failed", exc)
            return deps.response(
                content=json.dumps({"detail": "LeIsaac selection could not be saved"}),
                status_code=503,
                media_type="application/json",
            )
        return deps.response(
            content=json.dumps(
                {
                    "selected": True,
                    "run_id": manifest["run_id"],
                    "available": True,
                }
            ),
            status_code=200,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.get(LEISAAC_CLIENT_MODULE_PATH.removeprefix("/api"))
    def leisaac_client_module(run_id: str = "") -> Any:
        manifest, reason = cached_resolve(run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        health, reason = _health(deps, manifest)
        if not health:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=503,
                media_type="application/json",
            )
        try:
            response = deps.http_get(
                f"{manifest['service_url']}/client/index.js",
                timeout=10.0,
                follow_redirects=False,
            )
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac client module fetch failed", exc)
            response = None
        if response is None or int(response.status_code) != 200:
            return deps.response(
                content=json.dumps({"detail": "LeIsaac WebRTC client is unavailable"}),
                status_code=502,
                media_type="application/json",
            )
        content = bytes(response.content)
        if (
            len(content) > 2 * 1024 * 1024
            or hashlib.sha256(content).hexdigest() != LEISAAC_CLIENT_JS_SHA256
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "LeIsaac WebRTC client failed integrity validation"}
                ),
                status_code=502,
                media_type="application/json",
            )
        return deps.response(
            content=content,
            status_code=200,
            media_type="text/javascript",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/leisaac/ws-session")
    async def leisaac_ws_session(request: Request, run_id: str = "") -> Any:
        """Issue a short-lived HttpOnly credential for nginx-auth-free WS upgrades."""

        if (
            not _same_origin_session_request(request.headers)
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "same-origin authenticated HTTPS is required"}
                ),
                status_code=403,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        client_address = _client_address(request.headers, request.client)
        if not client_address:
            return deps.response(
                content=json.dumps(
                    {"detail": "trusted public client address is required"}
                ),
                status_code=403,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        response = deps.response(
            content=b"",
            status_code=204,
            headers={"Cache-Control": "private, no-store"},
        )
        for audience, cookie in _WS_SESSION_COOKIES.items():
            response.set_cookie(
                cookie,
                _mint_ws_session(
                    ws_session_secret,
                    str(manifest["run_id"]),
                    client_address,
                    audience,
                ),
                max_age=_WS_SESSION_TTL_SECONDS,
                path=f"/api/leisaac/transport/{audience}",
                secure=True,
                httponly=True,
                samesite="strict",
            )
        return response

    @app.post(LEISAAC_CONTROL_DATACHANNEL_PATH.removeprefix("/api"))
    @app.post(LEISAAC_VIDEO_DATACHANNEL_PATH.removeprefix("/api"))
    async def leisaac_video_datachannel(request: Request) -> Any:
        """Negotiate one authenticated direct WebRTC data channel."""

        control_offer = str(request.url.path).endswith("/control-webrtc")

        if (
            not _same_origin_session_request(request.headers)
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "same-origin authenticated HTTPS is required"}
                ),
                status_code=403,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        try:
            content_length = int(request.headers.get("content-length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 131_072:
            return deps.response(
                content=json.dumps({"detail": "invalid WebRTC video offer size"}),
                status_code=400,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        try:
            payload = await request.json()
        except ValueError:
            payload = None
        run_id = str(payload.get("run_id") if isinstance(payload, dict) else "")
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        try:
            offer_sdp = parse_video_datachannel_offer(
                payload, expected_run_id=str(manifest["run_id"])
            )
        except VideoDataChannelError as exc:
            return deps.response(
                content=json.dumps({"detail": str(exc)}),
                status_code=400,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        health, reason = await asyncio.to_thread(_health, deps, manifest)
        if not health or str(health.get("stream_transport") or "") != "websocket-v1":
            return deps.response(
                content=json.dumps(
                    {"detail": reason or "preferred video transport is unavailable"}
                ),
                status_code=503,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        if deps.http_post is None:
            return deps.response(
                content=json.dumps({"detail": "WebRTC relay is unavailable"}),
                status_code=503,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )

        try:
            upstream = await asyncio.to_thread(
                deps.http_post,
                f"{manifest['service_url']}/transport/"
                + ("control-webrtc" if control_offer else "video-webrtc"),
                json={
                    "v": 1,
                    "run_id": str(manifest["run_id"]),
                    "type": "offer",
                    "sdp": offer_sdp,
                },
                headers={
                    "X-NPA-LeIsaac-Nonce": manifest["session_nonce"],
                    "X-NPA-LeIsaac-Run-ID": str(manifest["run_id"]),
                },
                timeout=10.0,
                follow_redirects=False,
            )
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac WebRTC offer relay failed", exc)
            upstream = None
        if upstream is None or int(upstream.status_code) != 200:
            return deps.response(
                content=json.dumps({"detail": "WebRTC video relay is unavailable"}),
                status_code=503,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        try:
            answer = upstream.json()
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac WebRTC answer was invalid", exc)
            answer = None
        if (
            not isinstance(answer, dict)
            or set(answer) != {"v", "type", "sdp"}
            or answer.get("v") != 1
            or answer.get("type") != "answer"
            or not isinstance(answer.get("sdp"), str)
            or not 1 <= len(answer["sdp"].encode("utf-8")) <= 65_536
            or "m=application" not in answer["sdp"]
            or "UDP/DTLS/SCTP" not in answer["sdp"]
        ):
            return deps.response(
                content=json.dumps({"detail": "WebRTC video relay is unavailable"}),
                status_code=503,
                media_type="application/json",
                headers={"Cache-Control": "private, no-store"},
            )
        return deps.response(
            content=json.dumps(answer),
            status_code=200,
            media_type="application/json",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/leisaac/frame.jpg")
    def leisaac_frame(
        request: Request, run_id: str = "", camera: str = "workspace"
    ) -> Any:
        if str(request.headers.get("x-forwarded-proto") or "").lower() != "https":
            return deps.response(
                content=json.dumps({"detail": "public HTTPS is required"}),
                status_code=400,
                media_type="application/json",
            )
        if camera not in {"workspace", "overview"}:
            return deps.response(
                content=json.dumps({"detail": "invalid LeIsaac camera"}),
                status_code=400,
                media_type="application/json",
            )
        manifest, reason = cached_resolve(run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        try:
            response = deps.http_get(
                f"{manifest['service_url']}/frame.jpg",
                params={"camera": camera},
                timeout=5.0,
                follow_redirects=False,
                headers={"X-NPA-LeIsaac-Nonce": manifest["session_nonce"]},
            )
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac fallback frame fetch failed", exc)
            response = None
        if response is None or int(response.status_code) != 200:
            return deps.response(
                content=json.dumps({"detail": "LeIsaac frame is unavailable"}),
                status_code=503,
                media_type="application/json",
            )
        content = bytes(response.content)
        if (
            not content.startswith(b"\xff\xd8")
            or not content.endswith(b"\xff\xd9")
            or len(content) > 4 * 1024 * 1024
        ):
            return deps.response(
                content=json.dumps({"detail": "LeIsaac frame failed validation"}),
                status_code=502,
                media_type="application/json",
            )
        response_headers = getattr(response, "headers", {})
        return deps.response(
            content=content,
            status_code=200,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                **{
                    header: str(response_headers.get(header))
                    for header in (
                        "X-NPA-Frame-Sequence",
                        "X-NPA-Frame-Capture-Wall-Ns",
                        "X-NPA-Frame-SHA256",
                        "X-NPA-Camera",
                    )
                    if response_headers.get(header)
                },
            },
        )

    @app.post(LEISAAC_VIEW_PATH.removeprefix("/api"))
    async def leisaac_view(request: Request, run_id: str = "") -> Any:
        if (
            str(request.headers.get("x-forwarded-proto") or "").lower() != "https"
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "authenticated HTTPS view control is required"}
                ),
                status_code=403,
                media_type="application/json",
            )
        try:
            content_length = int(request.headers.get("content-length") or "0")
        except ValueError:
            content_length = 0
        if not 1 <= content_length <= 4096:
            return deps.response(
                content=json.dumps({"detail": "invalid view command size"}),
                status_code=400,
                media_type="application/json",
            )
        payload: Any = None
        try:
            payload = await request.json()
            exact = isinstance(payload, dict) and set(payload) == {
                "camera",
                "sequence",
                "yaw_delta",
                "pitch_delta",
                "distance_delta",
            }
            sequence = int(payload["sequence"])
            values = [
                float(payload["yaw_delta"]),
                float(payload["pitch_delta"]),
                float(payload["distance_delta"]),
            ]
        except (ValueError, TypeError, KeyError, OverflowError):
            exact = False
            sequence = 0
            values = []
        if (
            not exact
            or not isinstance(payload, dict)
            or payload.get("camera") != "overview"
            or not 1 <= sequence <= 2**53 - 1
            or len(values) != 3
            or not all(math.isfinite(value) for value in values)
            or abs(values[0]) > 0.5
            or abs(values[1]) > 0.5
            or abs(values[2]) > 1.0
        ):
            return deps.response(
                content=json.dumps({"detail": "invalid view command"}),
                status_code=400,
                media_type="application/json",
            )
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        if deps.http_post is None:
            upstream = None
        else:
            try:
                upstream = await asyncio.to_thread(
                    deps.http_post,
                    f"{manifest['service_url']}/view",
                    json=payload,
                    headers={"X-NPA-LeIsaac-Nonce": manifest["session_nonce"]},
                    timeout=5.0,
                    follow_redirects=False,
                )
            except Exception as exc:
                _log_exception(
                    logging.WARNING, "LeIsaac view control request failed", exc
                )
                upstream = None
        if upstream is None or int(upstream.status_code) != 202:
            return deps.response(
                content=json.dumps({"detail": "LeIsaac view control is unavailable"}),
                status_code=503,
                media_type="application/json",
            )
        return deps.response(
            content=json.dumps({"accepted": True, "sequence": sequence}),
            status_code=202,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post("/leisaac/input")
    async def leisaac_input(request: Request, run_id: str = "") -> Any:
        if (
            str(request.headers.get("x-forwarded-proto") or "").lower() != "https"
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "authenticated HTTPS control is required"}
                ),
                status_code=403,
                media_type="application/json",
            )
        try:
            content_length = int(request.headers.get("content-length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_CONTROL_MESSAGE_BYTES:
            return deps.response(
                content=json.dumps({"detail": "invalid LeIsaac input size"}),
                status_code=400,
                media_type="application/json",
            )
        try:
            payload = await request.json()
        except ValueError:
            payload = None
        direct_action = (
            isinstance(payload, dict)
            and payload.get("v") == 1
            and payload.get("type") == "action"
        )
        key = str(payload.get("key") if isinstance(payload, dict) else "").upper()
        event = str(payload.get("event") if isinstance(payload, dict) else "")
        if not direct_action and (
            key
            not in {
                "W",
                "S",
                "A",
                "D",
                "Q",
                "E",
                "J",
                "L",
                "I",
                "K",
                "U",
                "O",
            }
            or event not in {"press", "release"}
        ):
            return deps.response(
                content=json.dumps({"detail": "invalid LeIsaac input"}),
                status_code=400,
                media_type="application/json",
            )
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        if isinstance(payload, dict) and payload.get("v") == 1:
            try:
                forwarded_payload = parse_control_message(
                    json.dumps(payload, separators=(",", ":")),
                    expected_run_id=str(manifest["run_id"]),
                )
            except TransportProtocolError as exc:
                return deps.response(
                    content=json.dumps(exc.payload()),
                    status_code=400,
                    media_type="application/json",
                )
        else:
            forwarded_payload = {"key": key, "event": event}
        if deps.http_post is None:
            upstream = None
        else:
            try:
                upstream = await asyncio.to_thread(
                    deps.http_post,
                    f"{manifest['service_url']}/input",
                    json=forwarded_payload,
                    headers={"X-NPA-LeIsaac-Nonce": manifest["session_nonce"]},
                    timeout=5.0,
                    follow_redirects=False,
                )
            except Exception as exc:
                _log_exception(
                    logging.WARNING, "LeIsaac fallback control request failed", exc
                )
                upstream = None
        if upstream is None or int(upstream.status_code) != 202:
            return deps.response(
                content=json.dumps({"detail": "LeIsaac control is unavailable"}),
                status_code=503,
                media_type="application/json",
            )
        try:
            acknowledgement = upstream.json()
        except Exception as exc:
            _log_exception(
                logging.DEBUG,
                "LeIsaac fallback control acknowledgement was invalid",
                exc,
            )
            acknowledgement = {
                "detail": "LeIsaac control returned an invalid acknowledgement"
            }
        return deps.response(
            content=json.dumps(acknowledgement),
            status_code=202,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post(LEISAAC_RECORDER_PATH.removeprefix("/api"))
    async def leisaac_recorder(request: Request, run_id: str = "") -> Any:
        if (
            str(request.headers.get("x-forwarded-proto") or "").lower() != "https"
            or request.headers.get("x-npa-leisaac-control") != "1"
        ):
            return deps.response(
                content=json.dumps(
                    {"detail": "authenticated HTTPS control is required"}
                ),
                status_code=403,
                media_type="application/json",
            )
        try:
            payload = await request.json()
        except ValueError:
            payload = None
        command = str(payload.get("command") if isinstance(payload, dict) else "")
        if command not in {"start", "mark-success", "mark-failure", "finalize"}:
            return deps.response(
                content=json.dumps({"detail": "invalid recorder command"}),
                status_code=400,
                media_type="application/json",
            )
        request_id = str(
            payload.get("request_id") if isinstance(payload, dict) else ""
        ) or secrets.token_hex(16)
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            return deps.response(
                content=json.dumps({"detail": "invalid recorder request ID"}),
                status_code=400,
                media_type="application/json",
            )
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return deps.response(
                content=json.dumps({"detail": reason}),
                status_code=404,
                media_type="application/json",
            )
        if deps.http_post is None:
            upstream = None
        else:
            try:
                upstream = await asyncio.to_thread(
                    deps.http_post,
                    f"{manifest['service_url']}/recorder/control",
                    json={"command": command, "request_id": request_id},
                    headers={"X-NPA-LeIsaac-Nonce": manifest["session_nonce"]},
                    timeout=10.0,
                    follow_redirects=False,
                )
            except Exception as exc:
                _log_exception(logging.WARNING, "LeIsaac recorder request failed", exc)
                upstream = None
        if upstream is None:
            status_code = 503
            content = {"detail": "LeIsaac recorder is unavailable"}
        else:
            status_code = int(upstream.status_code)
            try:
                content = upstream.json()
            except Exception as exc:
                _log_exception(
                    logging.DEBUG, "LeIsaac recorder response body was invalid", exc
                )
                content = {"detail": "LeIsaac recorder returned an invalid response"}
                status_code = 502
        if status_code not in {202, 400, 409, 503}:
            status_code = 502
            content = {"detail": "LeIsaac recorder returned an invalid status"}
        return deps.response(
            content=json.dumps(content),
            status_code=status_code,
            media_type="application/json",
            headers={"Cache-Control": "private, no-store"},
        )

    async def prepare_transport(
        websocket: WebSocket, subprotocol: str
    ) -> tuple[dict[str, Any] | None, str]:
        if not _same_origin_websocket(websocket, subprotocol):
            return None, "same-origin authenticated HTTPS WebSocket is required"
        if set(websocket.query_params.keys()) != {"run_id"}:
            return None, "only run_id is accepted"
        run_id = str(websocket.query_params.get("run_id") or "")
        audience = "control" if subprotocol == CONTROL_SUBPROTOCOL else "video"
        client_address = _client_address(websocket.headers, websocket.client)
        if not client_address or not _valid_ws_session(
            ws_session_secret,
            str(websocket.cookies.get(_WS_SESSION_COOKIES[audience]) or ""),
            run_id,
            client_address,
            audience,
            consumed_nonces=consumed_ws_nonces,
        ):
            return None, "valid short-lived transport session is required"
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            return None, reason
        health, reason = await asyncio.to_thread(_health, deps, manifest)
        if not health:
            return None, reason
        if str(health.get("stream_transport") or "") != "websocket-v1":
            return None, "preferred transport is unavailable for this session"
        return manifest, ""

    @app.websocket(LEISAAC_CONTROL_WS_PATH.removeprefix("/api"))
    async def leisaac_transport_control(websocket: WebSocket) -> None:
        manifest, reason = await prepare_transport(websocket, CONTROL_SUBPROTOCOL)
        if not manifest:
            LOG.warning("LeIsaac control transport rejected: %s", reason)
            await websocket.close(code=1008)
            return
        run_id = str(manifest["run_id"])
        try:
            async with deps.websocket_connect(
                _runtime_ws_uri(manifest, "/transport/control"),
                subprotocols=[CONTROL_SUBPROTOCOL],
                additional_headers={
                    "X-NPA-LeIsaac-Nonce": manifest["session_nonce"],
                    "X-NPA-LeIsaac-Run-ID": run_id,
                },
                open_timeout=5,
                close_timeout=2,
                max_size=MAX_CONTROL_MESSAGE_BYTES,
                max_queue=4,
                ping_interval=10,
                ping_timeout=10,
                compression=None,
            ) as upstream:
                if upstream.subprotocol != CONTROL_SUBPROTOCOL:
                    raise TransportProtocolError(
                        "subprotocol", "runtime rejected the control subprotocol"
                    )
                await websocket.accept(subprotocol=CONTROL_SUBPROTOCOL)
                transport_metrics.increment("control_connections")

                async def browser_to_runtime() -> None:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        raw = message.get("text")
                        if raw is None:
                            raise TransportProtocolError(
                                "invalid_message", "control messages must be text"
                            )
                        parsed = parse_control_message(raw, expected_run_id=run_id)
                        await upstream.send(json.dumps(parsed, separators=(",", ":")))

                async def runtime_to_browser() -> None:
                    async for raw in upstream:
                        if (
                            not isinstance(raw, str)
                            or len(raw.encode("utf-8")) > MAX_CONTROL_MESSAGE_BYTES
                        ):
                            raise TransportProtocolError(
                                "invalid_message",
                                "runtime control acknowledgement is invalid",
                            )
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise TransportProtocolError(
                                "invalid_message", "runtime acknowledgement is not JSON"
                            ) from exc
                        if (
                            not isinstance(payload, dict)
                            or str(payload.get("run_id") or run_id) != run_id
                        ):
                            raise TransportProtocolError(
                                "run_mismatch", "runtime acknowledgement run mismatch"
                            )
                        payload["agent_received_mono_ns"] = str(time.monotonic_ns())
                        payload["agent_send_mono_ns"] = str(time.monotonic_ns())
                        await asyncio.wait_for(
                            websocket.send_text(
                                json.dumps(payload, separators=(",", ":"))
                            ),
                            timeout=2.0,
                        )

                tasks = {
                    asyncio.create_task(browser_to_runtime()),
                    asyncio.create_task(runtime_to_browser()),
                }
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                results = await asyncio.gather(*done, *pending, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        raise result
        except TransportProtocolError as exc:
            transport_metrics.increment("control_errors")
            try:
                await websocket.send_text(
                    json.dumps(exc.payload(), separators=(",", ":"))
                )
            except Exception as send_exc:
                _log_exception(
                    logging.DEBUG,
                    "LeIsaac control protocol error could not be returned",
                    send_exc,
                )
            try:
                await websocket.close(code=1008)
            except Exception as close_exc:
                _log_exception(
                    logging.DEBUG,
                    "LeIsaac control WebSocket was already closed",
                    close_exc,
                )
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac control transport closed", exc)
            try:
                await websocket.close(code=1013)
            except Exception as close_exc:
                _log_exception(
                    logging.DEBUG,
                    "LeIsaac control WebSocket was already closed",
                    close_exc,
                )

    @app.websocket(LEISAAC_VIDEO_WS_PATH.removeprefix("/api"))
    async def leisaac_transport_video(websocket: WebSocket) -> None:
        manifest, reason = await prepare_transport(websocket, VIDEO_SUBPROTOCOL)
        if not manifest:
            LOG.warning("LeIsaac video transport rejected: %s", reason)
            await websocket.close(code=1008)
            return
        run_id = str(manifest["run_id"])
        latest = AsyncLatestByKey(("workspace", "overview"))
        try:
            async with deps.websocket_connect(
                _runtime_ws_uri(manifest, "/transport/video"),
                subprotocols=[VIDEO_SUBPROTOCOL],
                additional_headers={
                    "X-NPA-LeIsaac-Nonce": manifest["session_nonce"],
                    "X-NPA-LeIsaac-Run-ID": run_id,
                },
                open_timeout=5,
                close_timeout=2,
                max_size=MAX_FRAME_BYTES + 256,
                max_queue=2,
                ping_interval=10,
                ping_timeout=10,
                compression=None,
            ) as upstream:
                if upstream.subprotocol != VIDEO_SUBPROTOCOL:
                    raise TransportProtocolError(
                        "subprotocol", "runtime rejected the video subprotocol"
                    )
                await websocket.accept(subprotocol=VIDEO_SUBPROTOCOL)
                transport_metrics.increment("video_connections")

                async def read_runtime() -> None:
                    async for raw in upstream:
                        if (
                            not isinstance(raw, bytes)
                            or len(raw) > MAX_FRAME_BYTES + 256
                        ):
                            raise TransportProtocolError(
                                "invalid_frame", "runtime video message is invalid"
                            )
                        envelope, content = await asyncio.to_thread(unpack_frame, raw)
                        # Credit the runtime as soon as the authenticated relay has
                        # accepted a frame into its bounded latest-value queue.  A
                        # browser receipt must not stall capture for every viewer:
                        # slow clients are independently bounded and coalesced by
                        # ``latest`` below.  Browser ACKs remain validated and
                        # counted, but describe browser receipt rather than runtime
                        # flow control.
                        await upstream.send(
                            json.dumps(
                                {
                                    "v": 1,
                                    "type": "frame-ack",
                                    "run_id": run_id,
                                    "sequence": envelope.sequence,
                                },
                                separators=(",", ":"),
                            )
                        )
                        transport_metrics.increment("frames_relay_acked")
                        camera = "overview" if envelope.flags & 1 else "workspace"
                        await latest.publish(
                            camera, (envelope, content, time.monotonic_ns())
                        )

                async def acknowledge_runtime() -> None:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        raw = message.get("text")
                        if raw is None:
                            raise TransportProtocolError(
                                "invalid_message",
                                "video acknowledgements must be text",
                            )
                        parse_video_ack(raw, expected_run_id=run_id)
                        transport_metrics.increment("frames_browser_acked")

                async def send_browser() -> None:
                    generations: dict[str, int] = {}
                    next_camera_index = 0
                    while True:
                        (
                            camera,
                            generation,
                            item,
                            skipped,
                            next_camera_index,
                        ) = await latest.wait_after(
                            generations,
                            next_index=next_camera_index,
                            preferred_key="workspace",
                            timeout=20.0,
                        )
                        generations[camera] = generation
                        envelope, content, received_mono_ns = item
                        if skipped:
                            transport_metrics.increment("frames_coalesced", skipped)
                        stamped = stamp_verified_frame(
                            envelope,
                            content,
                            received_mono_ns=received_mono_ns,
                            send_mono_ns=time.monotonic_ns(),
                            additional_dropped=skipped,
                        )
                        try:
                            await asyncio.wait_for(
                                websocket.send_bytes(stamped), timeout=2.0
                            )
                        except asyncio.TimeoutError:
                            transport_metrics.increment("slow_client_disconnects")
                            raise
                        transport_metrics.increment("frames_sent")

                tasks = {
                    asyncio.create_task(read_runtime()),
                    asyncio.create_task(acknowledge_runtime()),
                    asyncio.create_task(send_browser()),
                }
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        raise result
        except asyncio.CancelledError:
            LOG.debug("LeIsaac video transport cancelled after relay cleanup")
            raise
        except Exception as exc:
            _log_exception(logging.WARNING, "LeIsaac video transport closed", exc)
            try:
                await websocket.close(code=1013)
            except Exception as close_exc:
                _log_exception(
                    logging.DEBUG,
                    "LeIsaac video WebSocket was already closed",
                    close_exc,
                )

    @app.websocket("/leisaac/signal")
    @app.websocket("/leisaac/signal/{signal_path:path}")
    async def leisaac_signal(websocket: WebSocket, signal_path: str = "") -> None:
        if not _same_https_origin(websocket.headers):
            LOG.warning(
                "LeIsaac signaling rejected: exact same-origin HTTPS is required"
            )
            await websocket.close(code=1008)
            return
        # Isaac Sim's 5.1 browser client opens its signaling WebSocket at
        # ``<configured-path>/sign_in``.  Keep the bare path for protocol
        # compatibility tests, but do not turn this into an arbitrary upstream
        # path proxy.
        if signal_path not in ("", "sign_in"):
            LOG.warning("LeIsaac signaling rejected: unsupported upstream path")
            await websocket.close(code=1008)
            return
        run_id = str(websocket.query_params.get("run_id") or "")
        # Storage discovery and the loopback health request are synchronous.
        # In agent-relay mode their response also traverses the backhaul route
        # on this ASGI event loop, so running either call inline can deadlock
        # the WebSocket that is needed to return its own response.
        manifest, reason = await asyncio.to_thread(cached_resolve, run_id)
        if not manifest:
            LOG.warning("LeIsaac signaling rejected: %s", reason)
            await websocket.close(code=1008)
            return
        health, reason = await asyncio.to_thread(_health, deps, manifest)
        if not health:
            LOG.warning("LeIsaac signaling rejected: %s", reason)
            await websocket.close(code=1013)
            return

        requested = str(websocket.headers.get("sec-websocket-protocol") or "")
        protocols = [item.strip() for item in requested.split(",") if item.strip()]
        protocols = [
            item for item in protocols if len(item) <= 128 and "\n" not in item
        ]
        query = str(websocket.url.query or "")
        if len(query) > 4096 or any(char in query for char in "\r\n"):
            await websocket.close(code=1008)
            return
        upstream_path = f"/{signal_path}" if signal_path else ""
        uri = f"ws://{manifest['signal_host']}:{LEISAAC_SIGNAL_PORT}{upstream_path}"
        if query:
            uri += f"?{query}"
        try:
            async with deps.websocket_connect(
                uri,
                subprotocols=protocols or None,
                open_timeout=5,
                close_timeout=2,
                max_size=None,
            ) as upstream:
                accepted = (
                    upstream.subprotocol if upstream.subprotocol in protocols else None
                )
                await websocket.accept(subprotocol=accepted)
                tasks = {
                    asyncio.create_task(
                        _relay_browser_to_upstream(websocket, upstream)
                    ),
                    asyncio.create_task(
                        _relay_upstream_to_browser(websocket, upstream)
                    ),
                }
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
        except Exception as exc:
            _log_exception(
                logging.WARNING, "LeIsaac signaling upstream connection failed", exc
            )
            try:
                await websocket.close(code=1011)
            except Exception as close_exc:
                _log_exception(
                    logging.DEBUG,
                    "LeIsaac browser WebSocket was already closed",
                    close_exc,
                )

    @app.websocket("/leisaac/backhaul")
    async def leisaac_backhaul(websocket: WebSocket) -> None:
        """Bridge the authenticated pod WSS backhaul to the loopback relay."""

        # nginx Basic auth is necessary but not sufficient. Reject browser-shaped,
        # unscoped, or ambiguously attributed upgrades before accepting them.
        protocols = {
            item.strip()
            for item in str(
                websocket.headers.get("sec-websocket-protocol") or ""
            ).split(",")
            if item.strip()
        }
        if (
            str(websocket.headers.get("x-forwarded-proto") or "").lower() != "https"
            or websocket.headers.get("origin") is not None
            or protocols != {_BACKHAUL_SUBPROTOCOL}
            or websocket.url.query
            or not _client_address(websocket.headers, websocket.client)
        ):
            LOG.warning("LeIsaac backhaul rejected by application-layer checks")
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol=_BACKHAUL_SUBPROTOCOL)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 48081)
        except OSError:
            await websocket.close(code=1013)
            return

        async def websocket_to_relay() -> None:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                payload = message.get("bytes")
                if (
                    payload is None
                    or len(payload) > _BACKHAUL_MAX_FRAME + _BACKHAUL_HEADER_SIZE
                ):
                    raise ValueError("invalid LeIsaac backhaul frame")
                writer.write(payload)
                await writer.drain()

        async def relay_to_websocket() -> None:
            while True:
                header = await reader.readexactly(_BACKHAUL_HEADER_SIZE)
                size = int.from_bytes(header[5:9], "big")
                if size > _BACKHAUL_MAX_FRAME:
                    raise ValueError("invalid LeIsaac backhaul frame")
                await websocket.send_bytes(header + await reader.readexactly(size))

        tasks = {
            asyncio.create_task(websocket_to_relay()),
            asyncio.create_task(relay_to_websocket()),
        }
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
        except asyncio.CancelledError:
            # A client can disappear while both relay directions are still blocked.
            # Treat that cancellation as the disconnect signal, but do not let the
            # ASGI task finish until both children and the loopback writer are settled.
            LOG.debug("LeIsaac backhaul session cancelled; cleaning up")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    _log_exception(
                        logging.WARNING, "LeIsaac backhaul relay task failed", result
                    )
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as exc:
                _log_exception(
                    logging.WARNING, "LeIsaac backhaul writer cleanup failed", exc
                )
