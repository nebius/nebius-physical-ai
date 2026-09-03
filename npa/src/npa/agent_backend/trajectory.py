"""Goal-level NPA agent trajectory collection.

This module is shipped to the agent VM, so it intentionally depends only on the
standard library plus the runtime's boto3 installation. Product work must never
fail merely because telemetry cannot be uploaded: upload failures enter an
owner-only outbox and are retried at the next episode boundary.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

SCHEMA_VERSION = "npa.agent.trajectory.v1"
LOGGER = logging.getLogger(__name__)
_OUTBOX_LOCK = threading.RLock()
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|secret[_-]?key|password|authorization|credential|"
    r"(^|[_-])token($|[_-]))"
)
_SECRET_TEXT_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|password|token|authorization)"
        r"\s*[:=]\s*['\"]?[^\s'\"]+"
    ),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_S3_URI_RE = re.compile(r"s3://[^/\s]+(?:/[^\s?#]*)?(?:\?[^\s]*)?")
_HTTP_URI_RE = re.compile(r"https?://[^\s]+")


class AgentRunDataError(RuntimeError):
    """Trajectory collection configuration or integrity error."""


class CollectionStatus:
    COLLECTED = "collected"
    PENDING = "pending"
    DISABLED = "disabled"


@dataclass(frozen=True)
class DatasetConfig:
    tenant_id: str
    dataset_uri: str
    bucket: str
    prefix: str


def _outbox_dir() -> Path:
    return Path(
        os.environ.get("NPA_AGENT_DATASET_OUTBOX", "~/.npa/agent-dataset-outbox")
    ).expanduser()


def resolve_dataset_config(
    *, active_tenant_id: str = "", active_bucket: str = ""
) -> DatasetConfig | None:
    """Resolve and scope-check owner-provided collection configuration."""
    tenant_id = os.environ.get("NPA_AGENT_DATASET_TENANT_ID", "").strip()
    dataset_uri = os.environ.get("NPA_AGENT_DATASET_URI", "").strip()
    if not tenant_id and not dataset_uri:
        return None
    if not tenant_id or not dataset_uri:
        raise AgentRunDataError(
            "NPA_AGENT_DATASET_TENANT_ID and NPA_AGENT_DATASET_URI must be set together"
        )
    parsed = urlparse(dataset_uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise AgentRunDataError("NPA_AGENT_DATASET_URI must be an unsigned s3:// URI")
    if active_tenant_id and tenant_id != active_tenant_id.strip():
        raise AgentRunDataError("agent dataset tenant does not match the active deployment")
    if active_bucket and parsed.netloc != active_bucket.strip():
        raise AgentRunDataError("agent dataset bucket does not match the active deployment")
    return DatasetConfig(
        tenant_id=tenant_id,
        dataset_uri=dataset_uri,
        bucket=parsed.netloc,
        prefix=parsed.path.lstrip("/").rstrip("/"),
    )


def _storage_client() -> Any:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise AgentRunDataError("boto3 is unavailable for agent run data") from exc
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get(
        "NEBIUS_S3_ENDPOINT"
    )
    kwargs = {"endpoint_url": endpoint or None}
    return boto3.client("s3", **kwargs)


def _s3(storage: Any | None) -> Any:
    client = storage or _storage_client()
    return getattr(client, "s3", client)


def verify_destination(config: DatasetConfig, *, storage: Any | None = None) -> None:
    """Prove bucket access and write/read/delete behavior before enabling."""
    s3 = _s3(storage)
    probe_key = "/".join(
        item
        for item in (config.prefix, ".npa-probes", uuid.uuid4().hex)
        if item
    )
    body = b"npa-agent-dataset-probe"
    try:
        s3.head_bucket(Bucket=config.bucket)
        s3.put_object(Bucket=config.bucket, Key=probe_key, Body=body)
        fetched = s3.get_object(Bucket=config.bucket, Key=probe_key)["Body"].read()
        if fetched != body:
            raise AgentRunDataError("dataset probe read-after-write mismatch")
    except AgentRunDataError:
        raise
    except Exception as exc:
        raise AgentRunDataError("agent dataset destination is not writable") from exc
    finally:
        try:
            s3.delete_object(Bucket=config.bucket, Key=probe_key)
        except Exception:
            LOGGER.debug("agent dataset probe cleanup failed")


def _redact_string(value: str) -> str:
    text = value
    for pattern in _SECRET_TEXT_PATTERNS:
        text = pattern.sub("<redacted>", text)

    def redact_s3(match: re.Match[str]) -> str:
        parsed = urlparse(match.group(0))
        return urlunparse(("s3", "<bucket-ref>", parsed.path, "", "", ""))

    text = _S3_URI_RE.sub(redact_s3, text)

    def redact_http(match: re.Match[str]) -> str:
        parsed = urlparse(match.group(0))
        return urlunparse((parsed.scheme, "<host-ref>", parsed.path, "", "", ""))

    return _HTTP_URI_RE.sub(redact_http, text)


def redact(value: Any) -> Any:
    """Redact secrets, signed URLs, and concrete S3 bucket names recursively."""
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if _SECRET_KEY_RE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_identifiers(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_identifiers(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_identifiers(item, replacements) for item in value]
    if isinstance(value, str):
        text = value
        for identifier, replacement in replacements.items():
            if identifier:
                text = text.replace(identifier, replacement)
        return text
    return value


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _content_sha256(payload: dict[str, Any]) -> str:
    candidate = json.loads(_canonical_json(payload))
    candidate["collection"]["content_sha256"] = ""
    return hashlib.sha256(_canonical_json(candidate).encode()).hexdigest()


def _object_key(config: DatasetConfig, payload: dict[str, Any]) -> str:
    started = datetime.fromisoformat(
        str(payload["timing"]["started_at"]).replace("Z", "+00:00")
    )
    stem = f"{payload['episode_id']}-{payload['collection']['content_sha256']}.json"
    return "/".join(
        item
        for item in (
            config.prefix,
            "episodes",
            f"{started:%Y}",
            f"{started:%m}",
            f"{started:%d}",
            stem,
        )
        if item
    )


def _secure_outbox() -> Path:
    outbox = _outbox_dir()
    with _OUTBOX_LOCK:
        outbox.mkdir(mode=0o700, parents=True, exist_ok=True)
        if stat.S_ISLNK(outbox.lstat().st_mode):
            raise AgentRunDataError("agent dataset outbox must not be a symlink")
        outbox.chmod(0o700)
    return outbox


def _write_outbox(payload: dict[str, Any]) -> Path:
    """Atomically preserve a pending, twice-redacted immutable record."""
    pending = redact(payload)
    pending["collection"]["status"] = CollectionStatus.PENDING
    pending["collection"]["content_sha256"] = _content_sha256(pending)
    body = _canonical_json(redact(pending)).encode()
    outbox = _secure_outbox()
    path = outbox / (
        f"{pending['episode_id']}-{pending['collection']['content_sha256']}.json"
    )
    with _OUTBOX_LOCK:
        if path.exists():
            if path.is_symlink():
                raise AgentRunDataError("agent dataset outbox record must not be a symlink")
            if path.read_bytes() != body:
                raise AgentRunDataError("outbox key conflicts with different bytes")
            path.chmod(0o600)
            return path
        temporary = outbox / f".{path.name}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
    return path


def _put_and_verify(config: DatasetConfig, payload: dict[str, Any], s3: Any) -> str:
    uploaded = redact(payload)
    uploaded["collection"]["status"] = CollectionStatus.COLLECTED
    uploaded["collection"]["content_sha256"] = _content_sha256(uploaded)
    key = _object_key(config, uploaded)
    body = _canonical_json(redact(uploaded)).encode()
    try:
        existing = s3.get_object(Bucket=config.bucket, Key=key)["Body"].read()
    except Exception:
        existing = None
    if existing is not None:
        if existing != body:
            raise AgentRunDataError("immutable trajectory key contains different bytes")
        return key
    try:
        s3.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=body,
            IfNoneMatch="*",
        )
    except TypeError:  # simple injected fakes may not model conditional writes
        s3.put_object(Bucket=config.bucket, Key=key, Body=body)
    except Exception:
        # A concurrent identical writer may have won the conditional put.
        LOGGER.debug("conditional trajectory write did not create a new object")
    fetched = s3.get_object(Bucket=config.bucket, Key=key)["Body"].read()
    if fetched != body:
        raise AgentRunDataError("trajectory read-after-write mismatch")
    return key


def emit_trajectory(
    *,
    episode_id: str,
    session_id: str,
    request_content: str,
    intent: str,
    trajectory: list[dict[str, Any]],
    outcome: dict[str, Any],
    routing: dict[str, Any],
    versions: dict[str, Any],
    initial_state: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    storage: Any | None = None,
    active_tenant_id: str = "",
    active_bucket: str = "",
) -> tuple[str, str]:
    """Finalize exactly one episode and return collection status plus id."""
    config = resolve_dataset_config(
        active_tenant_id=active_tenant_id, active_bucket=active_bucket
    )
    if config is None:
        return CollectionStatus.DISABLED, episode_id
    now = datetime.now(timezone.utc)
    started = started_at or now.isoformat()
    ended = ended_at or now.isoformat()
    try:
        latency_ms = max(
            0,
            int(
                (
                    datetime.fromisoformat(ended.replace("Z", "+00:00"))
                    - datetime.fromisoformat(started.replace("Z", "+00:00"))
                ).total_seconds()
                * 1000
            ),
        )
    except ValueError:
        latency_ms = 0
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "session_id": session_id,
        "scope": {
            "tenant_id": config.tenant_id,
            "dataset_role": "agent-finetuning-raw",
        },
        "timing": {
            "started_at": started,
            "ended_at": ended,
            "latency_ms": latency_ms,
        },
        "request": {"content": request_content, "intent": intent},
        "initial_state": initial_state or {},
        "trajectory": trajectory,
        "outcome": outcome,
        "routing": routing,
        "versions": versions,
        "redaction": {"applied": True, "fields_removed": ["secret-shaped-values", "s3-bucket-names", "signed-url-queries"]},
        "collection": {"status": CollectionStatus.COLLECTED, "content_sha256": ""},
    }
    payload = _redact_identifiers(
        redact(payload),
        {config.tenant_id: "<tenant-ref>", config.bucket: "<bucket-ref>"},
    )
    # Preserve the configured tenant only in its access-controlled scope field.
    payload["scope"]["tenant_id"] = config.tenant_id
    payload["collection"]["content_sha256"] = _content_sha256(payload)
    try:
        verify_destination(config, storage=storage)
        _put_and_verify(config, payload, _s3(storage))
        return CollectionStatus.COLLECTED, episode_id
    except Exception:
        _write_outbox(payload)
        return CollectionStatus.PENDING, episode_id


def flush_outbox(
    *,
    storage: Any | None = None,
    active_tenant_id: str = "",
    active_bucket: str = "",
) -> list[str]:
    """Flush pending records only after exact read-after-write proof."""
    config = resolve_dataset_config(
        active_tenant_id=active_tenant_id, active_bucket=active_bucket
    )
    if config is None:
        return []
    outbox = _secure_outbox()
    if not outbox.exists():
        return []
    s3 = _s3(storage)
    verify_destination(config, storage=s3)
    flushed: list[str] = []
    for path in sorted(outbox.glob("*.json")):
        if path.is_symlink():
            continue
        try:
            payload = redact(json.loads(path.read_text(encoding="utf-8")))
            payload["scope"]["tenant_id"] = config.tenant_id
            _put_and_verify(config, payload, s3)
            path.unlink()
            flushed.append(str(payload["episode_id"]))
        except Exception:
            continue
    return flushed


def _request_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages") if isinstance(payload, dict) else []
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _episode_record(
    *,
    payload: dict[str, Any],
    response: dict[str, Any] | None,
    error: BaseException | None,
    episode_id: str,
    session_id: str,
    started_at: str,
) -> dict[str, Any]:
    ended_at = datetime.now(timezone.utc).isoformat()
    steps = response.get("steps", []) if isinstance(response, dict) else []
    events = [
        {
            "sequence": 0,
            "phase": "plan",
            "tool": "",
            "arguments": {},
            "observation": {"accepted": True},
            "status": "ok",
        }
    ]
    if isinstance(steps, list):
        for step in steps:
            events.append(
                {
                    "sequence": len(events),
                    "phase": "tool",
                    "tool": str(step.get("tool") or "") if isinstance(step, dict) else "",
                    "arguments": step.get("arguments", {}) if isinstance(step, dict) else {},
                    "observation": step.get("observation", {}) if isinstance(step, dict) else {},
                    "status": "ok" if isinstance(step, dict) and step.get("ok", True) else "error",
                }
            )
    apis_used = response.get("apis_used", []) if isinstance(response, dict) else []
    if isinstance(apis_used, list):
        for api in apis_used:
            events.append(
                {
                    "sequence": len(events),
                    "phase": "tool",
                    "tool": str(api),
                    "arguments": {},
                    "observation": {"reported_by_endpoint": True},
                    "status": "ok",
                }
            )
    stopped = str(response.get("stopped_reason") or "") if isinstance(response, dict) else ""
    refused = bool(response and (response.get("refused") or response.get("needs_confirmation")))
    cancelled = isinstance(error, asyncio.CancelledError) or stopped in {"cancelled", "canceled"}
    failed = error is not None or (isinstance(response, dict) and response.get("ok") is False)
    outcome_status = "cancelled" if cancelled else "refused" if refused else "failed" if failed else "succeeded"
    if refused and isinstance(response, dict):
        events.append(
            {
                "sequence": len(events),
                "phase": "confirm",
                "tool": "",
                "arguments": response.get("proposed_action") or {},
                "observation": {"needs_confirmation": True},
                "status": "rejected",
            }
        )
    events.append(
        {
            "sequence": len(events),
            "phase": "final",
            "tool": "",
            "arguments": {},
            "observation": {
                "response_ok": response.get("ok") if isinstance(response, dict) else False,
                "error_type": type(error).__name__ if error else "",
            },
            "status": "cancelled" if cancelled else "rejected" if refused else "error" if failed else "ok",
        }
    )
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    grounded = bool(response.get("grounded")) if isinstance(response, dict) else False
    routing: dict[str, Any] = {
        "grounded": grounded,
        "tier": str(response.get("tier") or "") if isinstance(response, dict) else "",
        "model": str(response.get("model") or "") if isinstance(response, dict) else "",
    }
    if grounded:
        routing.update({"input_tokens": 0, "output_tokens": 0})
    elif isinstance(usage, dict):
        for source, target in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            if source in usage:
                routing[target] = int(usage[source])
    verified = error is not None or (
        isinstance(response, dict) and "ok" in response
    )
    return {
        "episode_id": episode_id,
        "session_id": session_id,
        "request_content": _request_text(payload),
        "intent": str(response.get("semantic_intent") or response.get("mode") or "chat") if isinstance(response, dict) else "chat",
        "trajectory": events,
        "outcome": {
            "status": outcome_status,
            "verified": verified,
            "verified_by": (
                ["agent endpoint exception"]
                if error is not None
                else ["agent endpoint response"]
                if verified
                else []
            ),
            "artifact_uris": (
                response.get("artifact_uris", [])
                if isinstance(response, dict)
                and isinstance(response.get("artifact_uris"), list)
                else []
            ),
            "operator_interventions": [],
            "preference_pairs": [],
        },
        "routing": routing,
        "versions": {"agent": os.environ.get("NPA_AGENT_UI_VERSION", ""), "tools": {}},
        "initial_state": {},
        "started_at": started_at,
        "ended_at": ended_at,
    }


def goal_episode_boundary(
    *,
    active_tenant_id: Callable[[], str] = lambda: "",
    active_bucket: Callable[[], str] = lambda: "",
    storage_factory: Callable[[], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a synchronous goal endpoint with one terminal trajectory."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapped(payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            episode_id = f"episode-{uuid.uuid4().hex}"
            session_id = str(payload.get("session_id") or "default")
            started_at = datetime.now(timezone.utc).isoformat()
            storage = storage_factory() if storage_factory else None
            try:
                flush_outbox(
                    storage=storage,
                    active_tenant_id=active_tenant_id(),
                    active_bucket=active_bucket(),
                )
            except Exception:
                LOGGER.debug("pending trajectory flush failed at episode start")
            response: dict[str, Any] | None = None
            error: BaseException | None = None
            try:
                result = function(payload, *args, **kwargs)
                response = result if isinstance(result, dict) else {"ok": True}
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                record = _episode_record(
                    payload=payload,
                    response=response,
                    error=error,
                    episode_id=episode_id,
                    session_id=session_id,
                    started_at=started_at,
                )
                try:
                    emit_trajectory(
                        **record,
                        storage=storage,
                        active_tenant_id=active_tenant_id(),
                        active_bucket=active_bucket(),
                    )
                except Exception:
                    LOGGER.debug("terminal trajectory emission failed")

        return wrapped

    return decorate


__all__ = [
    "AgentRunDataError",
    "CollectionStatus",
    "DatasetConfig",
    "emit_trajectory",
    "flush_outbox",
    "goal_episode_boundary",
    "redact",
    "resolve_dataset_config",
    "verify_destination",
]
