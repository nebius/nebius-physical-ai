"""Foxglove embedded-viewer helpers for the NPA agent backend.

Pure/deterministic helpers behind the agent's Foxglove viewer pane, embedded
verbatim into the agent-VM ``backend.py`` the same way as ``agent_rrd_proxy`` /
``agent_routing``. They cover:

- resolving the ``/api/foxglove/config`` payload from environment + installed
  SDK assets (never inventing a viewer that cannot load),
- building the ``DataSource`` objects documented by the Foxglove embedding SDK
  (https://docs.foxglove.dev/docs/embed/typescript-sdk),
- validating recordings before they are published on the unauthenticated,
  CORS-enabled ``/foxglove/data/`` path the cross-origin viewer iframe reads.

Trust model: ``/foxglove/data/`` mirrors the existing public ``/rerun/recordings/``
path — the embedded viewer runs on a different origin and cannot send the agent's
basic-auth credentials — so published names are random and unguessable, only
recognized recording formats are published, and old publications are pruned.

The module deliberately has **no** intra-agent imports so it unit-tests as plain
Python and stays safe to inline into the backend template.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

# Kept in sync with npa.workbench.foxglove by npa/tests/cli/test_agent_foxglove.py
# (this module cannot import it: it is inlined into the agent backend).
FOXGLOVE_SDK_FILES: tuple[str, ...] = (
    "index.js",
    "FoxgloveViewer.js",
    "types.js",
    "layout.generated.js",
)
FOXGLOVE_SDK_MANIFEST = "npa-sdk-manifest.json"
FOXGLOVE_DEFAULT_EMBED_SRC = "https://embed.foxglove.dev/"
FOXGLOVE_DEFAULT_LAYOUT_KEY = "npa-agent-foxglove"
FOXGLOVE_ARTIFACT_EXTENSIONS: tuple[str, ...] = (".mcap", ".bag", ".db3", ".ulg", ".ulog")
MCAP_MAGIC = b"\x89MCAP0\r\n"

# Viewer backends the pane can mount.
#   foxglove-sdk : the official Foxglove app embedded with @foxglove/embed
#                  (cross-origin iframe; users sign in to your Foxglove org)
#   self-hosted  : the OSS, Foxglove-compatible viewer this agent already runs
#                  in-page (Lichtblick) — renders MCAP with no account at all
FOXGLOVE_BACKEND_SDK = "foxglove-sdk"
FOXGLOVE_BACKEND_SELF_HOSTED = "self-hosted"
FOXGLOVE_BACKENDS = (FOXGLOVE_BACKEND_SDK, FOXGLOVE_BACKEND_SELF_HOSTED)

# Public (same-origin) URLs served by nginx on the agent VM.
FOXGLOVE_SDK_URL = "/foxglove/sdk/index.js"
FOXGLOVE_HOST_MODULE_URL = "/foxglove/app/npa-foxglove-host.js"
FOXGLOVE_DATA_URL_PREFIX = "/foxglove/data/"
# The OSS, Foxglove-compatible viewer this agent serves in-page (Lichtblick).
FOXGLOVE_SELF_HOSTED_BASE = "/lichtblick/"
# Same-origin path the agent publishes the run recording on for that viewer.
LICHTBLICK_RECORDING_PATH = "/lichtblick/recordings/sim2real.mcap"

# Live data-source protocols the embedded viewer understands.
FOXGLOVE_LIVE_PROTOCOLS: tuple[str, ...] = ("foxglove-websocket", "rosbridge-websocket")

_BLOCKED_HOSTNAMES = frozenset(
    {"localhost", "metadata", "metadata.google.internal", "metadata.internal"}
)
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_foxglove_artifact(key: str) -> bool:
    """Return True when ``key`` names a recording the Foxglove viewer can open."""
    suffix = Path(str(key or "").strip()).suffix.lower()
    return suffix in FOXGLOVE_ARTIFACT_EXTENSIONS


def looks_like_mcap(data: bytes | None) -> bool:
    """Return True when ``data`` starts with the MCAP magic record."""
    if not data:
        return False
    return bytes(data[: len(MCAP_MAGIC)]) == MCAP_MAGIC


def published_data_name(key: str, *, token: str = "") -> str:
    """Return an unguessable, traversal-safe basename for a published recording.

    The published name keeps the original extension (Foxglove picks its reader
    from it) and a short sanitized stem for operator recognition, prefixed with a
    random token because the path is served without authentication.
    """
    raw = str(key or "").strip()
    suffix = Path(raw).suffix.lower()
    if suffix not in FOXGLOVE_ARTIFACT_EXTENSIONS:
        suffix = ".mcap"
    stem = _UNSAFE_NAME_RE.sub("-", Path(raw).stem).strip("-._")[:48] or "recording"
    prefix = str(token or "").strip() or secrets.token_hex(8)
    prefix = _UNSAFE_NAME_RE.sub("", prefix)[:32] or secrets.token_hex(8)
    return f"{prefix}-{stem}{suffix}"


def prune_published(data_dir: str | Path, *, keep: int = 3) -> list[str]:
    """Delete all but the ``keep`` newest published recordings; return removals."""
    base = Path(str(data_dir or "")).expanduser()
    if not base.is_dir():
        return []
    entries = [
        path
        for path in base.iterdir()
        if path.is_file() and path.suffix.lower() in FOXGLOVE_ARTIFACT_EXTENSIONS
    ]
    entries.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    removed: list[str] = []
    for path in entries[max(0, int(keep)) :]:
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    return removed


def publish_recording(
    local_path: str | Path,
    key: str,
    *,
    data_dir: str | Path,
    keep: int = 3,
) -> str:
    """Publish a recording on the public data path; return its same-origin URL.

    Returns "" when the artifact is not a recognized recording (never publish an
    arbitrary file on the unauthenticated path) or the copy fails.
    """
    import shutil

    source = Path(str(local_path or "")).expanduser()
    if not is_foxglove_artifact(key) or not source.is_file():
        return ""
    if Path(str(key)).suffix.lower() == ".mcap":
        try:
            with source.open("rb") as handle:
                if not looks_like_mcap(handle.read(len(MCAP_MAGIC))):
                    return ""
        except OSError:
            return ""
    base = Path(str(data_dir or "")).expanduser()
    name = published_data_name(key)
    target = base / name
    try:
        base.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o644)
    except OSError:
        return ""
    prune_published(base, keep=keep)
    return f"{FOXGLOVE_DATA_URL_PREFIX}{name}" if target.is_file() else ""


def convert_run_request(payload: dict | None, sim_viz: dict | None) -> dict:
    """Normalize a ``/api/foxglove/convert-run`` body into safe parameters."""
    body = payload if isinstance(payload, dict) else {}
    state = sim_viz if isinstance(sim_viz, dict) else {}
    try:
        fps = float(body.get("fps") or 10.0)
    except (TypeError, ValueError):
        fps = 10.0
    try:
        max_frames = int(body.get("max_frames") or 0)
    except (TypeError, ValueError):
        max_frames = 0
    return {
        "run_id": str(body.get("run_id") or state.get("run_id") or "").strip(),
        "fps": fps if fps > 0 else 10.0,
        "max_frames": max(0, max_frames),
    }


def converted_recording_update(
    sim_viz: dict | None,
    *,
    run_id: str,
    name: str,
    summary: dict,
    now: str,
) -> dict:
    """Return the sim_viz update for a freshly converted MCAP recording."""
    state = dict(sim_viz) if isinstance(sim_viz, dict) else {}
    url = f"{FOXGLOVE_DATA_URL_PREFIX}{name}"
    state.update(
        {
            "run_id": run_id,
            "artifact_render": "mcap",
            "artifact_key": f"{run_id}/foxglove/{name}",
            "foxglove_url": url,
            "foxglove_ready": True,
            "mcap_updated_at": now,
            "artifact_preview_url": url,
            "artifact_download_url": url,
            "rrd_uri": "",
            "rerun_ready": False,
            "visualization_note": (
                f"Converted {summary.get('frames', 0)} frame(s), "
                f"{summary.get('metrics', 0)} metric doc(s) and "
                f"{summary.get('logs', 0)} log line(s) from run artifacts into MCAP. "
                f"Frame timestamps are synthetic ({summary.get('fps', 0)} fps) because the "
                "source artifacts carry no capture time."
            ),
        }
    )
    return state


def live_source_update(
    payload: dict | None, sim_viz: dict | None, *, now: str
) -> tuple[dict, dict] | None:
    """Validate a live-source request and return ``(source, updated_sim_viz)``.

    Returns None when the URL is not an allowed public ws/wss target. A live
    source replaces any published recording for the viewer session.
    """
    body = payload if isinstance(payload, dict) else {}
    source = live_data_source(
        str(body.get("url") or "").strip(), protocol=str(body.get("protocol") or "").strip()
    )
    if source is None:
        return None
    state = dict(sim_viz) if isinstance(sim_viz, dict) else {}
    state.update({"foxglove_url": "", "foxglove_ready": True, "mcap_updated_at": now})
    return source, state


def sdk_assets_state(assets_dir: str | Path) -> dict:
    """Return ``{ready, reason, version, integrity, source}`` for installed assets."""
    base = Path(str(assets_dir or "")).expanduser()
    state = {"ready": False, "reason": "", "version": "", "integrity": "", "source": ""}
    if not base.is_dir():
        state["reason"] = (
            "Foxglove SDK assets are not installed on this agent VM "
            f"({base}). Re-run `npa agent bootstrap`, or install them with "
            "`npa workbench foxglove install-sdk`."
        )
        return state
    missing = [name for name in FOXGLOVE_SDK_FILES if not (base / name).is_file()]
    if missing:
        state["reason"] = (
            f"Foxglove SDK assets are incomplete ({base}): missing {', '.join(missing)}."
        )
        return state
    manifest_path = base / FOXGLOVE_SDK_MANIFEST
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            state["version"] = str(payload.get("version") or "")
            state["integrity"] = str(payload.get("integrity") or "")
            state["source"] = str(payload.get("source") or "")
    state["ready"] = True
    return state


def live_url_allowed(url: str) -> bool:
    """Return True when ``url`` is a safe ``ws``/``wss`` live data-source target.

    Mirrors the ``.rrd`` proxy allowlist rules (no loopback / private / link-local
    / metadata targets) so a configured live URL cannot be pointed at agent-VM
    internals.

    Deliberately does **not** resolve DNS, so a public hostname that resolves to a
    private address is not rejected. That is a conscious trade-off, not an
    oversight: the *operator's browser* opens this WebSocket, not the agent, so
    this is a UI-config guardrail rather than an SSRF boundary (contrast
    ``agent_rrd_proxy``, where the agent itself fetches and therefore resolves).
    """
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    if parsed.scheme not in {"ws", "wss"}:
        return False
    host = str(parsed.hostname or "").strip().lower()
    if not host or host in _BLOCKED_HOSTNAMES or host.endswith(".internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_unspecified
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def live_data_source(url: str, *, protocol: str = "") -> dict | None:
    """Return a Foxglove ``live`` data source, or None when ``url`` is unusable."""
    if not live_url_allowed(url):
        return None
    chosen = str(protocol or "").strip()
    if chosen not in FOXGLOVE_LIVE_PROTOCOLS:
        chosen = FOXGLOVE_LIVE_PROTOCOLS[0]
    return {"type": "live", "protocol": chosen, "url": str(url).strip()}


def remote_file_data_source(
    urls: list[str] | tuple[str, ...],
    *,
    autoplay: bool = False,
    start_time: float | None = None,
) -> dict | None:
    """Return a Foxglove ``remote-file`` data source for one or more recordings."""
    cleaned = [str(item).strip() for item in (urls or []) if str(item).strip()]
    if not cleaned:
        return None
    source: dict = {"type": "remote-file", "urls": cleaned}
    if autoplay:
        source["autoplay"] = True
    if start_time is not None:
        source["startTime"] = start_time
    return source


def data_source_for_state(
    sim_viz: dict | None,
    *,
    origin: str = "",
    env: Mapping[str, str] | None = None,
) -> dict | None:
    """Return the data source for the agent's current viewer state.

    A published recording wins over a configured live URL: the operator loaded it
    explicitly. Returns None when neither is available (the UI then mounts an
    empty viewer rather than claiming data it does not have).
    """
    state = sim_viz if isinstance(sim_viz, dict) else {}
    published = str(state.get("foxglove_url") or "").strip()
    if published:
        urls = [_absolute(published, origin)]
        return remote_file_data_source(urls)
    # A live URL set for this session (POST /api/foxglove/live) wins over the
    # deploy-time default; session state survives a backend restart, the process
    # environment does not.
    session_live = str(state.get("foxglove_live_url") or "").strip()
    if session_live:
        return live_data_source(
            session_live, protocol=str(state.get("foxglove_live_protocol") or "")
        )
    environ = env if env is not None else os.environ
    return live_data_source(str(environ.get("NPA_FOXGLOVE_LIVE_URL", "")).strip())


def _absolute(url: str, origin: str) -> str:
    raw = str(url or "").strip()
    if not raw or "://" in raw:
        return raw
    base = str(origin or "").rstrip("/")
    if not base:
        return raw
    return f"{base}{raw}" if raw.startswith("/") else f"{base}/{raw}"


def _truthy(value: str, *, default: bool = True) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def resolve_foxglove_config(
    env: Mapping[str, str] | None = None,
    *,
    assets_dir: str | Path,
    origin: str = "",
    sim_viz: dict | None = None,
    self_hosted_ready: bool = False,
) -> dict:
    """Return the ``/api/foxglove/config`` payload.

    ``available`` is only True when the SDK assets are installed **and** an embed
    source is configured; otherwise ``reason`` explains what the operator must do.
    No credentials or environment secrets are ever echoed.
    """
    environ = env if env is not None else os.environ
    state = sim_viz if isinstance(sim_viz, dict) else {}
    assets = sdk_assets_state(assets_dir)
    enabled = _truthy(str(environ.get("NPA_FOXGLOVE_ENABLED", "")), default=True)
    # No implicit default: embedding the Foxglove-hosted app requires an account,
    # so it is only selected when an operator configured it (flag/env). The
    # documented default value is offered by the CLI help, not assumed here.
    embed_src = str(environ.get("NPA_FOXGLOVE_EMBED_SRC", "")).strip()
    org_slug = str(environ.get("NPA_FOXGLOVE_ORG_SLUG", "")).strip()
    color_scheme = str(environ.get("NPA_FOXGLOVE_COLOR_SCHEME", "")).strip().lower()
    if color_scheme not in {"light", "dark", "auto"}:
        color_scheme = "dark"
    layout_key = (
        str(environ.get("NPA_FOXGLOVE_LAYOUT_STORAGE_KEY", "")).strip()
        or FOXGLOVE_DEFAULT_LAYOUT_KEY
    )
    live_url = str(state.get("foxglove_live_url") or "").strip() or str(
        environ.get("NPA_FOXGLOVE_LIVE_URL", "")
    ).strip()
    if live_url and not live_url_allowed(live_url):
        live_url = ""

    backend, reason = "", ""
    if not enabled:
        reason = "Foxglove viewer is disabled on this agent (NPA_FOXGLOVE_ENABLED=0)."
    else:
        backend, reason = select_viewer_backend(
            environ,
            sdk_ready=bool(assets["ready"]),
            embed_src=embed_src,
            self_hosted_ready=bool(self_hosted_ready),
        )
        if backend == FOXGLOVE_BACKEND_SELF_HOSTED and not assets["ready"]:
            # Not an error: the OSS viewer renders without the SDK. Surface the
            # SDK gap so an operator who wanted the official app knows why.
            reason = ""

    data_source = data_source_for_state(state, origin=origin, env=environ)
    # The in-page OSS viewer must read a same-origin recording; the published
    # Lichtblick path is exactly that (the CORS copy is for the official app).
    self_hosted_recording = str(state.get("mcap_uri") and LICHTBLICK_RECORDING_PATH or "")
    self_hosted_url = (
        self_hosted_viewer_url(self_hosted_recording)
        if backend == FOXGLOVE_BACKEND_SELF_HOSTED
        else ""
    )
    payload = {
        "available": bool(backend) and not reason,
        "reason": reason,
        "viewer_backend": backend,
        "viewer_backends": list(FOXGLOVE_BACKENDS),
        "self_hosted_ready": bool(self_hosted_ready),
        "self_hosted_url": self_hosted_url,
        "enabled": enabled,
        "sdk_url": FOXGLOVE_SDK_URL,
        "host_module_url": FOXGLOVE_HOST_MODULE_URL,
        "sdk_version": assets["version"] or str(environ.get("NPA_FOXGLOVE_SDK_VERSION", "")).strip(),
        "sdk_integrity": assets["integrity"],
        "sdk_source": assets["source"],
        "sdk_ready": bool(assets["ready"]),
        "embed_src": embed_src,
        "org_slug": org_slug,
        "color_scheme": color_scheme,
        "layout_storage_key": layout_key,
        "live_url": live_url,
        "data_source": data_source,
        "run_id": str(state.get("run_id") or ""),
        "artifact_key": str(state.get("artifact_key") or ""),
        "artifact_uri": str(state.get("artifact_uri") or ""),
        "recording_url": _absolute(str(state.get("foxglove_url") or ""), origin),
        "updated_at": str(state.get("mcap_updated_at") or ""),
        "requires_account_note": (
            "The embedded viewer application is hosted by Foxglove (or your "
            "self-hosted deployment); users sign in there. NPA serves only the "
            "MIT-licensed @foxglove/embed SDK and the recording."
        ),
    }
    return payload


def select_viewer_backend(
    env: Mapping[str, str],
    *,
    sdk_ready: bool,
    embed_src: str,
    self_hosted_ready: bool,
) -> tuple[str, str]:
    """Choose the viewer backend and explain the choice.

    Order (honest, never a dead end):

    1. an operator override (``NPA_FOXGLOVE_VIEWER_BACKEND``) that is actually usable;
    2. the official Foxglove app when its SDK assets are installed and an embed
       source is configured;
    3. the self-hosted OSS viewer when it is healthy — so the pane renders the
       recording out of the box, with no Foxglove account;
    4. nothing, with a reason.
    """
    sdk_usable = bool(sdk_ready and _valid_embed_src(embed_src))
    requested = str(env.get("NPA_FOXGLOVE_VIEWER_BACKEND", "")).strip().lower()
    if requested == FOXGLOVE_BACKEND_SDK and sdk_usable:
        return FOXGLOVE_BACKEND_SDK, ""
    if requested == FOXGLOVE_BACKEND_SELF_HOSTED and self_hosted_ready:
        return FOXGLOVE_BACKEND_SELF_HOSTED, ""
    if sdk_usable:
        return FOXGLOVE_BACKEND_SDK, ""
    if self_hosted_ready:
        return FOXGLOVE_BACKEND_SELF_HOSTED, ""
    if not sdk_ready:
        return "", (
            "No MCAP viewer is available: the Foxglove SDK assets are not installed "
            "and the self-hosted viewer is not running."
        )
    return "", (
        "No Foxglove embed source is configured and the self-hosted viewer is not "
        "running. Set NPA_FOXGLOVE_EMBED_SRC (or `npa agent bootstrap "
        "--foxglove-embed-src ...`) to https://embed.foxglove.dev/ or your own "
        "Foxglove deployment, or start the self-hosted viewer sidecar."
    )


def self_hosted_viewer_url(recording_url: str, *, base: str = FOXGLOVE_SELF_HOSTED_BASE) -> str:
    """Return the self-hosted viewer URL that opens ``recording_url``.

    Same contract the Lichtblick pane uses: ``?ds=remote-file&ds.url=<mcap>``.
    The recording must be same-origin for the in-page viewer to read it.
    """
    from urllib.parse import quote

    root = str(base or FOXGLOVE_SELF_HOSTED_BASE)
    recording = str(recording_url or "").strip()
    if not recording:
        return root
    return f"{root}?ds=remote-file&ds.url={quote(recording, safe='')}"


def _valid_embed_src(src: str) -> bool:
    raw = str(src or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def foxglove_status_payload(config: dict, sim_viz: dict | None = None) -> dict:
    """Return the compact ``/api/foxglove/status`` payload for UI + chat grounding."""
    state = sim_viz if isinstance(sim_viz, dict) else {}
    source = config.get("data_source") if isinstance(config, dict) else None
    return {
        "available": bool(config.get("available")),
        "reason": str(config.get("reason") or ""),
        "viewer_backend": str(config.get("viewer_backend") or ""),
        "self_hosted_ready": bool(config.get("self_hosted_ready")),
        "self_hosted_url": str(config.get("self_hosted_url") or ""),
        "sdk_version": str(config.get("sdk_version") or ""),
        "embed_src": str(config.get("embed_src") or ""),
        "org_slug": str(config.get("org_slug") or ""),
        "foxglove_ready": bool(state.get("foxglove_ready")),
        "run_id": str(state.get("run_id") or ""),
        "artifact_key": str(state.get("artifact_key") or ""),
        "artifact_render": str(state.get("artifact_render") or ""),
        "recording_url": str(config.get("recording_url") or ""),
        "updated_at": str(state.get("mcap_updated_at") or ""),
        "data_source_type": str((source or {}).get("type") or ""),
        "data_source": source,
    }


def describe_foxglove_context(config: dict | None, sim_viz: dict | None = None) -> str:
    """Return text-only viewer context for the UI's "Describe this" action.

    The Foxglove viewer renders inside a cross-origin iframe, so the browser
    cannot capture its pixels. Instead of pretending to attach a frame, the agent
    describes exactly what is loaded and says the capture is unavailable.
    """
    cfg = config if isinstance(config, dict) else {}
    state = sim_viz if isinstance(sim_viz, dict) else {}
    source = cfg.get("data_source") or {}
    backend = str(cfg.get("viewer_backend") or "")
    capture_note = (
        "Foxglove viewer context (self-hosted, same-origin viewer: a frame capture "
        "is possible)."
        if backend == FOXGLOVE_BACKEND_SELF_HOSTED
        else "Foxglove viewer context (no pixel capture: the official embed is a "
        "cross-origin iframe, so the browser cannot read its canvas)."
    )
    lines = [
        capture_note,
        f"- viewer_backend: `{backend or 'none'}`",
        f"- available: `{bool(cfg.get('available'))}`",
    ]
    if cfg.get("reason"):
        lines.append(f"- reason: {cfg['reason']}")
    lines.extend(
        [
            f"- embed_src: `{cfg.get('embed_src') or '(unset)'}`",
            f"- sdk_version: `{cfg.get('sdk_version') or '(unknown)'}`",
            f"- data_source: `{source.get('type') or 'none'}`",
        ]
    )
    urls = source.get("urls") if isinstance(source, dict) else None
    if urls:
        lines.append(f"- recording: `{urls[0]}`")
    elif isinstance(source, dict) and source.get("url"):
        lines.append(f"- live: `{source['url']}`")
    if state.get("run_id"):
        lines.append(f"- run_id: `{state['run_id']}`")
    if state.get("artifact_key"):
        lines.append(f"- artifact_key: `{state['artifact_key']}`")
    return "\n".join(lines)


__all__ = [
    "self_hosted_viewer_url",
    "select_viewer_backend",
    "LICHTBLICK_RECORDING_PATH",
    "FOXGLOVE_SELF_HOSTED_BASE",
    "FOXGLOVE_BACKEND_SELF_HOSTED",
    "FOXGLOVE_BACKEND_SDK",
    "FOXGLOVE_BACKENDS",
    "FOXGLOVE_ARTIFACT_EXTENSIONS",
    "FOXGLOVE_DATA_URL_PREFIX",
    "FOXGLOVE_DEFAULT_EMBED_SRC",
    "FOXGLOVE_DEFAULT_LAYOUT_KEY",
    "FOXGLOVE_HOST_MODULE_URL",
    "FOXGLOVE_LIVE_PROTOCOLS",
    "FOXGLOVE_SDK_FILES",
    "FOXGLOVE_SDK_MANIFEST",
    "FOXGLOVE_SDK_URL",
    "MCAP_MAGIC",
    "convert_run_request",
    "converted_recording_update",
    "data_source_for_state",
    "describe_foxglove_context",
    "foxglove_status_payload",
    "is_foxglove_artifact",
    "live_data_source",
    "live_source_update",
    "live_url_allowed",
    "looks_like_mcap",
    "prune_published",
    "publish_recording",
    "published_data_name",
    "remote_file_data_source",
    "resolve_foxglove_config",
    "sdk_assets_state",
]
