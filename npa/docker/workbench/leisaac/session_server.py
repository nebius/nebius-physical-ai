#!/usr/bin/env python3
"""Start the real LeIsaac teleoperator and expose its browser-streaming assets.

Isaac Sim, task assets, and NVIDIA's WebRTC browser client are fetched only at
runtime after the shared default/opt-out EULA preflight. None of those bytes are
part of the distributable image.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

LOGGER = logging.getLogger("npa.leisaac.session")

try:
    from leisaac_dataset import resolve_s3_endpoint
except ImportError:  # Repository unit tests import the script directly.
    from npa.workbench.leisaac.dataset import resolve_s3_endpoint


class _RedactedException(RuntimeError):
    pass


def _log_exception(level: int, event: str, exc: BaseException) -> None:
    safe = _RedactedException(type(exc).__name__)
    LOGGER.log(
        level,
        "%s",
        event,
        extra={"exception_type": type(exc).__name__},
        exc_info=(type(safe), safe, exc.__traceback__),
    )


try:
    from leisaac_registry import (
        DEFAULT_TASK,
        REGISTRY_FINGERPRINT,
        RUNTIME_ASSETS,
        registry_payload,
        resolve_configuration,
        validate_environment_id,
        validate_environment_index,
        validate_num_envs,
        validate_seed,
        validate_task,
    )
except ImportError:  # Repository unit tests import the script directly.
    from npa.agent_backend.leisaac_registry import (
        DEFAULT_TASK,
        REGISTRY_FINGERPRINT,
        RUNTIME_ASSETS,
        registry_payload,
        resolve_configuration,
        validate_environment_id,
        validate_environment_index,
        validate_num_envs,
        validate_seed,
        validate_task,
    )

try:
    from leisaac_transport import (
        AsyncLatestByKey,
        AsyncFrameCreditWindow,
        CONTROL_SUBPROTOCOL,
        CLIENT_ID_PATTERN,
        ControlLedger,
        DEFAULT_RECORDING_CAMERA_MODE,
        DEFAULT_VIEW_MODE,
        FrameEnvelope,
        MAX_CLIENT_HISTORY,
        MAX_CONTROL_MESSAGE_BYTES,
        MAX_FRAME_BYTES,
        TransportMetrics,
        TransportProtocolError,
        VIDEO_SUBPROTOCOL,
        RECORDING_CAMERA_CONTRACT,
        VIEW_MODE_CONTRACT,
        pack_frame,
        parse_control_message,
        parse_video_ack,
    )

except ImportError:  # Repository unit tests import the script directly.
    from npa.agent_backend.leisaac_transport import (
        AsyncLatestByKey,
        AsyncFrameCreditWindow,
        CONTROL_SUBPROTOCOL,
        CLIENT_ID_PATTERN,
        ControlLedger,
        DEFAULT_RECORDING_CAMERA_MODE,
        DEFAULT_VIEW_MODE,
        FrameEnvelope,
        MAX_CLIENT_HISTORY,
        MAX_CONTROL_MESSAGE_BYTES,
        MAX_FRAME_BYTES,
        TransportMetrics,
        TransportProtocolError,
        VIDEO_SUBPROTOCOL,
        RECORDING_CAMERA_CONTRACT,
        VIEW_MODE_CONTRACT,
        pack_frame,
        parse_control_message,
        parse_video_ack,
    )

try:
    from leisaac_bundles import BundleError, BundleStore
except ImportError:  # Repository unit tests import the script directly.
    from npa.agent_backend.leisaac_bundles import BundleError, BundleStore

try:
    from leisaac_datachannel import (
        ControlDataChannelPeerPool,
        VideoDataChannelError,
        VideoDataChannelPeerPool,
        parse_video_datachannel_offer,
    )
except ImportError:  # Repository unit tests import the script directly.
    from npa.agent_backend.leisaac_datachannel import (
        ControlDataChannelPeerPool,
        VideoDataChannelError,
        VideoDataChannelPeerPool,
        parse_video_datachannel_offer,
    )

SCHEMA = "npa.leisaac.health.v2"
TASK = os.environ.get("NPA_LEISAAC_TASK", DEFAULT_TASK)
ENVIRONMENT_ID = os.environ.get("NPA_LEISAAC_ENVIRONMENT_ID", "operator-0")
ENVIRONMENT_INDEX = int(os.environ.get("NPA_LEISAAC_ENVIRONMENT_INDEX", "0"))
TELEOP_DEVICE = "keyboard"
TELEOP_SEED = int(os.environ.get("NPA_LEISAAC_SEED", "42"))
NUM_ENVS = int(os.environ.get("NPA_LEISAAC_NUM_ENVS", "1"))
SOURCE_COMMIT = "1651c321e9b0c1bb54233211fc7b3cd70d8373d5"
SOURCE_VERSION = "0.4.0"
ISAAC_SIM_VERSION = "5.1.0.0"
ISAAC_LAB_VERSION = "2.3.2.post1"
SIGNAL_PORT = 49100
MEDIA_PORT = 47998
SERVICE_PORT = 8080
FRAME_STALL_SECONDS = 30.0

_ASSET_BY_ID = {item["id"]: item for item in RUNTIME_ASSETS}
ROBOT_URL = _ASSET_BY_ID["so101_follower"]["url"]
ROBOT_SHA256 = _ASSET_BY_ID["so101_follower"]["sha256"]
KITCHEN_URL = _ASSET_BY_ID["kitchen_with_orange"]["url"]
KITCHEN_SHA256 = _ASSET_BY_ID["kitchen_with_orange"]["sha256"]
TABLE_URL = _ASSET_BY_ID["table_with_cube"]["url"]
TABLE_SHA256 = _ASSET_BY_ID["table_with_cube"]["sha256"]
CLIENT_URL = (
    "https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/"
    "@nvidia/omniverse-webrtc-streaming-library/-/"
    "@nvidia/omniverse-webrtc-streaming-library-5.6.0.tgz"
)
CLIENT_SHA512 = "37bd827a8194bfec2ccfbc656d10e42e83deebd682ac134095b2a8126901faa0966773752dd017353a1a5f7d1bc0b53be668d474ad5a14fd016c01df649f85dd"
CLIENT_SOURCE_JS_SHA256 = (
    "93cf2b328bcaaf9cf5a864c5b51f62e1bafcc533da9432ccc85633892f79ed86"
)
CLIENT_JS_SHA256 = CLIENT_SOURCE_JS_SHA256
UPSTREAM_OBSERVABILITY_PATCH_SHA256 = (
    "429d554f7d9145c2ffe5c0cdf3ff422c81f4d795004631cdd870c0e2f9df989a"
)
UPSTREAM_PACKAGING_PATCH_SHA256 = (
    "6940dce429c7d03d7576b4c56836a8a2cad2102a77658c1480acfcd27e8f5783"
)

CACHE_ROOT = Path(os.environ.get("NPA_LEISAAC_CACHE_DIR", "/opt/leisaac-cache"))
ASSETS_ROOT = CACHE_ROOT / "assets" / "runtime"
CLIENT_ROOT = CACHE_ROOT / "client" / "5.6.0"
PROVENANCE_PATH = CACHE_ROOT / "provenance.json"
READY_PATH = Path("/tmp/npa-leisaac-ready")
INPUT_COUNTER_PATH = Path("/tmp/npa-leisaac-input-events")
APPLIED_COUNTER_PATH = Path("/tmp/npa-leisaac-applied-inputs")
INPUT_QUEUE_PATH = Path("/tmp/npa-leisaac-input-queue.jsonl")
FRAME_PATH = Path("/tmp/npa-leisaac-frame.jpg")
FRAME_META_PATH = Path("/tmp/npa-leisaac-frame.json")
SECONDARY_FRAME_PATH = Path("/tmp/npa-leisaac-frame-overview.jpg")
SECONDARY_FRAME_META_PATH = Path("/tmp/npa-leisaac-frame-overview.json")
VIEW_COMMAND_PATH = Path("/tmp/npa-leisaac-view-command.json")
MODE_COMMAND_PATH = Path("/tmp/npa-leisaac-mode-command.json")
MODE_STATUS_PATH = Path("/tmp/npa-leisaac-mode-status.json")
HEARTBEAT_PATH = Path("/tmp/npa-leisaac-simulator-heartbeat")
CUSTOM_BUNDLE_ROOT = CACHE_ROOT / "custom"
CAMERA_PATHS = {
    "workspace": (FRAME_PATH, FRAME_META_PATH),
    "overview": (SECONDARY_FRAME_PATH, SECONDARY_FRAME_META_PATH),
}
APPLIED_ACK_PATH = Path("/tmp/npa-leisaac-input-applied.jsonl")
IPC_EVENT_PATH = Path("/tmp/npa-leisaac-events.sock")
IPC_FRAME_HEADER = struct.Struct("!4sI")
IPC_FRAME_MAGIC = b"NPF1"
IPC_EVENT_MAX_BYTES = 1024 * 1024
RECORDER_ROOT = Path("/tmp/npa-leisaac-recorder")
RECORDER_STATUS_PATH = RECORDER_ROOT / "status.json"
RECORDER_CONTROL_PATH = RECORDER_ROOT / "control.jsonl"
RECORDER_PENDING_PATH = RECORDER_ROOT / "pending-command.json"
STATE_LOCK = threading.Lock()
INPUT_LOCK = threading.Lock()
RECORDER_COMMAND_LOCK = threading.Lock()
APPLIED_ACK_LOCK = threading.Lock()
BUNDLE_APPLY_LOCK = threading.Lock()
MODE_COMMAND_LOCK = threading.Lock()
CONTROL_OWNER_LOCK = threading.Lock()
MODE_OWNER_LOCK = threading.Lock()
CONTROL_OWNER: dict[str, str | int] = {
    "token": "",
    "client_id": "",
    "lease_id": "",
    "lease_generation": 0,
    "leased_at_monotonic_ns": 0,
}
MODE_OWNER: dict[str, str] = {"client_id": ""}
BUNDLE_RESTART = threading.Event()
FORCE_SAFE_RESTART = threading.Event()
SERVER_STOP = threading.Event()
BUNDLE_SELECTION: dict[str, dict[str, Any]] = {}
STATE: dict[str, Any] = {
    "state": "starting",
    "detail": "staging runtime",
    "webrtc_ready": False,
    "pid": 0,
    "gpu": "",
    "started_at": "",
}
VIDEO_PATH: dict[str, str | bool] = {
    "hardware": False,
    "fallback_reason": "hardware encoder has not been verified",
}
CHILD: subprocess.Popen[str] | None = None
CONTROL_LEDGER = ControlLedger()
TRANSPORT_METRICS = TransportMetrics()
FRAME_LATEST = AsyncLatestByKey(("workspace", "overview"))
VIDEO_DATACHANNEL_PEERS = VideoDataChannelPeerPool()
CONTROL_DATACHANNEL_PEERS = ControlDataChannelPeerPool()
APPLIED_ACK_OFFSET = 0
APPLIED_GENERATION = 0
APPLIED_EVENT: asyncio.Event | None = None
RUNTIME_EVENT_QUEUE: asyncio.Queue[dict[str, Any]] | None = None


def _default_mode_state() -> dict[str, Any]:
    return {
        "mode_schema": "npa.leisaac.view-mode.v1",
        "requested_view_mode": DEFAULT_VIEW_MODE.value,
        "applied_view_mode": DEFAULT_VIEW_MODE.value,
        "requested_recording_camera_mode": DEFAULT_RECORDING_CAMERA_MODE.value,
        "applied_recording_camera_mode": DEFAULT_RECORDING_CAMERA_MODE.value,
        "view_revision": 0,
        "applied_view_revision": 0,
        "recording_revision": 0,
        "applied_recording_revision": 0,
        "mode_transition_latency_ms": 0.0,
        "camera_switches": 0,
        "capture_counts": {"workspace": 0, "overview": 0},
        "capture_fps": {"workspace": 0.0, "overview": 0.0},
        "capture_coalesced": 0,
        "queue_depths": {
            "active": 0,
            "priority": 0,
            "background": 0,
            "encode": 0,
        },
    }


def _mode_state() -> dict[str, Any]:
    try:
        payload = json.loads(MODE_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default_mode_state()
    if not isinstance(payload, dict):
        return _default_mode_state()
    normalized = _default_mode_state()
    normalized.update(payload)
    normalized["mode_schema"] = str(
        normalized.pop("schema", "") or "npa.leisaac.view-mode.v1"
    )
    return normalized


HEALTH_MODE_FIELDS = frozenset(_default_mode_state())


def _mode_request_state() -> dict[str, Any]:
    """Read the latest request, which may be newer than scheduler status."""

    try:
        payload = json.loads(MODE_COMMAND_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _mode_state()
    return payload if isinstance(payload, dict) else _mode_state()


def _retain_mode_owner(client_id: str) -> None:
    """Retain controller identity across simulator-child file resets."""

    with MODE_OWNER_LOCK:
        MODE_OWNER["client_id"] = str(client_id)


def _retained_mode_owner() -> str:
    with MODE_OWNER_LOCK:
        return str(MODE_OWNER.get("client_id") or "")


def _queue_mode_request(message: dict[str, Any]) -> bool:
    """Atomically publish the latest controller-owned scheduler request."""

    with MODE_COMMAND_LOCK:
        try:
            command = json.loads(MODE_COMMAND_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            command = _default_mode_state()
        if not isinstance(command, dict):
            command = _default_mode_state()
        revision = int(message["revision"])
        revision_key = (
            "view_revision" if message["type"] == "view-mode" else "recording_revision"
        )
        # Fallback requests use independent HTTP exchanges and can complete out
        # of order. Never let an older revision replace the latest scheduler
        # request; equal revisions remain idempotently replayable after the
        # disconnect safety reset.
        if revision < int(command.get(revision_key) or 0):
            return False
        if message["type"] == "view-mode":
            command["requested_view_mode"] = str(message["mode"])
            command["view_revision"] = revision
        else:
            command["requested_recording_camera_mode"] = str(message["mode"])
            command["recording_revision"] = revision
        command.update(
            schema="npa.leisaac.view-mode-command.v1",
            owner_client_id=str(message["client_id"]),
            requested_monotonic_ns=time.monotonic_ns(),
        )
        _write_json_atomic(MODE_COMMAND_PATH, command)
        # Bundle changes restart only the simulator child. _reset_runtime_files()
        # must replace its command/status files, but it must not erase the
        # surrounding authenticated session server's retained lease identity.
        _retain_mode_owner(str(message["client_id"]))
        return True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def enqueue_recorder_command(
    command: str, request_id: str
) -> tuple[int, dict[str, Any]]:
    """Validate and reserve exactly one asynchronous recorder transition."""

    if command not in {"start", "mark-success", "mark-failure", "finalize"}:
        return 400, {"detail": "invalid recorder command"}
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
        return 400, {"detail": "invalid recorder request ID"}
    with RECORDER_COMMAND_LOCK:
        try:
            status = json.loads(RECORDER_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 503, {"detail": "recorder unavailable"}
        if not isinstance(status, dict):
            return 503, {"detail": "recorder unavailable"}
        processed_commands = status.get("processed_commands", {})
        if not isinstance(processed_commands, dict):
            processed_commands = {}
        processed_command = processed_commands.get(request_id)
        if processed_command is None and status.get("last_command_id") == request_id:
            processed_command = status.get("last_command")
        if processed_command is not None:
            if processed_command != command:
                return 409, {"detail": "recorder request ID was reused"}
            RECORDER_PENDING_PATH.unlink(missing_ok=True)
            return 202, {
                "accepted": True,
                "duplicate": True,
                "processed": True,
                "request_id": request_id,
            }
        try:
            pending = json.loads(RECORDER_PENDING_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pending = None
        if isinstance(pending, dict):
            if (
                pending.get("request_id") == request_id
                and pending.get("command") == command
            ):
                return 202, {
                    "accepted": True,
                    "duplicate": True,
                    "processed": False,
                    "request_id": request_id,
                }
            return 409, {
                "detail": "another recorder transition is already in progress",
                "pending_command": str(pending.get("command") or ""),
            }
        state = str(status.get("state") or "")
        valid = {
            "start": state == "idle",
            "mark-success": state in {"recording", "outcome-pending"},
            "mark-failure": state in {"recording", "outcome-pending"},
            "finalize": state in {"outcome-pending", "upload-failed"},
        }
        if not valid[command]:
            return 409, {
                "detail": "invalid recorder transition",
                "state": state,
            }
        pending = {
            "schema": "npa.leisaac.recorder-command.v1",
            "request_id": request_id,
            "command": command,
            "state_before": state,
            "accepted_at": utc_now(),
        }
        try:
            _write_json_atomic(RECORDER_PENDING_PATH, pending)
            record = (
                json.dumps(
                    {"command": command, "request_id": request_id},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            with RECORDER_CONTROL_PATH.open("a", encoding="utf-8") as queue:
                queue.write(record)
                queue.flush()
                os.fsync(queue.fileno())
        except OSError:
            RECORDER_PENDING_PATH.unlink(missing_ok=True)
            return 503, {"detail": "recorder command queue is unavailable"}
        return 202, {
            "accepted": True,
            "duplicate": False,
            "processed": False,
            "request_id": request_id,
        }


def require_operator_eula() -> None:
    """Apply the shared ACCEPT_EULA default before any runtime asset download."""

    raw = os.environ.get("ACCEPT_EULA", "Y").strip()
    normalized = raw.upper()
    if normalized in {"Y", "YES", "1", "TRUE"}:
        os.environ["ACCEPT_EULA"] = "Y"
        return
    if normalized in {"", "N", "NO", "0", "FALSE"}:
        print(
            "LeIsaac refuses to download runtime assets because ACCEPT_EULA "
            "explicitly opts out; set ACCEPT_EULA=Y to continue.",
            file=sys.stderr,
        )
        raise SystemExit(78)
    print(
        f"Invalid ACCEPT_EULA value {raw!r}; expected Y, YES, 1, TRUE, N, NO, "
        "0, FALSE, or an empty string. Nothing has been downloaded.",
        file=sys.stderr,
    )
    raise SystemExit(78)


def validate_runtime_configuration() -> None:
    try:
        validate_task(TASK)
        validate_environment_id(ENVIRONMENT_ID)
        validate_environment_index(ENVIRONMENT_INDEX)
        validate_seed(TELEOP_SEED)
        validate_num_envs(NUM_ENVS)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if os.environ.get("NPA_LEISAAC_REGISTRY_FINGERPRINT") != REGISTRY_FINGERPRINT:
        raise RuntimeError("task registry fingerprint mismatch")
    output = os.environ.get("NPA_LEISAAC_OUTPUT_PATH", "")
    if not output.startswith("s3://"):
        raise RuntimeError(
            "NPA_LEISAAC_OUTPUT_PATH must be an operator-owned S3 prefix"
        )


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str, destination: Path, expected: str, algorithm: str = "sha256"
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hash_file(destination, algorithm) == expected:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "npa-leisaac/0.4.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:  # noqa: S310 - fixed URLs
        shutil.copyfileobj(response, output)
    observed = hash_file(temporary, algorithm)
    if observed != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"hash mismatch for {url}: expected {expected}, got {observed}"
        )
    temporary.replace(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe asset archive member: {member.filename}")
        bundle.extractall(destination)


def safe_extract_client(archive: Path, destination: Path) -> None:
    wanted = {
        "package/dist/omniverse-webrtc-streaming-library.umd.cjs": "index.js",
        "package/LICENSE.txt": "LICENSE.txt",
    }
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for source, target in wanted.items():
            member = members.get(source)
            if member is None or not member.isfile():
                raise RuntimeError(f"NVIDIA client archive is missing {source}")
            handle = bundle.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read {source}")
            with (destination / target).open("wb") as output:
                shutil.copyfileobj(handle, output)

    client_js = destination / "index.js"
    if hash_file(client_js) != CLIENT_SOURCE_JS_SHA256:
        raise RuntimeError("NVIDIA streaming client source hash mismatch")


def stage_runtime() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    downloads = CACHE_ROOT / "downloads"
    robot = downloads / "so101_follower.usd"
    kitchen = downloads / "kitchen_with_orange.zip"
    table = downloads / "table_with_cube.zip"
    client = downloads / "omniverse-webrtc-streaming-library-5.6.0.tgz"
    download_verified(ROBOT_URL, robot, ROBOT_SHA256)
    download_verified(KITCHEN_URL, kitchen, KITCHEN_SHA256)
    download_verified(TABLE_URL, table, TABLE_SHA256)
    download_verified(CLIENT_URL, client, CLIENT_SHA512, "sha512")

    robot_target = ASSETS_ROOT / "robots" / "so101_follower.usd"
    if not robot_target.is_file() or hash_file(robot_target) != ROBOT_SHA256:
        robot_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(robot, robot_target)
    scene = ASSETS_ROOT / "scenes" / "kitchen_with_orange" / "scene.usd"
    if not scene.is_file():
        scenes = ASSETS_ROOT / "scenes"
        scenes.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(kitchen, scenes)
    if not scene.is_file():
        raise RuntimeError(f"asset archive did not produce {scene}")
    table_scene = ASSETS_ROOT / "scenes" / "table_with_cube" / "scene.usd"
    if not table_scene.is_file():
        scenes = ASSETS_ROOT / "scenes"
        scenes.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(table, scenes)
    if not table_scene.is_file():
        raise RuntimeError(f"asset archive did not produce {table_scene}")

    client_js = CLIENT_ROOT / "index.js"
    if not client_js.is_file() or hash_file(client_js) != CLIENT_JS_SHA256:
        shutil.rmtree(CLIENT_ROOT, ignore_errors=True)
        safe_extract_client(client, CLIENT_ROOT)
    if hash_file(client_js) != CLIENT_JS_SHA256:
        raise RuntimeError(
            "NVIDIA streaming client JavaScript hash mismatch after extraction"
        )

    provenance = {
        "schema": "npa.leisaac.provenance.v1",
        "staged_at": utc_now(),
        "leisaac": {
            "repository": "https://github.com/LightwheelAI/leisaac",
            "version": SOURCE_VERSION,
            "commit": SOURCE_COMMIT,
            "license": "Apache-2.0",
            "npa_observability_patch": {
                "path": "upstream-observability.patch",
                "sha256": UPSTREAM_OBSERVABILITY_PATCH_SHA256,
            },
            "npa_packaging_patch": {
                "path": "upstream-packaging.patch",
                "sha256": UPSTREAM_PACKAGING_PATCH_SHA256,
            },
        },
        "assets": [
            {"url": ROBOT_URL, "sha256": ROBOT_SHA256, "bytes": robot.stat().st_size},
            {
                "url": KITCHEN_URL,
                "sha256": KITCHEN_SHA256,
                "bytes": kitchen.stat().st_size,
            },
            {
                "url": TABLE_URL,
                "sha256": TABLE_SHA256,
                "bytes": table.stat().st_size,
            },
        ],
        "browser_client": {
            "url": CLIENT_URL,
            "version": "5.6.0",
            "sha512": CLIENT_SHA512,
            "source_index_js_sha256": CLIENT_SOURCE_JS_SHA256,
            "index_js_sha256": CLIENT_JS_SHA256,
            "transport_adapter": (
                "NPA-owned browser adapter selects WSS for same-origin port 443; "
                "vendor bytes remain pristine"
            ),
            "license": "NVIDIA proprietary; operator-fetched at runtime",
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def detect_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout.splitlines() or [""])[0].strip()
    except OSError:
        return ""


def update_state(**values: Any) -> None:
    with STATE_LOCK:
        STATE.update(values)


def apply_bundle_selection(selection: Any) -> dict[str, dict[str, Any]]:
    """Materialize immutable S3 bundles and request a safe simulator restart."""

    if (
        not isinstance(selection, dict)
        or len(selection) > 3
        or any(kind not in {"robot", "scene", "device"} for kind in selection)
        or any(
            not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
            for digest in selection.values()
        )
    ):
        raise BundleError("bundle selection is invalid")
    try:
        recorder_status = json.loads(RECORDER_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorder_status = {}
    if recorder_status.get("state") != "idle":
        raise BundleError(
            "finish or discard the active recording before applying bundles",
            status_code=409,
        )
    materialized: dict[str, dict[str, Any]] = {}
    if selection:
        output_uri = os.environ.get("NPA_LEISAAC_OUTPUT_PATH", "")
        parsed = urlparse(output_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise BundleError("bundle storage is unavailable", status_code=503)
        try:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=resolve_s3_endpoint(),
                region_name=os.environ.get("AWS_REGION") or "eu-north1",
            )
            store = BundleStore(client, output_uri, allowed_buckets=[parsed.netloc])
            for kind, digest in sorted(selection.items()):
                bundle = store.materialize(digest, CUSTOM_BUNDLE_ROOT / digest)
                if bundle.get("kind") != kind:
                    raise BundleError("bundle selection kind does not match")
                materialized[kind] = bundle
        except BundleError:
            raise
        except Exception as exc:
            raise BundleError("bundle storage is unavailable", status_code=503) from exc
    public_selection = {
        kind: {
            "bundle_sha256": item["bundle_sha256"],
            "name": item["name"],
            "entrypoint": item["entrypoint"],
        }
        for kind, item in materialized.items()
    }
    with BUNDLE_APPLY_LOCK:
        BUNDLE_SELECTION.clear()
        BUNDLE_SELECTION.update(materialized)
    # Publish the restart event and unavailable health state atomically. The
    # child monitor must never overwrite this with a final `ready` update from
    # the old simulator iteration after a bundle apply has been accepted.
    with STATE_LOCK:
        STATE.update(
            state="restarting",
            detail=(
                "applying checksum-verified custom bundles"
                if public_selection
                else "restoring built-in defaults"
            ),
            selected_bundles=public_selection,
            webrtc_ready=False,
            stream_ready=False,
        )
        BUNDLE_RESTART.set()
    return public_selection


def _selected_bundle_environment() -> dict[str, str]:
    with BUNDLE_APPLY_LOCK:
        selection = {kind: dict(value) for kind, value in BUNDLE_SELECTION.items()}
    public_selection = {
        kind: {
            "bundle_sha256": item["bundle_sha256"],
            "name": item["name"],
            "entrypoint": item["entrypoint"],
        }
        for kind, item in selection.items()
    }
    configuration = resolve_configuration(TASK, public_selection)
    environment: dict[str, str] = {
        "NPA_LEISAAC_ROBOT": str(configuration["robot"]["id"]),
        "NPA_LEISAAC_SCENE": str(configuration["scene"]["id"]),
        "NPA_LEISAAC_DEVICE": str(configuration["device"]["id"]),
    }
    for kind in ("robot", "scene"):
        item = selection.get(kind)
        if item:
            environment[f"NPA_LEISAAC_CUSTOM_{kind.upper()}_USD"] = str(
                item["entrypoint_path"]
            )
            environment[f"NPA_LEISAAC_{kind.upper()}"] = str(item["name"])
    device = selection.get("device")
    if device:
        environment["NPA_LEISAAC_DEVICE"] = str(device["name"])
        environment["NPA_LEISAAC_DEVICE_DESCRIPTOR"] = str(device["entrypoint_path"])
    digests = {kind: item["bundle_sha256"] for kind, item in sorted(selection.items())}
    environment["NPA_LEISAAC_BUNDLE"] = (
        json.dumps(digests, sort_keys=True, separators=(",", ":"))
        if digests
        else "built-in"
    )
    return environment


def _simulation_launch() -> tuple[list[str], dict[str, str]]:
    media_host = os.environ.get("NPA_LEISAAC_MEDIA_HOST", "").strip()
    if not media_host:
        raise RuntimeError("NPA_LEISAAC_MEDIA_HOST is required")
    kit_arguments = [
        "--no-window",
        "--/renderer/multiGpu/enabled=False",
    ]
    if bool(VIDEO_PATH["hardware"]):
        kit_arguments.extend(
            [
                "--enable omni.kit.livestream.webrtc",
                "--/app/livestream/webrtc/logQosStatus=true",
                f"--/app/livestream/publicEndpointAddress={media_host}",
                f"--/app/livestream/publicEndpointPort={MEDIA_PORT}",
                f"--/app/livestream/fixedHostPort={MEDIA_PORT}",
                f"--/app/livestream/minHostPort={MEDIA_PORT}",
                f"--/app/livestream/maxHostPort={MEDIA_PORT}",
                f"--/app/livestream/port={SIGNAL_PORT}",
            ]
        )
    command = [
        "/isaac-sim/python.sh",
        "/opt/leisaac/scripts/environments/teleoperation/teleop_se3_agent.py",
        f"--task={TASK}",
        f"--teleop_device={TELEOP_DEVICE}",
        f"--num_envs={NUM_ENVS}",
        f"--seed={TELEOP_SEED}",
        "--device=cuda:0",
        "--enable_cameras",
        "--kit_args=" + " ".join(kit_arguments),
    ]
    environment = os.environ.copy()
    module_root = "/opt/npa/leisaac"
    inherited_pythonpath = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = (
        f"{module_root}:{inherited_pythonpath}" if inherited_pythonpath else module_root
    )
    environment.update(_selected_bundle_environment())
    environment.update(
        {
            "LEISAAC_ASSETS_ROOT": str(ASSETS_ROOT),
            "NPA_LEISAAC_SOURCE_COMMIT": SOURCE_COMMIT,
            "NPA_LEISAAC_SOURCE_VERSION": SOURCE_VERSION,
            "NPA_LEISAAC_ISAAC_SIM_VERSION": ISAAC_SIM_VERSION,
            "NPA_LEISAAC_ISAAC_LAB_VERSION": ISAAC_LAB_VERSION,
            "NPA_LEISAAC_IMAGE": os.environ.get("NPA_LEISAAC_IMAGE", ""),
            "NPA_LEISAAC_REGISTRY_FINGERPRINT": REGISTRY_FINGERPRINT,
            "NPA_LEISAAC_READY_PATH": str(READY_PATH),
            "NPA_LEISAAC_INPUT_COUNTER": str(INPUT_COUNTER_PATH),
            "NPA_LEISAAC_APPLIED_COUNTER": str(APPLIED_COUNTER_PATH),
            "NPA_LEISAAC_INPUT_QUEUE": str(INPUT_QUEUE_PATH),
            "NPA_LEISAAC_APPLIED_ACK_PATH": str(APPLIED_ACK_PATH),
            "NPA_LEISAAC_IPC_EVENT_PATH": str(IPC_EVENT_PATH),
            "NPA_LEISAAC_FRAME_PATH": str(FRAME_PATH),
            "NPA_LEISAAC_FRAME_META_PATH": str(FRAME_META_PATH),
            "NPA_LEISAAC_SECONDARY_FRAME_PATH": str(SECONDARY_FRAME_PATH),
            "NPA_LEISAAC_SECONDARY_FRAME_META_PATH": str(SECONDARY_FRAME_META_PATH),
            "NPA_LEISAAC_VIEW_COMMAND_PATH": str(VIEW_COMMAND_PATH),
            "NPA_LEISAAC_MODE_COMMAND_PATH": str(MODE_COMMAND_PATH),
            "NPA_LEISAAC_MODE_STATUS_PATH": str(MODE_STATUS_PATH),
            "NPA_LEISAAC_HEARTBEAT_PATH": str(HEARTBEAT_PATH),
            "NPA_LEISAAC_NATIVE_VIDEO": "1" if VIDEO_PATH["hardware"] else "0",
            "NPA_LEISAAC_RECORDER_ROOT": str(RECORDER_ROOT),
            "NPA_LEISAAC_CUSTOM_ROOT": str(CUSTOM_BUNDLE_ROOT),
        }
    )
    environment["NPA_LEISAAC_BROWSER_TELEOP"] = "1"
    return command, environment


def _reset_runtime_files() -> None:
    global APPLIED_ACK_OFFSET
    # A bundle selection restarts only the simulator child. Preserve the
    # surrounding authenticated session's latest revisioned mode command so
    # the replacement child converges without depending on a browser polling
    # cycle or a preferred-transport reconnect. Applied state remains at the
    # safe defaults until the replacement scheduler acknowledges the command.
    with MODE_COMMAND_LOCK:
        try:
            prior_mode = json.loads(MODE_COMMAND_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior_mode = {}
        restart_mode = _default_mode_state()
        if isinstance(prior_mode, dict) and not FORCE_SAFE_RESTART.is_set():
            requested_view = str(prior_mode.get("requested_view_mode") or "")
            requested_recording = str(
                prior_mode.get("requested_recording_camera_mode") or ""
            )
            if requested_view in VIEW_MODE_CONTRACT["values"]:
                restart_mode["requested_view_mode"] = requested_view
                restart_mode["view_revision"] = max(
                    0, int(prior_mode.get("view_revision") or 0)
                )
            if requested_recording in RECORDING_CAMERA_CONTRACT["values"]:
                restart_mode["requested_recording_camera_mode"] = requested_recording
                restart_mode["recording_revision"] = max(
                    0, int(prior_mode.get("recording_revision") or 0)
                )
            owner_client_id = str(prior_mode.get("owner_client_id") or "")
            if owner_client_id:
                restart_mode.update(
                    schema="npa.leisaac.view-mode-command.v1",
                    owner_client_id=owner_client_id,
                    requested_monotonic_ns=int(
                        prior_mode.get("requested_monotonic_ns") or 0
                    ),
                )
    READY_PATH.unlink(missing_ok=True)
    INPUT_COUNTER_PATH.write_text("0\n", encoding="utf-8")
    APPLIED_COUNTER_PATH.write_text("0\n", encoding="utf-8")
    INPUT_QUEUE_PATH.write_text("", encoding="utf-8")
    # The simulator rewrites its acknowledgement stream on every restart. Keep
    # the reader offset in the same critical section as truncation; otherwise a
    # reconnect can apply controls successfully while their sequence-specific
    # acknowledgements remain invisible behind the previous file's byte offset.
    with APPLIED_ACK_LOCK:
        APPLIED_ACK_PATH.write_text("", encoding="utf-8")
        APPLIED_ACK_OFFSET = 0
    for path in (
        FRAME_PATH,
        FRAME_META_PATH,
        SECONDARY_FRAME_PATH,
        SECONDARY_FRAME_META_PATH,
        VIEW_COMMAND_PATH,
        MODE_STATUS_PATH,
        HEARTBEAT_PATH,
    ):
        path.unlink(missing_ok=True)
    with MODE_COMMAND_LOCK:
        _write_json_atomic(MODE_COMMAND_PATH, restart_mode)
    FORCE_SAFE_RESTART.clear()
    _write_json_atomic(MODE_STATUS_PATH, _default_mode_state())
    shutil.rmtree(RECORDER_ROOT, ignore_errors=True)
    RECORDER_ROOT.mkdir(parents=True, exist_ok=True)


def _frame_stream_stalled(now: float | None = None) -> bool:
    """Detect a live child whose render/control loop stopped."""

    try:
        oldest = HEARTBEAT_PATH.stat().st_mtime
    except OSError:
        try:
            oldest = FRAME_PATH.stat().st_mtime
        except OSError:
            return False
    return (time.time() if now is None else now) - oldest > FRAME_STALL_SECONDS


def _runtime_stream_stalled(now: float | None = None) -> bool:
    """Apply freshness checks only to transports that publish frame files.

    Kit's native WebRTC renderer can stop advancing the simulator heartbeat
    while it has no consumer.  Treating that expected idle state as a JPEG
    publisher failure makes the supervisor withdraw readiness, which prevents
    the browser from attaching and resuming the renderer.  The native path is
    instead guarded by child-process and Kubernetes liveness; the file-age
    watchdog remains mandatory for the JPEG fallback it was designed for.
    """

    if VIDEO_PATH["hardware"]:
        return False
    return _frame_stream_stalled(now)


def _prepare_stall_recovery() -> int:
    """Revoke runtime-bound state before replacing a non-responsive simulator."""

    released = CONTROL_LEDGER.reset_for_runtime_restart()
    with CONTROL_OWNER_LOCK:
        CONTROL_OWNER.update(
            token="",
            client_id="",
            lease_id="",
            lease_generation=int(CONTROL_OWNER.get("lease_generation") or 0) + 1,
            leased_at_monotonic_ns=0,
        )
    with MODE_OWNER_LOCK:
        MODE_OWNER["client_id"] = ""
    FORCE_SAFE_RESTART.set()
    TRANSPORT_METRICS.increment("forced_safe_restarts")
    TRANSPORT_METRICS.increment("forced_releases", released)
    return released


def _mark_runtime_ready() -> bool:
    """Publish readiness only while no accepted restart is pending."""

    with STATE_LOCK:
        if BUNDLE_RESTART.is_set():
            return False
        hardware = bool(VIDEO_PATH["hardware"])
        STATE.update(
            state="ready",
            detail="live",
            webrtc_ready=True,
            stream_ready=True,
            stream_transport="webrtc" if hardware else "websocket-v1",
            requested_video_transport="webrtc-kit-h264",
            active_video_transport=(
                "webrtc-kit-h264" if hardware else "jpeg-websocket"
            ),
            video_codec="H264" if hardware else "JPEG",
            hardware_acceleration="runtime-nvenc" if hardware else "none",
            video_fallback_reason=(
                "" if hardware else str(VIDEO_PATH["fallback_reason"])
            ),
        )
        return True


def verify_runtime_nvenc() -> tuple[bool, str]:
    """Exercise runtime-injected NVENC before advertising the native path."""

    if not any(
        path.name.removeprefix("nvidia").isdigit()
        for path in Path("/dev").glob("nvidia*")
    ):
        return False, "NVIDIA device nodes are unavailable"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        # Blackwell NVENC rejects 64x64 even though older generations accept
        # it. Exercise a resolution above the cross-generation minimum so the
        # capability probe reflects real teleoperation frames.
        "color=size=256x256:rate=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-bf",
        "0",
        "-f",
        "h264",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"NVENC probe failed ({type(exc).__name__})"
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").splitlines()
        return False, (detail[-1][:160] if detail else "NVENC probe was rejected")
    return True, ""


def run_simulation() -> None:
    global CHILD
    try:
        update_state(detail="fetching operator-licensed Isaac runtime")
        subprocess.run(["/opt/npa/bin/isaac-bootstrap", "ensure"], check=True)
        hardware, fallback_reason = verify_runtime_nvenc()
        VIDEO_PATH.update(hardware=hardware, fallback_reason=fallback_reason)
        while not SERVER_STOP.is_set():
            BUNDLE_RESTART.clear()
            update_state(
                state="starting",
                detail=f"starting {TASK}",
                webrtc_ready=False,
                stream_ready=False,
            )
            command, environment = _simulation_launch()
            _reset_runtime_files()
            CHILD = subprocess.Popen(
                command,
                cwd="/opt/leisaac",
                env=environment,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                text=True,
            )
            update_state(pid=CHILD.pid, gpu=detect_gpu(), started_at=utc_now())
            while CHILD.poll() is None and not BUNDLE_RESTART.is_set():
                if (
                    READY_PATH.is_file()
                    and HEARTBEAT_PATH.is_file()
                    and HEARTBEAT_PATH.stat().st_size > 0
                ):
                    if _runtime_stream_stalled():
                        _prepare_stall_recovery()
                        update_state(
                            state="restarting",
                            detail="recovering stalled primary frame publisher",
                            webrtc_ready=False,
                            stream_ready=False,
                        )
                        CHILD.terminate()
                        BUNDLE_RESTART.wait(1)
                        continue
                    _mark_runtime_ready()
                else:
                    with STATE_LOCK:
                        ready = STATE.get("state") == "ready"
                    if not ready:
                        update_state(detail="warming RTX renderer")
                BUNDLE_RESTART.wait(1)
            if BUNDLE_RESTART.is_set() and CHILD.poll() is None:
                update_state(
                    state="restarting",
                    detail="applying checksum-verified custom bundles",
                    webrtc_ready=False,
                    stream_ready=False,
                )
                CHILD.terminate()
                try:
                    CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    CHILD.kill()
                    CHILD.wait(timeout=10)
                continue
            if SERVER_STOP.is_set():
                return
            exit_code = CHILD.returncode
            update_state(
                state="restarting",
                detail=(
                    f"LeIsaac exited with status {exit_code}; retrying the exact "
                    "task and immutable bundle selection"
                ),
                webrtc_ready=False,
                stream_ready=False,
            )
            if SERVER_STOP.wait(2):
                return
    except Exception as exc:
        _log_exception(logging.ERROR, "LeIsaac simulation process failed", exc)
        update_state(
            state="failed",
            detail=f"LeIsaac runtime failed ({type(exc).__name__})",
            webrtc_ready=False,
        )


def health_document() -> dict[str, Any]:
    with STATE_LOCK:
        state = dict(STATE)
    try:
        input_events = max(
            0, int(INPUT_COUNTER_PATH.read_text(encoding="utf-8").strip() or "0")
        )
    except (OSError, ValueError):
        input_events = 0
    try:
        applied_inputs = max(
            0, int(APPLIED_COUNTER_PATH.read_text(encoding="utf-8").strip() or "0")
        )
    except (OSError, ValueError):
        applied_inputs = 0
    try:
        frame_bytes = FRAME_PATH.stat().st_size
        frame_updated_at = (
            datetime.fromtimestamp(FRAME_PATH.stat().st_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except OSError:
        frame_bytes = 0
        frame_updated_at = ""
    try:
        frame_metadata = json.loads(FRAME_META_PATH.read_text(encoding="utf-8"))
        frame_sequence = int(frame_metadata.get("sequence") or 0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        frame_sequence = 0
    secondary = _read_consistent_frame("overview")
    secondary_metadata = secondary[0] if secondary is not None else {}
    try:
        recorder = json.loads(RECORDER_STATUS_PATH.read_text(encoding="utf-8"))
        if not isinstance(recorder, dict):
            recorder = {}
    except (OSError, ValueError):
        recorder = {
            "state": "starting",
            "dataset_uri": os.environ.get("NPA_LEISAAC_OUTPUT_PATH", ""),
            "task": TASK,
            "environment_id": ENVIRONMENT_ID,
            "environment_index": ENVIRONMENT_INDEX,
            "seed": TELEOP_SEED,
        }
    nonce = os.environ.get("NPA_LEISAAC_SESSION_NONCE", "")
    attestation = (
        hashlib.sha256(f"npa-leisaac-session:{nonce}".encode()).hexdigest()
        if len(nonce) == 64
        else ""
    )
    selected_bundles = state.get("selected_bundles")
    if not isinstance(selected_bundles, dict):
        selected_bundles = {}
    configuration = resolve_configuration(TASK, selected_bundles)
    mode = _mode_state()
    public_mode = {key: mode[key] for key in HEALTH_MODE_FIELDS if key in mode}
    dual_applied = mode.get("applied_view_mode") == "dual_slow"
    return {
        "schema": SCHEMA,
        "run_id": os.environ.get("NPA_LEISAAC_RUN_ID", ""),
        "task": TASK,
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "task_registry": registry_payload(),
        "configuration": configuration,
        "robot": str(configuration["robot"]["id"]),
        "scene": str(configuration["scene"]["id"]),
        "device": str(configuration["device"]["id"]),
        "selected_bundles": selected_bundles,
        "teleop_device": TELEOP_DEVICE,
        "seed": TELEOP_SEED,
        "environment_id": ENVIRONMENT_ID,
        "environment_index": ENVIRONMENT_INDEX,
        "num_envs": NUM_ENVS,
        "source_commit": SOURCE_COMMIT,
        "source_version": SOURCE_VERSION,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "isaac_lab_version": ISAAC_LAB_VERSION,
        "session_attestation": attestation,
        "signal_port": SIGNAL_PORT,
        "media_port": MEDIA_PORT,
        "input_events": input_events,
        "applied_inputs": applied_inputs,
        "stream_ready": bool(state.get("stream_ready")),
        "stream_transport": str(state.get("stream_transport") or "webrtc"),
        "frame_bytes": frame_bytes,
        "frame_updated_at": frame_updated_at,
        "frame_sequence": frame_sequence,
        "cameras": ["workspace", "overview"] if dual_applied else ["workspace"],
        "secondary_frame_bytes": (
            len(secondary[1]) if dual_applied and secondary is not None else 0
        ),
        "secondary_frame_sequence": (
            int(secondary_metadata.get("sequence") or 0) if dual_applied else 0
        ),
        "view_orbit": True,
        "view_mode_contract": VIEW_MODE_CONTRACT,
        "recording_camera_contract": RECORDING_CAMERA_CONTRACT,
        **public_mode,
        "transport_metrics": TRANSPORT_METRICS.snapshot(),
        "physics_device": "cuda:0",
        "render_device": "cuda",
        "recorder": recorder,
        **state,
    }


def liveness_status() -> int:
    """Keep a live simulator process alive while readiness is still pending."""

    with STATE_LOCK:
        state = str(STATE.get("state") or "")
        pid = int(STATE.get("pid") or 0)
    child = CHILD
    if state == "failed" or (
        state != "restarting"
        and pid > 0
        and (child is None or child.poll() is not None)
    ):
        return 503
    return 200


def _authorized(headers: Any) -> bool:
    expected = os.environ.get("NPA_LEISAAC_SESSION_NONCE", "")
    supplied = str(headers.get("x-npa-leisaac-nonce") or "")
    return bool(expected) and hmac.compare_digest(expected, supplied)


def _controller_lease_authorized(headers: Any, client_id: str) -> bool:
    """Apply the preferred-transport controller lease to HTTP fallbacks."""

    supplied_client = str(headers.get("x-npa-leisaac-client-id") or "")
    supplied_lease = str(headers.get("x-npa-leisaac-lease-id") or "")
    if (
        not _authorized(headers)
        or CLIENT_ID_PATTERN.fullmatch(supplied_client) is None
        or supplied_client != client_id
        or re.fullmatch(r"[a-f0-9]{64}", supplied_lease) is None
    ):
        return False
    with CONTROL_OWNER_LOCK:
        owner_client = str(CONTROL_OWNER.get("client_id") or "")
        owner_lease = str(CONTROL_OWNER.get("lease_id") or "")
    return (
        owner_client == supplied_client
        and bool(owner_lease)
        and hmac.compare_digest(owner_lease, supplied_lease)
    )


def _controller_busy() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "v": 1,
            "type": "error",
            "code": "controller_busy",
            "detail": "the active controller lease is required",
        },
        headers={"Cache-Control": "no-store"},
    )


def _increment_input_counter() -> int:
    return _advance_input_counter(1)


def _advance_input_counter(amount: int) -> int:
    """Atomically reserve a counter range shared with the Isaac child."""

    if amount < 1:
        raise ValueError("counter increment must be positive")
    lock_path = INPUT_COUNTER_PATH.with_suffix(INPUT_COUNTER_PATH.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                current = int(
                    INPUT_COUNTER_PATH.read_text(encoding="utf-8").strip() or "0"
                )
            except (OSError, ValueError):
                current = 0
            count = current + amount
            temporary = INPUT_COUNTER_PATH.with_suffix(".tmp")
            temporary.write_text(f"{count}\n", encoding="utf-8")
            temporary.replace(INPUT_COUNTER_PATH)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return count


def _append_inputs(records: list[dict[str, Any]]) -> list[int]:
    serialized = [
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ]
    counts: list[int] = []
    durable = any(
        str(record.get("event") or "") == "release"
        or (
            record.get("type") == "action"
            and all(float(value) == 0.0 for value in (record.get("action") or ()))
        )
        for record in records
    )
    with INPUT_LOCK:
        with INPUT_QUEUE_PATH.open("a", encoding="utf-8") as queue:
            queue.writelines(serialized)
            queue.flush()
            # Movement presses are transient and a simulator/host failure resets
            # the robot state. Safety releases and neutral direct actions are the
            # durable boundary: their fsync also commits every prior ordered press.
            if durable:
                os.fsync(queue.fileno())
        if records:
            final_count = _advance_input_counter(len(records))
            counts.extend(range(final_count - len(records) + 1, final_count + 1))
    return counts


def _append_input(record: dict[str, Any]) -> int:
    return _append_inputs([record])[0]


def _read_consistent_frame(
    camera: str = "workspace",
    after_sequence: int = 0,
    producer_pid: int = 0,
    trusted_metadata: dict[str, Any] | None = None,
    trusted_jpeg: bytes | None = None,
) -> tuple[dict[str, Any], bytes] | None:
    paths = CAMERA_PATHS.get(camera)
    if paths is None:
        return None
    frame_path, metadata_path = paths
    if trusted_metadata is not None:
        try:
            observed_producer = int(trusted_metadata.get("producer_pid") or 0)
            observed_sequence = int(trusted_metadata.get("sequence") or 0)
            if (
                observed_producer == producer_pid
                and observed_sequence <= after_sequence
            ):
                return None
            jpeg = trusted_jpeg if trusted_jpeg is not None else frame_path.read_bytes()
            declared_size = int(trusted_metadata.get("bytes") or 0)
            declared_sha256 = str(trusted_metadata.get("sha256") or "")
        except (OSError, TypeError, ValueError):
            return None
        if (
            0 < len(jpeg) <= MAX_FRAME_BYTES
            and jpeg.startswith(b"\xff\xd8")
            and jpeg.endswith(b"\xff\xd9")
            and len(jpeg) == declared_size
            and re.fullmatch(r"[a-f0-9]{64}", declared_sha256)
        ):
            # The Unix socket is a same-pod trust boundary. The producer's
            # digest remains in the envelope for storage/provenance consumers;
            # this relay does not recompute it on the hot frame hop.
            return trusted_metadata, jpeg
        return None
    for _attempt in range(3):
        try:
            first = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(first, dict):
                return None
            observed_producer = int(first.get("producer_pid") or 0)
            if (
                observed_producer == producer_pid
                and int(first.get("sequence") or 0) <= after_sequence
            ):
                return None
            jpeg = frame_path.read_bytes()
            second = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if first != second or not isinstance(first, dict):
            continue
        digest = hashlib.sha256(jpeg).hexdigest()
        if (
            0 < len(jpeg) <= MAX_FRAME_BYTES
            and jpeg.startswith(b"\xff\xd8")
            and jpeg.endswith(b"\xff\xd9")
            and digest == str(first.get("sha256") or "")
            and len(jpeg) == int(first.get("bytes") or 0)
        ):
            return first, jpeg
    return None


def _read_new_frames(
    sequences: dict[str, int],
    producers: dict[str, int],
) -> list[tuple[str, dict[str, Any], bytes]]:
    """Read and authenticate only frames newer than the watcher snapshot."""

    frames: list[tuple[str, dict[str, Any], bytes]] = []
    applied_mode = str(_mode_state().get("applied_view_mode") or "")
    for camera in CAMERA_PATHS:
        if camera == "overview" and applied_mode != "dual_slow":
            continue
        item = _read_consistent_frame(
            camera,
            sequences.get(camera, 0),
            producers.get(camera, 0),
        )
        if item is not None:
            metadata, jpeg = item
            frames.append((camera, metadata, jpeg))
    return frames


async def _watch_frames() -> None:
    sequences = {camera: 0 for camera in CAMERA_PATHS}
    producers = {camera: 0 for camera in CAMERA_PATHS}
    pending: dict[str, tuple[dict[str, Any], bytes]] = {}
    next_camera = "workspace"
    initial = _read_new_frames(dict(sequences), dict(producers))
    pending_events = list(initial)
    while True:
        if pending_events:
            discovered = [pending_events.pop(0)]
        else:
            queue = RUNTIME_EVENT_QUEUE
            if queue is None:
                # Unit/startup fallback only. Production lifespan always binds
                # the push socket before starting this watcher.
                await asyncio.sleep(0.1)
                discovered = _read_new_frames(dict(sequences), dict(producers))
            else:
                event = await queue.get()
                camera = str(event.get("camera") or "")
                metadata = event.get("metadata")
                jpeg = event.get("jpeg")
                trusted = str(event.get("type") or "") == "frame"
                item = (
                    _read_consistent_frame(
                        camera,
                        sequences.get(camera, 0),
                        producers.get(camera, 0),
                        metadata if trusted and isinstance(metadata, dict) else None,
                        bytes(jpeg)
                        if trusted and isinstance(jpeg, (bytes, bytearray))
                        else None,
                    )
                    if camera in CAMERA_PATHS
                    else None
                )
                discovered = [(camera, *item)] if item is not None else []
        for camera, metadata, jpeg in discovered:
            observed = int(metadata.get("sequence") or 0)
            producer = int(metadata.get("producer_pid") or 0)
            if producer != producers[camera]:
                producers[camera] = producer
                sequences[camera] = 0
            if observed > sequences[camera]:
                sequences[camera] = observed
                pending[camera] = (metadata, jpeg)
        if _mode_state().get("applied_view_mode") != "dual_slow":
            pending.pop("overview", None)
        selected = next_camera if next_camera in pending else next(iter(pending), "")
        if selected:
            metadata, jpeg = pending.pop(selected)
            await FRAME_LATEST.publish(selected, (selected, metadata, jpeg))
            # Keep metrics low-cardinality.  Dynamic per-camera names aren't part
            # of TransportMetrics.ALLOWED and used to terminate this long-lived
            # watcher immediately after its first publication.
            TRANSPORT_METRICS.increment("frames_published")
            TRANSPORT_METRICS.increment(f"{selected}_frames_published")
            next_camera = "overview" if selected == "workspace" else "workspace"


def _accept_applied_ack(payload: dict[str, Any]) -> None:
    global APPLIED_GENERATION
    if CONTROL_LEDGER.mark_applied(payload):
        TRANSPORT_METRICS.increment("controls_applied")
        APPLIED_GENERATION += 1
        if APPLIED_EVENT is not None:
            APPLIED_EVENT.set()


class _RuntimeEventProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data: bytes, _address: Any) -> None:
        if data.startswith(IPC_FRAME_MAGIC) and len(data) >= IPC_FRAME_HEADER.size:
            _magic, metadata_size = IPC_FRAME_HEADER.unpack_from(data)
            metadata_end = IPC_FRAME_HEADER.size + metadata_size
            if metadata_end > len(data):
                return
            try:
                metadata = json.loads(data[IPC_FRAME_HEADER.size : metadata_end])
            except (UnicodeDecodeError, ValueError):
                return
            if not isinstance(metadata, dict) or RUNTIME_EVENT_QUEUE is None:
                return
            payload = {
                "type": "frame",
                "camera": str(metadata.get("camera") or ""),
                "metadata": metadata,
                "jpeg": data[metadata_end:],
            }
            try:
                RUNTIME_EVENT_QUEUE.put_nowait(payload)
            except asyncio.QueueFull:
                TRANSPORT_METRICS.increment("frames_coalesced")
            return
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("type") or "")
        if kind == "applied":
            acknowledgement = payload.get("acknowledgement")
            if isinstance(acknowledgement, dict):
                _accept_applied_ack(acknowledgement)
            return
        if kind not in {"frame", "frame-file"} or RUNTIME_EVENT_QUEUE is None:
            return
        try:
            RUNTIME_EVENT_QUEUE.put_nowait(payload)
        except asyncio.QueueFull:
            TRANSPORT_METRICS.increment("frames_coalesced")


async def _receive_runtime_events(event_socket: socket.socket) -> None:
    """Receive Unix datagrams through APIs supported by asyncio and uvloop."""

    loop = asyncio.get_running_loop()
    protocol = _RuntimeEventProtocol()
    while True:
        data = await loop.sock_recv(event_socket, IPC_EVENT_MAX_BYTES)
        protocol.datagram_received(data, None)


def _scan_applied_acks() -> None:
    global APPLIED_ACK_OFFSET
    with APPLIED_ACK_LOCK:
        try:
            with APPLIED_ACK_PATH.open("r", encoding="utf-8") as source:
                source.seek(APPLIED_ACK_OFFSET)
                lines = source.readlines()
                APPLIED_ACK_OFFSET = source.tell()
        except OSError:
            return
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _accept_applied_ack(payload)


async def _wait_for_applied(
    client_id: str,
    seq: int,
    stop: threading.Event,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        applied = CONTROL_LEDGER.applied(client_id, seq)
        if applied is not None:
            return applied
        observed = APPLIED_GENERATION
        event = APPLIED_EVENT
        if event is None:
            await asyncio.to_thread(_scan_applied_acks)
            await asyncio.sleep(0.1)
            continue
        event.clear()
        if APPLIED_GENERATION != observed:
            continue
        try:
            await asyncio.wait_for(event.wait(), timeout=0.1)
        except asyncio.TimeoutError:
            # Durable JSONL is recovery truth if a datagram is lost or the
            # simulator was alive before the socket listener bound.
            await asyncio.to_thread(_scan_applied_acks)
    return None


async def _wait_for_mode_applied(message: dict[str, Any]) -> dict[str, Any]:
    """Wait for the simulator scheduler, never merely the HTTP relay."""

    revision_key = (
        "applied_view_revision"
        if message["type"] == "view-mode"
        else "applied_recording_revision"
    )
    mode_key = (
        "applied_view_mode"
        if message["type"] == "view-mode"
        else "applied_recording_camera_mode"
    )
    requested_revision_key = (
        "view_revision" if message["type"] == "view-mode" else "recording_revision"
    )
    while True:
        state = await asyncio.to_thread(_mode_state)
        applied_revision = int(state.get(revision_key) or 0)
        if (
            applied_revision == int(message["revision"])
            and state.get(mode_key) == message["mode"]
        ):
            return {
                "v": 1,
                "type": "ack",
                "phase": "applied",
                "request_type": message["type"],
                "run_id": message["run_id"],
                "client_id": message["client_id"],
                "revision": message["revision"],
                "mode": message["mode"],
                "mode_transition_latency_ms": float(
                    state.get("mode_transition_latency_ms") or 0.0
                ),
            }
        if int(state.get(requested_revision_key) or 0) > int(message["revision"]):
            return {
                "v": 1,
                "type": "ack",
                "phase": "superseded",
                "request_type": message["type"],
                "run_id": message["run_id"],
                "client_id": message["client_id"],
                "revision": message["revision"],
                "mode": message["mode"],
            }
        await asyncio.sleep(0.002)


async def _serve_control_protocol(
    receive: Callable[[], Awaitable[str]],
    emit: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    """Serve the one control protocol over WebSocket or reliable SCTP."""

    run_id = os.environ.get("NPA_LEISAAC_RUN_ID", "")
    applied_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(
        maxsize=MAX_CLIENT_HISTORY
    )
    stop = threading.Event()
    send_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await asyncio.wait_for(emit(payload), timeout=2.0)

    async def send_applied() -> None:
        while True:
            client_id, seq = await applied_queue.get()
            payload = await _wait_for_applied(client_id, seq, stop)
            if payload is None:
                await send(
                    {
                        "v": 1,
                        "type": "error",
                        "code": "application_timeout",
                        "run_id": run_id,
                        "client_id": client_id,
                        "seq": seq,
                    }
                )
                continue
            acknowledgement = dict(payload)
            acknowledgement.update(v=1, type="ack", phase="applied", run_id=run_id)
            await send(acknowledgement)

    sender = asyncio.create_task(send_applied())
    mode_ack_tasks: set[asyncio.Task[None]] = set()
    active_client_id = ""
    owner_token = secrets.token_hex(16)

    def claim_controller(
        client_id: str,
        *,
        resume: bool = False,
        lease_id: str = "",
    ) -> str:
        """Permit one authenticated control transport to own mutable runtime state."""

        with CONTROL_OWNER_LOCK:
            current = str(CONTROL_OWNER["token"])
            if current and current != owner_token:
                # Only the server-issued, single-use resume capability may replace
                # a half-open transport for the same browser lease. Client clocks
                # are telemetry only and never participate in authority decisions.
                if not (
                    resume
                    and client_id == str(CONTROL_OWNER.get("client_id") or "")
                    and lease_id
                    and secrets.compare_digest(
                        lease_id, str(CONTROL_OWNER.get("lease_id") or "")
                    )
                ):
                    raise TransportProtocolError(
                        "controller_busy",
                        "another authenticated control transport owns this session",
                    )
            elif not current and str(CONTROL_OWNER.get("client_id") or "") == client_id:
                retained_lease = str(CONTROL_OWNER.get("lease_id") or "")
                if retained_lease and not (
                    resume
                    and lease_id
                    and secrets.compare_digest(lease_id, retained_lease)
                ):
                    raise TransportProtocolError(
                        "controller_busy", "control lease resume capability is required"
                    )
            next_lease_id = (
                secrets.token_hex(32)
                if resume
                else str(CONTROL_OWNER.get("lease_id") or "")
            )
            CONTROL_OWNER.update(
                token=owner_token,
                client_id=client_id,
                lease_id=next_lease_id,
                lease_generation=int(CONTROL_OWNER.get("lease_generation") or 0)
                + (1 if resume else 0),
                leased_at_monotonic_ns=time.monotonic_ns(),
            )
            _retain_mode_owner(client_id)
            return next_lease_id

    async def send_mode_applied(message: dict[str, Any]) -> None:
        try:
            acknowledgement = await _wait_for_mode_applied(message)
        except TransportProtocolError as exc:
            TRANSPORT_METRICS.increment("mode_transition_errors")
            acknowledgement = exc.payload()
            acknowledgement.update(
                run_id=run_id,
                client_id=message["client_id"],
                request_type=message["type"],
                revision=message["revision"],
            )
        else:
            if acknowledgement["phase"] == "applied":
                TRANSPORT_METRICS.increment("mode_transitions_applied")
            else:
                TRANSPORT_METRICS.increment("mode_requests_coalesced")
        await send(acknowledgement)

    def release_all(
        client_id: str, client_mono_ns: int = 0, client_wall_ns: int = 0
    ) -> int:
        releases: list[tuple[int, dict[str, Any]]] = []
        for key in CONTROL_LEDGER.keys_down(client_id):
            next_seq = int(CONTROL_LEDGER.resume(client_id)["next_seq"])
            message = {
                "v": 1,
                "type": "control",
                "run_id": run_id,
                "client_id": client_id,
                "seq": next_seq,
                "key": key,
                "event": "release",
                "client_mono_ns": client_mono_ns,
                "client_wall_ns": client_wall_ns,
            }
            _accepted, queued = CONTROL_LEDGER.accept(message)
            if queued is not None:
                releases.append((next_seq, queued))
        if not releases:
            return 0
        # This synchronous flush/fsync is the cancellation-safe safety boundary
        # shared by WebSocket and SCTP disconnects.
        _append_inputs([queued for _seq, queued in releases])
        TRANSPORT_METRICS.increment("controls_accepted", len(releases))
        for seq, _queued in releases:
            applied_queue.put_nowait((client_id, seq))
        return len(releases)

    try:
        while True:
            raw = await receive()
            try:
                message = parse_control_message(raw, expected_run_id=run_id)
                message_client_id = str(message["client_id"])
                active_lease_id = claim_controller(
                    message_client_id,
                    resume=message["type"] == "resume",
                    lease_id=str(message.get("lease_id") or ""),
                )
                if active_client_id and message_client_id != active_client_id:
                    raise TransportProtocolError(
                        "client_mismatch",
                        "one control transport may own only one client ID",
                    )
                active_client_id = message_client_id
                if message["type"] == "ping":
                    await send(
                        {
                            "v": 1,
                            "type": "pong",
                            "run_id": run_id,
                            "client_id": message["client_id"],
                            "nonce": message.get("nonce", ""),
                            "client_mono_ns": str(message["client_mono_ns"]),
                            "client_wall_ns": str(message["client_wall_ns"]),
                            "runtime_mono_ns": str(time.monotonic_ns()),
                            "runtime_wall_ns": str(time.time_ns()),
                        }
                    )
                    continue
                if message["type"] == "resume":
                    # Reconnects may follow a disconnect release whose simulator
                    # acknowledgement landed after the prior socket's sender was
                    # cancelled. Ingest the durable log before reporting state.
                    await asyncio.to_thread(_scan_applied_acks)
                    response = CONTROL_LEDGER.resume(str(message["client_id"]))
                    response["run_id"] = run_id
                    response["client_mono_ns"] = str(message["client_mono_ns"])
                    response["client_wall_ns"] = str(message["client_wall_ns"])
                    mode = await asyncio.to_thread(_mode_request_state)
                    response["view_revision"] = int(mode.get("view_revision") or 0)
                    response["recording_revision"] = int(
                        mode.get("recording_revision") or 0
                    )
                    response["lease_id"] = active_lease_id
                    response["lease_generation"] = int(
                        CONTROL_OWNER.get("lease_generation") or 0
                    )
                    TRANSPORT_METRICS.increment("reconnects")
                    await send(response)
                    continue
                if message["type"] == "release-all":
                    released = release_all(
                        active_client_id,
                        int(message["client_mono_ns"]),
                        int(message["client_wall_ns"]),
                    )
                    await send(
                        {
                            "v": 1,
                            "type": "released",
                            "run_id": run_id,
                            "client_id": message["client_id"],
                            "runtime_mono_ns": str(time.monotonic_ns()),
                            "released_count": released,
                        }
                    )
                    continue
                if message["type"] in {"view-mode", "recording-cameras"}:
                    with STATE_LOCK:
                        ready = STATE.get("state") == "ready"
                    if not ready:
                        await send(
                            {
                                "v": 1,
                                "type": "error",
                                "code": "simulator_not_ready",
                                "detail": "simulator not ready",
                            }
                        )
                        continue
                    await asyncio.to_thread(_queue_mode_request, message)
                    TRANSPORT_METRICS.increment("mode_requests")
                    await send(
                        {
                            "v": 1,
                            "type": "ack",
                            "phase": "accepted",
                            "request_type": message["type"],
                            "run_id": run_id,
                            "client_id": message["client_id"],
                            "revision": message["revision"],
                            "mode": message["mode"],
                        }
                    )
                    task = asyncio.create_task(send_mode_applied(dict(message)))
                    mode_ack_tasks.add(task)
                    task.add_done_callback(mode_ack_tasks.discard)
                    continue
                with STATE_LOCK:
                    ready = STATE.get("state") == "ready"
                if not ready:
                    await send(
                        {
                            "v": 1,
                            "type": "error",
                            "code": "simulator_not_ready",
                            "detail": "simulator not ready",
                        }
                    )
                    continue
                accepted, queued = CONTROL_LEDGER.accept(message)
                if queued is not None:
                    await asyncio.to_thread(_append_input, queued)
                    TRANSPORT_METRICS.increment("controls_accepted")
                else:
                    TRANSPORT_METRICS.increment("controls_duplicate")
                await send(accepted)
                await applied_queue.put(
                    (str(message["client_id"]), int(message["seq"]))
                )
            except TransportProtocolError as exc:
                TRANSPORT_METRICS.increment("control_errors")
                await send(exc.payload())
    finally:
        try:
            with CONTROL_OWNER_LOCK:
                still_owner = CONTROL_OWNER["token"] == owner_token
            if active_client_id and still_owner:
                release_all(active_client_id)
                # A disconnected controller cannot leave hidden secondary GPU
                # work enabled. The same client may reassert its latest choice
                # after its ordered resume handshake. Like release_all(), this
                # tiny local atomic write is intentionally synchronous: ASGI
                # teardown cancellation must not strand secondary GPU work.
                mode = _mode_request_state()
                _queue_mode_request(
                    {
                        "type": "view-mode",
                        "mode": DEFAULT_VIEW_MODE.value,
                        "revision": int(mode.get("view_revision") or 0) + 1,
                        "client_id": active_client_id,
                    }
                )
        except Exception as exc:
            _log_exception(
                logging.CRITICAL,
                "Failed to release LeIsaac controls during disconnect",
                exc,
            )
        finally:
            stop.set()
            sender.cancel()
            for task in mode_ack_tasks:
                task.cancel()
            with CONTROL_OWNER_LOCK:
                if CONTROL_OWNER["token"] == owner_token:
                    CONTROL_OWNER.update(
                        token="",
                        leased_at_monotonic_ns=0,
                    )
            await asyncio.gather(sender, *mode_ack_tasks, return_exceptions=True)


async def _serve_control_datachannel(channel: Any) -> None:
    """Adapt one bounded reliable RTCDataChannel to the shared control service."""

    messages: asyncio.Queue[str | None] = asyncio.Queue(maxsize=MAX_CLIENT_HISTORY)

    @channel.on("message")
    def on_message(raw: Any) -> None:
        if (
            not isinstance(raw, str)
            or len(raw.encode("utf-8")) > MAX_CONTROL_MESSAGE_BYTES
        ):
            channel.close()
            return
        try:
            messages.put_nowait(raw)
        except asyncio.QueueFull:
            channel.close()

    @channel.on("close")
    def on_close() -> None:
        asyncio.create_task(messages.put(None))

    async def receive_datachannel() -> str:
        raw = await messages.get()
        if raw is None:
            raise ConnectionError("WebRTC control channel closed")
        return raw

    async def emit_datachannel(payload: dict[str, Any]) -> None:
        if str(channel.readyState) != "open":
            raise ConnectionError("WebRTC control channel is unavailable")
        channel.send(json.dumps(payload, separators=(",", ":")))

    await _serve_control_protocol(receive_datachannel, emit_datachannel)


def _runtime_ws_authorized(websocket: WebSocket, subprotocol: str) -> bool:
    requested = {
        item.strip()
        for item in str(websocket.headers.get("sec-websocket-protocol") or "").split(
            ","
        )
        if item.strip()
    }
    return (
        _authorized(websocket.headers)
        and str(websocket.headers.get("x-npa-leisaac-run-id") or "")
        == os.environ.get("NPA_LEISAAC_RUN_ID", "")
        and requested == {subprotocol}
    )


async def _video_datachannel_frames():
    """Yield only the newest independently decodable frame for each camera."""

    generations: dict[str, int] = {}
    previous_sequences = {camera: 0 for camera in CAMERA_PATHS}
    next_camera_index = 0
    while True:
        (
            camera,
            generation,
            item,
            coalesced,
            next_camera_index,
        ) = await FRAME_LATEST.wait_after(
            generations,
            next_index=next_camera_index,
            preferred_key="workspace",
            timeout=20.0,
        )
        generations[camera] = generation
        camera, metadata, jpeg = item
        if (
            camera == "overview"
            and _mode_state().get("applied_view_mode") != "dual_slow"
        ):
            continue
        sequence = int(metadata["sequence"])
        previous_sequence = previous_sequences.get(camera, 0)
        dropped = (
            max(coalesced, max(0, sequence - previous_sequence - 1))
            if previous_sequence
            else 0
        )
        previous_sequences[camera] = sequence
        relay_receive_ns = time.monotonic_ns()
        envelope = FrameEnvelope(
            sequence=sequence,
            capture_wall_ns=int(metadata["capture_wall_ns"]),
            capture_monotonic_ns=int(metadata["capture_monotonic_ns"]),
            encoded_wall_ns=int(metadata["encoded_wall_ns"]),
            encoded_monotonic_ns=int(metadata["encoded_monotonic_ns"]),
            runtime_send_monotonic_ns=relay_receive_ns,
            agent_receive_monotonic_ns=relay_receive_ns,
            agent_send_monotonic_ns=time.monotonic_ns(),
            causal_action_sequence=int(metadata.get("causal_action_sequence") or 0),
            causal_applied_monotonic_ns=int(
                metadata.get("causal_applied_monotonic_ns") or 0
            ),
            view_revision=int(metadata.get("view_revision") or 0),
            dropped_before=dropped,
            flags=1 if camera == "overview" else 0,
            sha256=bytes.fromhex(str(metadata["sha256"])),
        )
        if dropped:
            TRANSPORT_METRICS.increment("frames_coalesced", dropped)
        yield pack_frame(envelope, jpeg)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global APPLIED_EVENT, RUNTIME_EVENT_QUEUE
    IPC_EVENT_PATH.unlink(missing_ok=True)
    APPLIED_EVENT = asyncio.Event()
    RUNTIME_EVENT_QUEUE = asyncio.Queue(maxsize=256)
    event_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    event_socket.setblocking(False)
    event_socket.bind(str(IPC_EVENT_PATH))
    os.chmod(IPC_EVENT_PATH, 0o600)
    receiver = asyncio.create_task(_receive_runtime_events(event_socket))
    watcher = asyncio.create_task(_watch_frames())
    try:
        yield
    finally:
        receiver.cancel()
        watcher.cancel()
        await asyncio.gather(receiver, watcher, return_exceptions=True)
        event_socket.close()
        IPC_EVENT_PATH.unlink(missing_ok=True)
        RUNTIME_EVENT_QUEUE = None
        APPLIED_EVENT = None


def build_app() -> FastAPI:
    application = FastAPI(lifespan=_lifespan)

    @application.get("/healthz")
    def healthz() -> Response:
        status = liveness_status()
        return JSONResponse(status_code=status, content={"ok": status == 200})

    @application.get("/readyz")
    def readyz() -> Response:
        # Kubernetes needs simulator readiness without receiving the session
        # credential or regaining access to the detailed /status document.
        with STATE_LOCK:
            ready = STATE.get("state") == "ready"
        return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})

    @application.get("/status")
    def status(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        document = health_document()
        return JSONResponse(
            status_code=200 if document["state"] == "ready" else 503,
            content=document,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @application.get("/frame.jpg")
    def frame(request: Request, camera: str = "workspace") -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        if camera not in CAMERA_PATHS:
            return JSONResponse(status_code=400, content={"detail": "invalid camera"})
        if (
            camera == "overview"
            and _mode_state().get("applied_view_mode") != "dual_slow"
        ):
            return JSONResponse(
                status_code=409,
                content={"detail": "secondary camera is disabled in Fast single"},
            )
        item = _read_consistent_frame(camera)
        if item is None:
            return JSONResponse(
                status_code=503, content={"detail": "frame unavailable"}
            )
        metadata, content = item
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-NPA-Frame-Sequence": str(metadata["sequence"]),
                "X-NPA-Frame-Capture-Wall-Ns": str(metadata["capture_wall_ns"]),
                "X-NPA-Frame-SHA256": str(metadata["sha256"]),
                "X-NPA-Camera": camera,
            },
        )

    @application.post("/view")
    async def update_view(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        payload = await read_bounded_json(request)
        if payload is None or set(payload) != {
            "camera",
            "client_id",
            "sequence",
            "yaw_delta",
            "pitch_delta",
            "distance_delta",
        }:
            return JSONResponse(
                status_code=400, content={"detail": "invalid view command"}
            )
        try:
            camera = str(payload["camera"])
            client_id = str(payload["client_id"])
            sequence = int(payload["sequence"])
            yaw_delta = float(payload["yaw_delta"])
            pitch_delta = float(payload["pitch_delta"])
            distance_delta = float(payload["distance_delta"])
        except (TypeError, ValueError, OverflowError):
            return JSONResponse(
                status_code=400, content={"detail": "invalid view command"}
            )
        if not _controller_lease_authorized(request.headers, client_id):
            return _controller_busy()
        if (
            camera != "workspace"
            or not 1 <= sequence <= 2**53 - 1
            or not math.isfinite(yaw_delta)
            or not math.isfinite(pitch_delta)
            or not math.isfinite(distance_delta)
            or abs(yaw_delta) > 0.5
            or abs(pitch_delta) > 0.5
            or abs(distance_delta) > 1.0
        ):
            return JSONResponse(
                status_code=400, content={"detail": "invalid view command"}
            )
        _write_json_atomic(
            VIEW_COMMAND_PATH,
            {
                "camera": camera,
                "client_id": client_id,
                "sequence": sequence,
                "yaw_delta": yaw_delta,
                "pitch_delta": pitch_delta,
                "distance_delta": distance_delta,
                "received_monotonic_ns": time.monotonic_ns(),
            },
        )
        return JSONResponse(
            status_code=202, content={"accepted": True, "sequence": sequence}
        )

    @application.post("/bundles/apply")
    async def bundles_apply(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        client_id = str(request.headers.get("x-npa-leisaac-client-id") or "")
        if not _controller_lease_authorized(request.headers, client_id):
            return _controller_busy()
        payload = await read_bounded_json(request)
        if payload is None or set(payload) != {"selection"}:
            return JSONResponse(
                status_code=400, content={"detail": "invalid bundle selection"}
            )
        try:
            selected = await asyncio.to_thread(
                apply_bundle_selection, payload["selection"]
            )
        except BundleError as exc:
            return JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )
        return JSONResponse(
            status_code=202,
            content={"accepted": True, "selected": selected, "restarting": True},
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/client/index.js")
    def client_module(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        return Response(
            content=(CLIENT_ROOT / "index.js").read_bytes(),
            media_type="text/javascript",
        )

    @application.get("/client/LICENSE.txt")
    def client_license() -> Response:
        return Response(
            content=(CLIENT_ROOT / "LICENSE.txt").read_bytes(), media_type="text/plain"
        )

    @application.get("/provenance")
    def provenance(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        return Response(
            content=PROVENANCE_PATH.read_bytes(), media_type="application/json"
        )

    @application.post("/transport/control-webrtc")
    async def transport_control_webrtc(request: Request) -> Response:
        if not _authorized(request.headers) or str(
            request.headers.get("x-npa-leisaac-run-id") or ""
        ) != os.environ.get("NPA_LEISAAC_RUN_ID", ""):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        try:
            content_length = int(request.headers.get("content-length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 131_072:
            return JSONResponse(
                status_code=400, content={"detail": "invalid WebRTC control offer"}
            )
        body = await request.body()
        if len(body) != content_length:
            return JSONResponse(
                status_code=400, content={"detail": "invalid WebRTC control offer"}
            )
        try:
            payload = json.loads(body)
            offer_sdp = parse_video_datachannel_offer(
                payload,
                expected_run_id=os.environ.get("NPA_LEISAAC_RUN_ID", ""),
            )
            answer = await CONTROL_DATACHANNEL_PEERS.create_answer(
                offer_sdp=offer_sdp,
                ice_server=None,
                channel_handler=_serve_control_datachannel,
                metrics=TRANSPORT_METRICS,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, VideoDataChannelError):
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid WebRTC control offer"},
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(
            status_code=200,
            content=answer,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.post("/transport/video-webrtc")
    async def transport_video_webrtc(request: Request) -> Response:
        if not _authorized(request.headers) or str(
            request.headers.get("x-npa-leisaac-run-id") or ""
        ) != os.environ.get("NPA_LEISAAC_RUN_ID", ""):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        try:
            content_length = int(request.headers.get("content-length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 131_072:
            return JSONResponse(
                status_code=400, content={"detail": "invalid WebRTC video offer"}
            )
        body = await request.body()
        if len(body) != content_length:
            return JSONResponse(
                status_code=400, content={"detail": "invalid WebRTC video offer"}
            )
        try:
            payload = json.loads(body)
            offer_sdp = parse_video_datachannel_offer(
                payload,
                expected_run_id=os.environ.get("NPA_LEISAAC_RUN_ID", ""),
            )
            answer = await VIDEO_DATACHANNEL_PEERS.create_answer(
                offer_sdp=offer_sdp,
                ice_server=None,
                frame_source=lambda: _video_datachannel_frames(),
                metrics=TRANSPORT_METRICS,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, VideoDataChannelError):
            return JSONResponse(
                status_code=400,
                content={"detail": "invalid WebRTC video offer"},
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(
            status_code=200,
            content=answer,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def read_bounded_json(request: Request) -> dict[str, Any] | None:
        raw_length = str(request.headers.get("content-length") or "")
        try:
            length = int(raw_length)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_CONTROL_MESSAGE_BYTES:
            return None
        body = await request.body()
        if len(body) != length or len(body) > MAX_CONTROL_MESSAGE_BYTES:
            return None
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @application.post("/input")
    async def input_control(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        payload = await read_bounded_json(request)
        if payload is None:
            return JSONResponse(status_code=400, content={"detail": "invalid body"})
        run_id = os.environ.get("NPA_LEISAAC_RUN_ID", "")
        if payload.get("v") == 1:
            raw = json.dumps(payload, separators=(",", ":"))
        else:
            raw = json.dumps(
                {
                    "v": 1,
                    "type": "control",
                    "run_id": run_id,
                    "client_id": str(
                        payload.get("client_id")
                        or request.headers.get("x-npa-leisaac-client-id")
                        or ""
                    ),
                    "seq": payload.get("seq", 1),
                    "key": payload.get("key"),
                    "event": payload.get("event"),
                    "client_mono_ns": payload.get("client_mono_ns", 0),
                    "client_wall_ns": payload.get("client_wall_ns", 0),
                },
                separators=(",", ":"),
            )
        try:
            message = parse_control_message(raw, expected_run_id=run_id)
        except TransportProtocolError as exc:
            return JSONResponse(
                status_code=409
                if exc.code.startswith("sequence") or exc.code == "out_of_order"
                else 400,
                content=exc.payload(),
            )
        if not _controller_lease_authorized(request.headers, str(message["client_id"])):
            return _controller_busy()
        with STATE_LOCK:
            ready = STATE.get("state") == "ready"
        if not ready:
            return JSONResponse(
                status_code=503, content={"detail": "simulator not ready"}
            )
        if message["type"] in {"view-mode", "recording-cameras"}:
            queued = await asyncio.to_thread(_queue_mode_request, message)
            TRANSPORT_METRICS.increment("mode_requests")
            return JSONResponse(
                status_code=202,
                content={
                    "v": 1,
                    "type": "ack",
                    "phase": "accepted" if queued else "superseded",
                    "request_type": message["type"],
                    "run_id": run_id,
                    "client_id": message["client_id"],
                    "revision": message["revision"],
                    "mode": message["mode"],
                },
                headers={"Cache-Control": "no-store"},
            )
        try:
            accepted, queued = CONTROL_LEDGER.accept(message)
        except TransportProtocolError as exc:
            return JSONResponse(
                status_code=409
                if exc.code.startswith("sequence") or exc.code == "out_of_order"
                else 400,
                content=exc.payload(),
            )
        if queued is not None:
            await asyncio.to_thread(_append_input, queued)
            TRANSPORT_METRICS.increment("controls_accepted")
        else:
            TRANSPORT_METRICS.increment("controls_duplicate")
        return JSONResponse(
            status_code=202, content=accepted, headers={"Cache-Control": "no-store"}
        )

    @application.post("/recorder/control")
    async def recorder_control(request: Request) -> Response:
        if not _authorized(request.headers):
            return JSONResponse(status_code=403, content={"detail": "forbidden"})
        client_id = str(request.headers.get("x-npa-leisaac-client-id") or "")
        if not _controller_lease_authorized(request.headers, client_id):
            return _controller_busy()
        payload = await read_bounded_json(request)
        command = str(payload.get("command") if payload else "")
        request_id = str(
            payload.get("request_id") if payload else ""
        ) or secrets.token_hex(16)
        status_code, result = await asyncio.to_thread(
            enqueue_recorder_command, command, request_id
        )
        return JSONResponse(
            status_code=status_code,
            content=result,
            headers={"Cache-Control": "no-store"},
        )

    @application.websocket("/transport/control")
    async def transport_control(websocket: WebSocket) -> None:
        if not _runtime_ws_authorized(websocket, CONTROL_SUBPROTOCOL):
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol=CONTROL_SUBPROTOCOL)
        TRANSPORT_METRICS.increment("control_connections")

        async def receive_websocket() -> str:
            return await websocket.receive_text()

        async def emit_websocket(payload: dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload, separators=(",", ":")))

        try:
            await _serve_control_protocol(receive_websocket, emit_websocket)
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            # ASGI/TestClient may cancel a closed socket handler instead of
            # delivering WebSocketDisconnect. Cleanup in the shared protocol
            # has already flushed deterministic releases.
            return
        return

    @application.websocket("/transport/video")
    async def transport_video(websocket: WebSocket) -> None:
        if not _runtime_ws_authorized(websocket, VIDEO_SUBPROTOCOL):
            await websocket.close(code=1008)
            return
        run_id = os.environ.get("NPA_LEISAAC_RUN_ID", "")
        await websocket.accept(subprotocol=VIDEO_SUBPROTOCOL)
        TRANSPORT_METRICS.increment("video_connections")
        generations: dict[str, int] = {}
        next_camera_index = 0
        previous_sequences = {camera: 0 for camera in CAMERA_PATHS}
        credits = AsyncFrameCreditWindow()

        async def receive_credits() -> None:
            while True:
                acknowledgement = parse_video_ack(
                    await websocket.receive_text(), expected_run_id=run_id
                )
                credits.acknowledge(int(acknowledgement["sequence"]))

        async def send_frames() -> None:
            nonlocal next_camera_index
            while True:
                (
                    camera,
                    generation,
                    item,
                    coalesced,
                    next_camera_index,
                ) = await FRAME_LATEST.wait_after(
                    generations,
                    next_index=next_camera_index,
                    preferred_key="workspace",
                    timeout=20.0,
                )
                generations[camera] = generation
                camera, metadata, jpeg = item
                if (
                    camera == "overview"
                    and _mode_state().get("applied_view_mode") != "dual_slow"
                ):
                    continue
                sequence = int(metadata["sequence"])
                previous_sequence = previous_sequences.get(camera, 0)
                dropped = (
                    max(coalesced, max(0, sequence - previous_sequence - 1))
                    if previous_sequence
                    else 0
                )
                previous_sequences[camera] = sequence
                envelope = FrameEnvelope(
                    sequence=sequence,
                    capture_wall_ns=int(metadata["capture_wall_ns"]),
                    capture_monotonic_ns=int(metadata["capture_monotonic_ns"]),
                    encoded_wall_ns=int(metadata["encoded_wall_ns"]),
                    encoded_monotonic_ns=int(metadata["encoded_monotonic_ns"]),
                    runtime_send_monotonic_ns=time.monotonic_ns(),
                    causal_action_sequence=int(
                        metadata.get("causal_action_sequence") or 0
                    ),
                    causal_applied_monotonic_ns=int(
                        metadata.get("causal_applied_monotonic_ns") or 0
                    ),
                    view_revision=int(metadata.get("view_revision") or 0),
                    dropped_before=dropped,
                    flags=1 if camera == "overview" else 0,
                    sha256=bytes.fromhex(str(metadata["sha256"])),
                )
                if dropped:
                    TRANSPORT_METRICS.increment("frames_coalesced", dropped)
                depth = await credits.reserve(sequence)
                if depth == credits.limit:
                    TRANSPORT_METRICS.increment("video_window_saturated")
                try:
                    await asyncio.wait_for(
                        websocket.send_bytes(pack_frame(envelope, jpeg)), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    TRANSPORT_METRICS.increment("slow_client_disconnects")
                    await websocket.close(code=1013)
                    return
                TRANSPORT_METRICS.increment("frames_sent")
                TRANSPORT_METRICS.increment(f"{camera}_frames_sent")

        try:
            tasks = {
                asyncio.create_task(receive_credits()),
                asyncio.create_task(send_frames()),
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
        except TransportProtocolError:
            await websocket.close(code=1008)
            return
        except asyncio.TimeoutError:
            TRANSPORT_METRICS.increment("slow_client_disconnects")
            await websocket.close(code=1013)
            return
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            # ASGI servers may cancel a handler instead of delivering a final
            # WebSocketDisconnect while the peer is closing. Both protocol
            # workers have already been cancelled and gathered above, so this
            # is normal bounded disconnect cleanup rather than an application
            # failure that should escape through the server task.
            return

    return application


app = build_app()


def stop_child(*_args: Any) -> None:
    SERVER_STOP.set()
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()


def main() -> int:
    require_operator_eula()
    validate_runtime_configuration()
    stage_runtime()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_child)
    worker = threading.Thread(
        target=run_simulation, name="leisaac-simulation", daemon=True
    )
    worker.start()
    try:
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=SERVICE_PORT,
            ws="websockets",
            ws_max_size=MAX_FRAME_BYTES + 256,
            ws_max_queue=4,
            ws_ping_interval=10.0,
            ws_ping_timeout=10.0,
            ws_per_message_deflate=False,
            timeout_keep_alive=5,
            access_log=False,
        )
    finally:
        stop_child()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["golden-smoke"]:
        os.execv(
            "/opt/npa/sim/venv/bin/python",
            [
                "/opt/npa/sim/venv/bin/python",
                "/opt/npa/leisaac/smoke_functional.py",
            ],
        )
    raise SystemExit(main())
