"""Minimal Foxglove Cloud recording upload client for the agent backend.

The API token is accepted only as an in-memory constructor argument.  It is
sent in the Authorization header to ``api.foxglove.dev`` and is never included
in URLs, subprocess arguments, return payloads, or exception messages.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from npa.agent_backend.canonical_mcap import has_rich_visualization_contract


FOXGLOVE_API_ROOT = "https://api.foxglove.dev/v1"
FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV = "NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS"
DEFAULT_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS = 300.0
MAX_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS = 3600.0
FOXGLOVE_CLOUD_IMPORT_POLL_SECONDS = 2.0
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
FOXGLOVE_LAYOUT_NAME = "NPA Physical AI robot motion v3"
FOXGLOVE_LAYOUT_ID = (
    "lay_" + uuid.uuid5(uuid.NAMESPACE_URL, "npa/foxglove/robot-motion-v3").hex[:16]
)


class FoxgloveCloudError(RuntimeError):
    """Actionable, token-safe Foxglove Cloud failure."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_cloud_import_timeout_seconds(
    value: object | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> float:
    """Resolve and validate the bounded Cloud indexing wait."""
    env = environ if environ is not None else os.environ
    raw = value
    if raw is None or not str(raw).strip():
        raw = env.get(FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV, "")
    if raw is None or not str(raw).strip():
        return DEFAULT_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV} must be a positive finite number "
            f"of seconds no greater than {MAX_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS:g}"
        ) from exc
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or timeout > MAX_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"{FOXGLOVE_CLOUD_IMPORT_TIMEOUT_ENV} must be a positive finite number "
            f"of seconds no greater than {MAX_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS:g}"
        )
    return timeout


class FoxgloveCloudTimeoutError(FoxgloveCloudError):
    """A token-safe, context-rich deadline failure while indexing an import."""

    def __init__(
        self,
        *,
        recording_key: str,
        recording_status: str,
        import_status: str,
        elapsed_seconds: float,
    ) -> None:
        safe_key = (
            recording_key
            if re.fullmatch(r"npa-[0-9a-f]{64}", recording_key)
            else "unknown"
        )

        def safe_status(value: str, fallback: str) -> str:
            cleaned = str(value or "").strip().lower()
            return cleaned if re.fullmatch(r"[a-z0-9_-]{1,32}", cleaned) else fallback

        self.recording_key = safe_key
        self.recording_status = safe_status(recording_status, "unknown")
        self.import_status = safe_status(import_status, "not-observed")
        self.elapsed_seconds = max(0.0, round(float(elapsed_seconds), 3))
        self.context = {
            "recording_key": self.recording_key,
            "recording_status": self.recording_status,
            "import_status": self.import_status,
            "elapsed_seconds": self.elapsed_seconds,
        }
        super().__init__(
            "Foxglove Cloud import did not reach complete before the server deadline "
            f"(recording_key={self.recording_key}, "
            f"recording_status={self.recording_status}, "
            f"import_status={self.import_status}, "
            f"elapsed_seconds={self.elapsed_seconds:.3f}). Retry to reuse the same "
            "content-addressed import.",
            status_code=504,
        )


@dataclass(frozen=True)
class FoxgloveCloudRecording:
    recording_id: str
    recording_key: str
    import_status: str
    size_bytes: int
    uploaded: bool
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "recording_key": self.recording_key,
            "import_status": self.import_status,
            "size_bytes": self.size_bytes,
            "uploaded": self.uploaded,
            "reused": self.reused,
        }


@dataclass(frozen=True)
class FoxgloveCloudLayout:
    layout_id: str
    created: bool
    updated: bool
    reused: bool
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "created": self.created,
            "updated": self.updated,
            "reused": self.reused,
            "available": self.available,
            "reason": self.reason,
        }


def data_aware_layout_data(provenance: dict[str, Any]) -> dict[str, Any]:
    """Build an intentional, data-aware Foxglove v1 programmatic layout."""
    schemas = dict(provenance.get("schemas") or {})
    numeric_paths = dict(provenance.get("numeric_paths") or {})
    discovered_image_topics = sorted(
        topic
        for topic, schema in schemas.items()
        if schema == "foxglove.CompressedImage"
    )
    fixed_frame = str(provenance.get("visualization_fixed_frame") or "world")
    rich_topics = {
        "/robot/diagnostic_scene",
        "/robot/diagnostic_pose",
        "/robot/diagnostic_trajectory",
    }
    has_rich_3d = rich_topics.issubset(schemas)
    primary_image = (
        "/camera"
        if "/camera" in discovered_image_topics
        else (discovered_image_topics[0] if discovered_image_topics else "")
    )
    image_topics = ([primary_image] if primary_image else []) + [
        topic for topic in discovered_image_topics if topic != primary_image
    ]

    def panel(panel_type: str, title: str, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "panel",
            "panelType": panel_type,
            "config": config,
            "title": title,
            "version": 1,
        }

    def split(
        direction: str, items: list[tuple[float, dict[str, Any]]]
    ) -> dict[str, Any]:
        return {
            "type": "split",
            "direction": direction,
            "items": [
                {"proportion": proportion, "content": content}
                for proportion, content in items
            ],
        }

    def image_panel(topic: str, index: int) -> dict[str, Any]:
        camera_name = topic.strip("/").rsplit("/", 1)[-1] or "camera"
        label = camera_name.replace("_", " ").title()
        if topic == "/camera":
            label = "Primary"
        return panel(
            "Image",
            f"{label} camera — preserved source RGB",
            {
                "imageMode": {
                    "imageTopic": topic,
                    "imageSchemaName": "foxglove.CompressedImage",
                },
                "synchronize": True,
                "syncedTopics": {topic: True},
                "npaCamera": {
                    "index": index,
                    "label": label,
                    "topic": topic,
                    "sourceFidelity": "source-rgb-only",
                },
            },
        )

    def camera_tabs(topics: list[str]) -> dict[str, Any]:
        return {
            "type": "tabs",
            "selectedTabIndex": 0,
            "tabs": [
                {
                    "title": (
                        (
                            "Primary"
                            if topic == "/camera"
                            else topic.rsplit("/", 1)[-1].replace("_", " ").title()
                        )
                        + f" ({topic})"
                    ),
                    "content": image_panel(topic, index),
                }
                for index, topic in enumerate(topics)
            ],
        }

    if has_rich_3d and primary_image:
        three_dee = panel(
            "ThreeDee",
            "Robot motion and end-effector trajectory",
            {
                "fixedFrame": fixed_frame,
                "followMode": "follow-none",
                "cameraState": {
                    "distance": 2.15,
                    "perspective": True,
                    "phi": 58,
                    "thetaOffset": 42,
                    "target": [0.38, 0.0, 0.32],
                    "targetOffset": [0.0, 0.0, 0.0],
                    "targetOrientation": [0.0, 0.0, 0.0, 1.0],
                    "fovy": 45,
                },
                "layers": {
                    "npa-ground-grid": {
                        "instanceId": "npa-ground-grid",
                        "layerId": "foxglove.Grid",
                        "label": "Action-space reference grid",
                        "visible": True,
                        "drawBehind": True,
                        "frameId": fixed_frame,
                        "size": 2.5,
                        "divisions": 20,
                        "lineWidth": 1,
                        "color": "#52607088",
                        "position": [0.25, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                    }
                },
                "topics": {
                    "/robot/diagnostic_scene": {"visible": True},
                    "/robot/diagnostic_pose": {
                        "visible": True,
                        "type": "axis",
                        "axisScale": 0.18,
                    },
                    "/robot/diagnostic_trajectory": {
                        "visible": True,
                        "type": "line",
                        "lineWidth": 4,
                        "gradient": ["#22D3EE", "#FBBF24"],
                    },
                },
                "synchronize": True,
                "syncedTopics": {topic: True for topic in sorted(rich_topics)},
            },
        )
        cameras = camera_tabs(image_topics)
        preferred_fields = [
            ("/metrics/execution", "reward", "reward"),
            ("/metrics/execution", "object_lift_m", "object lift (m)"),
            (
                "/metrics/execution",
                "object_goal_distance_m",
                "object-goal distance (m)",
            ),
            ("/run/state", "progress", "run progress"),
        ]
        plot_paths = [
            {
                "value": f"{topic}.{field}",
                "label": label,
                "enabled": True,
                "showLine": True,
                "lineSize": 2,
            }
            for topic, field, label in preferred_fields
            if field in list(numeric_paths.get(topic) or [])
        ]
        plot = panel(
            "Plot",
            "Execution performance",
            {
                "paths": plot_paths,
                "showLegend": True,
                "legendDisplay": "top",
                "showPlotValuesInLegend": True,
                "timeWindowMode": "automatic",
                "isSynced": True,
                "showXAxisLabels": True,
                "showYAxisLabels": True,
            },
        )
        transitions = panel(
            "StateTransitions",
            "Run phase and grasp state",
            {
                "paths": [
                    {"value": "/run/state.phase", "label": "phase", "enabled": True},
                    {
                        "value": "/run/state.contact",
                        "label": "contact",
                        "enabled": True,
                    },
                    {
                        "value": "/run/state.stable_grasp",
                        "label": "stable grasp",
                        "enabled": True,
                    },
                    {
                        "value": "/run/state.success",
                        "label": "success",
                        "enabled": True,
                    },
                ],
                "timeWindowMode": "automatic",
                "isSynced": True,
                "showPoints": True,
            },
        )
        log_topic = next(
            (topic for topic, schema in schemas.items() if schema == "foxglove.Log"),
            "/log",
        )
        log = panel(
            "Log",
            "Run events",
            {"topicToRender": log_topic, "preload": True, "minLogLevel": 1},
        )
        return {
            "version": 1,
            "content": split(
                "column",
                [
                    (0.72, split("row", [(0.58, three_dee), (0.42, cameras)])),
                    (
                        0.28,
                        split("row", [(0.45, plot), (0.30, transitions), (0.25, log)]),
                    ),
                ],
            ),
        }

    fallback_content = (
        camera_tabs(image_topics)
        if image_topics
        else panel("RawMessages", "Messages", {})
    )
    return {
        "version": 1,
        "content": fallback_content,
    }


class FoxgloveCloudClient:
    """Upload an MCAP once and wait until Foxglove can stream it."""

    def __init__(
        self,
        token: str,
        *,
        project_id: str = "",
        api_root: str = FOXGLOVE_API_ROOT,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wait_timeout_seconds: float | str | None = None,
    ) -> None:
        cleaned = str(token or "").strip()
        if not cleaned:
            raise FoxgloveCloudError(
                "Open in Foxglove Web requires FOXGLOVE_API_TOKEN in "
                "~/.npa/credentials.yaml on the agent VM.",
                status_code=503,
            )
        self._token = cleaned
        self._project_id = str(project_id or "").strip()
        self._api_root = str(api_root or FOXGLOVE_API_ROOT).rstrip("/")
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=False)
        self._owns_client = client is None
        self._sleep = sleep
        self._clock = clock
        try:
            self._wait_timeout_seconds = resolve_cloud_import_timeout_seconds(
                wait_timeout_seconds
            )
        except ValueError as exc:
            raise FoxgloveCloudError(str(exc), status_code=503) from exc

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "FoxgloveCloudClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _api_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_not_found: bool = False,
        allow_conflict: bool = False,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(
                method,
                f"{self._api_root}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                json=json_body,
                params=params,
                **(
                    {"timeout": max(0.001, float(timeout_seconds))}
                    if timeout_seconds is not None
                    else {}
                ),
            )
        except httpx.HTTPError as exc:
            raise FoxgloveCloudError(
                f"Foxglove Cloud request failed ({type(exc).__name__}); check network access and retry."
            ) from exc
        if allow_not_found and response.status_code == 404:
            return response
        if allow_conflict and response.status_code == 409:
            return response
        if response.is_success:
            return response
        raise self._response_error(response)

    @staticmethod
    def _response_error(response: httpx.Response) -> FoxgloveCloudError:
        status = response.status_code
        if status in {401, 403}:
            detail = (
                "Foxglove rejected the API token or project access. Verify the token has "
                "recording upload permission and belongs to the configured project."
            )
        elif status in {402, 413, 429}:
            detail = (
                "Foxglove refused the recording because of plan, size, rate, or storage quota. "
                "Review Foxglove usage/quota, then retry; NPA will reuse the same content key."
            )
        elif status == 409:
            detail = "Foxglove already has this recording key; retry to reuse the existing import."
        else:
            detail = f"Foxglove Cloud API returned HTTP {status} while preparing the recording."
        return FoxgloveCloudError(detail, status_code=status)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise FoxgloveCloudError(
                "Foxglove Cloud returned an invalid JSON response."
            ) from exc

    def _resolve_project_id(self) -> str:
        if self._project_id:
            return self._project_id
        response = self._api_request("GET", "/projects")
        projects = self._json(response)
        if not isinstance(projects, list) or not projects:
            raise FoxgloveCloudError(
                "The Foxglove organization has no project available for recording upload.",
                status_code=422,
            )
        ids = [str(item.get("id") or "") for item in projects if isinstance(item, dict)]
        ids = [item for item in ids if item]
        if len(ids) != 1:
            raise FoxgloveCloudError(
                "Multiple Foxglove projects are available; set NPA_FOXGLOVE_PROJECT_ID on the "
                "agent VM to choose the upload project.",
                status_code=422,
            )
        return ids[0]

    def _get_recording(
        self, key: str, *, timeout_seconds: float | None = None
    ) -> dict[str, Any] | None:
        response = self._api_request(
            "GET",
            f"/recordings/{quote(key, safe='')}",
            allow_not_found=True,
            timeout_seconds=timeout_seconds,
        )
        if response.status_code == 404:
            return None
        payload = self._json(response)
        if not isinstance(payload, dict):
            raise FoxgloveCloudError("Foxglove returned an invalid recording response.")
        return payload

    def _pending_import(
        self, key: str, *, timeout_seconds: float | None = None
    ) -> dict[str, Any] | None:
        response = self._api_request(
            "GET",
            "/data/pending-imports",
            params={"key": key, "showCompleted": "true", "limit": 20},
            timeout_seconds=timeout_seconds,
        )
        payload = self._json(response)
        if not isinstance(payload, list):
            raise FoxgloveCloudError(
                "Foxglove returned an invalid pending-import response."
            )
        matches = [
            item for item in payload if isinstance(item, dict) and item.get("requestId")
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: str(item.get("updatedAt") or item.get("createdAt") or ""),
        )

    @staticmethod
    def _ready_recording(
        payload: dict[str, Any] | None,
        *,
        key: str,
        size_bytes: int,
        uploaded: bool,
    ) -> FoxgloveCloudRecording | None:
        if not payload:
            return None
        status = str(payload.get("importStatus") or "")
        if status == "failed":
            raise FoxgloveCloudError(
                "Foxglove failed to index the uploaded MCAP. Inspect the recording/import error "
                "in Foxglove and retry after correcting the file.",
                status_code=422,
            )
        if status != "complete":
            return None
        actual_size = int(payload.get("size") or 0)
        if actual_size and actual_size != size_bytes:
            raise FoxgloveCloudError(
                "Foxglove recording key exists with a different byte size; refusing to reuse it.",
                status_code=409,
            )
        recording_id = str(payload.get("id") or "")
        if not recording_id:
            raise FoxgloveCloudError(
                "Foxglove indexed the recording without returning an ID."
            )
        return FoxgloveCloudRecording(
            recording_id=recording_id,
            recording_key=key,
            import_status=status,
            size_bytes=size_bytes,
            uploaded=uploaded,
            reused=not uploaded,
        )

    def _wait_for_ready(
        self, key: str, *, size_bytes: int, uploaded: bool
    ) -> FoxgloveCloudRecording:
        started_at = self._clock()
        deadline = started_at + self._wait_timeout_seconds
        recording_status = "missing"
        import_status = "not-observed"

        def remaining_seconds() -> float:
            return deadline - self._clock()

        def timeout_error() -> FoxgloveCloudTimeoutError:
            return FoxgloveCloudTimeoutError(
                recording_key=key,
                recording_status=recording_status,
                import_status=import_status,
                elapsed_seconds=self._clock() - started_at,
            )

        while True:
            remaining = remaining_seconds()
            if remaining <= 0:
                raise timeout_error()
            try:
                recording = self._get_recording(key, timeout_seconds=remaining)
            except FoxgloveCloudError as exc:
                if isinstance(exc.__cause__, httpx.TimeoutException):
                    raise timeout_error() from exc
                raise
            recording_status = str((recording or {}).get("importStatus") or "missing")
            if remaining_seconds() <= 0:
                raise timeout_error()
            ready = self._ready_recording(
                recording,
                key=key,
                size_bytes=size_bytes,
                uploaded=uploaded,
            )
            if ready is not None:
                return ready
            remaining = remaining_seconds()
            if remaining <= 0:
                raise timeout_error()
            try:
                pending = self._pending_import(key, timeout_seconds=remaining)
            except FoxgloveCloudError as exc:
                if isinstance(exc.__cause__, httpx.TimeoutException):
                    raise timeout_error() from exc
                raise
            if pending:
                import_status = str(pending.get("status") or "unknown")
                if import_status.lower() in {"error", "failed"}:
                    raise FoxgloveCloudError(
                        "Foxglove could not import the MCAP. Review the Foxglove import error "
                        "and available storage quota, then retry.",
                        status_code=422,
                    )
            if remaining_seconds() <= 0:
                raise timeout_error()
            remaining = remaining_seconds()
            if remaining <= 0:
                raise timeout_error()
            self._sleep(min(FOXGLOVE_CLOUD_IMPORT_POLL_SECONDS, remaining))

    def ensure_recording(
        self, local_path: str | Path, *, run_id: str
    ) -> FoxgloveCloudRecording:
        path = Path(local_path)
        if not path.is_file():
            raise FoxgloveCloudError(
                "The exported MCAP is missing on the agent VM.", status_code=404
            )
        size_bytes = path.stat().st_size
        digest_state = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest_state.update(chunk)
        digest = digest_state.hexdigest()
        key = f"npa-{digest}"
        existing = self._ready_recording(
            self._get_recording(key), key=key, size_bytes=size_bytes, uploaded=False
        )
        if existing is not None:
            return existing
        if self._pending_import(key) is not None:
            return self._wait_for_ready(key, size_bytes=size_bytes, uploaded=False)

        safe_run = (
            _SAFE_FILENAME_RE.sub("-", str(run_id or "run")).strip("-._") or "run"
        )
        payload = {
            "filename": f"{safe_run[:80]}-{digest[:12]}.mcap",
            "key": key,
            "projectId": self._resolve_project_id(),
        }
        upload = self._api_request(
            "POST", "/data/upload", json_body=payload, allow_conflict=True
        )
        if upload.status_code == 409:
            return self._wait_for_ready(key, size_bytes=size_bytes, uploaded=False)
        upload_payload = self._json(upload)
        link = (
            str(upload_payload.get("link") or "")
            if isinstance(upload_payload, dict)
            else ""
        )
        if not link:
            raise FoxgloveCloudError(
                "Foxglove did not return a signed recording upload link."
            )
        try:
            with path.open("rb") as source:
                put = self._client.put(
                    link,
                    headers={"Content-Type": "application/octet-stream"},
                    content=source,
                )
        except (OSError, httpx.HTTPError) as exc:
            raise FoxgloveCloudError(
                f"Foxglove recording upload failed ({type(exc).__name__}); retry will reuse the content key."
            ) from exc
        if not put.is_success:
            raise self._response_error(put)
        return self._wait_for_ready(key, size_bytes=size_bytes, uploaded=True)

    def ensure_layout(self, provenance: dict[str, Any]) -> FoxgloveCloudLayout:
        """Create/update one shared layout; preserve recording access if unavailable."""
        try:
            return self._ensure_layout(provenance)
        except FoxgloveCloudError as exc:
            return FoxgloveCloudLayout("", False, False, False, False, str(exc))

    def _ensure_layout(self, provenance: dict[str, Any]) -> FoxgloveCloudLayout:
        """Create one schema-versioned org layout, or reuse it unchanged."""
        if not has_rich_visualization_contract(provenance):
            raise FoxgloveCloudError(
                "The selected MCAP does not expose the NPA robot-motion v3 topic contract; "
                "the canonical shared layout was not created.",
                status_code=409,
            )
        desired = data_aware_layout_data(provenance)
        response = self._api_request("GET", "/layouts", params={"includeData": "true"})
        payload = self._json(response)
        if not isinstance(payload, list):
            raise FoxgloveCloudError("Foxglove returned an invalid layout list.")
        existing = next(
            (
                item
                for item in payload
                if isinstance(item, dict) and item.get("name") == FOXGLOVE_LAYOUT_NAME
            ),
            None,
        )
        existing_id = str(existing.get("id") or "") if existing is not None else ""
        if existing is not None:
            if not existing_id:
                raise FoxgloveCloudError(
                    "Foxglove returned the shared layout without an ID."
                )
            # This versioned canonical layout is a seed, not an enforcement
            # mechanism. Once created, preserve any organization/user edits.
            # A future incompatible seed gets a new versioned name and ID.
            return FoxgloveCloudLayout(existing_id, False, False, True)
        body = {
            "id": FOXGLOVE_LAYOUT_ID,
            "name": FOXGLOVE_LAYOUT_NAME,
            "folderName": "NPA",
            # Foxglove's public API requires API-key-created layouts to use
            # ORG_WRITE. Basic-seat members can still view shared layouts.
            "permission": "ORG_WRITE",
            "data": desired,
        }
        create_response = self._api_request(
            "POST", "/layouts", json_body=body, allow_conflict=True
        )
        if create_response.status_code == 409:
            # Concurrent hosted-action clicks can both observe an empty list.
            # The deterministic ID makes the winning create safe to discover.
            retry = self._json(
                self._api_request("GET", "/layouts", params={"includeData": "false"})
            )
            winner = (
                next(
                    (
                        item
                        for item in retry
                        if isinstance(item, dict)
                        and (
                            item.get("name") == FOXGLOVE_LAYOUT_NAME
                            or item.get("id") == FOXGLOVE_LAYOUT_ID
                        )
                    ),
                    None,
                )
                if isinstance(retry, list)
                else None
            )
            winner_id = str(winner.get("id") or "") if winner else ""
            if winner_id:
                return FoxgloveCloudLayout(winner_id, False, False, True)
            raise FoxgloveCloudError(
                "Foxglove reported a shared-layout conflict but the versioned layout "
                "could not be found; retry the action.",
                status_code=409,
            )
        created = self._json(create_response)
        layout_id = str(created.get("id") or "") if isinstance(created, dict) else ""
        if not layout_id:
            raise FoxgloveCloudError(
                "Foxglove created the shared layout without returning an ID."
            )
        return FoxgloveCloudLayout(layout_id, True, False, False)


def ensure_recording_from_credentials(
    local_path: str | Path,
    run_id: str,
    *,
    credentials_path: str | Path = "/root/.npa/credentials.yaml",
) -> dict[str, Any]:
    """Load the VM-local token into memory and return a token-free recording payload."""
    from npa.clients.credentials import load_credentials

    try:
        credentials = load_credentials(
            path=Path(credentials_path), environ={}, export_to_environment=False
        )
    except Exception as exc:  # noqa: BLE001 - convert parser/I/O failures to a token-safe API error
        raise FoxgloveCloudError(
            f"Could not load the agent's Foxglove credentials ({type(exc).__name__}).",
            status_code=503,
        ) from exc
    with FoxgloveCloudClient(
        credentials.foxglove_api_token,
        project_id=str(os.environ.get("NPA_FOXGLOVE_PROJECT_ID", "")).strip(),
    ) as client:
        return client.ensure_recording(local_path, run_id=run_id).to_dict()


def ensure_recording_and_layout_from_credentials(
    local_path: str | Path,
    run_id: str,
    provenance: dict[str, Any],
    *,
    credentials_path: str | Path = "/root/.npa/credentials.yaml",
) -> dict[str, Any]:
    """Return token-free recording and best-effort shared-layout contracts."""
    from npa.clients.credentials import load_credentials

    try:
        credentials = load_credentials(
            path=Path(credentials_path), environ={}, export_to_environment=False
        )
    except Exception as exc:  # noqa: BLE001 - convert I/O/parser errors to safe API text
        raise FoxgloveCloudError(
            f"Could not load the agent's Foxglove credentials ({type(exc).__name__}).",
            status_code=503,
        ) from exc
    with FoxgloveCloudClient(
        credentials.foxglove_api_token,
        project_id=str(os.environ.get("NPA_FOXGLOVE_PROJECT_ID", "")).strip(),
    ) as client:
        recording = client.ensure_recording(local_path, run_id=run_id).to_dict()
        recording["layout"] = client.ensure_layout(provenance).to_dict()
        return recording


def ensure_layout_from_credentials(
    provenance: dict[str, Any],
    *,
    credentials_path: str | Path = "/root/.npa/credentials.yaml",
) -> dict[str, Any]:
    """Idempotently ensure only the shared layout, returning no secret material."""
    from npa.clients.credentials import load_credentials

    try:
        credentials = load_credentials(
            path=Path(credentials_path), environ={}, export_to_environment=False
        )
    except Exception as exc:  # noqa: BLE001 - token-safe operator error
        raise FoxgloveCloudError(
            f"Could not load the agent's Foxglove credentials ({type(exc).__name__}).",
            status_code=503,
        ) from exc
    with FoxgloveCloudClient(
        credentials.foxglove_api_token,
        project_id=str(os.environ.get("NPA_FOXGLOVE_PROJECT_ID", "")).strip(),
    ) as client:
        return client.ensure_layout(provenance).to_dict()
