"""Minimal Foxglove Cloud recording upload client for the agent backend.

The API token is accepted only as an in-memory constructor argument.  It is
sent in the Authorization header to ``api.foxglove.dev`` and is never included
in URLs, subprocess arguments, return payloads, or exception messages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx


FOXGLOVE_API_ROOT = "https://api.foxglove.dev/v1"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
FOXGLOVE_LAYOUT_NAME = "NPA Physical AI rich visualization v1"


class FoxgloveCloudError(RuntimeError):
    """Actionable, token-safe Foxglove Cloud failure."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    """Build Foxglove's documented v1 programmatic layout from real channels."""
    schemas = dict(provenance.get("schemas") or {})
    numeric_paths = dict(provenance.get("numeric_paths") or {})
    image_topics = sorted(
        topic
        for topic, schema in schemas.items()
        if schema == "foxglove.CompressedImage"
    )
    # A transform alone defines frames but paints no geometry. Never configure
    # an empty 3D panel unless a renderable schema is actually present.
    has_3d = any(
        schema
        in {
            "foxglove.PointCloud",
            "foxglove.SceneUpdate",
            "foxglove.PosesInFrame",
            "foxglove.PoseInFrame",
            "foxglove.LaserScan",
        }
        for schema in schemas.values()
    )
    panels: list[dict[str, Any]] = []
    for topic in image_topics[:2]:
        panels.append(
            {
                "type": "panel",
                "panelType": "Image",
                "config": {"imageMode": {"imageTopic": topic}},
                "title": topic.rsplit("/", 1)[-1].replace("_", " ").title(),
                "version": 1,
            }
        )
    if has_3d:
        panels.append(
            {
                "type": "panel",
                "panelType": "ThreeDee",
                "config": {"fixedFrame": "world"},
                "title": "State trajectory",
                "version": 1,
            }
        )
    paths = [
        {"value": f"{topic}.{field}", "label": field.replace("_", " ")}
        for topic, fields in sorted(numeric_paths.items())
        for field in list(fields)[:4]
    ][:8]
    if paths:
        panels.append(
            {
                "type": "panel",
                "panelType": "Plot",
                "config": {"paths": paths, "timeRange": "all", "showLegend": True},
                "title": "Execution metrics",
                "version": 1,
            }
        )
    if any(schema == "foxglove.Log" for schema in schemas.values()):
        log_topic = next(
            topic for topic, schema in schemas.items() if schema == "foxglove.Log"
        )
        panels.append(
            {
                "type": "panel",
                "panelType": "Log",
                "config": {"topicToRender": log_topic, "preload": True},
                "title": "Run events",
                "version": 1,
            }
        )
    if not panels:
        panels.append(
            {
                "type": "panel",
                "panelType": "RawMessages",
                "config": {},
                "title": "Messages",
                "version": 1,
            }
        )
    rows = [panels[index : index + 2] for index in range(0, len(panels), 2)]
    row_content = [
        {
            "type": "split",
            "direction": "row",
            "items": [{"proportion": 1, "content": panel} for panel in row],
        }
        for row in rows
    ]
    content = (
        row_content[0]
        if len(row_content) == 1
        else {
            "type": "split",
            "direction": "column",
            "items": [{"proportion": 1, "content": row} for row in row_content],
        }
    )
    return {"version": 1, "content": content}


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
    ) -> httpx.Response:
        try:
            response = self._client.request(
                method,
                f"{self._api_root}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                json=json_body,
                params=params,
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

    def _get_recording(self, key: str) -> dict[str, Any] | None:
        response = self._api_request(
            "GET", f"/recordings/{quote(key, safe='')}", allow_not_found=True
        )
        if response.status_code == 404:
            return None
        payload = self._json(response)
        if not isinstance(payload, dict):
            raise FoxgloveCloudError("Foxglove returned an invalid recording response.")
        return payload

    def _pending_import(self, key: str) -> dict[str, Any] | None:
        response = self._api_request(
            "GET",
            "/data/pending-imports",
            params={"key": key, "showCompleted": "true", "limit": 20},
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
        while True:
            ready = self._ready_recording(
                self._get_recording(key),
                key=key,
                size_bytes=size_bytes,
                uploaded=uploaded,
            )
            if ready is not None:
                return ready
            pending = self._pending_import(key)
            if pending:
                status = str(pending.get("status") or "")
                if status == "error":
                    raise FoxgloveCloudError(
                        "Foxglove could not import the MCAP. Review the Foxglove import error "
                        "and available storage quota, then retry.",
                        status_code=422,
                    )
            self._sleep(2.0)

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
        """Create/update one schema-versioned org layout, or reuse it unchanged."""
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
        if existing is not None and json.dumps(
            existing.get("data") or {}, sort_keys=True
        ) == json.dumps(desired, sort_keys=True):
            return FoxgloveCloudLayout(existing_id, False, False, True)
        body = {
            "name": FOXGLOVE_LAYOUT_NAME,
            "folderName": "NPA",
            # Foxglove's public API requires API-key-created layouts to use
            # ORG_WRITE. Basic-seat members can still view shared layouts.
            "permission": "ORG_WRITE",
            "data": desired,
        }
        if existing is None:
            created = self._json(self._api_request("POST", "/layouts", json_body=body))
            layout_id = (
                str(created.get("id") or "") if isinstance(created, dict) else ""
            )
            if not layout_id:
                raise FoxgloveCloudError(
                    "Foxglove created the shared layout without returning an ID."
                )
            return FoxgloveCloudLayout(layout_id, True, False, False)
        if not existing_id:
            raise FoxgloveCloudError(
                "Foxglove returned the shared layout without an ID."
            )
        self._api_request("PATCH", f"/layouts/{existing_id}", json_body=body)
        return FoxgloveCloudLayout(existing_id, False, True, False)


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
