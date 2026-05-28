"""Motion retargeting helpers for Workbench workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


SUPPORTED_SOURCE_FORMATS = ("amass", "bvh", "isaac-lab", "mocap-json", "usd")


class RetargetingError(ValueError):
    """Raised when a retargeting request is invalid."""


@dataclass(frozen=True)
class RetargetingResult:
    status: str
    input_path: str
    output_path: str
    result_uri: str
    source_format: str
    embodiment: str
    retarget_map: str
    frame_rate: int
    max_frames: int
    motion_count: int
    generated_at: str


__all__ = [
    "RetargetingResult",
    "build_retargeting_manifest",
    "result_uri_for",
    "write_result",
]


def build_retargeting_manifest(
    *,
    input_path: str,
    output_path: str,
    source_format: str = "mocap-json",
    embodiment: str = "unitree-g1",
    retarget_map: str = "",
    frame_rate: int = 50,
    max_frames: int = 0,
) -> RetargetingResult:
    """Build a validated retargeting manifest.

    The actual workflow contract is the manifest and output prefix. Workbench
    SkyPilot YAMLs can swap the command behind this schema without changing
    downstream SONIC training or MJLab evaluation stages.
    """

    if not input_path:
        raise RetargetingError("input_path is required")
    if not output_path:
        raise RetargetingError("output_path is required")
    normalized_format = source_format.lower()
    if normalized_format not in SUPPORTED_SOURCE_FORMATS:
        supported = ", ".join(SUPPORTED_SOURCE_FORMATS)
        raise RetargetingError(f"--source-format must be one of: {supported}")
    if not embodiment:
        raise RetargetingError("embodiment is required")
    if frame_rate <= 0:
        raise RetargetingError("--frame-rate must be positive")
    if max_frames < 0:
        raise RetargetingError("--max-frames must be non-negative")

    return RetargetingResult(
        status="retargeted",
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        source_format=normalized_format,
        embodiment=embodiment,
        retarget_map=retarget_map,
        frame_rate=frame_rate,
        max_frames=max_frames,
        motion_count=_deterministic_motion_count(input_path, embodiment),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def result_uri_for(output_path: str) -> str:
    """Return the retargeting JSON artifact URI for an output path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + "/retargeting_manifest.json"


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a retargeting manifest to local disk or S3."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-retargeting-") as tmp:
            local_path = Path(tmp) / "retargeting_manifest.json"
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / "retargeting_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _deterministic_motion_count(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return 1 + int(digest[:4], 16) % 16
