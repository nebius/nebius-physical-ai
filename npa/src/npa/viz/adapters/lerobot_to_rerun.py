"""Convert LeRobotDataset G1 trajectories to Rerun recordings."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import numpy as np

from npa.adapter.isaac_lab_lerobot import G1_BONE_PAIRS, G1_STATE_NAMES_43
from npa.clients.config import list_projects, resolve_project_storage
from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient
from npa.viz.lerobot import (
    g1_state_vectors_to_skeleton,
    load_lerobot_state_vectors,
)


DEFAULT_DURATION_CAP_S = 5.0
TIMELINE = "frame_time"
APPLICATION_ID = "npa_lerobot_to_rerun"
REPRESENTATIVE_JOINTS = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
)
_G1_INDEX = {name: index for index, name in enumerate(G1_STATE_NAMES_43)}


class RerunAdapterError(Exception):
    """Raised when a LeRobotDataset cannot be exported to Rerun."""


def lerobot_to_rerun(
    dataset_path: str | Path,
    output_rrd_path: Path,
    entity_root: str = "world/skeleton",
    color: tuple[int, int, int] = (0, 217, 255),
    duration_s: float | None = None,
) -> None:
    """Write a Rerun ``.rrd`` recording for a Unitree G1 LeRobotDataset."""
    output_ref = str(output_rrd_path)
    with ExitStack() as stack:
        local_dataset = _materialize_dataset(dataset_path, stack)
        local_output = _materialize_output(output_ref, stack)
        _write_lerobot_recording(
            local_dataset,
            local_output,
            entity_root=entity_root,
            color=color,
            duration_s=duration_s,
        )
        if _is_s3_uri(output_ref):
            _storage_client(output_ref).upload_file(str(local_output), output_ref)


def _write_lerobot_recording(
    dataset_path: Path,
    output_rrd_path: Path,
    *,
    entity_root: str,
    color: tuple[int, int, int],
    duration_s: float | None,
) -> None:
    rr, rrb = _import_rerun()
    color_rgb = _normalize_color(color)
    entity_root = _normalize_entity_root(entity_root)
    state_vectors, source_fps, _title = load_lerobot_state_vectors(dataset_path)
    selected_states, _indices, resolved_duration_s = _select_adapter_frames(
        state_vectors,
        fps=source_fps,
        duration_s=duration_s,
    )
    skeleton = g1_state_vectors_to_skeleton(selected_states)

    output_rrd_path = Path(output_rrd_path)
    if output_rrd_path.suffix.lower() != ".rrd":
        raise RerunAdapterError(f"Rerun output path must end in .rrd, got: {output_rrd_path}")
    output_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = _build_blueprint(rrb)
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd_path, default_blueprint=blueprint, recording=recording)
    rr.send_blueprint(blueprint, recording=recording)
    _log_angle_series_styles(rr, recording, entity_root, color_rgb)
    for frame_idx, (positions, state) in enumerate(zip(skeleton, selected_states, strict=True)):
        _set_time_seconds(rr, recording, frame_idx / float(source_fps))
        _log_frame(rr, recording, entity_root, positions, state, color_rgb)
    rr.disconnect(recording=recording)

    if not output_rrd_path.exists() or output_rrd_path.stat().st_size == 0:
        raise RerunAdapterError(f"Rerun recording was not written: {output_rrd_path}")
    if resolved_duration_s <= 0:
        raise RerunAdapterError("Resolved recording duration must be positive")


def _log_frame(
    rr: Any,
    recording: Any,
    entity_root: str,
    positions: np.ndarray,
    state: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    rr.log(
        f"{entity_root}/joints",
        rr.Points3D(
            positions,
            colors=[color] * int(positions.shape[0]),
            radii=0.04,
        ),
        recording=recording,
    )
    bone_segments = np.asarray(
        [[positions[parent], positions[child]] for parent, child in G1_BONE_PAIRS],
        dtype=np.float32,
    )
    rr.log(
        f"{entity_root}/bones",
        rr.LineStrips3D(bone_segments, colors=[color] * len(G1_BONE_PAIRS)),
        recording=recording,
    )
    for joint_name in REPRESENTATIVE_JOINTS:
        rr.log(
            f"{entity_root}/angles/{joint_name}",
            _scalar_archetype(rr, float(state[_G1_INDEX[joint_name]])),
            recording=recording,
        )


def _log_angle_series_styles(rr: Any, recording: Any, entity_root: str, color: tuple[int, int, int]) -> None:
    if not hasattr(rr, "SeriesLines"):
        return
    for joint_name in REPRESENTATIVE_JOINTS:
        rr.log(
            f"{entity_root}/angles/{joint_name}",
            rr.SeriesLines(colors=[color], names=[joint_name]),
            static=True,
            recording=recording,
        )


def _build_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                contents="world/**",
                name="G1 trajectory",
                background=rrb.Background(color=(26, 26, 26)),
                eye_controls=rrb.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=(2.6, -3.4, 2.2),
                    look_target=(0.0, 0.0, 0.8),
                    eye_up=(0.0, 0.0, 1.0),
                ),
            ),
            rrb.TimeSeriesView(
                origin="world",
                contents="world/**/angles/**",
                name="Representative joint angles",
            ),
            column_shares=[3.0, 1.0],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=TIMELINE),
        auto_layout=False,
    )


def _select_adapter_frames(
    state_vectors: np.ndarray,
    *,
    fps: int,
    duration_s: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if fps <= 0:
        raise RerunAdapterError(f"fps must be positive, got {fps}")
    frame_count = int(state_vectors.shape[0])
    if frame_count <= 0:
        raise RerunAdapterError("Cannot export an empty trajectory")
    if duration_s is not None and duration_s <= 0:
        raise RerunAdapterError(f"duration_s must be positive, got {duration_s}")

    source_duration_s = frame_count / float(fps)
    resolved_duration_s = min(
        source_duration_s,
        DEFAULT_DURATION_CAP_S,
        float(duration_s) if duration_s is not None else DEFAULT_DURATION_CAP_S,
    )
    target_frames = max(1, min(frame_count, int(round(resolved_duration_s * fps))))
    if target_frames == frame_count:
        indices = np.arange(frame_count, dtype=np.int64)
    else:
        indices = np.rint(np.linspace(0, frame_count - 1, target_frames)).astype(np.int64)
    return state_vectors[indices], indices, resolved_duration_s


def _set_time_seconds(rr: Any, recording: Any, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(TIMELINE, seconds, recording=recording)
    else:
        rr.set_time(TIMELINE, duration=seconds, recording=recording)


def _scalar_archetype(rr: Any, value: float) -> Any:
    if hasattr(rr, "Scalars"):
        return rr.Scalars(value)
    return rr.Scalar(value)


def _materialize_dataset(dataset_path: str | Path, stack: ExitStack) -> Path:
    dataset_ref = str(dataset_path)
    if not _is_s3_uri(dataset_ref):
        return Path(dataset_path)
    temp_dir = stack.enter_context(TemporaryDirectory(prefix="npa-rerun-dataset-"))
    return Path(_storage_client(dataset_ref).download_directory(dataset_ref, temp_dir))


def _materialize_output(output_ref: str, stack: ExitStack) -> Path:
    if not _is_s3_uri(output_ref):
        output = Path(output_ref)
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    temp_dir = stack.enter_context(TemporaryDirectory(prefix="npa-rerun-output-"))
    parsed = urlparse(output_ref)
    output_name = Path(parsed.path).name or "lerobot.rrd"
    return Path(temp_dir) / output_name


def _storage_client(bucket_uri: str | None = None) -> StorageClient:
    credentials = load_credentials()
    endpoint_url = credentials.s3_endpoint
    access_key_id = credentials.s3_access_key_id
    secret_access_key = credentials.s3_secret_access_key
    if not (endpoint_url and access_key_id and secret_access_key):
        storage = _matching_project_storage(bucket_uri)
        endpoint_url = endpoint_url or storage.get("endpoint_url", "")
        access_key_id = access_key_id or storage.get("aws_access_key_id", "")
        secret_access_key = secret_access_key or storage.get("aws_secret_access_key", "")
    return StorageClient.from_environment(
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )


def _matching_project_storage(bucket_uri: str | None) -> dict[str, str]:
    target_bucket = _bucket_name(bucket_uri) if bucket_uri else ""
    fallback: dict[str, str] = {}
    for project in list_projects():
        storage = resolve_project_storage(project)
        candidate = {
            "checkpoint_bucket": storage.checkpoint_bucket,
            "endpoint_url": storage.endpoint_url,
            "aws_access_key_id": storage.aws_access_key_id,
            "aws_secret_access_key": storage.aws_secret_access_key,
        }
        if not (
            candidate["endpoint_url"]
            and candidate["aws_access_key_id"]
            and candidate["aws_secret_access_key"]
        ):
            continue
        if not fallback:
            fallback = candidate
        if target_bucket and _bucket_name(candidate["checkpoint_bucket"]) == target_bucket:
            return candidate
    return fallback


def _bucket_name(uri_or_bucket: str | None) -> str:
    if not uri_or_bucket:
        return ""
    value = uri_or_bucket.strip()
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.split("/", 1)[0]


def _normalize_entity_root(entity_root: str) -> str:
    normalized = entity_root.strip("/")
    if not normalized:
        raise RerunAdapterError("entity_root must not be empty")
    return normalized


def _normalize_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(color) != 3:
        raise RerunAdapterError(f"color must be an RGB tuple, got: {color}")
    rgb = tuple(int(channel) for channel in color)
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise RerunAdapterError(f"color channels must be in 0..255, got: {color}")
    return rgb


def _is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def _import_rerun() -> tuple[Any, Any]:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RerunAdapterError("rerun-sdk is required to write .rrd recordings") from exc
    if not hasattr(rr, "Scalars") and not hasattr(rr, "Scalar"):
        raise RerunAdapterError("rerun-sdk does not expose a scalar archetype")
    return rr, rrb
