"""Convert LeRobotDataset trajectories to Rerun recordings."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq

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


@dataclass(frozen=True)
class LogicalRerunResult:
    """Rerun output and verified entity counts for a generic LeRobot recording."""

    output_rrd_path: str
    entity_counts: dict[str, int]


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


def lerobot_dataset_logical_to_rerun(
    dataset_path: str | Path,
    output_rrd_path: Path,
    *,
    input_episode_indices: list[int],
    rollout_episode_indices: list[int],
    feedback_by_episode: dict[int, dict[str, Any]],
    duration_s: float | None = None,
    max_frames_per_episode: int = 32,
) -> LogicalRerunResult:
    """Write a logical LeRobotDataset recording with demos, rollout, and eval paths.

    This adapter is intentionally generic: it maps LeRobot camera, state, action,
    timestamp, and feedback fields to stable Rerun entity paths without assuming
    a specific robot morphology.
    """

    output_ref = str(output_rrd_path)
    with ExitStack() as stack:
        local_dataset = _materialize_dataset(dataset_path, stack)
        local_output = _materialize_output(output_ref, stack)
        counts = _write_logical_lerobot_recording(
            local_dataset,
            local_output,
            input_episode_indices=input_episode_indices,
            rollout_episode_indices=rollout_episode_indices,
            feedback_by_episode=feedback_by_episode,
            duration_s=duration_s,
            max_frames_per_episode=max_frames_per_episode,
        )
        if _is_s3_uri(output_ref):
            _storage_client(output_ref).upload_file(str(local_output), output_ref)
        required = _required_logical_entities(
            input_episode_indices,
            rollout_episode_indices,
            feedback_by_episode,
            _camera_keys(_read_lerobot_metadata(local_dataset)),
        )
        counts = verify_rerun_entities(local_output, required, fallback_counts=counts)
    return LogicalRerunResult(output_rrd_path=output_ref, entity_counts=counts)


def verify_rerun_entities(
    rrd_path: Path,
    required_entities: list[str],
    *,
    fallback_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return row counts for required Rerun entities, raising on missing content."""

    try:
        from rerun.recording import load_recording
    except ImportError as exc:
        if fallback_counts is not None:
            _assert_required_entity_counts(rrd_path, fallback_counts, required_entities)
            return fallback_counts
        raise RerunAdapterError("rerun-sdk recording loader is required to verify .rrd content") from exc

    chunks = list(load_recording(rrd_path).chunks())
    counts: dict[str, int] = {}
    for entity in required_entities:
        normalized = "/" + entity.strip("/")
        counts[normalized] = sum(
            int(chunk.num_rows)
            for chunk in chunks
            if str(chunk.entity_path) == normalized and not chunk.is_static
        )
    _assert_required_entity_counts(rrd_path, counts, required_entities)
    return counts


def _assert_required_entity_counts(
    rrd_path: Path,
    counts: dict[str, int],
    required_entities: list[str],
) -> None:
    if not rrd_path.exists() or rrd_path.stat().st_size == 0:
        raise RerunAdapterError(f"Rerun recording was not written: {rrd_path}")
    missing = {
        _normalize_count_entity(entity): counts.get(_normalize_count_entity(entity), 0)
        for entity in required_entities
        if counts.get(_normalize_count_entity(entity), 0) <= 0
    }
    if missing:
        raise RerunAdapterError(f"Rerun recording is missing dynamic content for: {sorted(missing)}")


def _write_logical_lerobot_recording(
    dataset_path: Path,
    output_rrd_path: Path,
    *,
    input_episode_indices: list[int],
    rollout_episode_indices: list[int],
    feedback_by_episode: dict[int, dict[str, Any]],
    duration_s: float | None,
    max_frames_per_episode: int,
) -> dict[str, int]:
    rr, rrb = _import_rerun()
    if output_rrd_path.suffix.lower() != ".rrd":
        raise RerunAdapterError(f"Rerun output path must end in .rrd, got: {output_rrd_path}")
    if max_frames_per_episode <= 0:
        raise RerunAdapterError(f"max_frames_per_episode must be positive, got {max_frames_per_episode}")
    output_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _read_lerobot_metadata(dataset_path)
    frame_rows = _read_lerobot_rows(
        dataset_path,
        sorted(set(input_episode_indices + rollout_episode_indices)),
        duration_s=duration_s,
        fps=int(metadata.get("fps") or DEFAULT_DURATION_CAP_S),
        max_frames_per_episode=max_frames_per_episode,
    )
    camera_keys = _camera_keys(metadata)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="input_dataset",
                contents="input_dataset/**",
                name="Input demos",
            ),
            rrb.Spatial2DView(
                origin="policy_rollout",
                contents="policy_rollout/**",
                name="Policy rollout",
            ),
            rrb.TimeSeriesView(
                origin="eval",
                contents="eval/**",
                name="VLM/VLA eval",
            ),
            column_shares=[2.0, 2.0, 1.0],
        ),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=TIMELINE),
        auto_layout=False,
    )
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd_path, default_blueprint=blueprint, recording=recording)
    rr.send_blueprint(blueprint, recording=recording)
    video_entities = _log_dataset_videos(rr, recording, dataset_path, metadata, camera_keys)

    for episode in input_episode_indices:
        _log_episode_rows(
            rr,
            recording,
            rows=frame_rows.get(int(episode), []),
            root=f"input_dataset/episodes/episode_{int(episode):06d}",
            video_entities=video_entities,
            camera_keys=camera_keys,
        )
    for episode in rollout_episode_indices:
        _log_episode_rows(
            rr,
            recording,
            rows=frame_rows.get(int(episode), []),
            root=f"policy_rollout/episodes/episode_{int(episode):06d}",
            video_entities=video_entities,
            camera_keys=camera_keys,
        )
        _log_feedback(rr, recording, int(episode), feedback_by_episode.get(int(episode), {}))
    counts = _logical_entity_counts(
        frame_rows,
        input_episode_indices=input_episode_indices,
        rollout_episode_indices=rollout_episode_indices,
        feedback_by_episode=feedback_by_episode,
        video_entities=video_entities,
    )
    rr.disconnect(recording=recording)

    if not output_rrd_path.exists() or output_rrd_path.stat().st_size == 0:
        raise RerunAdapterError(f"Rerun recording was not written: {output_rrd_path}")
    return counts


def _logical_entity_counts(
    frame_rows: dict[int, list[dict[str, Any]]],
    *,
    input_episode_indices: list[int],
    rollout_episode_indices: list[int],
    feedback_by_episode: dict[int, dict[str, Any]],
    video_entities: dict[str, str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in input_episode_indices:
        _count_episode_rows(
            counts,
            rows=frame_rows.get(int(episode), []),
            root=f"input_dataset/episodes/episode_{int(episode):06d}",
            video_entities=video_entities,
        )
    for episode in rollout_episode_indices:
        _count_episode_rows(
            counts,
            rows=frame_rows.get(int(episode), []),
            root=f"policy_rollout/episodes/episode_{int(episode):06d}",
            video_entities=video_entities,
        )
        if int(episode) in feedback_by_episode:
            root = f"eval/episodes/episode_{int(episode):06d}"
            _increment_entity_count(counts, f"{root}/success")
            _increment_entity_count(counts, f"{root}/score")
            _increment_entity_count(counts, f"{root}/critique")
    return counts


def _count_episode_rows(
    counts: dict[str, int],
    *,
    rows: list[dict[str, Any]],
    root: str,
    video_entities: dict[str, str],
) -> None:
    for row in rows:
        for camera_key in video_entities:
            _increment_entity_count(counts, f"{root}/camera/{_entity_key(camera_key)}")
        state = _as_float_list(row.get("observation.state"))
        for index, _value in enumerate(state):
            _increment_entity_count(counts, f"{root}/state/dim_{index:02d}")
        if len(state) >= 2:
            _increment_entity_count(counts, f"{root}/state/transform")
        for index, _value in enumerate(_as_float_list(row.get("action"))):
            _increment_entity_count(counts, f"{root}/actions/dim_{index:02d}")


def _increment_entity_count(counts: dict[str, int], entity: str) -> None:
    normalized = _normalize_count_entity(entity)
    counts[normalized] = counts.get(normalized, 0) + 1


def _normalize_count_entity(entity: str) -> str:
    return "/" + entity.strip("/")


def _log_episode_rows(
    rr: Any,
    recording: Any,
    *,
    rows: list[dict[str, Any]],
    root: str,
    video_entities: dict[str, str],
    camera_keys: list[str],
) -> None:
    for row in rows:
        timestamp = float(row.get("timestamp", 0.0) or 0.0)
        _set_time_seconds(rr, recording, timestamp)
        for camera_key in camera_keys:
            video_entity = video_entities.get(camera_key, "")
            if video_entity:
                rr.log(
                    f"{root}/camera/{_entity_key(camera_key)}",
                    rr.VideoFrameReference(seconds=timestamp, video_reference=video_entity),
                    recording=recording,
                )
        for index, value in enumerate(_as_float_list(row.get("observation.state"))):
            rr.log(f"{root}/state/dim_{index:02d}", rr.Scalars(float(value)), recording=recording)
        state = _as_float_list(row.get("observation.state"))
        if len(state) >= 2:
            rr.log(
                f"{root}/state/transform",
                rr.Transform3D(translation=[float(state[0]), float(state[1]), 0.0]),
                recording=recording,
            )
        for index, value in enumerate(_as_float_list(row.get("action"))):
            rr.log(f"{root}/actions/dim_{index:02d}", rr.Scalars(float(value)), recording=recording)


def _log_feedback(rr: Any, recording: Any, episode: int, feedback: dict[str, Any]) -> None:
    root = f"eval/episodes/episode_{episode:06d}"
    _set_time_seconds(rr, recording, float(feedback.get("timestamp", 0.0) or 0.0))
    rr.log(f"{root}/success", rr.Scalars(1.0 if feedback.get("success") else 0.0), recording=recording)
    rr.log(f"{root}/score", rr.Scalars(float(feedback.get("score", 0.0))), recording=recording)
    rationale = str(feedback.get("rationale") or feedback.get("critique") or "No critique provided.")
    rr.log(f"{root}/critique", rr.TextDocument(rationale, media_type="text/plain"), recording=recording)


def _log_dataset_videos(
    rr: Any,
    recording: Any,
    dataset_path: Path,
    metadata: dict[str, Any],
    camera_keys: list[str],
) -> dict[str, str]:
    video_entities: dict[str, str] = {}
    video_path_pattern = str(metadata.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    for camera_key in camera_keys:
        candidate = dataset_path / video_path_pattern.format(
            video_key=camera_key,
            chunk_index=0,
            file_index=0,
        )
        if not candidate.exists():
            matches = sorted((dataset_path / "videos" / camera_key).rglob("*.mp4"))
            candidate = matches[0] if matches else candidate
        if candidate.exists():
            entity = f"input_dataset/videos/{_entity_key(camera_key)}"
            rr.log(entity, rr.AssetVideo(path=candidate), static=True, recording=recording)
            video_entities[camera_key] = entity
    return video_entities


def _read_lerobot_metadata(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise RerunAdapterError(f"LeRobotDataset meta/info.json is missing: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _camera_keys(metadata: dict[str, Any]) -> list[str]:
    features = metadata.get("features") or {}
    return sorted(
        key
        for key, value in features.items()
        if str(key).startswith("observation.") and isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    )


def _read_lerobot_rows(
    dataset_path: Path,
    episode_indices: list[int],
    *,
    duration_s: float | None,
    fps: int,
    max_frames_per_episode: int,
) -> dict[int, list[dict[str, Any]]]:
    data_dir = dataset_path / "data"
    parquet_paths = sorted(path for path in data_dir.rglob("*.parquet") if not path.name.startswith("._"))
    if not parquet_paths:
        raise RerunAdapterError(f"No LeRobot parquet files found under {data_dir}")
    requested = {int(index) for index in episode_indices}
    rows_by_episode: dict[int, list[dict[str, Any]]] = {episode: [] for episode in requested}
    columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
    for path in parquet_paths:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            if episode in requested:
                rows_by_episode[episode].append(row)
    frame_limit = max_frames_per_episode
    if duration_s is not None:
        if duration_s <= 0:
            raise RerunAdapterError(f"duration_s must be positive, got: {duration_s}")
        frame_limit = min(frame_limit, max(1, int(round(duration_s * max(1, fps)))))
    for episode, rows in rows_by_episode.items():
        rows.sort(key=lambda item: int(item.get("frame_index", 0) or 0))
        if not rows:
            raise RerunAdapterError(f"Episode {episode} has no rows")
        if len(rows) > frame_limit:
            indices = np.rint(np.linspace(0, len(rows) - 1, frame_limit)).astype(np.int64)
            rows_by_episode[episode] = [rows[int(index)] for index in indices]
    return rows_by_episode


def _required_logical_entities(
    input_episode_indices: list[int],
    rollout_episode_indices: list[int],
    feedback_by_episode: dict[int, dict[str, Any]],
    camera_keys: list[str],
) -> list[str]:
    required: list[str] = []
    first_camera = _entity_key(camera_keys[0]) if camera_keys else ""
    if input_episode_indices:
        episode = int(input_episode_indices[0])
        required.extend(
            [
                *(
                    [f"input_dataset/episodes/episode_{episode:06d}/camera/{first_camera}"]
                    if first_camera
                    else []
                ),
                f"input_dataset/episodes/episode_{episode:06d}/state/dim_00",
                f"input_dataset/episodes/episode_{episode:06d}/state/transform",
                f"input_dataset/episodes/episode_{episode:06d}/actions/dim_00",
            ]
        )
    if rollout_episode_indices:
        episode = int(rollout_episode_indices[0])
        required.extend(
            [
                *(
                    [f"policy_rollout/episodes/episode_{episode:06d}/camera/{first_camera}"]
                    if first_camera
                    else []
                ),
                f"policy_rollout/episodes/episode_{episode:06d}/state/dim_00",
                f"policy_rollout/episodes/episode_{episode:06d}/state/transform",
                f"policy_rollout/episodes/episode_{episode:06d}/actions/dim_00",
            ]
        )
        if episode in feedback_by_episode:
            required.extend(
                [
                    f"eval/episodes/episode_{episode:06d}/success",
                    f"eval/episodes/episode_{episode:06d}/score",
                    f"eval/episodes/episode_{episode:06d}/critique",
                ]
            )
    return required


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, list | tuple):
        return [float(item) for item in value]
    return [float(value)]


def _entity_key(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


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
