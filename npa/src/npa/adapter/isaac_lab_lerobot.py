"""Convert Isaac Lab numpy rollouts into standard LeRobotDataset v3 layout.

The writer half is robot-agnostic and driven by a ``LeRobotFeatureSpec``
(joint names, dimensions, robot type). The module ships a Unitree G1 default
spec so existing G1 callers keep working unchanged; other robots pass their
own spec to ``convert``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryFile
from threading import Thread
from typing import Any, BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_SIZE_MB = 100
DEFAULT_VIDEO_SIZE_MB = 500
DATA_PATH_TPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH_TPL = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EGO_VIEW_KEY = "observation.images.ego_view"
WORKSPACE_VIEW_KEY = "observation.images.workspace"
RGB_FRAMES_FILENAME = "rgb.npy"

G1_STATE_NAMES_43 = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_pinky_joint",
    "left_hand_ring_joint",
    "left_hand_middle_joint",
    "left_hand_index_joint",
    "left_hand_thumb_bend_joint",
    "left_hand_thumb_rotation_joint",
    "left_hand_aux_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_pinky_joint",
    "right_hand_ring_joint",
    "right_hand_middle_joint",
    "right_hand_index_joint",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rotation_joint",
    "right_hand_aux_joint",
]
G1_STATE_DIM = len(G1_STATE_NAMES_43)
_G1_STATE_INDEX = {name: index for index, name in enumerate(G1_STATE_NAMES_43)}


def _g1_idx(name: str) -> int:
    return _G1_STATE_INDEX[name]


G1_BONE_PAIRS = [
    (_g1_idx("waist_yaw_joint"), _g1_idx("waist_roll_joint")),
    (_g1_idx("waist_roll_joint"), _g1_idx("waist_pitch_joint")),
    (_g1_idx("waist_yaw_joint"), _g1_idx("left_hip_yaw_joint")),
    (_g1_idx("left_hip_yaw_joint"), _g1_idx("left_hip_roll_joint")),
    (_g1_idx("left_hip_roll_joint"), _g1_idx("left_hip_pitch_joint")),
    (_g1_idx("left_hip_pitch_joint"), _g1_idx("left_knee_joint")),
    (_g1_idx("left_knee_joint"), _g1_idx("left_ankle_pitch_joint")),
    (_g1_idx("left_ankle_pitch_joint"), _g1_idx("left_ankle_roll_joint")),
    (_g1_idx("waist_yaw_joint"), _g1_idx("right_hip_yaw_joint")),
    (_g1_idx("right_hip_yaw_joint"), _g1_idx("right_hip_roll_joint")),
    (_g1_idx("right_hip_roll_joint"), _g1_idx("right_hip_pitch_joint")),
    (_g1_idx("right_hip_pitch_joint"), _g1_idx("right_knee_joint")),
    (_g1_idx("right_knee_joint"), _g1_idx("right_ankle_pitch_joint")),
    (_g1_idx("right_ankle_pitch_joint"), _g1_idx("right_ankle_roll_joint")),
    (_g1_idx("waist_pitch_joint"), _g1_idx("left_shoulder_pitch_joint")),
    (_g1_idx("left_shoulder_pitch_joint"), _g1_idx("left_shoulder_roll_joint")),
    (_g1_idx("left_shoulder_roll_joint"), _g1_idx("left_shoulder_yaw_joint")),
    (_g1_idx("left_shoulder_yaw_joint"), _g1_idx("left_elbow_pitch_joint")),
    (_g1_idx("left_elbow_pitch_joint"), _g1_idx("left_elbow_roll_joint")),
    (_g1_idx("left_elbow_roll_joint"), _g1_idx("left_wrist_pitch_joint")),
    (_g1_idx("left_wrist_pitch_joint"), _g1_idx("left_wrist_yaw_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_pinky_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_ring_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_middle_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_index_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_thumb_bend_joint")),
    (_g1_idx("left_hand_thumb_bend_joint"), _g1_idx("left_hand_thumb_rotation_joint")),
    (_g1_idx("left_wrist_yaw_joint"), _g1_idx("left_hand_aux_joint")),
    (_g1_idx("waist_pitch_joint"), _g1_idx("right_shoulder_pitch_joint")),
    (_g1_idx("right_shoulder_pitch_joint"), _g1_idx("right_shoulder_roll_joint")),
    (_g1_idx("right_shoulder_roll_joint"), _g1_idx("right_shoulder_yaw_joint")),
    (_g1_idx("right_shoulder_yaw_joint"), _g1_idx("right_elbow_pitch_joint")),
    (_g1_idx("right_elbow_pitch_joint"), _g1_idx("right_elbow_roll_joint")),
    (_g1_idx("right_elbow_roll_joint"), _g1_idx("right_wrist_pitch_joint")),
    (_g1_idx("right_wrist_pitch_joint"), _g1_idx("right_wrist_yaw_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_pinky_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_ring_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_middle_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_index_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_thumb_bend_joint")),
    (
        _g1_idx("right_hand_thumb_bend_joint"),
        _g1_idx("right_hand_thumb_rotation_joint"),
    ),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_aux_joint")),
]


class IsaacLabLeRobotError(Exception):
    """Raised when an Isaac Lab rollout cannot be represented as LeRobot data."""


@dataclass(frozen=True)
class LeRobotFeatureSpec:
    """Robot-specific feature layout for the LeRobot writer.

    ``state_dim``/``action_dim`` default to the length of the corresponding
    name lists; pass them explicitly only when the arrays are wider than the
    named joints.
    """

    state_names: list[str]
    action_names: list[str]
    robot_type: str
    state_dim: int = field(default=0)
    action_dim: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.state_names or not self.action_names:
            raise IsaacLabLeRobotError(
                "LeRobotFeatureSpec requires state and action names"
            )
        if self.state_dim <= 0:
            object.__setattr__(self, "state_dim", len(self.state_names))
        if self.action_dim <= 0:
            object.__setattr__(self, "action_dim", len(self.action_names))


G1_FEATURE_SPEC = LeRobotFeatureSpec(
    state_names=list(G1_STATE_NAMES_43),
    action_names=list(G1_STATE_NAMES_43),
    robot_type="unitree_g1",
)


def discover_episodes(input_dir: Path) -> list[Path]:
    episodes = sorted(
        path
        for path in Path(input_dir).iterdir()
        if path.is_dir() and path.name.startswith("episode_")
    )
    if not episodes:
        raise IsaacLabLeRobotError(f"No episode_* directories found in {input_dir}")
    return episodes


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    fps: int = 50,
    robot_type: str = "unitree_g1",
    task: str = "",
    include_placeholder_video: bool = False,
    video_size: int = 64,
    spec: LeRobotFeatureSpec | None = None,
) -> Path:
    """Convert raw Isaac Lab numpy rollouts to standard LeRobotDataset v3.

    Raw input is a directory containing ``episode_*`` subdirectories, each with
    ``state.npy`` and ``actions.npy`` arrays matching ``spec`` (default: the
    canonical 43D Unitree G1 layout). Input and output must resolve to disjoint
    directories because conversion replaces the output directory.
    """
    if fps <= 0:
        raise IsaacLabLeRobotError(f"fps must be positive, got {fps}")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    resolved_input = input_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_input.is_relative_to(resolved_output) or resolved_output.is_relative_to(
        resolved_input
    ):
        raise IsaacLabLeRobotError("Input and output directories must not overlap")
    episodes = discover_episodes(input_dir)
    top_meta = _load_optional_json(input_dir / "meta.json")
    rgb_episode_paths = [episode / RGB_FRAMES_FILENAME for episode in episodes]
    rgb_presence = [path.is_file() for path in rgb_episode_paths]
    if any(rgb_presence) and not all(rgb_presence):
        missing = [
            episodes[index].name
            for index, present in enumerate(rgb_presence)
            if not present
        ]
        raise IsaacLabLeRobotError(
            "Isaac RGB capture must cover every episode; missing rgb.npy in "
            + ", ".join(missing)
        )
    has_simulator_rgb = all(rgb_presence)
    include_video = has_simulator_rgb or include_placeholder_video
    video_key = WORKSPACE_VIEW_KEY if has_simulator_rgb else EGO_VIEW_KEY
    video_source = (
        "isaac_sim_rgb_array" if has_simulator_rgb else "synthetic_placeholder"
    )
    resolved_spec = spec or G1_FEATURE_SPEC
    default_task = (
        "Isaac Lab G1 rollout"
        if resolved_spec.robot_type == "unitree_g1"
        else f"Isaac Lab {resolved_spec.robot_type} rollout"
    )
    task_text = task or str(
        top_meta.get("task") or top_meta.get("task_id") or default_task
    )
    if spec is not None and robot_type == "unitree_g1":
        robot = spec.robot_type
    else:
        robot = robot_type or str(
            top_meta.get("robot_type") or resolved_spec.robot_type
        )
    state_names = _names_from_meta(
        top_meta,
        "state_names",
        expected_dim=resolved_spec.state_dim,
        fallback=resolved_spec.state_names,
    )
    action_names = _names_from_meta(
        top_meta,
        "action_names",
        expected_dim=resolved_spec.action_dim,
        fallback=resolved_spec.action_names,
    )

    _reset_dir(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    stats_accum: dict[str, list[np.ndarray]] = {
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    video_stats = _RgbStats() if include_video else None

    global_index = 0
    video_shape: tuple[int, int, int] | None = None
    for episode_index, episode_dir in enumerate(episodes):
        state, actions = _load_episode_arrays(episode_dir, spec=resolved_spec)
        ep_len = int(state.shape[0])
        dataset_from_index = global_index

        for frame_index in range(ep_len):
            all_rows.append(
                {
                    "observation.state": state[frame_index].tolist(),
                    "action": actions[frame_index].tolist(),
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": frame_index / fps,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1

        dataset_to_index = global_index
        episode_row: dict[str, Any] = {
            "episode_index": episode_index,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            "length": ep_len,
            "tasks": [task_text],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        timestamps = np.arange(ep_len, dtype=np.float32) / fps
        stats_accum["observation.state"].append(state)
        stats_accum["action"].append(actions)
        stats_accum["timestamp"].append(timestamps)
        stats_accum["frame_index"].append(np.arange(ep_len, dtype=np.int64))
        stats_accum["episode_index"].append(
            np.full(ep_len, episode_index, dtype=np.int64)
        )
        stats_accum["index"].append(
            np.arange(dataset_from_index, dataset_to_index, dtype=np.int64)
        )
        stats_accum["task_index"].append(np.zeros(ep_len, dtype=np.int64))

        if include_video:
            frames = (
                _load_rgb_frames(episode_dir, expected_frames=ep_len)
                if has_simulator_rgb
                else _placeholder_video_frames(state, size=video_size)
            )
            current_shape = tuple(int(value) for value in frames.shape[1:])
            if video_shape is None:
                video_shape = current_shape
            elif current_shape != video_shape:
                raise IsaacLabLeRobotError(
                    f"{episode_dir.name}: RGB dimensions {current_shape} do not match "
                    f"the first episode {video_shape}"
                )
            video_path = (
                output_dir
                / "videos"
                / video_key
                / "chunk-000"
                / f"file-{episode_index:03d}.mp4"
            )
            _encode_video(frames, video_path, fps=fps)
            video_stats.update(frames)
            del frames
            episode_row[f"videos/{video_key}/chunk_index"] = 0
            episode_row[f"videos/{video_key}/file_index"] = episode_index
            episode_row[f"videos/{video_key}/from_timestamp"] = 0.0
            episode_row[f"videos/{video_key}/to_timestamp"] = ep_len / fps

        episode_rows.append(episode_row)

    total_frames = len(all_rows)
    _write_data_parquet(
        all_rows,
        output_dir / "data" / "chunk-000" / "file-000.parquet",
        spec=resolved_spec,
    )
    _write_episodes_parquet(
        episode_rows,
        output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    _write_tasks_parquet(task_text, output_dir / "meta" / "tasks.parquet")
    _write_stats(
        stats_accum,
        output_dir / "meta" / "stats.json",
        video_stats={video_key: video_stats.as_dict()}
        if video_stats is not None
        else None,
    )
    _write_info(
        output_dir / "meta" / "info.json",
        task=task_text,
        robot_type=robot,
        fps=fps,
        total_episodes=len(episodes),
        total_frames=total_frames,
        state_names=state_names,
        action_names=action_names,
        include_video=include_video,
        video_key=video_key,
        video_shape=video_shape,
        visual_provenance=_visual_provenance(
            top_meta,
            source=video_source,
            frame_count=total_frames if include_video else 0,
            video_shape=video_shape,
        ),
        spec=resolved_spec,
    )
    return output_dir


def _reset_dir(path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _names_from_meta(
    meta: dict[str, Any],
    key: str,
    *,
    expected_dim: int = G1_STATE_DIM,
    fallback: list[str] | None = None,
) -> list[str]:
    raw = meta.get(key)
    if (
        isinstance(raw, list)
        and len(raw) == expected_dim
        and all(isinstance(v, str) for v in raw)
    ):
        return [str(v) for v in raw]
    return list(fallback if fallback is not None else G1_STATE_NAMES_43)


def _load_episode_arrays(
    episode_dir: Path, *, spec: LeRobotFeatureSpec | None = None
) -> tuple[np.ndarray, np.ndarray]:
    state_path = episode_dir / "state.npy"
    action_path = episode_dir / "actions.npy"
    if not state_path.exists():
        raise IsaacLabLeRobotError(f"Missing state.npy in {episode_dir}")
    if not action_path.exists():
        raise IsaacLabLeRobotError(f"Missing actions.npy in {episode_dir}")

    state = np.load(state_path).astype(np.float32, copy=False)
    actions = np.load(action_path).astype(np.float32, copy=False)
    if state.ndim != 2:
        raise IsaacLabLeRobotError(
            f"{state_path} must be a 2D array, got shape {state.shape}"
        )
    if actions.ndim != 2:
        raise IsaacLabLeRobotError(
            f"{action_path} must be a 2D array, got shape {actions.shape}"
        )
    if state.shape[0] != actions.shape[0]:
        raise IsaacLabLeRobotError(
            f"{episode_dir.name}: state/action length mismatch "
            f"({state.shape[0]} != {actions.shape[0]})"
        )
    resolved = spec or G1_FEATURE_SPEC
    if state.shape[1] != resolved.state_dim or actions.shape[1] != resolved.action_dim:
        raise IsaacLabLeRobotError(
            f"{episode_dir.name}: expected {resolved.state_dim}D state / "
            f"{resolved.action_dim}D action arrays for {resolved.robot_type}, "
            f"got state={state.shape[1]} action={actions.shape[1]}"
        )
    if state.shape[0] == 0:
        raise IsaacLabLeRobotError(f"{episode_dir.name}: episode has zero frames")
    return state, actions


def _load_rgb_frames(episode_dir: Path, *, expected_frames: int) -> np.ndarray:
    path = episode_dir / RGB_FRAMES_FILENAME
    frames = np.load(path, allow_pickle=False, mmap_mode="r")
    if frames.ndim != 4 or frames.shape[-1] not in {3, 4}:
        raise IsaacLabLeRobotError(
            f"{path} must contain [frames,height,width,RGB(A)] pixels, got {frames.shape}"
        )
    if frames.shape[0] != expected_frames:
        raise IsaacLabLeRobotError(
            f"{episode_dir.name}: RGB/state length mismatch "
            f"({frames.shape[0]} != {expected_frames})"
        )
    if frames.dtype != np.uint8:
        raise IsaacLabLeRobotError(
            f"{path} must contain uint8 pixels, got {frames.dtype}"
        )
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[1] <= 1 or frames.shape[2] <= 1:
        raise IsaacLabLeRobotError(
            f"{path} has invalid image dimensions {frames.shape[1:3]}"
        )
    return frames


def _visual_provenance(
    top_meta: dict[str, Any],
    *,
    source: str,
    frame_count: int,
    video_shape: tuple[int, int, int] | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source": source,
        "genuine_simulator_pixels": source == "isaac_sim_rgb_array",
        "synchronized_timeline": "episode_index/frame_index/timestamp",
        "frame_count": frame_count,
        "dimensions": list(video_shape or ()),
    }
    for key in (
        "task",
        "runtime_version",
        "policy_loaded",
        "checkpoint_sha256",
        "renderer",
    ):
        if key in top_meta:
            provenance[key] = top_meta[key]
    return provenance


def _write_data_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    spec: LeRobotFeatureSpec | None = None,
) -> None:
    resolved = spec or G1_FEATURE_SPEC
    schema = pa.schema(
        [
            ("observation.state", pa.list_(pa.float32(), resolved.state_dim)),
            ("action", pa.list_(pa.float32(), resolved.action_dim)),
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("timestamp", pa.float32()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )
    table = pa.table(
        {
            "observation.state": pa.array(
                [row["observation.state"] for row in rows],
                type=pa.list_(pa.float32(), resolved.state_dim),
            ),
            "action": pa.array(
                [row["action"] for row in rows],
                type=pa.list_(pa.float32(), resolved.action_dim),
            ),
            "episode_index": pa.array(
                [row["episode_index"] for row in rows], type=pa.int64()
            ),
            "frame_index": pa.array(
                [row["frame_index"] for row in rows], type=pa.int64()
            ),
            "timestamp": pa.array(
                [row["timestamp"] for row in rows], type=pa.float32()
            ),
            "index": pa.array([row["index"] for row in rows], type=pa.int64()),
            "task_index": pa.array(
                [row["task_index"] for row in rows], type=pa.int64()
            ),
        },
        schema=schema,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def _write_episodes_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    columns: dict[str, list[Any]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)

    pa_columns: dict[str, pa.Array] = {}
    for key, values in columns.items():
        sample = values[0]
        if isinstance(sample, int):
            pa_columns[key] = pa.array(values, type=pa.int64())
        elif isinstance(sample, float):
            pa_columns[key] = pa.array(values, type=pa.float64())
        elif isinstance(sample, str):
            pa_columns[key] = pa.array(values, type=pa.string())
        else:
            pa_columns[key] = pa.array(values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(pa_columns), output_path, compression="snappy")


def _write_tasks_parquet(task: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([task], type=pa.string()),
            }
        ),
        output_path,
        compression="snappy",
    )


@dataclass
class _RgbStats:
    """Keep exact uint8 pixel counts without retaining episode images."""

    histogram: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 256), dtype=np.int64)
    )
    frame_count: int = 0

    def update(self, frames: np.ndarray) -> None:
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise IsaacLabLeRobotError(
                "RGB statistics require uint8 [frames,height,width,3]"
            )
        if any(size == 0 for size in frames.shape):
            raise IsaacLabLeRobotError("RGB statistics require nonempty frames")
        for frame in frames:
            for channel in range(3):
                self.histogram[channel] += np.bincount(
                    frame[..., channel].reshape(-1), minlength=256
                )
        self.frame_count += int(frames.shape[0])

    def as_dict(self) -> dict[str, Any]:
        if self.frame_count == 0:
            raise IsaacLabLeRobotError("RGB statistics require at least one frame")
        levels = np.arange(256, dtype=np.float64)
        pixel_count = self.histogram.sum(axis=1)
        mean = self.histogram @ levels / pixel_count
        # Center before squaring to preserve small variances and exact constant channels.
        variance = (self.histogram * (levels - mean[:, None]) ** 2).sum(
            axis=1
        ) / pixel_count
        occupied = self.histogram > 0
        values = {
            "min": occupied.argmax(axis=1) / 255.0,
            "max": (255 - occupied[:, ::-1].argmax(axis=1)) / 255.0,
            "mean": mean / 255.0,
            "std": np.sqrt(variance) / 255.0,
        }
        return {
            **{name: value.reshape(3, 1, 1).tolist() for name, value in values.items()},
            "count": [self.frame_count],
        }


def _write_stats(
    stats_accum: dict[str, list[np.ndarray]],
    output_path: Path,
    *,
    video_stats: dict[str, Any] | None = None,
) -> None:
    stats = {
        key: _compute_feature_stats(
            arrays, is_video=key.startswith("observation.images.")
        )
        for key, arrays in stats_accum.items()
    }
    stats.update(video_stats or {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))


def _compute_feature_stats(
    arrays: list[np.ndarray], *, is_video: bool = False
) -> dict[str, Any]:
    if is_video:
        video_stats = _RgbStats()
        for frames in arrays:
            video_stats.update(frames)
        return video_stats.as_dict()

    concat = np.concatenate(arrays, axis=0).astype(np.float64)
    if concat.ndim == 1:
        concat = concat.reshape(-1, 1)
    return {
        "min": concat.min(axis=0).tolist(),
        "max": concat.max(axis=0).tolist(),
        "mean": concat.mean(axis=0).tolist(),
        "std": concat.std(axis=0).tolist(),
        "count": [int(concat.shape[0])],
    }


def _write_info(
    output_path: Path,
    *,
    task: str,
    robot_type: str,
    fps: int,
    total_episodes: int,
    total_frames: int,
    state_names: list[str],
    action_names: list[str],
    include_video: bool,
    video_key: str,
    video_shape: tuple[int, int, int] | None,
    visual_provenance: dict[str, Any],
    spec: LeRobotFeatureSpec | None = None,
) -> None:
    resolved = spec or G1_FEATURE_SPEC
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [resolved.state_dim],
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": [resolved.action_dim],
            "names": [action_names],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info: dict[str, Any] = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH_TPL,
        "data_files_size_in_mb": DEFAULT_DATA_SIZE_MB,
        "features": features,
    }
    if include_video:
        if video_shape is None:
            raise IsaacLabLeRobotError("video_shape is required when video is enabled")
        height, width, channels = video_shape
        info["video_path"] = VIDEO_PATH_TPL
        info["video_files_size_in_mb"] = DEFAULT_VIDEO_SIZE_MB
        info["visual_provenance"] = visual_provenance
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2))


def _placeholder_video_frames(state: np.ndarray, *, size: int) -> np.ndarray:
    frames = np.zeros((state.shape[0], size, size, 3), dtype=np.uint8)
    mid = size // 2
    for idx, row in enumerate(state):
        color = int(np.clip((float(np.mean(row)) + 1.0) * 64.0, 0.0, 255.0))
        frames[idx, :, :, :] = [20, 20, 20]
        frames[idx, mid - 1 : mid + 1, :, :] = [color, 160, 220]
        frames[idx, :, mid - 1 : mid + 1, :] = [220, color, 80]
    return frames


def _encode_video(frames: np.ndarray, output_path: Path, *, fps: int) -> None:
    t, h, w, c = frames.shape
    if c != 3:
        raise IsaacLabLeRobotError(
            f"Expected RGB video frames, got shape {frames.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "28",
        "-g",
        "2",
        str(output_path),
    ]
    _run_video_encoder(cmd, frames, timeout=max(30, t))


def _write_video_frames(
    stream: BinaryIO, frames: np.ndarray, errors: list[OSError]
) -> None:
    try:
        with stream:
            for frame in frames:
                stream.write(frame.tobytes())
    except OSError as error:
        errors.append(error)


def _run_video_encoder(command: list[str], frames: np.ndarray, *, timeout: int) -> None:
    # A file drains diagnostics without retaining them or blocking the encoder's stderr pipe.
    with (
        TemporaryFile() as diagnostics,
        subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=diagnostics,
        ) as process,
    ):
        errors: list[OSError] = []
        writer = Thread(
            target=_write_video_frames,
            args=(process.stdin, frames, errors),
            daemon=True,
        )
        writer.start()
        try:
            process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            writer.join()
        if process.returncode != 0:
            diagnostics.seek(max(0, diagnostics.tell() - 500))
            message = diagnostics.read().decode(errors="ignore")
            raise IsaacLabLeRobotError(
                f"ffmpeg failed (exit {process.returncode}): {message}"
            )
        if errors:
            raise IsaacLabLeRobotError(
                "Failed to write RGB frames to ffmpeg"
            ) from errors[0]
