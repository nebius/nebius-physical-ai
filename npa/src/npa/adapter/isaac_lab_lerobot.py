"""Convert Isaac Lab G1 rollouts into standard LeRobotDataset v3 layout."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_SIZE_MB = 100
DEFAULT_VIDEO_SIZE_MB = 500
DATA_PATH_TPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_PATH_TPL = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EGO_VIEW_KEY = "observation.images.ego_view"

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
    (_g1_idx("right_hand_thumb_bend_joint"), _g1_idx("right_hand_thumb_rotation_joint")),
    (_g1_idx("right_wrist_yaw_joint"), _g1_idx("right_hand_aux_joint")),
]


class IsaacLabLeRobotError(Exception):
    """Raised when an Isaac Lab rollout cannot be represented as LeRobot data."""


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
) -> Path:
    """Convert raw Isaac Lab G1 numpy rollouts to standard LeRobotDataset v3.

    Raw input is a directory containing ``episode_*`` subdirectories, each with
    ``state.npy`` and ``actions.npy`` arrays in the canonical 43D G1 layout.
    """
    if fps <= 0:
        raise IsaacLabLeRobotError(f"fps must be positive, got {fps}")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    episodes = discover_episodes(input_dir)
    top_meta = _load_optional_json(input_dir / "meta.json")
    task_text = task or str(top_meta.get("task") or top_meta.get("task_id") or "Isaac Lab G1 rollout")
    robot = robot_type or str(top_meta.get("robot_type") or "unitree_g1")
    state_names = _names_from_meta(top_meta, "state_names")
    action_names = _names_from_meta(top_meta, "action_names")

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
    if include_placeholder_video:
        stats_accum[EGO_VIEW_KEY] = []

    global_index = 0
    for episode_index, episode_dir in enumerate(episodes):
        state, actions = _load_episode_arrays(episode_dir)
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
        stats_accum["episode_index"].append(np.full(ep_len, episode_index, dtype=np.int64))
        stats_accum["index"].append(np.arange(dataset_from_index, dataset_to_index, dtype=np.int64))
        stats_accum["task_index"].append(np.zeros(ep_len, dtype=np.int64))

        if include_placeholder_video:
            frames = _placeholder_video_frames(state, size=video_size)
            video_path = (
                output_dir
                / "videos"
                / EGO_VIEW_KEY
                / "chunk-000"
                / f"file-{episode_index:03d}.mp4"
            )
            _encode_video(frames, video_path, fps=fps)
            stats_accum[EGO_VIEW_KEY].append(frames)
            episode_row[f"videos/{EGO_VIEW_KEY}/chunk_index"] = 0
            episode_row[f"videos/{EGO_VIEW_KEY}/file_index"] = episode_index
            episode_row[f"videos/{EGO_VIEW_KEY}/from_timestamp"] = 0.0
            episode_row[f"videos/{EGO_VIEW_KEY}/to_timestamp"] = ep_len / fps

        episode_rows.append(episode_row)

    total_frames = len(all_rows)
    _write_data_parquet(all_rows, output_dir / "data" / "chunk-000" / "file-000.parquet")
    _write_episodes_parquet(
        episode_rows,
        output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    _write_tasks_parquet(task_text, output_dir / "meta" / "tasks.parquet")
    _write_stats(stats_accum, output_dir / "meta" / "stats.json")
    _write_info(
        output_dir / "meta" / "info.json",
        task=task_text,
        robot_type=robot,
        fps=fps,
        total_episodes=len(episodes),
        total_frames=total_frames,
        state_names=state_names,
        action_names=action_names,
        include_video=include_placeholder_video,
        video_size=video_size,
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


def _names_from_meta(meta: dict[str, Any], key: str) -> list[str]:
    raw = meta.get(key)
    if isinstance(raw, list) and len(raw) == G1_STATE_DIM and all(isinstance(v, str) for v in raw):
        return [str(v) for v in raw]
    return list(G1_STATE_NAMES_43)


def _load_episode_arrays(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    state_path = episode_dir / "state.npy"
    action_path = episode_dir / "actions.npy"
    if not state_path.exists():
        raise IsaacLabLeRobotError(f"Missing state.npy in {episode_dir}")
    if not action_path.exists():
        raise IsaacLabLeRobotError(f"Missing actions.npy in {episode_dir}")

    state = np.load(state_path).astype(np.float32, copy=False)
    actions = np.load(action_path).astype(np.float32, copy=False)
    if state.ndim != 2:
        raise IsaacLabLeRobotError(f"{state_path} must be a 2D array, got shape {state.shape}")
    if actions.ndim != 2:
        raise IsaacLabLeRobotError(f"{action_path} must be a 2D array, got shape {actions.shape}")
    if state.shape[0] != actions.shape[0]:
        raise IsaacLabLeRobotError(
            f"{episode_dir.name}: state/action length mismatch "
            f"({state.shape[0]} != {actions.shape[0]})"
        )
    if state.shape[1] != G1_STATE_DIM or actions.shape[1] != G1_STATE_DIM:
        raise IsaacLabLeRobotError(
            f"{episode_dir.name}: expected 43D G1 state/action arrays, "
            f"got state={state.shape[1]} action={actions.shape[1]}"
        )
    if state.shape[0] == 0:
        raise IsaacLabLeRobotError(f"{episode_dir.name}: episode has zero frames")
    return state, actions


def _write_data_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    schema = pa.schema(
        [
            ("observation.state", pa.list_(pa.float32(), G1_STATE_DIM)),
            ("action", pa.list_(pa.float32(), G1_STATE_DIM)),
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
                type=pa.list_(pa.float32(), G1_STATE_DIM),
            ),
            "action": pa.array(
                [row["action"] for row in rows],
                type=pa.list_(pa.float32(), G1_STATE_DIM),
            ),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float32()),
            "index": pa.array([row["index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
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


def _write_stats(stats_accum: dict[str, list[np.ndarray]], output_path: Path) -> None:
    stats = {
        key: _compute_feature_stats(arrays, is_video=(key == EGO_VIEW_KEY))
        for key, arrays in stats_accum.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))


def _compute_feature_stats(arrays: list[np.ndarray], *, is_video: bool = False) -> dict[str, Any]:
    if is_video:
        frames = np.concatenate(arrays, axis=0).astype(np.float64) / 255.0
        per_channel = frames.reshape(-1, frames.shape[-1])
        return {
            "min": [[[float(per_channel[:, channel].min())]] for channel in range(3)],
            "max": [[[float(per_channel[:, channel].max())]] for channel in range(3)],
            "mean": [[[float(per_channel[:, channel].mean())]] for channel in range(3)],
            "std": [[[float(per_channel[:, channel].std())]] for channel in range(3)],
            "count": [int(frames.shape[0])],
        }

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
    video_size: int,
) -> None:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [G1_STATE_DIM],
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": [G1_STATE_DIM],
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
        info["video_path"] = VIDEO_PATH_TPL
        info["video_files_size_in_mb"] = DEFAULT_VIDEO_SIZE_MB
        features[EGO_VIEW_KEY] = {
            "dtype": "video",
            "shape": [video_size, video_size, 3],
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
        raise IsaacLabLeRobotError(f"Expected RGB video frames, got shape {frames.shape}")
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
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, timeout=max(30, t))
    if proc.returncode != 0:
        raise IsaacLabLeRobotError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.decode(errors='ignore')[-500:]}"
        )
