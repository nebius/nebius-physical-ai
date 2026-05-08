"""Convert simulation demo numpy arrays to LeRobotDataset v3.0 format.

Sim-agnostic: any simulator that outputs the expected numpy array shapes
can use this adapter. The input directory must contain episode subdirs,
each with:
    obs_workspace.npy  (T, H, W, 3) uint8
    obs_wrist.npy      (T, H, W, 3) uint8
    state.npy          (T, n_joints) float32
    actions.npy        (T, n_actions) float32

The output is a valid LeRobotDataset v3.0 directory that can be loaded
with ``LeRobotDataset("path/to/output")``.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH_TPL = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


class AdapterError(Exception):
    pass


# ── Video encoding ──────────────────────────────────────────────────────


def encode_video(
    frames: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    """Encode (T, H, W, 3) uint8 frames to MP4 via ffmpeg subprocess."""
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise AdapterError(
            f"Expected (T, H, W, 3) uint8 frames, got shape {frames.shape}"
        )
    t, h, w, _ = frames.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-g", "2",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd,
        input=frames.tobytes(),
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AdapterError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.decode()[-500:]}"
        )


# ── Statistics ──────────────────────────────────────────────────────────


def _compute_feature_stats(
    arrays: list[np.ndarray],
    is_video: bool = False,
) -> dict[str, Any]:
    """Compute min/max/mean/std/count across a list of arrays.

    For video features the stats are per-channel with shape (C, 1, 1),
    computed on normalized [0, 1] float values.
    """
    if is_video:
        # Flatten all frames into (N, H, W, C), normalize to [0,1]
        all_frames = np.concatenate(arrays, axis=0).astype(np.float64) / 255.0
        n, h, w, c = all_frames.shape
        # Per-channel stats → shape (C, 1, 1)
        per_channel = all_frames.reshape(-1, c)  # (N*H*W, C)
        return {
            "min": [[[float(per_channel[:, ch].min())]] for ch in range(c)],
            "max": [[[float(per_channel[:, ch].max())]] for ch in range(c)],
            "mean": [[[float(per_channel[:, ch].mean())]] for ch in range(c)],
            "std": [[[float(per_channel[:, ch].std())]] for ch in range(c)],
            "count": [n],
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


def _compute_episode_stats(
    arrays: dict[str, np.ndarray],
    video_keys: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-episode stats for all features."""
    stats: dict[str, dict[str, Any]] = {}
    for key, arr in arrays.items():
        stats[key] = _compute_feature_stats([arr], is_video=key in video_keys)
    return stats


# ── Parquet schema helpers ──────────────────────────────────────────────


def _build_data_schema(
    n_state: int,
    n_actions: int,
) -> pa.Schema:
    """Build the Arrow schema for data parquet files."""
    return pa.schema([
        ("observation.state", pa.list_(pa.float32(), n_state)),
        ("action", pa.list_(pa.float32(), n_actions)),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.float32()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])


# ── Main conversion ────────────────────────────────────────────────────


def discover_episodes(input_dir: Path) -> list[Path]:
    """Find episode directories sorted by name."""
    episodes = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )
    if not episodes:
        raise AdapterError(f"No episode_* directories found in {input_dir}")
    return episodes


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    fps: int = 20,
    robot_type: str = "franka_panda",
    task: str = "Pick and place cube to target",
) -> Path:
    """Convert a directory of episode numpy arrays to LeRobotDataset v3.0.

    Args:
        input_dir: Directory containing episode_NNNN/ subdirectories.
        output_dir: Where to write the dataset.
        fps: Frame rate for video encoding and timestamps.
        robot_type: Robot identifier for metadata.
        task: Task description string.

    Returns:
        Path to the output directory.
    """
    episodes = discover_episodes(input_dir)
    n_episodes = len(episodes)

    # Peek at first episode to determine shapes
    first_state = np.load(episodes[0] / "state.npy")
    first_actions = np.load(episodes[0] / "actions.npy")
    n_state = first_state.shape[1]
    n_actions = first_actions.shape[1]

    first_workspace = np.load(episodes[0] / "obs_workspace.npy", mmap_mode="r")
    img_h, img_w = first_workspace.shape[1], first_workspace.shape[2]

    video_keys = {"observation.images.workspace", "observation.images.wrist"}

    # Prepare output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    data_schema = _build_data_schema(n_state, n_actions)

    # Accumulators
    all_data_rows: list[dict[str, Any]] = []
    episode_meta_rows: list[dict[str, Any]] = []
    global_stats: dict[str, list[np.ndarray]] = {
        "observation.images.workspace": [],
        "observation.images.wrist": [],
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    global_index = 0
    total_frames = 0

    for ep_idx, ep_dir in enumerate(episodes):
        _print_progress(f"Processing episode {ep_idx}/{n_episodes}")

        # Load numpy arrays
        obs_workspace = np.load(ep_dir / "obs_workspace.npy")
        obs_wrist = np.load(ep_dir / "obs_wrist.npy")
        state = np.load(ep_dir / "state.npy")
        actions = np.load(ep_dir / "actions.npy")

        ep_len = state.shape[0]
        if obs_workspace.shape[0] != ep_len:
            raise AdapterError(
                f"Episode {ep_idx}: obs_workspace has {obs_workspace.shape[0]} "
                f"frames but state has {ep_len}"
            )

        # ── Encode videos ───────────────────────────────────────────
        for cam_key, cam_frames in [
            ("observation.images.workspace", obs_workspace),
            ("observation.images.wrist", obs_wrist),
        ]:
            video_path = output_dir / "videos" / cam_key / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            encode_video(cam_frames, video_path, fps)

        # ── Build data rows ─────────────────────────────────────────
        dataset_from_index = global_index
        for frame_idx in range(ep_len):
            row = {
                "observation.state": state[frame_idx].tolist(),
                "action": actions[frame_idx].tolist(),
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
                "index": global_index,
                "task_index": 0,
            }
            all_data_rows.append(row)
            global_index += 1

        dataset_to_index = global_index
        total_frames += ep_len

        # ── Episode stats ───────────────────────────────────────────
        ep_arrays = {
            "observation.images.workspace": obs_workspace,
            "observation.images.wrist": obs_wrist,
            "observation.state": state,
            "action": actions,
            "timestamp": np.arange(ep_len, dtype=np.float32) / fps,
            "frame_index": np.arange(ep_len, dtype=np.int64),
            "episode_index": np.full(ep_len, ep_idx, dtype=np.int64),
            "index": np.arange(dataset_from_index, dataset_to_index, dtype=np.int64),
            "task_index": np.zeros(ep_len, dtype=np.int64),
        }
        ep_stats = _compute_episode_stats(ep_arrays, video_keys)

        # Accumulate for global stats
        for key in global_stats:
            global_stats[key].append(ep_arrays[key])

        # ── Episode metadata row ────────────────────────────────────
        ep_meta: dict[str, Any] = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_to_index,
            "length": ep_len,
            "tasks": [task],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for cam_key in ["observation.images.workspace", "observation.images.wrist"]:
            ep_meta[f"videos/{cam_key}/chunk_index"] = 0
            ep_meta[f"videos/{cam_key}/file_index"] = ep_idx
            ep_meta[f"videos/{cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{cam_key}/to_timestamp"] = ep_len / fps

        # Flatten per-episode stats into columns
        for feat_key, feat_stats in ep_stats.items():
            for stat_name, stat_val in feat_stats.items():
                ep_meta[f"stats/{feat_key}/{stat_name}"] = stat_val

        episode_meta_rows.append(ep_meta)

    _print_progress("Writing data parquet...")

    # ── Write data parquet ──────────────────────────────────────────
    arrays = {
        "observation.state": pa.array(
            [r["observation.state"] for r in all_data_rows],
            type=pa.list_(pa.float32(), n_state),
        ),
        "action": pa.array(
            [r["action"] for r in all_data_rows],
            type=pa.list_(pa.float32(), n_actions),
        ),
        "episode_index": pa.array(
            [r["episode_index"] for r in all_data_rows], type=pa.int64()
        ),
        "frame_index": pa.array(
            [r["frame_index"] for r in all_data_rows], type=pa.int64()
        ),
        "timestamp": pa.array(
            [r["timestamp"] for r in all_data_rows], type=pa.float32()
        ),
        "index": pa.array(
            [r["index"] for r in all_data_rows], type=pa.int64()
        ),
        "task_index": pa.array(
            [r["task_index"] for r in all_data_rows], type=pa.int64()
        ),
    }
    data_table = pa.table(arrays, schema=data_schema)
    data_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(data_table, data_path, compression="snappy")

    _print_progress("Writing episode metadata parquet...")

    # ── Write episodes parquet ──────────────────────────────────────
    _write_episodes_parquet(
        episode_meta_rows,
        output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    _print_progress("Writing tasks parquet...")

    # ── Write tasks parquet ─────────────────────────────────────────
    _write_tasks_parquet(task, output_dir / "meta" / "tasks.parquet")

    _print_progress("Computing global stats...")

    # ── Compute and write global stats ──────────────────────────────
    stats: dict[str, dict[str, Any]] = {}
    for key, arr_list in global_stats.items():
        stats[key] = _compute_feature_stats(arr_list, is_video=key in video_keys)

    stats_path = output_dir / "meta" / "stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    _print_progress("Writing info.json...")

    # ── Write info.json ─────────────────────────────────────────────
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": DATA_PATH_TPL,
        "video_path": VIDEO_PATH_TPL,
        "data_files_size_in_mb": DEFAULT_DATA_SIZE_MB,
        "video_files_size_in_mb": DEFAULT_VIDEO_SIZE_MB,
        "features": {
            "observation.images.workspace": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [n_state],
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": [n_actions],
                "names": None,
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }
    info_path = output_dir / "meta" / "info.json"
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)

    _print_progress(
        f"Done: {n_episodes} episodes, {total_frames} frames → {output_dir}"
    )
    return output_dir


# ── Internal helpers ────────────────────────────────────────────────────


def _write_episodes_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write the episodes metadata parquet (one row per episode)."""
    if not rows:
        return

    # Build columns dynamically from the first row's keys
    columns: dict[str, list[Any]] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            columns[k].append(v)

    # Convert to pyarrow arrays with appropriate types
    pa_columns: dict[str, pa.Array] = {}
    for key, values in columns.items():
        sample = values[0]
        if isinstance(sample, int):
            pa_columns[key] = pa.array(values, type=pa.int64())
        elif isinstance(sample, float):
            pa_columns[key] = pa.array(values, type=pa.float64())
        elif isinstance(sample, list):
            # For nested lists (stats, tasks), use default inference
            pa_columns[key] = pa.array(values)
        elif isinstance(sample, str):
            pa_columns[key] = pa.array(values, type=pa.string())
        else:
            pa_columns[key] = pa.array(values)

    table = pa.table(pa_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def _write_tasks_parquet(task: str, output_path: Path) -> None:
    """Write the tasks.parquet metadata file."""
    table = pa.table({
        "task_index": pa.array([0], type=pa.int64()),
        "task": pa.array([task], type=pa.string()),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def _print_progress(msg: str) -> None:
    sys.stderr.write(f"\r{msg}\033[K\n")
    sys.stderr.flush()
