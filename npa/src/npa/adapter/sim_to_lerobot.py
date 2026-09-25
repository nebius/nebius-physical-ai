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
EPISODES_PATH_TPL = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)


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
        "23",
        "-g",
        "2",
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


def _compute_video_stats(arrays: list[np.ndarray]) -> dict[str, Any]:
    """Merge per-frame moments without allocating a float64 copy of the dataset."""
    if not arrays or arrays[0].ndim != 4:
        raise ValueError("video statistics require arrays shaped (N, H, W, C)")
    shape = arrays[0].shape[1:]
    if not all(shape) or any(
        array.ndim != 4 or array.shape[1:] != shape for array in arrays
    ):
        raise ValueError("video statistics require matching nonempty frame shapes")
    channels = shape[-1]
    minimum, maximum = np.full(channels, np.inf), np.full(channels, -np.inf)
    mean, squared_deviations = np.zeros(channels), np.zeros(channels)
    pixels = frames = 0
    for array in arrays:
        for frame in array:
            frame_pixels = frame.shape[0] * frame.shape[1]
            frame_mean = frame.mean(axis=(0, 1), dtype=np.float64)
            delta = frame_mean - mean
            combined_pixels = pixels + frame_pixels
            # Parallel variance merge retains small differences between frames.
            squared_deviations += frame.var(
                axis=(0, 1), dtype=np.float64
            ) * frame_pixels + delta * delta * (pixels * frame_pixels / combined_pixels)
            mean += delta * (frame_pixels / combined_pixels)
            minimum = np.minimum(minimum, frame.min(axis=(0, 1)))
            maximum = np.maximum(maximum, frame.max(axis=(0, 1)))
            pixels = combined_pixels
            frames += 1
    if not frames:
        raise ValueError("video statistics require at least one frame")
    values = {
        "min": minimum / 255.0,
        "max": maximum / 255.0,
        "mean": mean / 255.0,
        "std": np.sqrt(squared_deviations / pixels) / 255.0,
    }
    result = {
        key: value.reshape(channels, 1, 1).tolist() for key, value in values.items()
    }
    result["count"] = [frames]
    return result


def _compute_feature_stats(
    arrays: list[np.ndarray],
    is_video: bool = False,
) -> dict[str, Any]:
    """Compute min/max/mean/std/count across a list of arrays.

    For video features the stats are per-channel with shape (C, 1, 1),
    computed on normalized [0, 1] float values.
    """
    if is_video:
        return _compute_video_stats(arrays)

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
    return pa.schema(
        [
            ("observation.state", _numeric_feature_type(n_state)),
            ("action", _numeric_feature_type(n_actions)),
            ("episode_index", pa.int64()),
            ("frame_index", pa.int64()),
            ("timestamp", pa.float32()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
    )


def _numeric_feature_type(width: int) -> pa.DataType:
    """Match LeRobot's Hugging Face feature encoding for 1-D numerics.

    LeRobot 0.5.x maps metadata shape ``[1]`` to ``datasets.Value`` rather
    than a one-element ``Sequence``.  The physical feature is still described
    as shape ``[1]`` in ``info.json``; only its Arrow storage is scalar.
    """

    return pa.float32() if width == 1 else pa.list_(pa.float32(), width)


def _numeric_feature_values(
    rows: list[dict[str, Any]], key: str, width: int
) -> list[Any]:
    values = [row[key] for row in rows]
    if width == 1:
        return [float(value[0]) for value in values]
    return values


# ── Main conversion ────────────────────────────────────────────────────


def discover_episodes(input_dir: Path) -> list[Path]:
    """Find episode directories sorted by name."""
    episodes = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")
    )
    if not episodes:
        raise AdapterError(f"No episode_* directories found in {input_dir}")
    return episodes


def _load_episode_array(ep_dir: Path, ep_idx: int, name: str) -> np.ndarray:
    """Load one episode array, mapping unreadable files onto AdapterError."""
    try:
        return np.load(ep_dir / f"{name}.npy")
    except FileNotFoundError:
        raise
    except OSError as error:
        raise AdapterError(f"Episode {ep_idx}: cannot read {name}: {error}") from error


def _validate_numeric_stream(
    name: str, array: np.ndarray, ep_len: int, ep_idx: int
) -> None:
    """Require a finite, nonempty (T, width) state or action stream."""
    if array.ndim != 2 or not all(array.shape):
        raise AdapterError(
            f"Episode {ep_idx}: {name} requires nonempty (T, width), got {array.shape}"
        )
    if array.shape[0] != ep_len:
        raise AdapterError(
            f"Episode {ep_idx}: {name} has {array.shape[0]} frames, expected "
            f"{ep_len}; got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise AdapterError(f"Episode {ep_idx}: {name} requires real numeric values")
    if not np.isfinite(array).all():
        raise AdapterError(f"Episode {ep_idx}: {name} contains non-finite values")


def _validate_camera_stream(
    name: str, array: np.ndarray, ep_len: int, ep_idx: int
) -> None:
    """Require uint8 RGB frames of shape (T, H, W, 3) covering every timestep."""
    if array.ndim != 4 or not all(array.shape) or array.shape[3] != 3:
        raise AdapterError(
            f"Episode {ep_idx}: {name} requires nonempty (T, H, W, 3), got {array.shape}"
        )
    if array.shape[0] != ep_len:
        raise AdapterError(
            f"Episode {ep_idx}: {name} has {array.shape[0]} frames, expected "
            f"{ep_len} of shape (T, H, W, 3); got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise AdapterError(
            f"Episode {ep_idx}: {name} has dtype {array.dtype}, expected uint8"
        )


def _load_episode_arrays(ep_dir: Path, ep_idx: int) -> dict[str, np.ndarray]:
    """Load the four required episode arrays and validate their streams."""
    arrays = {
        name: _load_episode_array(ep_dir, ep_idx, name)
        for name in ("obs_workspace", "obs_wrist", "state", "actions")
    }
    ep_len = int(arrays["state"].shape[0]) if arrays["state"].ndim == 2 else -1
    _validate_numeric_stream("state", arrays["state"], ep_len, ep_idx)
    _validate_numeric_stream("actions", arrays["actions"], ep_len, ep_idx)
    for name in ("obs_workspace", "obs_wrist"):
        _validate_camera_stream(name, arrays[name], ep_len, ep_idx)
    return arrays


def _episode_shape_reference(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """Derive the dataset-wide shape contract from one episode."""
    return {
        "n_state": arrays["state"].shape[1],
        "n_actions": arrays["actions"].shape[1],
        "img_h": arrays["obs_workspace"].shape[1],
        "img_w": arrays["obs_workspace"].shape[2],
    }


def _check_cameras_agree(arrays: dict[str, np.ndarray], ep_idx: int) -> None:
    """Require both cameras of one episode to share resolution."""
    workspace = arrays["obs_workspace"].shape[1:3]
    wrist = arrays["obs_wrist"].shape[1:3]
    if wrist != workspace:
        raise AdapterError(
            f"Episode {ep_idx}: obs_wrist resolution {wrist} differs from "
            f"obs_workspace resolution {workspace}"
        )


def _check_episode_matches_reference(
    arrays: dict[str, np.ndarray], ep_idx: int, reference: dict[str, int]
) -> None:
    """Reject episodes whose widths or camera resolutions drift from episode 0."""
    for name, key in (
        ("obs_workspace", "img_h"),
        ("obs_wrist", "img_h"),
        ("state", "n_state"),
        ("actions", "n_actions"),
    ):
        expected = (
            (reference[key], reference["img_w"]) if key == "img_h" else reference[key]
        )
        observed = arrays[name].shape[1:3] if key == "img_h" else arrays[name].shape[1]
        if observed != expected:
            raise AdapterError(
                f"Episode {ep_idx}: {name} has {observed}, expected {expected} "
                f"from episode 0"
            )


def _load_and_validate_episode(
    ep_dir: Path, ep_idx: int, reference: dict[str, int] | None
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load and fully validate one episode against the dataset-wide contract.

    Args:
        ep_dir: Episode directory holding the four numpy arrays.
        ep_idx: Zero-based episode position used in error messages.
        reference: Shapes of the first episode, or None to become the reference.
    Returns:
        The loaded arrays and the dataset-wide shape reference.
    Raises:
        AdapterError: Lengths, dtypes, dimensions or finiteness violate the contract.
    """
    arrays = _load_episode_arrays(ep_dir, ep_idx)
    _check_cameras_agree(arrays, ep_idx)
    shapes = _episode_shape_reference(arrays)
    if reference is None:
        return arrays, shapes
    _check_episode_matches_reference(arrays, ep_idx, reference)
    return arrays, reference


def _validate_fps(fps: Any) -> float:
    """Reject boolean, non-numeric, non-finite and non-positive frame rates.

    Returns the validated rate as a float. Called before any output directory
    is created or video is encoded so invalid rates can never produce partial
    datasets.
    """
    if isinstance(fps, bool) or not isinstance(
        fps, (int, float, np.integer, np.floating)
    ):
        raise AdapterError(f"fps must be a positive finite number, got {fps!r}")
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0:
        raise AdapterError(f"fps must be a positive finite number, got {fps!r}")
    return rate


def convert(
    input_dir: Path,
    output_dir: Path,
    *,
    fps: int = 20,
    robot_type: str = "franka_panda",
    task: str = "Pick and place cube to target",
    task_from_metadata: bool = False,
) -> Path:
    """Convert a directory of episode numpy arrays to LeRobotDataset v3.0.

    Args:
        input_dir: Directory containing episode_NNNN/ subdirectories.
        output_dir: Where to write the dataset.
        fps: Frame rate for video encoding and timestamps.
        robot_type: Robot identifier for metadata.
        task: Task description string.
        task_from_metadata: Read a task for every episode from metadata.json.

    Returns:
        Path to the output directory.
    Raises:
        AdapterError: Frame rate, episode streams, metadata or encoding is invalid.
        FileNotFoundError: A required input array is missing.
        OSError: Dataset output cannot be written.
    """
    fps = _validate_fps(fps)
    episodes = discover_episodes(input_dir)
    n_episodes = len(episodes)
    episode_tasks = [task] * n_episodes
    if task_from_metadata:
        metadata_path = input_dir / "metadata.json"
        if not metadata_path.is_file():
            raise AdapterError("--task-from-metadata requires metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = metadata.get("episodes") or []
        by_index = {
            int(record["episode_index"]): str(
                record.get("task") or record.get("env_id") or ""
            ).strip()
            for record in records
            if isinstance(record, dict) and "episode_index" in record
        }
        episode_tasks = [by_index.get(index, "") for index in range(n_episodes)]
        if any(not value for value in episode_tasks):
            raise AdapterError("metadata.json does not name a task for every episode")
    tasks = list(dict.fromkeys(episode_tasks))
    task_indices = {name: index for index, name in enumerate(tasks)}

    # Validate every episode fully before creating any output or encoding video.
    loaded: list[dict[str, np.ndarray]] = []
    reference: dict[str, int] | None = None
    for ep_idx, ep_dir in enumerate(episodes):
        arrays, reference = _load_and_validate_episode(ep_dir, ep_idx, reference)
        loaded.append(arrays)
    n_state = reference["n_state"]
    n_actions = reference["n_actions"]
    img_h, img_w = reference["img_h"], reference["img_w"]

    video_keys = {"observation.images.workspace", "observation.images.wrist"}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    data_schema = _build_data_schema(n_state, n_actions)

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

        obs_workspace = loaded[ep_idx]["obs_workspace"]
        obs_wrist = loaded[ep_idx]["obs_wrist"]
        state = loaded[ep_idx]["state"]
        actions = loaded[ep_idx]["actions"]

        ep_len = state.shape[0]
        # ── Encode videos ───────────────────────────────────────────
        for cam_key, cam_frames in [
            ("observation.images.workspace", obs_workspace),
            ("observation.images.wrist", obs_wrist),
        ]:
            video_path = (
                output_dir / "videos" / cam_key / "chunk-000" / f"file-{ep_idx:03d}.mp4"
            )
            encode_video(cam_frames, video_path, fps)

        episode_task = episode_tasks[ep_idx]
        episode_task_index = task_indices[episode_task]

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
                "task_index": episode_task_index,
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
            "task_index": np.full(ep_len, episode_task_index, dtype=np.int64),
        }
        ep_stats = _compute_episode_stats(ep_arrays, video_keys)

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
            "tasks": [episode_task],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for cam_key in ["observation.images.workspace", "observation.images.wrist"]:
            ep_meta[f"videos/{cam_key}/chunk_index"] = 0
            ep_meta[f"videos/{cam_key}/file_index"] = ep_idx
            ep_meta[f"videos/{cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{cam_key}/to_timestamp"] = ep_len / fps

        for feat_key, feat_stats in ep_stats.items():
            for stat_name, stat_val in feat_stats.items():
                ep_meta[f"stats/{feat_key}/{stat_name}"] = stat_val

        episode_meta_rows.append(ep_meta)

    _print_progress("Writing data parquet...")

    # ── Write data parquet ──────────────────────────────────────────
    arrays = {
        "observation.state": pa.array(
            _numeric_feature_values(all_data_rows, "observation.state", n_state),
            type=_numeric_feature_type(n_state),
        ),
        "action": pa.array(
            _numeric_feature_values(all_data_rows, "action", n_actions),
            type=_numeric_feature_type(n_actions),
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
        "index": pa.array([r["index"] for r in all_data_rows], type=pa.int64()),
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
    _write_tasks_parquet(tasks, output_dir / "meta" / "tasks.parquet")

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
        "total_tasks": len(tasks),
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


def _write_tasks_parquet(task: str | list[str], output_path: Path) -> None:
    """Write the tasks.parquet metadata file."""
    import pandas as pd

    tasks = [task] if isinstance(task, str) else task
    frame = pd.DataFrame(
        {"task_index": list(range(len(tasks)))},
        index=pd.Index(tasks, name="task"),
    )
    # Native LeRobot looks up the task text through the DataFrame index.
    table = pa.Table.from_pandas(frame, preserve_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")


def _print_progress(msg: str) -> None:
    sys.stderr.write(f"\r{msg}\033[K\n")
    sys.stderr.flush()
