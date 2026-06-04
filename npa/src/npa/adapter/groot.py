"""Convert between standard LeRobotDataset and GR00T LeRobot format.

GR00T uses LeRobot trajectory storage with a few extra conventions:

* ``meta/modality.json`` maps GR00T modality keys to slices of the flat
  LeRobot ``observation.state`` and ``action`` vectors.
* episode data is stored as one parquet per episode using the LeRobot v2
  pattern ``data/chunk-000/episode_000000.parquet``.
* tasks and episodes are JSONL files, while newer standard LeRobot datasets
  often store those metadata tables as parquet.

The adapter keeps numeric data and task text unchanged. It also records how the
flat LeRobot vectors should be interpreted by GR00T. Joint-space datasets use a
``single_arm``/``gripper`` split compatible with Isaac-GR00T's SO100 example.
Genesis Franka demos, however, use cartesian end-effector command actions
instead of joint targets. Those are written as absolute cartesian action keys
and accompanied by a generated modality config, so GR00T does not attempt to
compute relative joint deltas between vectors with different dimensions.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


GROOT_DATA_PATH_TPL = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
GROOT_VIDEO_PATH_TPL = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
LEROBOT_DATA_PATH_TPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
LEROBOT_VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
ADAPTER_MANIFEST = "npa_groot_adapter.json"
GENERATED_MODALITY_CONFIG = "npa_groot_modality_config.py"

ACTION_SPACE_JOINT = "joint"
ACTION_SPACE_CARTESIAN_XYZ = "cartesian_xyz"
ACTION_SPACE_CARTESIAN_XYZ_GRIPPER = "cartesian_xyz_gripper"
ACTION_SPACE_CARTESIAN_POSE = "cartesian_pose"
ACTION_SPACE_CARTESIAN_POSE_GRIPPER = "cartesian_pose_gripper"

CARTESIAN_ACTION_KEYS: dict[int, list[str]] = {
    3: ["x", "y", "z"],
    4: ["x", "y", "z", "gripper"],
    6: ["x", "y", "z", "roll", "pitch", "yaw"],
    7: ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
}

BUILTIN_CARTESIAN_TAGS = {
    "libero_panda",
    "libero_sim",
    "simpler_env_google",
    "simpler_env_widowx",
}

NEW_EMBODIMENT_TAGS = {"new", "new_embodiment", "new-embodiment", "newembodiment"}
REAL_G1_TAGS = {"real_g1", "real_g1_relative_eef_relative_joints"}

# Canonical REAL_G1 schema from nvidia/GR00T-N1.7-3B processor_config.json:
# processor_kwargs.modality_configs.real_g1_relative_eef_relative_joints.
REAL_G1_STATE_KEYS = [
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
]
REAL_G1_ACTION_KEYS = [
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
    "base_height_command",
    "navigate_command",
]
REAL_G1_GROUP_DIMS = {
    "left_wrist_eef_9d": 9,
    "right_wrist_eef_9d": 9,
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}
REAL_G1_IDENTITY_EEF_9D = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class GR00TAdapterError(Exception):
    """Raised when a dataset cannot be converted safely."""


def lerobot_to_groot(
    input_dir: Path,
    output_dir: Path,
    *,
    robot_embodiment: str = "NEW_EMBODIMENT",
) -> Path:
    """Convert a standard LeRobotDataset directory to GR00T LeRobot format."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise GR00TAdapterError(f"Input dataset does not exist: {input_dir}")

    _reset_dir(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = _load_json(input_dir / "meta" / "info.json")
    data_table = _read_lerobot_data_table(input_dir, info)
    episode_rows = _read_lerobot_episode_rows(input_dir)
    task_rows = _read_lerobot_task_rows(input_dir)

    action_space = _infer_action_space(info, robot_embodiment=robot_embodiment)
    modality = _build_modality_json(
        info,
        robot_embodiment=robot_embodiment,
        action_space=action_space,
    )
    data_table = _add_synthetic_feature_columns(data_table, modality, info)

    _write_groot_episode_parquets(output_dir, data_table, episode_rows, info)
    _copy_or_write_stats(input_dir, output_dir)
    _write_synthetic_feature_stats(output_dir, modality, data_table)
    _write_jsonl(meta_dir / "episodes.jsonl", _groot_episode_rows(episode_rows, data_table))
    _write_jsonl(meta_dir / "tasks.jsonl", task_rows or [{"task_index": 0, "task": ""}])

    groot_info = {**info, "features": dict(info.get("features", {}))}
    _add_synthetic_info_features(groot_info, modality)
    groot_info["codebase_version"] = "v2.1"
    groot_info["data_path"] = GROOT_DATA_PATH_TPL
    if _video_features(info):
        groot_info["video_path"] = GROOT_VIDEO_PATH_TPL
    groot_info["total_episodes"] = len(_episode_indices(data_table))
    groot_info["total_frames"] = data_table.num_rows
    groot_info["total_tasks"] = len(task_rows) if task_rows else 1
    groot_info["robot_type"] = robot_embodiment
    _write_json(meta_dir / "info.json", groot_info)

    _write_json(meta_dir / "modality.json", modality)
    if _should_write_generated_modality_config(robot_embodiment, action_space):
        (meta_dir / GENERATED_MODALITY_CONFIG).write_text(
            _render_generated_modality_config(modality, robot_embodiment=robot_embodiment)
        )
    _copy_videos_lerobot_to_groot(input_dir, output_dir, info, episode_rows, modality)
    _write_manifest(
        output_dir,
        "lerobot-to-groot",
        robot_embodiment,
        action_space=action_space,
        state_dim=_feature_dim(info, "observation.state"),
        action_dim=_feature_dim(info, "action"),
    )
    return output_dir


def groot_to_lerobot(input_dir: Path, output_dir: Path) -> Path:
    """Convert a GR00T LeRobot-format dataset back to standard LeRobot layout."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise GR00TAdapterError(f"Input dataset does not exist: {input_dir}")

    _reset_dir(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = _load_json(input_dir / "meta" / "info.json")
    episode_rows = _read_groot_episode_rows(input_dir)
    task_rows = _read_groot_task_rows(input_dir)
    data_table = _read_groot_data_table(input_dir, info, episode_rows)

    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(data_table, output_dir / "data" / "chunk-000" / "file-000.parquet")
    _write_lerobot_episodes_parquet(output_dir, episode_rows, input_dir, info)
    _write_lerobot_tasks_parquet(output_dir, task_rows)
    _copy_or_write_stats(input_dir, output_dir)

    lerobot_info = dict(info)
    lerobot_info["codebase_version"] = "v3.0"
    lerobot_info["data_path"] = LEROBOT_DATA_PATH_TPL
    if _video_features(info):
        lerobot_info["video_path"] = LEROBOT_VIDEO_PATH_TPL
    lerobot_info["total_episodes"] = len(_episode_indices(data_table))
    lerobot_info["total_frames"] = data_table.num_rows
    lerobot_info["total_tasks"] = len(task_rows) if task_rows else 1
    _write_json(meta_dir / "info.json", lerobot_info)

    _copy_videos_groot_to_lerobot(input_dir, output_dir, info)
    _write_manifest(output_dir, "groot-to-lerobot", _manifest_embodiment(input_dir))
    return output_dir


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GR00TAdapterError(f"Missing metadata file: {path}")
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _is_sidecar_path(path: Path) -> bool:
    return any(part.startswith("._") or part == ".DS_Store" for part in path.parts)


def _parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if not _is_sidecar_path(path))


def _read_lerobot_data_table(input_dir: Path, info: dict[str, Any]) -> pa.Table:
    data_path = info.get("data_path", LEROBOT_DATA_PATH_TPL)
    files = _parquet_files(input_dir / "data")
    if not files:
        raise GR00TAdapterError(f"No parquet data files found under {input_dir / 'data'}")
    if "episode_" in data_path:
        return pa.concat_tables([pq.read_table(path) for path in files], promote_options="default")
    return pa.concat_tables([pq.read_table(path) for path in files], promote_options="default")


def _read_groot_data_table(
    input_dir: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> pa.Table:
    tables: list[pa.Table] = []
    data_pattern = info.get("data_path", GROOT_DATA_PATH_TPL)
    if not episode_rows:
        files = _parquet_files(input_dir / "data")
    else:
        files = [
            input_dir
            / data_pattern.format(
                episode_chunk=int(row.get("episode_index", 0)) // int(info.get("chunks_size", 1000)),
                episode_index=int(row.get("episode_index", 0)),
            )
            for row in episode_rows
        ]
    for path in files:
        if path.exists():
            tables.append(pq.read_table(path))
    if not tables:
        raise GR00TAdapterError(f"No GR00T episode parquet files found under {input_dir / 'data'}")
    return pa.concat_tables(tables, promote_options="default")


def _read_lerobot_episode_rows(input_dir: Path) -> list[dict[str, Any]]:
    jsonl_rows = _read_jsonl(input_dir / "meta" / "episodes.jsonl")
    if jsonl_rows:
        return jsonl_rows
    rows: list[dict[str, Any]] = []
    for path in _parquet_files(input_dir / "meta" / "episodes"):
        rows.extend(_table_rows(path))
    return rows


def _read_lerobot_task_rows(input_dir: Path) -> list[dict[str, Any]]:
    jsonl_rows = _read_jsonl(input_dir / "meta" / "tasks.jsonl")
    if jsonl_rows:
        return jsonl_rows
    rows = _table_rows(input_dir / "meta" / "tasks.parquet")
    return [
        {"task_index": int(row.get("task_index", idx)), "task": str(row.get("task", ""))}
        for idx, row in enumerate(rows)
    ]


def _read_groot_episode_rows(input_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(input_dir / "meta" / "episodes.jsonl")


def _read_groot_task_rows(input_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(input_dir / "meta" / "tasks.jsonl")


def _episode_indices(table: pa.Table) -> list[int]:
    if "episode_index" not in table.column_names:
        raise GR00TAdapterError("Dataset data is missing required episode_index column")
    return sorted({int(value) for value in table["episode_index"].to_pylist()})


def _take_rows(table: pa.Table, row_indices: list[int]) -> pa.Table:
    return table.take(pa.array(row_indices, type=pa.int64()))


def _write_groot_episode_parquets(
    output_dir: Path,
    data_table: pa.Table,
    episode_rows: list[dict[str, Any]],
    info: dict[str, Any],
) -> None:
    chunk_size = int(info.get("chunks_size", 1000) or 1000)
    episode_values = [int(value) for value in data_table["episode_index"].to_pylist()]
    for episode_index in _episode_indices(data_table):
        row_indices = [idx for idx, value in enumerate(episode_values) if value == episode_index]
        episode_table = _take_rows(data_table, row_indices)
        chunk_index = episode_index // chunk_size
        target = output_dir / GROOT_DATA_PATH_TPL.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode_table, target)


def _groot_episode_rows(
    episode_rows: list[dict[str, Any]],
    data_table: pa.Table,
) -> list[dict[str, Any]]:
    by_episode = {int(row.get("episode_index", idx)): row for idx, row in enumerate(episode_rows)}
    frame_episode_values = [int(value) for value in data_table["episode_index"].to_pylist()]
    rows = []
    for episode_index in _episode_indices(data_table):
        source = by_episode.get(episode_index, {})
        length = int(source.get("length") or frame_episode_values.count(episode_index))
        tasks = source.get("tasks") or [source.get("task", "")]
        rows.append({"episode_index": episode_index, "tasks": tasks, "length": length})
    return rows


def _copy_or_write_stats(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "meta" / "stats.json"
    dst = output_dir / "meta" / "stats.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        _write_json(dst, {})
    rel_src = input_dir / "meta" / "relative_stats.json"
    if rel_src.exists():
        shutil.copy2(rel_src, output_dir / "meta" / "relative_stats.json")


def _synthetic_feature_specs(
    modality: dict[str, Any],
    existing_columns: set[str] | None = None,
) -> dict[str, int]:
    specs: dict[str, int] = {}
    existing_columns = existing_columns or set()
    for section in ("state", "action"):
        for group in modality.get(section, {}).values():
            original_key = group.get("original_key")
            if not original_key or original_key in existing_columns:
                continue
            specs[str(original_key)] = int(group["end"]) - int(group["start"])
    return specs


def _real_g1_synthetic_group(feature_key: str) -> tuple[str, str] | None:
    for modality, prefix in (
        ("state", "observation.real_g1."),
        ("action", "action.real_g1."),
    ):
        if feature_key.startswith(prefix):
            return modality, feature_key.removeprefix(prefix)
    return None


def _synthetic_feature_values(
    data_table: pa.Table,
    info: dict[str, Any],
    feature_key: str,
    dim: int,
) -> list[list[float]]:
    real_g1_group = _real_g1_synthetic_group(feature_key)
    if real_g1_group is not None:
        modality, group = real_g1_group
        if group in {"left_wrist_eef_9d", "right_wrist_eef_9d"}:
            return [list(REAL_G1_IDENTITY_EEF_9D) for _ in range(data_table.num_rows)]
        source_key = "observation.state" if modality == "state" else "action"
        mapping = _g1_source_group(info, source_key=source_key, group=group)
        if mapping is not None and source_key in data_table.column_names:
            start = int(mapping["start"])
            end = int(mapping["end"])
            values = []
            for row in data_table[source_key].to_pylist():
                segment = [float(value) for value in (row or [])[start:end]]
                if len(segment) < dim:
                    segment.extend([0.0] * (dim - len(segment)))
                values.append(segment[:dim])
            return values

    return [[0.0] * dim for _ in range(data_table.num_rows)]


def _add_synthetic_feature_columns(
    data_table: pa.Table,
    modality: dict[str, Any],
    info: dict[str, Any],
) -> pa.Table:
    specs = _synthetic_feature_specs(modality, set(data_table.column_names))
    for key, dim in specs.items():
        values = _synthetic_feature_values(data_table, info, key, dim)
        data_table = data_table.append_column(
            key,
            pa.array(values, type=pa.list_(pa.float32(), dim)),
        )
    return data_table


def _add_synthetic_info_features(info: dict[str, Any], modality: dict[str, Any]) -> None:
    features = info.setdefault("features", {})
    for key, dim in _synthetic_feature_specs(modality, set(features)).items():
        features[key] = {
            "dtype": "float32",
            "shape": [dim],
            "names": None,
        }


def _zero_feature_stats(dim: int, total_frames: int) -> dict[str, Any]:
    zeros = [0.0] * dim
    return {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * dim,
        "count": [int(total_frames)],
        "q01": zeros,
        "q10": zeros,
        "q50": zeros,
        "q90": zeros,
        "q99": zeros,
    }


def _feature_stats_from_table(data_table: pa.Table, key: str, dim: int) -> dict[str, Any]:
    if key not in data_table.column_names or data_table.num_rows == 0:
        return _zero_feature_stats(dim, data_table.num_rows)
    values = np.asarray(data_table[key].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        values = values.reshape(data_table.num_rows, dim)
    if values.shape[1] < dim:
        values = np.pad(values, ((0, 0), (0, dim - values.shape[1])), constant_values=0.0)
    values = values[:, :dim]
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": std.tolist(),
        "count": [int(data_table.num_rows)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_synthetic_feature_stats(
    output_dir: Path,
    modality: dict[str, Any],
    data_table: pa.Table,
) -> None:
    path = output_dir / "meta" / "stats.json"
    stats = _load_json(path) if path.exists() else {}
    for key, dim in _synthetic_feature_specs(modality, set(stats)).items():
        stats[key] = _feature_stats_from_table(data_table, key, dim)
    _write_json(path, stats)


def _feature_dim(info: dict[str, Any], key: str) -> int:
    shape = info.get("features", {}).get(key, {}).get("shape") or []
    if not shape:
        return 0
    return int(shape[0])


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names") or []
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names if isinstance(name, str)]


def _video_features(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [
        key
        for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") == "video"
    ]


def _safe_modality_key(feature_key: str) -> str:
    name = feature_key.removeprefix("observation.images.")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "image"


def _video_modality(info: dict[str, Any]) -> dict[str, dict[str, str]]:
    video_features = _video_features(info)
    if not video_features:
        return {}
    result: dict[str, dict[str, str]] = {}
    used: set[str] = set()
    non_wrist_count = 0
    for feature in video_features:
        raw = _safe_modality_key(feature)
        if "wrist" in raw.lower() and "wrist" not in used:
            key = "wrist"
        elif non_wrist_count == 0 and "front" not in used:
            key = "front"
            non_wrist_count += 1
        else:
            key = raw
        if key in used:
            suffix = 2
            base = key
            while f"{base}_{suffix}" in used:
                suffix += 1
            key = f"{base}_{suffix}"
        used.add(key)
        result[key] = {"original_key": feature}
    return result


def _split_vector_modality(dim: int) -> dict[str, dict[str, int]]:
    if dim <= 0:
        return {}
    if dim == 1:
        return {"single_arm": {"start": 0, "end": 1}}
    return {
        "single_arm": {"start": 0, "end": dim - 1},
        "gripper": {"start": dim - 1, "end": dim},
    }


def _cartesian_action_space(action_dim: int) -> str:
    if action_dim == 3:
        return ACTION_SPACE_CARTESIAN_XYZ
    if action_dim == 4:
        return ACTION_SPACE_CARTESIAN_XYZ_GRIPPER
    if action_dim == 6:
        return ACTION_SPACE_CARTESIAN_POSE
    if action_dim == 7:
        return ACTION_SPACE_CARTESIAN_POSE_GRIPPER
    raise GR00TAdapterError(
        "GR00T cartesian action data must have 3, 4, 6, or 7 action dimensions; "
        f"got {action_dim}"
    )


def _tag_key(robot_embodiment: str) -> str:
    return robot_embodiment.strip().lower().replace("-", "_")


def _is_real_g1(robot_embodiment: str) -> bool:
    return _tag_key(robot_embodiment) in REAL_G1_TAGS


def _infer_action_space(info: dict[str, Any], *, robot_embodiment: str) -> str:
    state_dim = _feature_dim(info, "observation.state")
    action_dim = _feature_dim(info, "action")
    if action_dim <= 0:
        raise GR00TAdapterError("Dataset action feature is missing or has zero dimensions")
    if state_dim <= 0:
        raise GR00TAdapterError("Dataset observation.state feature is missing or has zero dimensions")

    tag = _tag_key(robot_embodiment)
    if tag in BUILTIN_CARTESIAN_TAGS:
        if state_dim != 8 or action_dim != 7:
            raise GR00TAdapterError(
                f"Embodiment tag {robot_embodiment!r} uses GR00T's built-in cartesian "
                f"7D action layout, but this dataset has state_dim={state_dim} and "
                f"action_dim={action_dim}. Use --embodiment-tag NEW_EMBODIMENT for "
                "custom cartesian data or provide data matching the built-in tag."
            )
        return ACTION_SPACE_CARTESIAN_POSE_GRIPPER

    if action_dim == state_dim:
        return ACTION_SPACE_JOINT

    if action_dim in CARTESIAN_ACTION_KEYS:
        if tag not in NEW_EMBODIMENT_TAGS:
            raise GR00TAdapterError(
                f"Dataset action_dim={action_dim} does not match state_dim={state_dim}. "
                f"That looks like cartesian/end-effector command data, but embodiment tag "
                f"{robot_embodiment!r} is not a custom NEW_EMBODIMENT tag. Use "
                "--embodiment-tag NEW_EMBODIMENT or convert data whose action dimensions "
                "match the requested embodiment."
            )
        return _cartesian_action_space(action_dim)

    raise GR00TAdapterError(
        f"Cannot infer a GR00T action layout for state_dim={state_dim}, action_dim={action_dim}, "
        f"embodiment={robot_embodiment!r}. Joint-space data must have matching state/action "
        "dimensions. Cartesian data must have 3, 4, 6, or 7 action dimensions."
    )


def _cartesian_modality(dim: int) -> dict[str, dict[str, int]]:
    return {key: {"start": idx, "end": idx + 1} for idx, key in enumerate(CARTESIAN_ACTION_KEYS[dim])}


def _builtin_cartesian_state_modality(robot_embodiment: str) -> dict[str, dict[str, int]]:
    tag = _tag_key(robot_embodiment)
    if tag == "simpler_env_google":
        keys = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
    elif tag in {"simpler_env_widowx"}:
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
    else:
        keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        return {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 8},
        }
    return {key: {"start": idx, "end": idx + 1} for idx, key in enumerate(keys)}


def _real_g1_synthetic_key(modality: str, group: str) -> str:
    prefix = "observation.real_g1" if modality == "state" else "action.real_g1"
    return f"{prefix}.{group}"


def _normalized_joint_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _contiguous_span(indices: list[int]) -> tuple[int, int] | None:
    if not indices:
        return None
    ordered = sorted(indices)
    start, end = ordered[0], ordered[-1] + 1
    if ordered != list(range(start, end)):
        return None
    return start, end


def _named_joint_span(info: dict[str, Any], feature_key: str, group: str) -> tuple[int, int] | None:
    names = [_normalized_joint_name(name) for name in _feature_names(info, feature_key)]
    if not names:
        return None

    def matches(name: str) -> bool:
        if group == "left_arm":
            return "left" in name and "hand" not in name and any(
                token in name for token in ("shoulder", "elbow", "wrist")
            )
        if group == "right_arm":
            return "right" in name and "hand" not in name and any(
                token in name for token in ("shoulder", "elbow", "wrist")
            )
        if group == "left_hand":
            return "left" in name and "hand" in name
        if group == "right_hand":
            return "right" in name and "hand" in name
        if group == "waist":
            return "waist" in name
        return False

    return _contiguous_span([idx for idx, name in enumerate(names) if matches(name)])


def _fallback_g1_joint_span(dim: int, group: str) -> tuple[int, int] | None:
    if dim == 26:
        spans = {
            "left_arm": (0, 7),
            "right_arm": (7, 14),
            "left_hand": (14, 20),
            "right_hand": (20, 26),
        }
    elif dim == 43:
        spans = {
            "waist": (12, 15),
            "left_arm": (15, 22),
            "left_hand": (22, 29),
            "right_arm": (29, 36),
            "right_hand": (36, 43),
        }
    else:
        spans = {}
    return spans.get(group)


def _g1_source_group(
    info: dict[str, Any],
    *,
    source_key: str,
    group: str,
) -> dict[str, int | str] | None:
    span = _named_joint_span(info, source_key, group) or _fallback_g1_joint_span(
        _feature_dim(info, source_key),
        group,
    )
    if span is None:
        return None
    start, end = span
    return {"start": start, "end": end, "original_key": source_key}


def _real_g1_zero_group(modality: str, group: str) -> dict[str, int | str]:
    dim = REAL_G1_GROUP_DIMS[group]
    return {
        "start": 0,
        "end": dim,
        "original_key": _real_g1_synthetic_key(modality, group),
    }


def _real_g1_required_source_group(
    info: dict[str, Any],
    *,
    source_key: str,
    group: str,
    modality: str,
) -> dict[str, int | str]:
    mapping = _g1_source_group(info, source_key=source_key, group=group)
    if mapping is None:
        raise GR00TAdapterError(
            f"Cannot convert dataset to REAL_G1: unable to locate {group!r} in "
            f"{source_key!r}. Provide G1 joint names or a supported 26D/43D G1 layout."
        )
    expected_dim = REAL_G1_GROUP_DIMS[group]
    actual_dim = int(mapping["end"]) - int(mapping["start"])
    if actual_dim != expected_dim:
        return {
            "start": 0,
            "end": expected_dim,
            "original_key": _real_g1_synthetic_key(modality, group),
        }
    return mapping


def _real_g1_optional_source_group(
    info: dict[str, Any],
    *,
    source_key: str,
    group: str,
    modality: str,
) -> dict[str, int | str] | None:
    mapping = _g1_source_group(info, source_key=source_key, group=group)
    if mapping is None:
        return None
    expected_dim = REAL_G1_GROUP_DIMS[group]
    actual_dim = int(mapping["end"]) - int(mapping["start"])
    if actual_dim != expected_dim:
        return {
            "start": 0,
            "end": expected_dim,
            "original_key": _real_g1_synthetic_key(modality, group),
        }
    return mapping


def _real_g1_state_modality(info: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    return {
        "left_wrist_eef_9d": _real_g1_zero_group("state", "left_wrist_eef_9d"),
        "right_wrist_eef_9d": _real_g1_zero_group("state", "right_wrist_eef_9d"),
        "left_hand": _real_g1_required_source_group(
            info, source_key="observation.state", group="left_hand", modality="state"
        ),
        "right_hand": _real_g1_required_source_group(
            info, source_key="observation.state", group="right_hand", modality="state"
        ),
        "left_arm": _real_g1_required_source_group(
            info, source_key="observation.state", group="left_arm", modality="state"
        ),
        "right_arm": _real_g1_required_source_group(
            info, source_key="observation.state", group="right_arm", modality="state"
        ),
        "waist": _real_g1_optional_source_group(
            info, source_key="observation.state", group="waist", modality="state"
        )
        or _real_g1_zero_group("state", "waist"),
    }


def _real_g1_action_modality(info: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    return {
        "left_wrist_eef_9d": _real_g1_zero_group("action", "left_wrist_eef_9d"),
        "right_wrist_eef_9d": _real_g1_zero_group("action", "right_wrist_eef_9d"),
        "left_hand": _real_g1_required_source_group(
            info, source_key="action", group="left_hand", modality="action"
        ),
        "right_hand": _real_g1_required_source_group(
            info, source_key="action", group="right_hand", modality="action"
        ),
        "left_arm": _real_g1_required_source_group(
            info, source_key="action", group="left_arm", modality="action"
        ),
        "right_arm": _real_g1_required_source_group(
            info, source_key="action", group="right_arm", modality="action"
        ),
        "waist": _real_g1_optional_source_group(
            info, source_key="action", group="waist", modality="action"
        )
        or _real_g1_zero_group("action", "waist"),
        "base_height_command": _real_g1_zero_group("action", "base_height_command"),
        "navigate_command": _real_g1_zero_group("action", "navigate_command"),
    }


def _real_g1_video_modality(info: dict[str, Any]) -> dict[str, dict[str, str]]:
    video_features = _video_features(info)
    if not video_features:
        return {}
    return {"ego_view": {"original_key": video_features[0]}}


def _state_modality_for_action_space(
    info: dict[str, Any],
    *,
    robot_embodiment: str,
    action_space: str,
) -> dict[str, dict[str, int | str]]:
    if _is_real_g1(robot_embodiment):
        return _real_g1_state_modality(info)
    state_dim = _feature_dim(info, "observation.state")
    if action_space == ACTION_SPACE_JOINT:
        return _split_vector_modality(state_dim)
    if _tag_key(robot_embodiment) in BUILTIN_CARTESIAN_TAGS:
        return _builtin_cartesian_state_modality(robot_embodiment)
    return {"joint_position": {"start": 0, "end": state_dim}}


def _action_modality_for_action_space(
    info: dict[str, Any],
    *,
    robot_embodiment: str,
    action_space: str,
) -> dict[str, dict[str, int | str]]:
    if _is_real_g1(robot_embodiment):
        return _real_g1_action_modality(info)
    action_dim = _feature_dim(info, "action")
    if action_space == ACTION_SPACE_JOINT:
        return _split_vector_modality(action_dim)
    return _cartesian_modality(action_dim)


def _language_key(robot_embodiment: str) -> str:
    key = robot_embodiment.strip().lower()
    if key in {"libero_panda", "libero_sim", "simpler_env_google", "simpler_env_widowx"}:
        return "human.action.task_description"
    return "human.task_description"


def _build_modality_json(
    info: dict[str, Any],
    *,
    robot_embodiment: str,
    action_space: str,
) -> dict[str, Any]:
    return {
        "state": _state_modality_for_action_space(
            info,
            robot_embodiment=robot_embodiment,
            action_space=action_space,
        ),
        "action": _action_modality_for_action_space(
            info,
            robot_embodiment=robot_embodiment,
            action_space=action_space,
        ),
        "video": (
            _real_g1_video_modality(info)
            if _is_real_g1(robot_embodiment)
            else _video_modality(info)
        ),
        "annotation": {
            _language_key(robot_embodiment): {
                "original_key": "task_index",
            }
        },
    }


def _should_write_generated_modality_config(robot_embodiment: str, action_space: str) -> bool:
    return action_space != ACTION_SPACE_JOINT and _tag_key(robot_embodiment) in NEW_EMBODIMENT_TAGS


def _render_modality_config_section(
    *,
    keys: list[str],
    delta_indices: str,
    action_configs: bool = False,
) -> str:
    key_lines = "\n".join(f'            "{key}",' for key in keys)
    if not action_configs:
        return (
            "    ModalityConfig(\n"
            f"        delta_indices={delta_indices},\n"
            "        modality_keys=[\n"
            f"{key_lines}\n"
            "        ],\n"
            "    )"
        )

    config_lines = "\n".join(
        "            ActionConfig(\n"
        "                rep=ActionRepresentation.ABSOLUTE,\n"
        "                type=ActionType.NON_EEF,\n"
        "                format=ActionFormat.DEFAULT,\n"
        "            ),"
        for _ in keys
    )
    return (
        "    ModalityConfig(\n"
        f"        delta_indices={delta_indices},\n"
        "        modality_keys=[\n"
        f"{key_lines}\n"
        "        ],\n"
        "        action_configs=[\n"
        f"{config_lines}\n"
        "        ],\n"
        "    )"
    )


def _render_generated_modality_config(
    modality: dict[str, Any],
    *,
    robot_embodiment: str,
) -> str:
    video_keys = list(modality.get("video", {}).keys())
    state_keys = list(modality.get("state", {}).keys())
    action_keys = list(modality.get("action", {}).keys())
    language_keys = [f"annotation.{key}" for key in modality.get("annotation", {}).keys()]
    sections: list[str] = []
    if video_keys:
        sections.append(
            '    "video": '
            + _render_modality_config_section(keys=video_keys, delta_indices="[0]")
            + ","
        )
    sections.extend(
        [
            '    "state": '
            + _render_modality_config_section(keys=state_keys, delta_indices="[0]")
            + ",",
            '    "action": '
            + _render_modality_config_section(
                keys=action_keys,
                delta_indices="list(range(0, 16))",
                action_configs=True,
            )
            + ",",
            '    "language": '
            + _render_modality_config_section(keys=language_keys, delta_indices="[0]")
            + ",",
        ]
    )
    body = "\n".join(sections)
    return f'''"""Generated by npa workbench groot convert.

This config registers the converted dataset's cartesian action layout with
Isaac-GR00T. The source LeRobot data stores action commands that do not share
the same dimension as observation.state, so all action keys are treated as
absolute NON_EEF scalars instead of SO100-style relative joint deltas.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


embodiment_tag = EmbodimentTag.resolve({json.dumps(robot_embodiment)})
npa_groot_config = {{
{body}
}}

if embodiment_tag.value in MODALITY_CONFIGS:
    MODALITY_CONFIGS[embodiment_tag.value] = npa_groot_config
else:
    register_modality_config(npa_groot_config, embodiment_tag=embodiment_tag)
'''


def _render_lerobot_video_path(
    input_dir: Path,
    info: dict[str, Any],
    feature_key: str,
    episode_index: int,
    episode_row: dict[str, Any],
) -> Path | None:
    pattern = info.get("video_path") or LEROBOT_VIDEO_PATH_TPL
    chunk_index = int(episode_row.get(f"videos/{feature_key}/chunk_index", 0) or 0)
    file_index = int(episode_row.get(f"videos/{feature_key}/file_index", episode_index) or episode_index)
    candidates = [
        input_dir
        / pattern.format(
            video_key=feature_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=episode_index // int(info.get("chunks_size", 1000) or 1000),
            episode_index=episode_index,
        ),
        input_dir / "videos" / feature_key / "chunk-000" / f"file-{file_index:03d}.mp4",
        input_dir / "videos" / "chunk-000" / feature_key / f"episode_{episode_index:06d}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _copy_videos_lerobot_to_groot(
    input_dir: Path,
    output_dir: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    modality: dict[str, Any],
) -> None:
    if not modality.get("video"):
        return
    rows_by_episode = {int(row.get("episode_index", idx)): row for idx, row in enumerate(episode_rows)}
    episode_indices = sorted(rows_by_episode) or list(range(int(info.get("total_episodes", 0) or 0)))
    chunk_size = int(info.get("chunks_size", 1000) or 1000)
    for episode_index in episode_indices:
        episode_row = rows_by_episode.get(episode_index, {})
        episode_chunk = episode_index // chunk_size
        for meta in modality["video"].values():
            original_key = meta["original_key"]
            src = _render_lerobot_video_path(input_dir, info, original_key, episode_index, episode_row)
            if src is None:
                continue
            dst = output_dir / GROOT_VIDEO_PATH_TPL.format(
                episode_chunk=episode_chunk,
                video_key=original_key,
                episode_index=episode_index,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _copy_videos_groot_to_lerobot(input_dir: Path, output_dir: Path, info: dict[str, Any]) -> None:
    modality_path = input_dir / "meta" / "modality.json"
    if not modality_path.exists():
        return
    modality = _load_json(modality_path)
    episode_rows = _read_groot_episode_rows(input_dir)
    chunk_size = int(info.get("chunks_size", 1000) or 1000)
    for episode_row in episode_rows:
        episode_index = int(episode_row.get("episode_index", 0))
        episode_chunk = episode_index // chunk_size
        for meta in modality.get("video", {}).values():
            original_key = meta["original_key"]
            src = input_dir / GROOT_VIDEO_PATH_TPL.format(
                episode_chunk=episode_chunk,
                video_key=original_key,
                episode_index=episode_index,
            )
            if not src.exists():
                continue
            dst = output_dir / LEROBOT_VIDEO_PATH_TPL.format(
                video_key=original_key,
                chunk_index=0,
                file_index=episode_index,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _write_lerobot_tasks_parquet(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"task_index": 0, "task": ""}]
    table = pa.table(
        {
            "task_index": pa.array([int(row.get("task_index", idx)) for idx, row in enumerate(rows)], type=pa.int64()),
            "task": pa.array([str(row.get("task", "")) for row in rows], type=pa.string()),
        }
    )
    target = output_dir / "meta" / "tasks.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


def _write_lerobot_episodes_parquet(
    output_dir: Path,
    rows: list[dict[str, Any]],
    input_dir: Path,
    info: dict[str, Any],
) -> None:
    rows = rows or []
    columns: dict[str, list[Any]] = {
        "episode_index": [],
        "data/chunk_index": [],
        "data/file_index": [],
        "dataset_from_index": [],
        "dataset_to_index": [],
        "length": [],
        "tasks": [],
        "meta/episodes/chunk_index": [],
        "meta/episodes/file_index": [],
    }
    cursor = 0
    for row in rows:
        episode_index = int(row.get("episode_index", len(columns["episode_index"])))
        length = int(row.get("length", 0))
        columns["episode_index"].append(episode_index)
        columns["data/chunk_index"].append(0)
        columns["data/file_index"].append(0)
        columns["dataset_from_index"].append(cursor)
        cursor += length
        columns["dataset_to_index"].append(cursor)
        columns["length"].append(length)
        columns["tasks"].append(row.get("tasks", []))
        columns["meta/episodes/chunk_index"].append(0)
        columns["meta/episodes/file_index"].append(0)
        for feature in _video_features(info):
            columns.setdefault(f"videos/{feature}/chunk_index", []).append(0)
            columns.setdefault(f"videos/{feature}/file_index", []).append(episode_index)
            columns.setdefault(f"videos/{feature}/from_timestamp", []).append(0.0)
            fps = float(info.get("fps", 30) or 30)
            columns.setdefault(f"videos/{feature}/to_timestamp", []).append(length / fps)

    target = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), target)


def _write_manifest(
    output_dir: Path,
    direction: str,
    robot_embodiment: str,
    *,
    action_space: str | None = None,
    state_dim: int | None = None,
    action_dim: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "direction": direction,
        "robot_embodiment": robot_embodiment,
        "format": "groot-lerobot" if direction == "lerobot-to-groot" else "lerobot",
    }
    if action_space is not None:
        payload["action_space"] = action_space
    if state_dim is not None:
        payload["state_dim"] = state_dim
    if action_dim is not None:
        payload["action_dim"] = action_dim
    _write_json(output_dir / "meta" / ADAPTER_MANIFEST, payload)


def _manifest_embodiment(input_dir: Path) -> str:
    path = input_dir / "meta" / ADAPTER_MANIFEST
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text()).get("robot_embodiment", "")
    except json.JSONDecodeError:
        return ""
