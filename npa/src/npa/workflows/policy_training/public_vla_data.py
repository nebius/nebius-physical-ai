"""Pin public SmolVLA inputs and derive episode splits with training-only statistics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

PINS = {
    "assets": {
        "repo": "lerobot/libero-assets",
        "revision": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
    },
    "dataset": {
        "repo": "lerobot/libero",
        "revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
    },
    "model": {
        "repo": "HuggingFaceVLA/smolvla_libero",
        "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    },
    "backbone": {
        "repo": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    },
}


def write_json(path: Path, value: object) -> None:
    """Persist finite evidence atomically.

    Args:
        path: Destination file.
        value: JSON-compatible evidence.
    Returns:
        None.
    Raises:
        ValueError: Evidence contains nonfinite numbers.
        OSError: Evidence cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fetch_inputs(root: Path) -> dict[str, Path]:
    """Download immutable public inputs without credentials or publication.

    Args:
        root: Private run directory.
    Returns:
        Paths to the four revision-pinned snapshots.
    Raises:
        OSError: Public payload access or local storage fails.
    """
    from huggingface_hub import snapshot_download

    result = {}
    for kind, pin in PINS.items():
        path = root / "inputs" / kind
        snapshot_download(
            pin["repo"],
            revision=pin["revision"],
            token=False,
            repo_type="dataset" if kind in {"dataset", "assets"} else "model",
            local_dir=path,
            ignore_patterns=["onnx/*"],
        )
        result[kind] = path
    write_json(root / "inputs.json", PINS)
    return result


def prepare_corpus(dataset: Path, output: Path, seed: int) -> dict:
    """Flag invalid trajectories and split duplicate groups within each public task.

    Args:
        dataset: Downloaded LeRobot v3 corpus.
        output: Destination for the curation and split manifest.
        seed: Stable split seed.
    Returns:
        Manifest retaining every episode, its quality flags and assigned partition.
    Raises:
        ValueError: Data is incomplete or a task cannot support three partitions.
    """
    frames = _read_frames(dataset)
    episodes = pq.read_table(dataset / "meta" / "episodes").to_pylist()
    tasks = pq.read_table(dataset / "meta" / "tasks.parquet").to_pandas()
    task_names = {int(row.task_index): str(index) for index, row in tasks.iterrows()}
    records = [_episode_record(row, frames, task_names) for row in episodes]
    _assign_partitions(records, seed)
    manifest = {
        "schema": "npa.public-vla.corpus.v1",
        "source": PINS["dataset"],
        "seed": seed,
        "episodes": records,
    }
    write_json(output, manifest)
    return manifest


def _read_frames(dataset):
    tables = [
        pq.read_table(
            path, columns=["episode_index", "task_index", "action", "observation.state"]
        )
        for path in sorted((dataset / "data").rglob("*.parquet"))
    ]
    import pyarrow as pa

    return pa.concat_tables(tables).to_pandas().groupby("episode_index", sort=True)


def _episode_record(row, frames, task_names):
    index = int(row["episode_index"])
    data = frames.get_group(index)
    arrays = [
        np.stack(data[key]).astype("<f4") for key in ("observation.state", "action")
    ]
    valid = all(np.isfinite(array).all() for array in arrays)
    if len(data) != row["length"]:
        raise ValueError("episode metadata does not match its frame count")
    checksum = hashlib.sha256(b"".join(array.tobytes() for array in arrays)).hexdigest()
    return {
        "episode_index": index,
        "tasks": [task_names[int(index)] for index in sorted(data.task_index.unique())],
        "frames": len(data),
        "trajectory_sha256": checksum,
        "quality_flags": [] if valid else ["nonfinite"],
        "partition": "excluded",
    }


def _assign_partitions(records, seed):
    tasks = sorted({tuple(row["tasks"]) for row in records})
    assigned = {}
    for task in tasks:
        selected = [
            r for r in records if tuple(r["tasks"]) == task and not r["quality_flags"]
        ]
        groups = sorted(
            {r["trajectory_sha256"] for r in selected},
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
        )
        count = max(1, math.ceil(len(groups) * 0.05))
        if len(groups) <= 2 * count:
            raise ValueError(
                "each task needs independent training, validation and test groups"
            )
        for index, group in enumerate(groups):
            partition = (
                "test"
                if index < count
                else "validation"
                if index < 2 * count
                else "train"
            )
            if group in assigned and assigned[group] != partition:
                raise ValueError(
                    "cross-task duplicate would cross a partition boundary"
                )
            assigned[group] = partition
        for row in selected:
            row["partition"] = assigned[row["trajectory_sha256"]]


def training_view(
    dataset: Path, manifest: dict, output: Path, task: str | None
) -> dict:
    """Build a read-only corpus view with normalization fitted on selected training rows.

    Args:
        dataset: Original immutable public corpus.
        manifest: Complete split manifest.
        output: New phase-specific view.
        task: Exact task description, or None for the whole corpus.
    Returns:
        Selected training episodes, frame count and statistics digest.
    Raises:
        ValueError: No matching training episodes remain.
        OSError: The view cannot be materialized.
    """
    rows = [
        r
        for r in manifest["episodes"]
        if r["partition"] == "train" and (task is None or task in r["tasks"])
    ]
    if not rows:
        raise ValueError("training selection is empty")
    selected = [r["episode_index"] for r in rows]
    stats = _training_stats(dataset, selected)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset / "meta", output / "meta", dirs_exist_ok=True)
    for directory in ("data", "videos"):
        link = output / directory
        if not link.exists():
            link.symlink_to((dataset / directory).resolve(), target_is_directory=True)
    write_json(output / "meta" / "stats.json", stats)
    return {
        "episodes": selected,
        "frames": sum(r["frames"] for r in rows),
        "stats_sha256": hashlib.sha256(
            (output / "meta" / "stats.json").read_bytes()
        ).hexdigest(),
    }


def _training_stats(dataset, selected):
    frames = _read_frames(dataset)
    result = {}
    for key in ("observation.state", "action"):
        values = np.concatenate(
            [np.stack(frames.get_group(index)[key]) for index in selected]
        )
        result[key] = {
            "mean": values.mean(axis=0, dtype=np.float64).tolist(),
            "std": values.std(axis=0, dtype=np.float64).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "count": [len(values)],
        }
    return result


def local_policy(source: Path, backbone: Path, output: Path) -> Path:
    """Localize model and tokenizer references while preserving the downloaded weights.

    Args:
        source: Immutable public policy snapshot.
        backbone: Immutable public VLM snapshot.
        output: Writable policy configuration directory.
    Returns:
        Local policy directory suitable for native LeRobot training and evaluation.
    Raises:
        ValueError: The source is not the supported LIBERO SmolVLA schema.
        OSError: Required model files are unavailable.
    """
    config = json.loads((source / "config.json").read_text())
    if config["input_features"]["observation.state"]["shape"] != [8]:
        raise ValueError("expected the LIBERO eight-dimensional state contract")
    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix == ".safetensors":
            target = output / path.name
            if not target.exists():
                target.symlink_to(path.resolve())
    config.update(
        vlm_model_name=str(backbone.resolve()), push_to_hub=False, n_action_steps=10
    )
    write_json(output / "config.json", config)
    for path in source.glob("policy_*processor.json"):
        value = json.loads(path.read_text())
        _localize_tokenizer(value, backbone)
        write_json(output / path.name, value)
    return output


def _localize_tokenizer(value, backbone):
    if isinstance(value, dict):
        if "tokenizer_name" in value:
            value["tokenizer_name"] = str(backbone.resolve())
        for child in value.values():
            _localize_tokenizer(child, backbone)
    elif isinstance(value, list):
        for child in value:
            _localize_tokenizer(child, backbone)
