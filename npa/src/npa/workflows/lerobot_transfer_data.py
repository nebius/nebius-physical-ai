"""Seal PushT demonstration splits and exchange verified transfer-stage artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from npa.clients.storage import StorageClient
from npa.workflows.lerobot_dataset import (
    DEFAULT_PUBLIC_LEROBOT_REPO,
    DEFAULT_PUBLIC_LEROBOT_REVISION,
    download_public_lerobot_dataset,
    seeded_episode_split,
)

LEROBOT_VERSION = "0.6.0"


def write_json(path: Path, payload: object) -> None:
    """Write finite, deterministic JSON.

    Args:
        path: Destination file.
        payload: JSON-serializable evidence.
    Returns:
        None.
    Raises:
        ValueError: Evidence contains nonfinite values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    """Hash file bytes without loading a checkpoint into memory.

    Args:
        path: File to hash.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: File cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Identify artifact files with relative names and content hashes.

    Args:
        root: Artifact directory.
    Returns:
        Sorted relative file names and SHA-256 digests.
    Raises:
        OSError: An artifact is unreadable.
    """
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.parts and path != root / "checksums.json"
    }


def materialize(source: str, target: Path) -> Path:
    """Read a local or S3 stage and verify its complete checksum manifest.

    Args:
        source: Local directory or S3 prefix.
        target: Empty local staging directory for S3.
    Returns:
        Verified artifact directory.
    Raises:
        ValueError: Files differ from the published manifest.
        OSError: An artifact cannot be read.
    """
    root = Path(source)
    if source.startswith("s3://"):
        StorageClient.from_environment().download_directory(source, str(target))
        root = target
    expected = json.loads((root / "checksums.json").read_text())
    if not expected or expected != tree_hashes(root):
        raise ValueError("Stage artifact checksum mismatch or empty manifest")
    return root


def publish(root: Path, destination: str) -> None:
    """Publish a sealed stage, verifying every uploaded file by readback.

    Args:
        root: Completed local stage directory.
        destination: New local directory or run-scoped S3 prefix.
    Returns:
        None.
    Raises:
        ValueError: Uploaded bytes do not match the local stage.
        OSError: Publication fails or a local destination exists.
    """
    write_json(root / "checksums.json", tree_hashes(root))
    if not destination.startswith("s3://"):
        shutil.copytree(root, destination)
        return
    storage = StorageClient.from_environment()
    storage.upload_directory(str(root), destination)
    readback = root.parent / f"{root.name}-readback"
    materialize(destination, readback)


def prepare_dataset(output: Path, recipe: dict) -> None:
    """Download pinned PushT data and seal an episode-disjoint training recipe.

    Args:
        output: New stage directory.
        recipe: Training and evaluation settings chosen before the experiment.
    Returns:
        None.
    Raises:
        ValueError: Data fails the PushT action, timing, or coverage contract.
        OSError: Data cannot be downloaded or written.
    """
    snapshot = download_public_lerobot_dataset(output)
    dataset = output / "dataset"
    snapshot.rename(dataset)
    shutil.rmtree(dataset / ".cache", ignore_errors=True)
    source_hashes = tree_hashes(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    rows = _read_rows(dataset, info)
    train, heldout = seeded_episode_split(
        sorted({row["episode_index"] for row in rows}), train_fraction=0.8, seed=recipe["seed"],
    )
    selected = [row for row in rows if row["episode_index"] in set(train)]
    write_json(dataset / "meta/stats.json", _training_statistics(selected))
    recipe.update({
        "schema": "npa.lerobot-transfer.recipe.v1", "dataset_repo": DEFAULT_PUBLIC_LEROBOT_REPO,
        "dataset_revision": DEFAULT_PUBLIC_LEROBOT_REVISION, "train_episodes": train,
        "reserved_episodes": heldout, "train_frames": len(selected), "total_frames": len(rows),
        "normalization": "training episodes only; fixed ImageNet camera statistics",
        "dataset_source_hashes": source_hashes, "lerobot_version": LEROBOT_VERSION,
        "video_backend": "torchcodec",
        "lerobot_release_commit": "30da8e687a6dfc617fcd94afc367ac7071c376ce",
        "conditions": ["clean", "dim", "warm", "delay"],
    })
    write_json(output / "recipe.json", recipe)


def _read_rows(dataset: Path, info: dict) -> list[dict]:
    if info["codebase_version"] != "v3.0" or info["fps"] != 10:
        raise ValueError("This benchmark requires LeRobot v3 PushT data at 10 Hz")
    for feature in ("action", "observation.state"):
        if info["features"][feature]["shape"] != [2]:
            raise ValueError(f"PushT requires two-dimensional {feature}")
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    rows = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    if len(rows) != info["total_frames"]:
        raise ValueError("Dataset frame count differs from metadata")
    episodes = sorted({row["episode_index"] for row in rows})
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Dataset episode count differs from metadata")
    for episode in episodes:
        _check_episode([row for row in rows if row["episode_index"] == episode])
    return rows


def _check_episode(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: row["frame_index"])
    if [row["frame_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Episode contains missing or duplicate frame indices")
    timestamps = np.array([row["timestamp"] for row in rows])
    if not np.allclose(timestamps, np.arange(len(rows)) / 10, atol=1e-4):
        raise ValueError("Episode timestamps differ from the 10 Hz control contract")
    for feature in ("action", "observation.state"):
        values = np.array([row[feature] for row in rows])
        if values.shape != (len(rows), 2) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {feature} vectors")
        if np.any(values < 0) or np.any(values > 512):
            raise ValueError(f"{feature} leaves the absolute PushT workspace [0, 512]")


def _training_statistics(rows: list[dict]) -> dict:
    stats = {}
    for feature in ("action", "observation.state"):
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        stats[feature] = {
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "count": [len(values)],
        }
    stats["observation.image"] = {
        "mean": [[[v]] for v in (0.485, 0.456, 0.406)],
        "std": [[[v]] for v in (0.229, 0.224, 0.225)],
        "min": [[[0.0]]] * 3, "max": [[[1.0]]] * 3, "count": [len(rows)],
    }
    return stats
