"""Verify completed upstream checkpoints, training metrics, and export hashes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .schemas import Recipe


def sha256(path: Path) -> str:
    """Hash an artifact without loading large checkpoints into memory.

    Args: path is a regular file.
    Returns: A hexadecimal SHA-256 digest.
    Raises: OSError if the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(output: Path, recipe: Recipe) -> Path:
    """Require the requested final update and finite recorded training metrics.

    Args: output is the artifact root; recipe contains the target update count.
    Returns: The completed checkpoint directory.
    Raises: ValueError for absent, incomplete, or nonfinite training evidence.
    """
    checkpoint = output / "train" / f"step-{recipe.training.steps}"
    for name in (
        "COMPLETE",
        "state.json",
        "config.json",
        "model/.metadata",
        "optimizer/.metadata",
    ):
        if not (checkpoint / name).is_file():
            raise ValueError(f"missing completed checkpoint artifact: {name}")
    state = json.loads((checkpoint / "state.json").read_text())
    if state.get("step") != recipe.training.steps:
        raise ValueError("checkpoint step does not match requested updates")
    _verify_metrics(output / "train/metrics.jsonl", recipe.training.steps)
    return checkpoint


def verify_export(directory: Path) -> dict:
    """Check the native export manifest against both published files.

    Args: directory contains the native standalone export.
    Returns: Its parsed manifest.
    Raises: ValueError if the export contract or hashes disagree.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("kind") != "policy_export" or manifest.get("format_version") != 1:
        raise ValueError("unsupported FLUX policy export manifest")
    for name in ("config.json", "model.safetensors"):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"empty or missing export artifact: {name}")
        if manifest.get("sha256", {}).get(name) != sha256(path):
            raise ValueError(f"export checksum mismatch: {name}")
    return manifest


def _verify_metrics(metrics: Path, steps: int) -> None:
    if not metrics.is_file():
        raise ValueError("training metrics.jsonl is missing")
    rows = [
        json.loads(line) for line in metrics.read_text().splitlines() if line.strip()
    ]
    updates = [row for row in rows if "loss" in row]
    if not updates or not any(row.get("step") == steps for row in updates):
        raise ValueError("metrics do not include the final training update")
    if not any(
        row.get("lr_trunk", 0) > 0 and row.get("lr_heads", 0) > 0 for row in updates
    ):
        raise ValueError("metrics do not prove full trunk and head training")
    for row in updates:
        if any(
            isinstance(value, (int, float)) and not math.isfinite(value)
            for value in row.values()
        ):
            raise ValueError("training metrics contain nonfinite values")
