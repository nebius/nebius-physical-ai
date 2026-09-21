"""Validate a policy export against immutable files and fixed-output evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def file_identity(path: Path) -> dict[str, Any]:
    """Describe one immutable file.

    Args:
        path: Regular file to hash.

    Returns:
        Its byte length and SHA-256 digest.

    Raises:
        ValueError: If ``path`` is absent, a symlink, or not a regular file.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"immutable file is absent or unsafe: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def validate_export_files(
    export_root: Path, expected_files: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Require an export tree to equal its declared regular-file inventory.

    Args:
        export_root: Root containing the model export.
        expected_files: Relative paths mapped to byte lengths and SHA-256 digests.

    Returns:
        The independently measured inventory in sorted path order.

    Raises:
        ValueError: If paths are unsafe, missing, extra, or byte-different.
    """
    actual_paths = [path for path in export_root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in export_root.rglob("*")):
        raise ValueError("export tree contains a symlink")
    actual_names = {path.relative_to(export_root).as_posix() for path in actual_paths}
    if actual_names != set(expected_files):
        raise ValueError("export file set differs")
    actual = {name: file_identity(export_root / name) for name in sorted(actual_names)}
    if actual != dict(sorted(expected_files.items())):
        raise ValueError("export file identities differ")
    return actual


def array_identity(value: Any) -> dict[str, Any]:
    """Describe an array using its exact host bytes.

    Args:
        value: Array-like value already transferred to the host when necessary.

    Returns:
        Shape, dtype, byte length, and SHA-256 digest.

    Raises:
        ValueError: If the value has an object dtype.
    """
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not stable qualification inputs")
    data = array.tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def stage_mask_proof(
    stage_prediction: Any, task_ids: Any, task_num_stages: Any
) -> dict[str, Any]:
    """Validate finite stage logits and the model's documented ``-inf`` mask.

    Args:
        stage_prediction: Batch by maximum-stage logits.
        task_ids: Task ID for every batch row.
        task_num_stages: Frozen number of valid stages for every task ID.

    Returns:
        Auditable task IDs, stage counts, mask identity, and array identity.

    Raises:
        ValueError: If shapes, IDs, counts, finite logits, or masks differ.
    """
    stages = np.asarray(stage_prediction)
    ids = np.asarray(task_ids)
    counts = np.asarray(task_num_stages)
    _validate_stage_inputs(stages, ids, counts)
    selected_counts = counts[ids]
    if np.any(selected_counts <= 0) or np.any(selected_counts > stages.shape[1]):
        raise ValueError("task stage count differs")
    valid_mask = np.arange(stages.shape[1])[None, :] < selected_counts[:, None]
    if not np.isfinite(stages[valid_mask]).all():
        raise ValueError("valid stage logits must be finite")
    if not np.isneginf(stages[~valid_mask]).all():
        raise ValueError("invalid stage logits must be negative infinity")
    return {
        "task_ids": ids.tolist(),
        "selected_task_stage_counts": selected_counts.tolist(),
        "task_num_stages": array_identity(counts),
        "expected_valid_mask": array_identity(valid_mask),
        "stage_prediction": array_identity(stages),
        "semantically_valid": True,
    }


def qualify_policy_output(
    output: tuple[Any, Any],
    *,
    task_ids: Any,
    task_num_stages: Any,
    expected_actions: dict[str, Any],
    expected_stage_prediction: dict[str, Any],
) -> dict[str, Any]:
    """Require a fixed policy call to reproduce archived native/export bytes.

    Args:
        output: ``(actions, stage_prediction)`` from the policy.
        task_ids: Task ID for every batch row.
        task_num_stages: Frozen number of valid stages for every task.
        expected_actions: Exact archived action identity.
        expected_stage_prediction: Exact archived stage-logit identity.

    Returns:
        A complete action and documented-stage-mask qualification proof.

    Raises:
        ValueError: If the output contract, finiteness, mask, or bytes differ.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("policy must return (actions, stage_prediction)")
    actions, stages = output
    action_array = np.asarray(actions)
    if not np.isfinite(action_array).all():
        raise ValueError("actions must be finite")
    action_identity = array_identity(action_array)
    mask = stage_mask_proof(stages, task_ids, task_num_stages)
    if action_identity != expected_actions:
        raise ValueError("fixed-seed action bytes differ")
    if mask["stage_prediction"] != expected_stage_prediction:
        raise ValueError("fixed-seed stage-prediction bytes differ")
    return {"actions": action_identity, "stage_prediction": mask, "exact": True}


def _validate_stage_inputs(stages: Any, ids: Any, counts: Any) -> None:
    if stages.ndim != 2:
        raise ValueError("stage predictions must be a two-dimensional array")
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("task IDs must be a one-dimensional integer array")
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("task stage counts must be a one-dimensional integer array")
    if stages.shape[0] != ids.shape[0] or ids.size == 0:
        raise ValueError("stage prediction batch differs from task IDs")
    if np.any(ids < 0) or np.any(ids >= counts.size):
        raise ValueError("task ID is outside the frozen stage-count table")
