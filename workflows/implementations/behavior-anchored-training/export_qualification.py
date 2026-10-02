"""Validate a policy export against immutable files and fixed-output evidence."""

from __future__ import annotations

import hashlib
import json
import os
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


def serialize_prng_key(jax: Any, key: Any) -> dict[str, Any]:
    """Serialize a typed JAX key through its public raw-key API.

    Args:
        jax: Imported JAX module.
        key: Typed PRNG key.

    Returns:
        Key implementation, raw data, and exact byte identity.
    """
    data = np.asarray(jax.device_get(jax.random.key_data(key)))
    return {
        "implementation": str(jax.random.key_impl(key)),
        "data": data.tolist(),
        **array_identity(data),
    }


def restore_prng_key(jax: Any, record: dict[str, Any]) -> Any:
    """Restore a typed JAX key after validating its exact raw data.

    Args:
        jax: Imported JAX module.
        record: Exact record from :func:`serialize_prng_key`.

    Returns:
        Typed key with the recorded implementation and raw data.

    Raises:
        ValueError: Record fields, raw data, or restored identity differ.
    """
    required = {"implementation", "data", "shape", "dtype", "bytes", "sha256"}
    if set(record) != required:
        raise ValueError("serialized PRNG key fields differ")
    data = np.asarray(record["data"], dtype=record["dtype"])
    if array_identity(data) != {
        name: record[name] for name in ("shape", "dtype", "bytes", "sha256")
    }:
        raise ValueError("serialized PRNG key data differs")
    key = jax.random.wrap_key_data(data, impl=record["implementation"])
    if serialize_prng_key(jax, key) != record:
        raise ValueError("restored typed PRNG key differs")
    return key


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


def flatten_array_tree(value: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten a nested string-keyed mapping into stable array paths.

    Args:
        value: Nested mapping whose leaves are array-compatible or ``None``.

    Returns:
        Stable slash-delimited paths mapped to NumPy arrays.

    Raises:
        ValueError: The tree is empty, has unsafe keys, or contains objects.
    """
    result: dict[str, np.ndarray] = {}

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            if not node:
                raise ValueError("empty mappings are not stable diagnostic inputs")
            for key in sorted(node):
                if not isinstance(key, str) or not key or "/" in key:
                    raise ValueError("diagnostic mapping keys must be safe strings")
                visit(node[key], (*path, key))
            return
        if node is None:
            return
        array = np.asarray(node)
        if array.dtype.hasobject:
            raise ValueError("diagnostic arrays cannot have object dtype")
        result["/".join(path)] = array

    visit(value, ())
    if not result or "" in result:
        raise ValueError("diagnostic array tree is empty")
    return result


def unflatten_array_tree(value: dict[str, np.ndarray]) -> dict[str, Any]:
    """Restore a mapping produced by :func:`flatten_array_tree`.

    Args:
        value: Flat slash-delimited array mapping.

    Returns:
        Nested mapping with the original arrays.

    Raises:
        TypeError: A flat path traverses an existing leaf.
        ValueError: A path is empty, malformed, or duplicated.
    """
    result: dict[str, Any] = {}
    for path, array in sorted(value.items()):
        parts = path.split("/")
        if not path or any(not part for part in parts):
            raise ValueError("diagnostic array path differs")
        cursor = result
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise TypeError("diagnostic array path aliases a leaf")
            cursor = child
        if parts[-1] in cursor:
            raise ValueError("duplicate diagnostic array path")
        cursor[parts[-1]] = array
    return result


def _write_npz_new(path, flattened):
    encoded = {
        f"leaf_{ordinal:03d}": array for ordinal, array in enumerate(flattened.values())
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_entries(flattened):
    return {
        name: {"archive_key": f"leaf_{ordinal:03d}", **array_identity(array)}
        for ordinal, (name, array) in enumerate(flattened.items())
    }


def _write_json_new(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_array_bundle(
    path: Path,
    manifest_path: Path,
    *,
    schema: str,
    arrays: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically persist raw arrays with a byte-verifiable manifest.

    Args:
        path: Destination NPZ path.
        manifest_path: Destination JSON manifest path.
        schema: Caller-owned schema identifier.
        arrays: Nested string-keyed array mapping.
        metadata: JSON-compatible immutable input metadata.

    Returns:
        The exact manifest written to disk.

    Raises:
        FileExistsError: Either output already exists.
        ValueError: The array tree is empty, unsafe, or contains objects.
    """
    for target in (path, manifest_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
    flattened = flatten_array_tree(arrays)
    _write_npz_new(path, flattened)
    manifest = {
        "schema": schema,
        "arrays": _array_entries(flattened),
        "metadata": metadata,
        "npz": file_identity(path),
    }
    _write_json_new(manifest_path, manifest)
    return manifest


def _load_npz_arrays(path, entries, expected_keys):
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("diagnostic NPZ members differ")
        for name, entry in sorted(entries.items()):
            array = _restore_array_dtype(
                np.asarray(archive[entry["archive_key"]]), entry["dtype"]
            )
            identity = {
                key: entry[key] for key in ("shape", "dtype", "bytes", "sha256")
            }
            if array_identity(array) != identity:
                raise ValueError(f"diagnostic array differs: {name}")
            arrays[name] = array
    return arrays


def load_array_bundle(
    path: Path, manifest_path: Path, *, expected_schema: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load raw arrays after checking manifest, archive, and leaf identities.

    Args:
        path: NPZ archive to verify and load.
        manifest_path: Exact array manifest.
        expected_schema: Required caller-owned schema.

    Returns:
        Flat arrays and the validated manifest.

    Raises:
        ValueError: Schema, archive identity, members, dtype, or bytes differ.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != expected_schema:
        raise ValueError("diagnostic array schema differs")
    if manifest.get("npz") != file_identity(path):
        raise ValueError("diagnostic NPZ identity differs")
    entries = manifest.get("arrays")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("diagnostic array manifest is empty")
    expected_keys = {entry.get("archive_key") for entry in entries.values()}
    if None in expected_keys or len(expected_keys) != len(entries):
        raise ValueError("diagnostic archive keys differ")
    return _load_npz_arrays(path, entries, expected_keys), manifest


def _restore_array_dtype(array: np.ndarray, expected: str) -> np.ndarray:
    """Restore extension dtypes that NumPy NPZ exposes as raw void bytes."""
    if str(array.dtype) == expected:
        return array
    try:
        dtype = np.dtype(expected)
    except TypeError:
        import ml_dtypes

        extension = getattr(ml_dtypes, expected, None)
        if extension is None:
            raise ValueError(f"unsupported diagnostic dtype: {expected}") from None
        dtype = np.dtype(extension)
    if array.dtype.kind != "V" or array.dtype.itemsize != dtype.itemsize:
        raise ValueError(f"diagnostic array dtype differs: {array.dtype} != {expected}")
    return array.view(dtype)


def exact_array(left: Any, right: Any) -> bool:
    """Return whether dtype, shape, and C-order bytes are identical.

    Args:
        left: First array-compatible value.
        right: Second array-compatible value.

    Returns:
        True only when dtype, shape, and C-order bytes match.
    """
    first = np.asarray(left)
    second = np.asarray(right)
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def require_tree_exact(label: str, left: dict[str, Any], right: dict[str, Any]) -> None:
    """Require two flattened array trees to have identical leaves and bytes.

    Args:
        label: Human-readable comparison boundary.
        left: First flat array tree.
        right: Second flat array tree.

    Returns:
        None after every path and array matches.

    Raises:
        ValueError: Leaf paths or any array bytes differ.
    """
    if set(left) != set(right):
        raise ValueError(f"{label} leaf paths differ")
    for name in sorted(left):
        if not exact_array(left[name], right[name]):
            raise ValueError(f"{label} leaf {name} bytes differ")


def _require_pairs(values, pairs, prefix):
    for left, right in pairs:
        if not exact_array(values[left], values[right]):
            raise ValueError(f"{prefix}{left} bytes differ")


def _require_repeatability(capture, repeat):
    _require_pairs(
        capture,
        (
            ("actions", "same_load_repeat_actions"),
            ("stage_prediction", "same_load_repeat_stage_prediction"),
        ),
        "capture same-load ",
    )
    _require_pairs(
        repeat,
        (
            ("frozen_actions", "frozen_same_load_repeat_actions"),
            ("frozen_stage_prediction", "frozen_same_load_repeat_stage_prediction"),
            ("reconstructed_actions", "reconstructed_same_load_repeat_actions"),
            (
                "reconstructed_stage_prediction",
                "reconstructed_same_load_repeat_stage_prediction",
            ),
        ),
        "repeat ",
    )


def _require_independent_load(capture, repeat):
    pairs = (
        ("actions", "frozen_actions"),
        ("stage_prediction", "frozen_stage_prediction"),
        ("actions", "reconstructed_actions"),
        ("stage_prediction", "reconstructed_stage_prediction"),
    )
    for captured, repeated in pairs:
        if not exact_array(capture[captured], repeat[repeated]):
            raise ValueError(f"independent loader {captured} bytes differ")


def _stage_proofs(capture, repeat, task_ids, task_num_stages):
    names = (
        ("capture/stage_prediction", capture, "stage_prediction"),
        (
            "capture/same_load_repeat_stage_prediction",
            capture,
            "same_load_repeat_stage_prediction",
        ),
        ("repeat/frozen_stage_prediction", repeat, "frozen_stage_prediction"),
        (
            "repeat/frozen_same_load_repeat_stage_prediction",
            repeat,
            "frozen_same_load_repeat_stage_prediction",
        ),
        (
            "repeat/reconstructed_stage_prediction",
            repeat,
            "reconstructed_stage_prediction",
        ),
        (
            "repeat/reconstructed_same_load_repeat_stage_prediction",
            repeat,
            "reconstructed_same_load_repeat_stage_prediction",
        ),
    )
    return {
        label: stage_mask_proof(values[name], task_ids, task_num_stages)
        for label, values, name in names
    }


def _repeatability_receipt(actions, stages):
    return {
        "schema": "npa.behavior.export-loader-repeatability.v1",
        "status": "exact_loader_repeatability_passed_native_parity_unresolved",
        "observation_reconstruction_exact": True,
        "same_load_repeats_exact": True,
        "independent_load_repeat_exact": True,
        "actions": array_identity(actions),
        "stage_semantics": stages,
        "native_export_parity_resolved": False,
        "rollout_eligibility_claimed": False,
    }


def compare_loader_diagnostics(
    *,
    captured_observation: dict[str, Any],
    reconstructed_observation: dict[str, Any],
    capture: dict[str, Any],
    repeat: dict[str, Any],
    task_ids: Any,
    task_num_stages: Any,
) -> dict[str, Any]:
    """Compare saved loader diagnostics without claiming native parity.

    Args:
        captured_observation: Flattened post-transform arrays from process one.
        reconstructed_observation: Independently reconstructed flattened arrays.
        capture: First-process actions, stages, and same-load repeats.
        repeat: Second-process frozen/reconstructed actions, stages, and repeats.
        task_ids: Task ID for each output row.
        task_num_stages: Full frozen task-stage-count registry.

    Returns:
        Exact repeatability identities and task-aware stage proofs.

    Raises:
        ValueError: Any observation, output, repeat, finiteness, or mask differs.
    """
    require_tree_exact(
        "reconstructed observation", captured_observation, reconstructed_observation
    )
    _require_repeatability(capture, repeat)
    _require_independent_load(capture, repeat)
    actions = np.asarray(capture["actions"])
    if not np.isfinite(actions).all():
        raise ValueError("captured actions must be finite")
    stages = _stage_proofs(capture, repeat, task_ids, task_num_stages)
    return _repeatability_receipt(actions, stages)
