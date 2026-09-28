"""Load a validated selected-RLC correlation asset before Policy JIT setup."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "npa.behavior.rlc-selected-native-correlation-adapter.v1"
VALIDATION_SCHEMA = "npa.behavior.rlc-selected-serving-validation.v1"
LEGACY_MANIFEST_SCHEMA = "npa.behavior.selected-3599-native-correlation-adapter.v1"
LEGACY_VALIDATION_SCHEMA = "npa.behavior.selected-3599-serving-validation.v1"
EXPORT_SCHEMA = "npa.behavior.rlc-selected-export.v1"
CORRELATION_SHAPE = (960, 960)
CORRELATION_DTYPE = "bfloat16"
CORRELATION_BYTES = 960 * 960 * 2
ARRAY_IDENTITY_FIELDS = ("dtype", "shape", "bytes", "sha256")
TYPED_IDENTITY_FIELDS = (*ARRAY_IDENTITY_FIELDS, "path", "variable_type")


def file_digest(path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: Regular file to hash.
    Returns:
        Lowercase hexadecimal SHA-256 digest.
    Raises:
        OSError: The file cannot be read.
    """

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_identity(value: Any) -> dict[str, Any]:
    """Describe an array's serving-relevant byte identity.

    Args:
        value: Array-compatible value.
    Returns:
        Dtype, shape, byte count, and SHA-256 identity.
    Raises:
        ValueError: The value cannot be represented as an array.
    """

    import numpy as np

    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _array_projection(identity: object, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict) or any(
        field not in identity for field in ARRAY_IDENTITY_FIELDS
    ):
        raise ValueError(f"{label} array identity fields differ")
    return {field: identity[field] for field in ARRAY_IDENTITY_FIELDS}


def _typed_correlation(identity: object, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict) or set(identity) != set(TYPED_IDENTITY_FIELDS):
        raise ValueError(f"{label} typed-state identity fields differ")
    if (
        identity["path"] != "action_correlation_cholesky"
        or identity["variable_type"] != "Intermediate"
    ):
        raise ValueError(f"{label} is not the correlation Intermediate")
    return _array_projection(identity, label)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not set(value) - set("0123456789abcdef")
    )


def _validate_hashes(values: object, label: str) -> dict[str, str]:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{label} hash mapping is empty")
    if any(
        not isinstance(name, str) or not _is_sha256(value)
        for name, value in values.items()
    ):
        raise ValueError(f"{label} contains an invalid SHA-256")
    return values


def _validate_artifact(manifest: dict[str, Any], manifest_path: Path) -> Path:
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "path",
        "sha256",
        "bytes",
        "dtype",
        "shape",
    }:
        raise ValueError("correlation artifact manifest fields differ")
    if Path(artifact["path"]).name != artifact["path"]:
        raise ValueError("correlation artifact path must be a basename")
    expected = (CORRELATION_BYTES, CORRELATION_DTYPE, list(CORRELATION_SHAPE))
    if (artifact["bytes"], artifact["dtype"], artifact["shape"]) != expected:
        raise ValueError("correlation artifact shape or dtype differs")
    if not _is_sha256(artifact["sha256"]):
        raise ValueError("correlation artifact SHA-256 is invalid")
    artifact_path = manifest_path.parent / artifact["path"]
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError("correlation artifact must be a regular file")
    if (
        artifact_path.stat().st_size != artifact["bytes"]
        or file_digest(artifact_path) != artifact["sha256"]
    ):
        raise ValueError("correlation artifact bytes differ")
    return artifact_path


def _validate_relations(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    export: dict[str, Any],
    manifest_path: Path,
    export_path: Path,
) -> None:
    step = manifest.get("selected_step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("selected step must be a positive integer")
    schema = manifest.get("schema")
    if schema == MANIFEST_SCHEMA:
        validation_schema = VALIDATION_SCHEMA
        validation_status = "selected_serving_validated"
    elif schema == LEGACY_MANIFEST_SCHEMA and step == 3599:
        validation_schema = LEGACY_VALIDATION_SCHEMA
        validation_status = "selected_3599_serving_validated"
    else:
        raise ValueError("unsupported selected-RLC correlation manifest")
    if manifest.get("adapter_version") != 1:
        raise ValueError("unsupported selected-RLC correlation adapter version")
    if export.get("schema") != EXPORT_SCHEMA or export.get("selected_step") != step:
        raise ValueError("selected export does not bind the correlation step")
    if export.get("status") != "holdout_selected_not_rollout_evaluated":
        raise ValueError("selected export status differs")
    if (
        validation.get("schema") != validation_schema
        or validation.get("selected_step") != step
    ):
        raise ValueError("serving validation does not bind the correlation step")
    if validation.get("status") != validation_status:
        raise ValueError("selected checkpoint has not passed serving validation")
    if validation.get("candidate_rollout_eligible") is not True:
        raise ValueError("selected checkpoint is not rollout eligible")
    adapter = validation.get("adapter", {})
    if adapter.get("manifest_sha256") != file_digest(manifest_path):
        raise ValueError("serving validation does not bind the correlation manifest")
    if manifest["evidence_sha256"].get("selected_export_receipt") != file_digest(
        export_path
    ):
        raise ValueError("correlation manifest does not bind the selected export")


def _validate_adapter_sources(validation: dict[str, Any], adapter_root: Path) -> None:
    expected = validation.get("existing_adapter", {}).get("source_sha256")
    actual = {
        name: file_digest(adapter_root / name)
        for name in ("rlc_server.py", "rlc_observations.py")
    }
    if expected != actual:
        raise ValueError("serving validation binds different RLC adapter sources")


def _validate_serving_claims(validation: dict[str, Any]) -> None:
    typed_state = validation.get("typed_state", {})
    after = typed_state.get("after_adapter", {})
    if (
        not isinstance(typed_state.get("leaves"), int)
        or isinstance(typed_state["leaves"], bool)
        or typed_state["leaves"] <= 0
        or after.get("equal") is not True
        or after.get("differing") != []
        or after.get("direct_only") != []
        or after.get("native_only") != []
    ):
        raise ValueError("serving validation did not prove typed-state equality")
    metrics = validation.get("metrics", {}).get("native_vs_aligned_selected_export", {})
    actions = validation.get("actions", {})
    selected_actions = actions.get("native_vs_aligned_selected_export", {})
    existing_actions = actions.get("native_vs_existing_adapter_policy_jit", {})
    smoke = actions.get("existing_adapter_smoke", {})
    metric_parts = metrics.get("parts")
    if (
        metrics.get("equal") is not True
        or not isinstance(metric_parts, list)
        or len(metric_parts) != 4
        or any(
            part.get("equal") is not True
            or part.get("dtype_equal") is not True
            or part.get("shape_equal") is not True
            or part.get("max_abs_delta") != 0.0
            or part.get("mean_abs_delta") != 0.0
            for part in metric_parts
        )
    ):
        raise ValueError("serving validation did not prove fixed-batch metric equality")
    action_parts = selected_actions.get("parts")
    existing_parts = existing_actions.get("parts")

    def equal_actions(parts: object) -> bool:
        if not isinstance(parts, list) or len(parts) != 2:
            return False
        if any(
            part.get("equal") is not True
            or part.get("dtype_equal") is not True
            or part.get("shape_equal") is not True
            for part in parts
        ):
            return False
        if (parts[0].get("max_abs_delta"), parts[0].get("mean_abs_delta")) != (
            0.0,
            0.0,
        ):
            return False
        auxiliary = (
            parts[1].get("max_abs_delta"),
            parts[1].get("mean_abs_delta"),
        )
        return auxiliary == (0.0, 0.0) or all(
            isinstance(value, float) and math.isnan(value) for value in auxiliary
        )

    if (
        selected_actions.get("equal") is not True
        or existing_actions.get("equal") is not True
        or not equal_actions(action_parts)
        or not equal_actions(existing_parts)
        or smoke.get("finite") is not True
        or smoke.get("dtype") != "float32"
        or smoke.get("shape") != [23]
    ):
        raise ValueError("serving validation did not prove the action contract")


def _validate_finite_bfloat16(data: bytes) -> None:
    """Reject BF16 infinity and NaN bit patterns without numeric dependencies."""

    if len(data) % 2:
        raise ValueError("correlation artifact has a partial BF16 element")
    if any(
        int.from_bytes(data[offset : offset + 2], "little") & 0x7F80 == 0x7F80
        for offset in range(0, len(data), 2)
    ):
        raise ValueError("correlation artifact contains non-finite values")


def validate_selected_correlation(
    manifest_path: Path,
    validation_path: Path,
    export_path: Path,
    adapter_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate a selected correlation asset using the Python standard library.

    Args:
        manifest_path: Correlation artifact manifest.
        validation_path: Successful selected-checkpoint serving receipt.
        export_path: Holdout-selected model export receipt.
        adapter_root: Directory containing the exact RLC server sources.
    Returns:
        The correlation artifact path and its manifest.
    Raises:
        OSError: A required file cannot be read.
        ValueError: Any schema, source, relationship, or byte identity differs.
    """

    manifest = _load_json(manifest_path, "correlation manifest")
    validation = _load_json(validation_path, "serving validation receipt")
    export = _load_json(export_path, "selected export receipt")
    _validate_hashes(manifest.get("source_sha256"), "manifest source")
    _validate_hashes(manifest.get("evidence_sha256"), "manifest evidence")
    if validation.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("validation and manifest source identities differ")
    if validation.get("evidence_sha256") != manifest["evidence_sha256"]:
        raise ValueError("validation and manifest evidence identities differ")
    _validate_relations(manifest, validation, export, manifest_path, export_path)
    _validate_adapter_sources(validation, adapter_root)
    _validate_serving_claims(validation)
    artifact_path = _validate_artifact(manifest, manifest_path)
    data = artifact_path.read_bytes()
    _validate_finite_bfloat16(data)
    native = _typed_correlation(
        manifest.get("native_correlation"), "native correlation"
    )
    direct = _typed_correlation(
        manifest.get("direct_correlation_before"), "direct correlation"
    )
    if native != {key: manifest["artifact"][key] for key in ARRAY_IDENTITY_FIELDS}:
        raise ValueError("correlation artifact differs from the native state identity")
    if (
        direct["dtype"] != "float32"
        or direct["shape"] != list(CORRELATION_SHAPE)
        or direct["bytes"] != CORRELATION_SHAPE[0] * CORRELATION_SHAPE[1] * 4
        or not _is_sha256(direct["sha256"])
    ):
        raise ValueError("direct correlation identity fields differ")
    return artifact_path, manifest


def load_validated_correlation(
    manifest_path: Path,
    validation_path: Path,
    export_path: Path,
    adapter_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Decode a validated selected correlation in the policy runtime.

    Args:
        manifest_path: Correlation artifact manifest.
        validation_path: Successful selected-checkpoint serving receipt.
        export_path: Holdout-selected model export receipt.
        adapter_root: Directory containing the exact RLC server sources.
    Returns:
        The BF16 correlation array and its manifest.
    Raises:
        ImportError: The policy runtime lacks its numeric dependencies.
        OSError: A required file cannot be read.
        ValueError: Any schema, source, relationship, or byte identity differs.
    """

    import ml_dtypes
    import numpy as np

    artifact_path, manifest = validate_selected_correlation(
        manifest_path, validation_path, export_path, adapter_root
    )
    array = np.frombuffer(artifact_path.read_bytes(), dtype=ml_dtypes.bfloat16).reshape(
        CORRELATION_SHAPE
    )
    if array_identity(array) != _array_projection(
        manifest.get("native_correlation"), "native correlation"
    ):
        raise ValueError("loaded correlation identity differs from the native state")
    return array, manifest


@contextlib.contextmanager
def pre_policy_correlation(pi_behavior_type: type, native_correlation: Any):
    """Install native correlation state before Policy captures its JIT sampler.

    Args:
        pi_behavior_type: PiBehavior class used by the selected checkpoint.
        native_correlation: Validated BF16 correlation matrix.
    Yields:
        The single observed direct-before and native-after identity record.
    Raises:
        ValueError: Installation differs or the initializer does not call once.
    """

    import jax.numpy as jnp

    original = pi_behavior_type.load_correlation_matrix
    expected = array_identity(native_correlation)
    calls: list[dict[str, Any]] = []

    def load(instance, norm_stats):
        original(instance, norm_stats)
        before = array_identity(instance.action_correlation_cholesky.value)
        instance.action_correlation_cholesky.value = jnp.asarray(native_correlation)
        after = array_identity(instance.action_correlation_cholesky.value)
        if after != expected:
            raise ValueError("pre-Policy correlation installation differs")
        calls.append({"direct_before": before, "direct_after": after})

    pi_behavior_type.load_correlation_matrix = load
    try:
        yield calls
    finally:
        pi_behavior_type.load_correlation_matrix = original
    if len(calls) != 1:
        raise ValueError("expected exactly one pre-Policy correlation installation")


def load_selected_policy(
    adapter: Any, args: Any, native_correlation: Any, manifest: dict
):
    """Construct the existing RLC wrapper with validated native correlation.

    Args:
        adapter: Imported unchanged RLC server module.
        args: Server arguments consumed by ``adapter._load_policy``.
        native_correlation: Validated BF16 correlation matrix.
        manifest: Validated correlation manifest.
    Returns:
        Existing RLC policy wrapper.
    Raises:
        ValueError: Direct initialization or captured model state differs.
    """

    from b1k.models.pi_behavior import PiBehavior

    with pre_policy_correlation(PiBehavior, native_correlation) as calls:
        wrapper = adapter._load_policy(args)
    before = _array_projection(
        manifest.get("direct_correlation_before"), "direct correlation"
    )
    if calls[0]["direct_before"] != before:
        raise ValueError("direct correlation identity differs from validation")
    if array_identity(
        wrapper.policy._model.action_correlation_cholesky.value
    ) != array_identity(native_correlation):
        raise ValueError("RLC Policy did not capture validated correlation state")
    return wrapper
