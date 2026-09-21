"""Install an explicit FP32 RLC correlation artifact before state capture."""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
import re
from typing import Any

CORRELATION_SHAPE = (960, 960)
CORRELATION_BYTES = 960 * 960 * 4
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STATE_PATH = ("action_correlation_cholesky",)


def correlation_identity(value: Any) -> dict[str, Any]:
    """Describe correlation bytes after transferring them to the host.

    Args:
        value: NumPy-compatible correlation array.
    Returns:
        Dtype, shape, byte count, and SHA-256 identity.
    Raises:
        ValueError: The value is not finite FP32 with the expected shape.
    """
    import numpy as np

    array = np.asarray(value)
    if (
        str(array.dtype) != "float32"
        or tuple(array.shape) != CORRELATION_SHAPE
        or not np.isfinite(array).all()
    ):
        raise ValueError("correlation array must be finite FP32 [960, 960]")
    payload = array.astype("<f4", copy=False).tobytes(order="C")
    return {
        "dtype": "float32",
        "shape": list(CORRELATION_SHAPE),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_fp32_correlation(path: Path, expected_sha256: str) -> Any:
    """Load one byte-bound little-endian FP32 correlation artifact.

    Args:
        path: Regular raw matrix file in C order.
        expected_sha256: Required lowercase SHA-256 of the raw bytes.
    Returns:
        A writable NumPy FP32 array with shape ``[960, 960]``.
    Raises:
        OSError: The artifact cannot be read.
        ValueError: The path, digest, size, dtype, shape, or values differ.
    """
    import numpy as np

    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("correlation SHA-256 must be lowercase hexadecimal")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != CORRELATION_BYTES
    ):
        raise ValueError("correlation artifact must be an exact-size regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("correlation artifact SHA-256 differs")
    value = np.frombuffer(payload, dtype="<f4").reshape(CORRELATION_SHAPE).copy()
    if correlation_identity(value)["sha256"] != expected_sha256:
        raise ValueError("decoded correlation identity differs")
    return value


def _require_intermediate(variable: Any) -> None:
    from flax import nnx

    if not isinstance(variable, nnx.Intermediate):
        raise TypeError("action correlation must be an NNX Intermediate")


def _stock_configuration(model: Any) -> dict[str, Any]:
    fields = {
        "use_correlated_noise": model.use_correlated_noise,
        "action_horizon": model.action_horizon,
        "action_dim": model.action_dim,
        "correlation_beta": model.correlation_beta,
    }
    expected = {
        "use_correlated_noise": True,
        "action_horizon": 30,
        "action_dim": 32,
        "correlation_beta": 0.5,
    }
    if fields != expected:
        raise ValueError("stock correlation configuration differs")
    return fields


@contextlib.contextmanager
def pre_policy_fp32_correlation(
    pi_behavior_type: type, correlation: Any, expected_sha256: str
):
    """Install FP32 state during one model load, before Policy captures it.

    Args:
        pi_behavior_type: Pinned ``PiBehavior`` class used by the stock loader.
        correlation: Validated FP32 correlation matrix.
        expected_sha256: Required identity of the installed matrix.
    Yields:
        A list receiving the single installation record.
    Raises:
        TypeError: The live model does not expose an NNX Intermediate.
        ValueError: Configuration, installed bytes, or call count differs.
    """
    expected = correlation_identity(correlation)
    if expected["sha256"] != expected_sha256:
        raise ValueError("correlation input SHA-256 differs")
    original = pi_behavior_type.load_correlation_matrix
    calls: list[dict[str, Any]] = []
    pi_behavior_type.load_correlation_matrix = _pre_policy_loader(
        correlation, expected, calls
    )
    try:
        yield calls
    finally:
        pi_behavior_type.load_correlation_matrix = original
    if len(calls) != 1:
        raise ValueError("expected exactly one pre-Policy correlation installation")


def _pre_policy_loader(
    correlation: Any, expected: dict[str, Any], calls: list[dict[str, Any]]
):
    import jax
    import jax.numpy as jnp

    def load(model: Any, norm_stats: dict[str, Any]) -> None:
        if not isinstance(norm_stats, dict) or "actions" not in norm_stats:
            raise ValueError("stock normalization contract differs")
        variable = model.action_correlation_cholesky
        _require_intermediate(variable)
        variable.value = jnp.asarray(correlation)
        model.correlation_loaded = True
        actual = correlation_identity(jax.device_get(variable.value))
        if actual != expected:
            raise ValueError("installed stock correlation identity differs")
        calls.append({"configuration": _stock_configuration(model), "identity": actual})

    return load


def verify_captured_correlation(policy: Any, expected_sha256: str) -> dict[str, Any]:
    """Verify the model state captured by the serving Policy.

    Args:
        policy: RLC wrapper containing ``policy._model``.
        expected_sha256: Required FP32 correlation identity.
    Returns:
        The captured correlation identity.
    Raises:
        TypeError: The captured state is not an NNX Intermediate.
        ValueError: The captured bytes differ.
    """
    import jax

    variable = policy.policy._model.action_correlation_cholesky
    _require_intermediate(variable)
    identity = correlation_identity(jax.device_get(variable.value))
    if identity["sha256"] != expected_sha256:
        raise ValueError("Policy captured a different stock correlation")
    return identity


def _inspect_training_state(state: Any) -> tuple[dict, dict]:
    before = dict(state.flat_state().items())
    if _STATE_PATH not in before:
        raise ValueError("training state lacks action correlation")
    target = before[_STATE_PATH]
    variable_type = getattr(target, "type", None)
    from flax import nnx

    if not isinstance(variable_type, type) or not issubclass(
        variable_type, nnx.Intermediate
    ):
        raise TypeError("training correlation state is not an NNX Intermediate")
    original_values = {path: variable.value for path, variable in before.items()}
    return before, original_values


def _restore_state(
    state: Any,
    correlation: Any,
    expected: dict[str, Any],
    inspection: tuple[dict, dict],
) -> dict[str, Any]:
    import jax

    before, original_values = inspection
    target = before[_STATE_PATH]
    target.value = jax.device_put(correlation, getattr(target.value, "sharding", None))
    after = dict(state.flat_state().items())
    if set(after) != set(before):
        raise ValueError("training state paths changed during correlation installation")
    for path, variable in before.items():
        if path != _STATE_PATH and (
            after[path] is not variable
            or after[path].value is not original_values[path]
        ):
            raise ValueError("non-correlation training state changed")
    actual = correlation_identity(jax.device_get(after[_STATE_PATH].value))
    if actual != expected:
        raise ValueError("training correlation device readback differs")
    return actual


def install_training_correlation(
    train_state: Any, artifact_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], ...]:
    """Install canonical state into parameters and EMA before teacher capture.

    Args:
        train_state: Native state with ``params`` and optional ``ema_params``.
        artifact_path: Raw little-endian FP32 correlation artifact.
        expected_sha256: Required artifact and device-readback SHA-256.
    Returns:
        One installation record per updated state, in params/EMA order.
    Raises:
        OSError: The artifact cannot be read.
        TypeError: A target leaf is not an NNX Intermediate.
        ValueError: Artifact, topology, non-target state, or readback differs.
    """
    correlation = load_fp32_correlation(artifact_path, expected_sha256)
    expected = correlation_identity(correlation)
    states = [("params", train_state.params)]
    if train_state.ema_params is not None:
        states.append(("ema_params", train_state.ema_params))
    inspections = [_inspect_training_state(state) for _, state in states]
    return tuple(
        {
            "state": name,
            "identity": _restore_state(state, correlation, expected, inspection),
        }
        for (name, state), inspection in zip(states, inspections, strict=True)
    )
