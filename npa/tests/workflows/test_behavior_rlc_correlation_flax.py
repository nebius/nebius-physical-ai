"""Exercise RLC correlation installation with the pinned JAX/Flax state API."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

SOURCE = (
    Path(__file__).parents[2]
    / "src/npa/workflows/behavior_challenge/rlc_correlation.py"
)
SPEC = importlib.util.spec_from_file_location("public_rlc_correlation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
correlation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correlation)


class CorrelationModel(nnx.Module):
    """Expose the actual NNX variable types used by the pinned policy."""

    def __init__(self, *, intermediate: bool = True) -> None:
        value = jnp.zeros(correlation.CORRELATION_SHAPE, dtype=jnp.float32)
        variable = nnx.Intermediate(value) if intermediate else nnx.Param(value)
        self.action_correlation_cholesky = variable
        self.other = nnx.Param(jnp.ones((2,), dtype=jnp.float32))
        self.use_correlated_noise = True
        self.action_horizon = 30
        self.action_dim = 32
        self.correlation_beta = 0.5
        self.correlation_loaded = False

    def load_correlation_matrix(self, norm_stats) -> None:
        raise AssertionError(f"native numerical reconstruction ran: {norm_stats}")


def artifact(tmp_path: Path) -> tuple[Path, str, np.ndarray]:
    """Write a deterministic nonzero FP32 matrix for identity assertions."""
    value = np.arange(1, np.prod(correlation.CORRELATION_SHAPE) + 1, dtype=np.float32)
    value = value.reshape(correlation.CORRELATION_SHAPE) / 1_000_000
    path = tmp_path / "correlation.float32.bin"
    path.write_bytes(value.astype("<f4").tobytes(order="C"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), value


def test_pre_policy_hook_installs_actual_intermediate_before_capture(tmp_path) -> None:
    path, expected, value = artifact(tmp_path)
    loaded = correlation.load_fp32_correlation(path, expected)

    with correlation.pre_policy_fp32_correlation(
        CorrelationModel, loaded, expected
    ) as calls:
        model = CorrelationModel()
        model.load_correlation_matrix({"actions": object()})
        policy = CapturedPolicy(model)
        wrapper = SimpleNamespace(policy=policy)

    late_model = CorrelationModel()
    late_policy = CapturedPolicy(late_model)
    late_model.action_correlation_cholesky.value = jnp.asarray(value)

    assert len(calls) == 1
    assert isinstance(model.action_correlation_cholesky, nnx.Intermediate)
    assert np.array_equal(np.asarray(model.action_correlation_cholesky.value), value)
    assert float(policy.sample()) == float(value[0, 0])
    assert float(late_policy.sample()) == 0.0
    assert (
        correlation.verify_captured_correlation(wrapper, expected)["sha256"] == expected
    )
    with pytest.raises(AssertionError, match="native numerical reconstruction"):
        CorrelationModel().load_correlation_matrix({"actions": object()})


def test_training_installer_updates_params_and_ema_before_teacher(tmp_path) -> None:
    path, expected, value = artifact(tmp_path)
    params = nnx.state(CorrelationModel())
    ema_params = nnx.state(CorrelationModel())
    other_before = (
        params["other"],
        params["other"].value,
        ema_params["other"],
        ema_params["other"].value,
    )
    train_state = SimpleNamespace(params=params, ema_params=ema_params)

    records = correlation.install_training_correlation(train_state, path, expected)
    teacher = train_state.params.filter(nnx.Intermediate)

    assert [record["state"] for record in records] == ["params", "ema_params"]
    assert all(record["identity"]["sha256"] == expected for record in records)
    params_flat = dict(params.flat_state().items())
    ema_flat = dict(ema_params.flat_state().items())
    assert np.array_equal(np.asarray(params_flat[_path()].value), value)
    assert np.array_equal(np.asarray(ema_flat[_path()].value), value)
    assert dict(teacher.flat_state().items())[_path()].type is nnx.Intermediate
    assert params["other"] is other_before[0]
    assert params["other"].value is other_before[1]
    assert ema_params["other"] is other_before[2]
    assert ema_params["other"].value is other_before[3]


def test_training_installer_rejects_param_in_target_slot(tmp_path) -> None:
    path, expected, _ = artifact(tmp_path)
    state = SimpleNamespace(
        params=nnx.state(CorrelationModel(intermediate=False)), ema_params=None
    )

    with pytest.raises(TypeError, match="NNX Intermediate"):
        correlation.install_training_correlation(state, path, expected)


def test_invalid_ema_is_rejected_before_params_mutation(tmp_path) -> None:
    path, expected, _ = artifact(tmp_path)
    params = nnx.state(CorrelationModel())
    params_target = dict(params.flat_state().items())[_path()]
    original_value = params_target.value
    state = SimpleNamespace(
        params=params,
        ema_params=nnx.state(CorrelationModel(intermediate=False)),
    )

    with pytest.raises(TypeError, match="NNX Intermediate"):
        correlation.install_training_correlation(state, path, expected)

    assert params_target.value is original_value
    assert np.count_nonzero(np.asarray(params_target.value)) == 0


class CapturedPolicy:
    """Capture the model array in a real JIT closure like the serving Policy."""

    def __init__(self, model: CorrelationModel) -> None:
        self._model = model
        captured = model.action_correlation_cholesky.value
        self.sample = jax.jit(lambda: captured[0, 0])


def _path() -> tuple[str, ...]:
    return ("action_correlation_cholesky",)
