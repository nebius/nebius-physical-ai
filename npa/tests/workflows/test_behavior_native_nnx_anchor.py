"""Test paired NNX execution for the public stock-anchor objective."""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-anchored-training"
)
sys.path.insert(0, str(IMPLEMENTATION))
SPEC = importlib.util.spec_from_file_location(
    "behavior_native_nnx_anchor", IMPLEMENTATION / "native_nnx_anchor.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TinyFlow(nnx.Module):
    """Provide one trainable action leaf and one typed correlation leaf."""

    def __init__(self, weight: float = 1.0):
        self.weight = nnx.Param(jnp.asarray(weight, dtype=jnp.float32))
        self.action_correlation_cholesky = nnx.Intermediate(
            jnp.eye(2, dtype=jnp.float32)
        )


def detailed_loss(model, rng, observation, actions, *, num_flow_samples):
    noise = jax.random.normal(rng, (num_flow_samples, *observation.shape))
    velocity = model.weight.value * observation[None] + noise
    return {
        "total_loss": jnp.square(velocity - actions[None]),
        "_flow_velocity": velocity,
    }


def parent_state(model):
    state = SimpleNamespace(params=nnx.state(model))
    return MODULE.capture_parent_action_state(state, nnx.Param)


def anchored(model, parent):
    paired = MODULE.paired_detailed_losses(
        model,
        parent,
        nnx.Param,
        jax.random.key(11),
        jnp.ones((2, 3, 4), dtype=jnp.float32),
        jnp.zeros((2, 3, 4), dtype=jnp.float32),
        loss_method=detailed_loss,
        num_flow_samples=3,
    )
    return MODULE.native_anchored_loss(
        paired, array_module=jnp, stop_gradient=jax.lax.stop_gradient
    )


def test_paired_call_has_zero_anchor_and_positive_student_gradient() -> None:
    model = TinyFlow()
    parent = parent_state(model)

    def loss(selected):
        objective, metrics = anchored(selected, parent)
        return objective.total_loss, metrics

    update = nnx.jit(nnx.value_and_grad(loss, has_aux=True))
    (value, metrics), gradients = update(model)
    assert np.isfinite(float(value))
    assert np.array_equal(np.asarray(metrics["anchor_loss"]), np.zeros(2))
    assert float(jnp.abs(gradients.weight.value)) > 0.0
    assert "_flow_velocity" not in metrics

    def anchor_only(selected, frozen):
        objective, _ = anchored(selected, frozen)
        return objective.anchor_loss

    anchor_value, (student_anchor_gradient, parent_gradient) = nnx.jit(
        nnx.value_and_grad(anchor_only, argnums=(0, 1))
    )(model, parent)
    assert float(anchor_value) == 0.0
    assert float(student_anchor_gradient.weight.value) == 0.0
    assert float(parent_gradient.weight.value) == 0.0


def test_drift_produces_anchor_gradient_and_parent_stays_independent() -> None:
    model = TinyFlow()
    parent = parent_state(model)
    captured = np.asarray(parent.weight.value).copy()

    @nnx.jit(donate_argnums=(0,))
    def drift_student(selected):
        selected.weight.value = selected.weight.value + 0.25

    drift_student(model)

    def anchor_only(selected):
        objective, _ = anchored(selected, parent)
        return objective.anchor_loss

    anchor_loss, gradients = nnx.jit(nnx.value_and_grad(anchor_only))(model)
    assert float(anchor_loss) > 0.0
    assert float(jnp.abs(gradients.weight.value)) > 0.0
    assert np.array_equal(np.asarray(parent.weight.value), captured)


def test_capture_requires_fp32_intermediate_correlation() -> None:
    model = TinyFlow()
    model.action_correlation_cholesky = nnx.Param(jnp.eye(2, dtype=jnp.float32))
    with pytest.raises(ValueError, match="Intermediate"):
        parent_state(model)
    model.action_correlation_cholesky = nnx.Intermediate(jnp.eye(2, dtype=jnp.bfloat16))
    with pytest.raises(ValueError, match="FP32"):
        parent_state(model)


def test_capture_rejects_different_variable_class_named_intermediate() -> None:
    fake_intermediate = type("Intermediate", (nnx.Variable,), {})
    model = TinyFlow()
    model.action_correlation_cholesky = fake_intermediate(jnp.eye(2, dtype=jnp.float32))
    with pytest.raises(ValueError, match="NNX Intermediate"):
        parent_state(model)


@pytest.mark.parametrize(
    "difference", ["path", "missing", "extra", "type", "shape", "dtype"]
)
def test_action_metadata_drift_fails_before_mapping(difference: str) -> None:
    student = nnx.state(TinyFlow()).filter(nnx.Param)
    if difference == "path":
        parent = nnx.state(nnx.Dict(other=nnx.Param(jnp.asarray(1.0))))
    elif difference == "missing":
        parent = nnx.State({})
    elif difference == "extra":
        parent = nnx.state(
            nnx.Dict(
                weight=nnx.Param(jnp.asarray(1.0)),
                other=nnx.Param(jnp.asarray(2.0)),
            )
        )
    elif difference == "type":
        parent = nnx.state(nnx.Dict(weight=nnx.Intermediate(jnp.asarray(1.0))))
    elif difference == "shape":
        parent = nnx.state(nnx.Dict(weight=nnx.Param(jnp.ones((2,)))))
    else:
        parent = nnx.state(
            nnx.Dict(weight=nnx.Param(jnp.asarray(1.0, dtype=jnp.float16)))
        )
    with pytest.raises(ValueError, match="differ"):
        MODULE._validate_action_parity(student, parent)


def test_paired_flow_velocity_has_model_axis() -> None:
    model = TinyFlow()
    paired = MODULE.paired_detailed_losses(
        model,
        parent_state(model),
        nnx.Param,
        jax.random.key(3),
        jnp.ones((2, 3, 4), dtype=jnp.float32),
        jnp.zeros((2, 3, 4), dtype=jnp.float32),
        loss_method=detailed_loss,
        num_flow_samples=3,
    )
    assert paired["_flow_velocity"].shape == (2, 3, 2, 3, 4)


def test_invalid_flow_sample_count_fails_before_model_call() -> None:
    model = TinyFlow()
    with pytest.raises(ValueError, match="positive"):
        MODULE.paired_detailed_losses(
            model,
            parent_state(model),
            nnx.Param,
            jax.random.key(0),
            jnp.ones((1, 1, 1)),
            jnp.zeros((1, 1, 1)),
            loss_method=detailed_loss,
            num_flow_samples=0,
        )


def test_reordered_action_paths_fail_before_mapping() -> None:
    first = nnx.state(nnx.Dict(a=nnx.Param(jnp.asarray(1.0)))).a
    second = nnx.state(nnx.Dict(b=nnx.Param(jnp.asarray(2.0)))).b

    class OrderedState:
        def __init__(self, rows):
            self.rows = rows

        def flat_state(self):
            return OrderedDict(self.rows)

    student = OrderedState([(("a",), first), (("b",), second)])
    parent = OrderedState([(("b",), second), (("a",), first)])
    with pytest.raises(ValueError, match="paths differ"):
        MODULE._validate_action_parity(student, parent)
