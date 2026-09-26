"""Test the operative public native anchored-trainer wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
optax = pytest.importorskip("optax")

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-anchored-training"
)
PARTITION_IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-matched-training"
)
sys.path.insert(0, str(IMPLEMENTATION))
sys.path.insert(0, str(PARTITION_IMPLEMENTATION))


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"behavior_{name}", IMPLEMENTATION / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load("native_trainer")
PREFLIGHT = load("anchor_preflight")

from action_partition import EXPECTED_TRAINABLE_PATHS, ExactPathSet  # noqa: E402


class TinyFlow(nnx.Module):
    """Provide one selected parameter, one frozen parameter, and correlation."""

    def __init__(self):
        self.weight = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))
        self.frozen = nnx.Param(jnp.asarray(2.0, dtype=jnp.float32))
        self.action_correlation_cholesky = nnx.Intermediate(
            jnp.eye(2, dtype=jnp.float32)
        )


class WeightOnly:
    """Select the exact weight path from NNX state."""

    __hash__ = object.__hash__

    def __call__(self, path, variable):
        del variable
        return tuple(str(part) for part in path) == ("weight",)

    def __eq__(self, other):
        return isinstance(other, WeightOnly)


def detailed_loss(model, rng, observation, actions, *, num_flow_samples):
    noise = jax.random.normal(rng, (num_flow_samples, *observation.shape))
    velocity = model.weight.value * observation[None] + noise
    return {
        "total_loss": jnp.square(velocity - actions[None]),
        "_flow_velocity": velocity,
    }


def frozen_only_loss(model, rng, observation, actions, *, num_flow_samples):
    noise = jax.random.normal(rng, (num_flow_samples, *observation.shape))
    velocity = model.frozen.value * observation[None] + noise
    return {
        "total_loss": jnp.square(velocity - actions[None]),
        "_flow_velocity": velocity,
    }


def wiring(model):
    return TRAINER.NativeTrainer.initialize(
        SimpleNamespace(params=nnx.state(model)),
        WeightOnly(),
        detailed_loss,
        num_flow_samples=2,
        allowed_paths={("weight",)},
    )


def batch():
    return (
        jax.random.key(7),
        jnp.ones((2, 3, 4), dtype=jnp.float32),
        jnp.zeros((2, 3, 4), dtype=jnp.float32),
    )


def test_native_trainer_runs_real_filtered_update_and_keeps_frozen_bytes():
    model = TinyFlow()
    trainer = wiring(model)
    optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=WeightOnly())
    frozen = np.asarray(model.frozen.value).copy()
    correlation = np.asarray(model.action_correlation_cholesky.value).copy()

    loss, metrics, changed = trainer.update(optimizer, *batch())

    assert np.isfinite(float(loss))
    assert changed == (("weight",),)
    assert "_flow_velocity" not in metrics
    assert np.array_equal(np.asarray(model.frozen.value), frozen)
    assert np.array_equal(
        np.asarray(model.action_correlation_cholesky.value), correlation
    )


def test_compiled_preflight_proves_paired_contract_and_parent_stability():
    model = TinyFlow()
    receipt = PREFLIGHT.compiled_initialization_preflight(
        wiring(model), model, *batch()
    )
    assert receipt["paired_anchor_loss_exact_zero"] is True
    assert receipt["student_demonstration_gradient_norm"] > 0
    assert receipt["student_anchor_gradient_norm"] == 0
    assert receipt["parent_gradient_norm"] == 0
    assert receipt["parent_state_byte_equal"] is True
    assert receipt["separate_topology_is_gate"] is False


def test_compiled_preflight_rejects_frozen_only_demonstration_gradient():
    model = TinyFlow()
    trainer = TRAINER.NativeTrainer.initialize(
        SimpleNamespace(params=nnx.state(model)),
        WeightOnly(),
        frozen_only_loss,
        num_flow_samples=2,
        allowed_paths={("weight",)},
    )
    with pytest.raises(ValueError, match="demonstration gradient"):
        PREFLIGHT.compiled_initialization_preflight(trainer, model, *batch())


def test_update_validator_rejects_frozen_leaf_change():
    model = TinyFlow()
    before = TRAINER.state_inventory(nnx.state(model))
    model.frozen.value = model.frozen.value + 1
    after = TRAINER.state_inventory(nnx.state(model))
    with pytest.raises(ValueError, match="frozen state changed"):
        TRAINER.validate_state_update(
            before, after, {("weight",)}, require_change=False
        )


def test_native_trainer_rejects_nonfinite_post_update_parameter():
    model = TinyFlow()
    trainer = wiring(model)
    optimizer = nnx.Optimizer(model, optax.sgd(float("inf")), wrt=WeightOnly())
    with pytest.raises(FloatingPointError, match="updated parameter"):
        trainer.update(optimizer, *batch())


def test_preserve_frozen_state_keeps_only_allowlisted_new_values():
    before_model = TinyFlow()
    after_model = TinyFlow()
    after_model.weight.value = after_model.weight.value + 1
    after_model.frozen.value = after_model.frozen.value + 2
    restored = TRAINER.preserve_frozen_state(
        nnx.state(before_model), nnx.state(after_model), {("weight",)}
    )
    values = dict(restored.flat_state().items())
    assert float(values[("weight",)].value) == 2.0
    assert float(values[("frozen",)].value) == 2.0


def nested_action_model():
    """Build the real 23-path action partition with two frozen state leaves."""
    tree = {}
    for ordinal, path in enumerate(sorted(EXPECTED_TRAINABLE_PATHS), start=1):
        cursor = tree
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = nnx.Param(jnp.asarray(ordinal / 100, dtype=jnp.float32))

    def convert(value):
        return (
            nnx.Dict(**{key: convert(child) for key, child in value.items()})
            if isinstance(value, dict)
            else value
        )

    model = convert(tree)
    model.frozen_stage = nnx.Param(jnp.asarray(5.0, dtype=jnp.float32))
    model.action_correlation_cholesky = nnx.Intermediate(jnp.eye(2, dtype=jnp.float32))
    return model


def read_path(model, path):
    value = model
    for part in path:
        value = value[part]
    return value


def full_partition_loss(model, rng, observation, actions, *, num_flow_samples):
    total = sum(
        read_path(model, path).value for path in sorted(EXPECTED_TRAINABLE_PATHS)
    )
    noise = jax.random.normal(rng, (num_flow_samples, *observation.shape))
    velocity = total * observation[None] + noise
    return {
        "total_loss": jnp.square(velocity - actions[None]),
        "_flow_velocity": velocity,
    }


def test_real_23_leaf_filter_runs_two_updates_and_selects_parent_sharding():
    model = nested_action_model()
    trainable = ExactPathSet(EXPECTED_TRAINABLE_PATHS)
    trainer = TRAINER.NativeTrainer.initialize(
        SimpleNamespace(params=nnx.state(model)),
        trainable,
        full_partition_loss,
        num_flow_samples=2,
        allowed_paths=EXPECTED_TRAINABLE_PATHS,
    )
    selected_sharding = trainer.parent_input_sharding(
        SimpleNamespace(params=nnx.state(model))
    )
    assert set(selected_sharding.flat_state()) == EXPECTED_TRAINABLE_PATHS

    optimizer = nnx.Optimizer(model, optax.sgd(1e-4), wrt=trainable)
    frozen = np.asarray(model.frozen_stage.value).copy()
    correlation = np.asarray(model.action_correlation_cholesky.value).copy()
    for seed in (11, 12):
        _, _, changed = trainer.update(
            optimizer,
            jax.random.key(seed),
            jnp.ones((1, 1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
        )
        assert set(changed) == EXPECTED_TRAINABLE_PATHS
        assert np.array_equal(np.asarray(model.frozen_stage.value), frozen)
        assert np.array_equal(
            np.asarray(model.action_correlation_cholesky.value), correlation
        )

    before_ema = nnx.state(nested_action_model())
    after_ema_model = nested_action_model()
    for path in EXPECTED_TRAINABLE_PATHS:
        variable = read_path(after_ema_model, path)
        variable.value = variable.value + 1
    after_ema_model.frozen_stage.value += 3
    restored = TRAINER.preserve_frozen_state(
        before_ema, nnx.state(after_ema_model), EXPECTED_TRAINABLE_PATHS
    )
    restored_values = dict(restored.flat_state().items())
    assert float(restored_values[("frozen_stage",)].value) == 5.0
    for path in EXPECTED_TRAINABLE_PATHS:
        assert float(restored_values[path].value) > float(
            dict(before_ema.flat_state().items())[path].value
        )
