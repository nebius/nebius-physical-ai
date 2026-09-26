"""Test the publication-safe frozen-parent flow objective."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-anchored-training"
)
ACTION_PARTITION = (
    IMPLEMENTATION.parent / "behavior-matched-training/action_partition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "behavior_anchored_objective", IMPLEMENTATION / "anchored_objective.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_zero_at_initialization_and_positive_after_drift() -> None:
    rng = np.random.default_rng(7)
    parent = rng.normal(size=(4, 3, 5, 23)).astype(np.float32)
    native_loss = np.full((3, 5, 23), 2.0, dtype=np.float32)
    losses = {"total_loss": native_loss, "_flow_velocity": parent.copy()}
    initial, metrics = MODULE.compute_anchored_objective(
        losses,
        parent,
        array_module=np,
        stop_gradient=lambda value: value.copy(),
    )
    assert float(initial.demonstration_loss) == 2.0
    assert float(initial.anchor_loss) == 0.0
    assert float(initial.total_loss) == 2.0
    assert "_flow_velocity" not in metrics
    drifted = dict(losses, _flow_velocity=parent + 0.25)
    changed, _ = MODULE.compute_anchored_objective(
        drifted,
        parent,
        array_module=np,
        stop_gradient=lambda value: value.copy(),
    )
    assert np.isclose(float(changed.anchor_loss), 0.0625)
    assert float(changed.total_loss) > float(changed.demonstration_loss)


def test_objective_rejects_mismatched_stochastic_tensor_shapes() -> None:
    losses = {
        "total_loss": np.ones((2,), dtype=np.float32),
        "_flow_velocity": np.ones((4, 2, 3, 23), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="share"):
        MODULE.compute_anchored_objective(
            losses,
            np.ones((1, 2, 3, 23), dtype=np.float32),
            array_module=np,
            stop_gradient=lambda value: value,
        )


def test_parent_is_stopped_in_real_jax_autodiff() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def loss(student, parent):
        objective, _ = MODULE.compute_anchored_objective(
            {"total_loss": jnp.ones((1, 3, 4)), "_flow_velocity": student},
            parent,
            array_module=jnp,
            stop_gradient=jax.lax.stop_gradient,
        )
        return objective.total_loss

    student = jnp.ones((2, 1, 3, 4))
    parent = jnp.zeros_like(student)
    student_gradient, parent_gradient = jax.grad(loss, argnums=(0, 1))(student, parent)
    assert np.any(np.asarray(student_gradient) != 0)
    assert np.array_equal(np.asarray(parent_gradient), np.zeros(parent.shape))


@pytest.mark.parametrize("anchor_weight", [-1.0, np.nan, np.inf, -np.inf])
def test_objective_rejects_invalid_anchor_weight(anchor_weight: float) -> None:
    velocity = np.ones((1, 1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MODULE.compute_anchored_objective(
            {"total_loss": velocity[0], "_flow_velocity": velocity},
            velocity,
            array_module=np,
            stop_gradient=lambda value: value,
            anchor_weight=anchor_weight,
        )


def test_frozen_recipe_keeps_curriculum_change_disabled() -> None:
    recipe = json.loads((IMPLEMENTATION / "recipe.json").read_text())
    assert recipe["anchor_weight"] == 1.0
    assert recipe["late_stage_oversampling"] is False
    assert recipe["candidate_checkpoint_steps"] == [400, 800, 1200, 1600, 2000, 2399]
    assert recipe["trainable_partition"]["leaves"] == 23
    assert (
        recipe["trainable_partition"]["source_sha256"]
        == hashlib.sha256(ACTION_PARTITION.read_bytes()).hexdigest()
    )
    assert recipe["gpu_results"] == "pending"
