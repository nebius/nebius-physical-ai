"""Run compiled initialization gates for the public anchored trainer."""

from __future__ import annotations

import math
from typing import Any


def _gradient_norm(gradients: Any) -> float:
    import jax
    import jax.numpy as jnp
    import numpy as np

    leaves = jax.tree.leaves(gradients.to_pure_dict())
    squared = sum(jnp.sum(jnp.square(value)) for value in leaves)
    return float(np.asarray(jax.device_get(jnp.sqrt(squared))))


def _separate_topology_anchor(
    trainer: Any, model: Any, rng: Any, observation: Any, actions: Any
) -> float:
    """Measure the separate-call topology as a diagnostic only."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    shared = nnx.state(model).filter(nnx.Not(trainer.trainable_filter))
    parent_model = nnx.merge(
        nnx.graphdef(model), nnx.State.merge(shared, trainer.parent_action_state)
    )
    student = trainer.detailed_loss(
        model,
        rng,
        observation,
        actions,
        num_flow_samples=trainer.num_flow_samples,
    )
    parent = trainer.detailed_loss(
        parent_model,
        rng,
        observation,
        actions,
        num_flow_samples=trainer.num_flow_samples,
    )
    if "_flow_velocity" not in student or "_flow_velocity" not in parent:
        raise ValueError("native detailed loss lacks flow velocity")
    value = jnp.mean(jnp.square(student["_flow_velocity"] - parent["_flow_velocity"]))
    return float(np.asarray(jax.device_get(value)))


def _student_preflight(trainer, model, rng, observation, actions):
    import jax
    import numpy as np
    from flax import nnx

    def total_loss(selected):
        return trainer.loss_with_parent(
            selected, trainer.parent_action_state, rng, observation, actions
        )

    student_diff = nnx.DiffState(0, trainer.trainable_filter)
    (loss, metrics), gradients = nnx.jit(
        nnx.value_and_grad(total_loss, argnums=student_diff, has_aux=True)
    )(model)
    anchor = np.asarray(jax.device_get(metrics["anchor_loss"]))
    if not np.array_equal(anchor, np.zeros_like(anchor)):
        raise ValueError("paired initialization anchor loss must be bit zero")
    norm = _gradient_norm(gradients)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("student demonstration gradient must be finite and nonzero")
    return loss, norm


def _anchor_gradient_preflight(trainer, model, rng, observation, actions):
    import jax
    import numpy as np
    from flax import nnx

    def anchor_only(selected, parent):
        _, metrics = trainer.loss_with_parent(
            selected, parent, rng, observation, actions
        )
        return metrics["anchor_loss"].mean()

    student_diff = nnx.DiffState(0, trainer.trainable_filter)
    parent_diff = nnx.DiffState(1, trainer.trainable_filter)
    scalar, gradients = nnx.jit(
        nnx.value_and_grad(anchor_only, argnums=(student_diff, parent_diff))
    )(model, trainer.parent_action_state)
    if float(np.asarray(jax.device_get(scalar))) != 0.0:
        raise ValueError("paired scalar anchor loss must be exactly zero")
    student_norm, parent_norm = map(_gradient_norm, gradients)
    if student_norm != 0.0 or parent_norm != 0.0:
        raise ValueError("initial anchor gradients must be exactly zero")
    return student_norm, parent_norm


def _preflight_receipt(loss, demonstration_norm, student_norm, parent_norm, separate):
    import numpy as np

    return {
        "schema": "npa.behavior.anchored-native-preflight.v1",
        "status": "paired_compiled_initialization_passed",
        "paired_anchor_loss_exact_zero": True,
        "student_demonstration_gradient_norm": demonstration_norm,
        "student_anchor_gradient_norm": student_norm,
        "parent_gradient_norm": parent_norm,
        "parent_state_byte_equal": True,
        "separate_topology_anchor_loss_diagnostic": separate,
        "separate_topology_is_gate": False,
        "total_loss": float(np.asarray(loss)),
    }


def compiled_initialization_preflight(
    trainer: Any, model: Any, rng: Any, observation: Any, actions: Any
) -> dict[str, Any]:
    """Prove the paired compiled initialization and gradient contract.

    Args:
        trainer: Initialized ``NativeTrainer``.
        model: Student model at the captured initialization state.
        rng, observation, actions: One real native batch and shared random key.

    Returns:
        Exact paired, gradient, parent-stability, and diagnostic measurements.

    Raises:
        ValueError: Paired loss is not bit zero or gradients violate the contract.
        FloatingPointError: A diagnostic or gradient is non-finite.
    """
    from native_trainer import state_inventory

    parent_before = state_inventory(trainer.parent_action_state)
    loss, student_gradient_norm = _student_preflight(
        trainer, model, rng, observation, actions
    )
    student_anchor_norm, parent_gradient_norm = _anchor_gradient_preflight(
        trainer, model, rng, observation, actions
    )
    if state_inventory(trainer.parent_action_state) != parent_before:
        raise ValueError("compiled preflight changed frozen parent state")
    separate = _separate_topology_anchor(trainer, model, rng, observation, actions)
    if not math.isfinite(separate):
        raise FloatingPointError("separate-topology diagnostic is non-finite")
    return _preflight_receipt(
        loss,
        student_gradient_norm,
        student_anchor_norm,
        parent_gradient_norm,
        separate,
    )
