"""Run student and frozen-parent NNX losses in one mapped model call."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _semantic_type(variable: Any) -> type:
    """Return the NNX variable type stored by a state leaf."""
    return getattr(variable, "type", type(variable))


def _flat_variables(state: Any) -> dict[tuple[Any, ...], Any]:
    """Return state variables indexed by their complete NNX paths."""
    return dict(state.flat_state().items())


def _validate_action_parity(student: Any, parent: Any) -> None:
    """Reject structural or numerical metadata drift before compilation."""
    student_variables = _flat_variables(student)
    parent_variables = _flat_variables(parent)
    if tuple(student_variables) != tuple(parent_variables):
        raise ValueError("student and parent action paths differ")
    for path, student_variable in student_variables.items():
        parent_variable = parent_variables[path]
        if _semantic_type(student_variable) is not _semantic_type(parent_variable):
            raise ValueError(f"student and parent variable types differ at {path}")
        if student_variable.value.shape != parent_variable.value.shape:
            raise ValueError(f"student and parent shapes differ at {path}")
        if student_variable.value.dtype != parent_variable.value.dtype:
            raise ValueError(f"student and parent dtypes differ at {path}")


def _validate_correlation(state: Any) -> None:
    """Require one FP32 Intermediate correlation leaf in initialized state."""
    from flax import nnx

    rows = []
    for path, variable in state.flat_state().items():
        if tuple(str(part) for part in path)[-1:] == ("action_correlation_cholesky",):
            rows.append(variable)
    if len(rows) != 1:
        raise ValueError("initialized state must contain one action correlation leaf")
    variable = rows[0]
    if _semantic_type(variable) is not nnx.Intermediate:
        raise ValueError("action correlation must remain an NNX Intermediate")
    if str(variable.value.dtype) != "float32":
        raise ValueError("action correlation must remain FP32")


def capture_parent_action_state(train_state: Any, trainable_filter: Any) -> Any:
    """Clone the initialized trainable action state for a frozen parent.

    Args:
        train_state: Native state whose ``params`` include installed correlation.
        trainable_filter: NNX filter selecting the action parameters.

    Returns:
        An independently stored NNX state containing the selected variables.

    Raises:
        ValueError: The initialized state lacks the typed FP32 correlation.
    """
    import jax
    import numpy as np

    _validate_correlation(train_state.params)
    selected = train_state.params.filter(trainable_filter)
    return selected.map(
        lambda _path, variable: variable.replace(
            value=jax.device_put(np.asarray(jax.device_get(variable.value)).copy())
        )
    )


def _stack_action_states(student: Any, parent: Any) -> Any:
    """Stack matching action leaves on the leading model axis."""
    import jax
    import jax.numpy as jnp

    _validate_action_parity(student, parent)
    parent_variables = _flat_variables(parent)
    return student.map(
        lambda path, variable: variable.replace(
            value=jnp.stack(
                (variable.value, jax.lax.stop_gradient(parent_variables[path].value)),
                axis=0,
            )
        )
    )


def _mapped_losses(
    paired_model: Any,
    state_axes: Any,
    rng: Any,
    observation: Any,
    actions: Any,
    loss_method: Callable[..., dict[str, Any]],
    num_flow_samples: int,
) -> dict[str, Any]:
    """Map one detailed-loss call over the student and parent model axis."""
    from flax import nnx

    @nnx.vmap(in_axes=(state_axes, None, None, None), out_axes=0)
    def paired_forward(selected, selected_rng, selected_obs, selected_actions):
        return loss_method(
            selected,
            selected_rng,
            selected_obs,
            selected_actions,
            num_flow_samples=num_flow_samples,
        )

    return paired_forward(paired_model, rng, observation, actions)


def paired_detailed_losses(
    model: Any,
    parent_action_state: Any,
    trainable_filter: Any,
    rng: Any,
    observation: Any,
    actions: Any,
    *,
    loss_method: Callable[..., dict[str, Any]],
    num_flow_samples: int,
) -> dict[str, Any]:
    """Compute student and parent losses with matched stochastic inputs.
    Args:
        model and parent_action_state: Student model and frozen action state.
        trainable_filter and rng: Action selector and shared random key.
        observation, actions, loss_method, num_flow_samples: Native call inputs.
    Returns: Detailed losses with leading axis ``[student, parent]``.
    Raises: ValueError for invalid state metadata or sample count.
    """
    from flax import nnx

    if num_flow_samples <= 0:
        raise ValueError("num_flow_samples must be positive")
    student_action = nnx.state(model).filter(trainable_filter)
    paired_action = _stack_action_states(student_action, parent_action_state)
    shared = nnx.state(model).filter(nnx.Not(trainable_filter))
    paired_model = nnx.merge(
        nnx.graphdef(model), nnx.State.merge(shared, paired_action)
    )
    state_axes = nnx.StateAxes({trainable_filter: 0, ...: None})
    return _mapped_losses(
        paired_model,
        state_axes,
        rng,
        observation,
        actions,
        loss_method,
        num_flow_samples,
    )


def native_anchored_loss(
    paired_losses: dict[str, Any],
    *,
    array_module: Any,
    stop_gradient: Callable[[Any], Any],
    anchor_weight: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Reduce paired native losses with the public anchored objective.

    Args:
        paired_losses: Detailed losses with student and parent on axis zero.
        array_module: NumPy-compatible operations used by the objective.
        stop_gradient: Backend gradient barrier for the parent velocity.
        anchor_weight: Finite nonnegative anchor multiplier.

    Returns:
        The public anchored objective and scalar-compatible metrics.

    Raises:
        ValueError: Paired outputs lack exactly two model results.
    """
    from anchored_objective import compute_anchored_objective

    velocity = paired_losses.get("_flow_velocity")
    if velocity is None or velocity.shape[0] != 2:
        raise ValueError("paired flow velocity must contain student and parent")
    student_losses = {key: value[0] for key, value in paired_losses.items()}
    parent_velocity = stop_gradient(velocity[1])
    return compute_anchored_objective(
        student_losses,
        parent_velocity,
        array_module=array_module,
        stop_gradient=stop_gradient,
        anchor_weight=anchor_weight,
    )
