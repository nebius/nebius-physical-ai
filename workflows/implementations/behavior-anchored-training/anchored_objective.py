"""Compute the frozen-parent flow anchor prepared for the BEHAVIOR training ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


class ArrayModule(Protocol):
    """Describe the NumPy-compatible operations required by the objective.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    def mean(self, value: Any, axis: Any = None) -> Any:
        """Return an arithmetic mean.

        Args:
            value: Array to reduce.
            axis: Optional axes to reduce.

        Returns:
            The reduced array.

        Raises:
            None.
        """

    def square(self, value: Any) -> Any:
        """Square an array elementwise.

        Args:
            value: Array to square.

        Returns:
            The squared array.

        Raises:
            None.
        """


@dataclass(frozen=True)
class AnchoredObjective:
    """Hold the three loss values produced by frozen-parent anchoring.

    Args:
        demonstration_loss: Mean native demonstration flow loss.
        anchor_loss: Mean student-to-parent flow-velocity error.
        total_loss: Weighted sum used for differentiation.

    Returns:
        None.

    Raises:
        None.
    """

    demonstration_loss: Any
    anchor_loss: Any
    total_loss: Any


def _validate_inputs(
    detailed_losses: dict[str, Any], parent_velocity: Any, anchor_weight: float
) -> Any:
    """Return the student velocity after checking the objective contract."""
    if not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError("anchor_weight must be finite and nonnegative")
    required = {"total_loss", "_flow_velocity"}
    if not required.issubset(detailed_losses):
        raise ValueError("native detailed losses lack the flow objective tensors")
    student_velocity = detailed_losses["_flow_velocity"]
    if student_velocity.shape != parent_velocity.shape or student_velocity.ndim != 4:
        raise ValueError("student and parent velocities must share [N,B,H,D] shape")
    return student_velocity


def _loss_metrics(
    detailed_losses: dict[str, Any], demonstration_loss: Any, anchor: Any, total: Any
) -> dict[str, Any]:
    """Remove the private velocity tensor and add logger-compatible losses."""
    metrics = {
        key: value for key, value in detailed_losses.items() if key != "_flow_velocity"
    }
    metrics.update(
        demonstration_loss=demonstration_loss,
        anchor_loss=anchor,
        anchored_total_loss=total,
    )
    return metrics


def compute_anchored_objective(
    detailed_losses: dict[str, Any],
    parent_velocity: Any,
    *,
    array_module: ArrayModule,
    stop_gradient: Any,
    anchor_weight: float = 1.0,
) -> tuple[AnchoredObjective, dict[str, Any]]:
    """Add stopped-parent flow anchoring to the demonstration loss.

    Args:
        detailed_losses: Native losses plus the student's flow velocity.
        parent_velocity: Parent flow velocity from identical stochastic inputs.
        array_module: NumPy-compatible mean and square operations.
        stop_gradient: Backend gradient barrier applied to the parent output.
        anchor_weight: Finite nonnegative multiplier for the anchor loss.

    Returns:
        The differentiable objective and logger-compatible metrics.

    Raises:
        ValueError: If the weight or flow-velocity tensors violate the contract.
    """
    student_velocity = _validate_inputs(detailed_losses, parent_velocity, anchor_weight)
    per_example_anchor = array_module.mean(
        array_module.square(student_velocity - stop_gradient(parent_velocity)),
        axis=(0, 2, 3),
    )
    demonstration_loss = array_module.mean(detailed_losses["total_loss"])
    anchor_loss = array_module.mean(per_example_anchor)
    total_loss = demonstration_loss + anchor_weight * anchor_loss
    objective = AnchoredObjective(demonstration_loss, anchor_loss, total_loss)
    metrics = _loss_metrics(
        detailed_losses, demonstration_loss, per_example_anchor, total_loss
    )
    return objective, metrics
