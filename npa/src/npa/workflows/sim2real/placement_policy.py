"""Deterministic last-mile controller for a learned Isaac placement actor.

The PPO actor remains responsible for the complete reach, grasp, lift, and
transport trajectory.  Once it brings a genuinely lifted, still-grasped object
inside a tighter 4 cm goal basin, this controller latches the actor's current
joint target.  Holding that target lets the position-controlled arm and object
settle instead of allowing a later actor update to drive through the goal.

The latch is deliberately stricter than the authoritative 5 cm distance
threshold and does not declare success.  Validation and gold still require the
unchanged object speed and consecutive-step conditions at episode end.
"""

from __future__ import annotations

from typing import Any


SETTLE_HOLD_TRIGGER_DISTANCE_M = 0.04
SETTLE_HOLD_MINIMAL_LIFT_M = 0.04


def settle_hold_trigger(
    goal_distance: Any,
    lift_height: Any,
    contact: Any,
    gripper_closed: Any,
    *,
    trigger_distance_m: float = SETTLE_HOLD_TRIGGER_DISTANCE_M,
    minimal_lift_m: float = SETTLE_HOLD_MINIMAL_LIFT_M,
) -> Any:
    """Return the scalar or vector mask that enters the settle-hold phase.

    NumPy and Torch arrays both implement the comparison and bitwise operations
    used here, keeping the exact live predicate directly testable on CPU.
    """

    if trigger_distance_m <= 0:
        raise ValueError("settle-hold trigger distance must be positive")
    if minimal_lift_m <= 0:
        raise ValueError("settle-hold minimal lift must be positive")
    return (
        (goal_distance < float(trigger_distance_m))
        & (lift_height >= float(minimal_lift_m))
        & contact
        & gripper_closed
    )


def advance_settle_hold(latched: Any, trigger: Any) -> tuple[Any, Any]:
    """Latch new eligible environments and return ``(latched, newly_latched)``."""

    newly_latched = (~latched) & trigger
    return latched | newly_latched, newly_latched
