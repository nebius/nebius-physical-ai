"""Post-success retention for a learned Isaac placement actor.

The PPO actor remains responsible for reach, grasp, lift, transport, braking,
and the complete strict placement event.  Only after the actor independently
keeps the object below 5 cm and 0.03 m/s for three consecutive steps may this
controller hold the measured joint position.  It preserves a learned result at
episode end; it cannot create or declare one.
"""

from __future__ import annotations

from typing import Any


SETTLE_HOLD_TRIGGER_DISTANCE_M = 0.05
SETTLE_HOLD_TRIGGER_SPEED_MPS = 0.03
SETTLE_HOLD_REQUIRED_STEPS = 3
SETTLE_HOLD_MINIMAL_LIFT_M = 0.04


def settle_hold_trigger(
    goal_distance: Any,
    object_speed: Any,
    stable_steps: Any,
    lift_height: Any,
    contact: Any,
    gripper_closed: Any,
    *,
    trigger_distance_m: float = SETTLE_HOLD_TRIGGER_DISTANCE_M,
    trigger_speed_mps: float = SETTLE_HOLD_TRIGGER_SPEED_MPS,
    required_steps: int = SETTLE_HOLD_REQUIRED_STEPS,
    minimal_lift_m: float = SETTLE_HOLD_MINIMAL_LIFT_M,
) -> Any:
    """Return the mask that may retain an already-complete strict placement.

    NumPy and Torch arrays both implement the comparison and bitwise operations
    used here, keeping the exact live predicate directly testable on CPU.
    """

    if trigger_distance_m <= 0:
        raise ValueError("settle-hold trigger distance must be positive")
    if trigger_speed_mps <= 0:
        raise ValueError("settle-hold trigger speed must be positive")
    if required_steps <= 0:
        raise ValueError("settle-hold required steps must be positive")
    if minimal_lift_m <= 0:
        raise ValueError("settle-hold minimal lift must be positive")
    return (
        (goal_distance < float(trigger_distance_m))
        & (object_speed < float(trigger_speed_mps))
        & (stable_steps >= int(required_steps))
        & (lift_height >= float(minimal_lift_m))
        & contact
        & gripper_closed
    )


def advance_settle_hold(latched: Any, trigger: Any) -> tuple[Any, Any]:
    """Latch new eligible environments and return ``(latched, newly_latched)``."""

    newly_latched = (~latched) & trigger
    return latched | newly_latched, newly_latched


def joint_position_hold_action(
    joint_position: Any,
    scale: Any,
    offset: Any,
) -> Any:
    """Invert Isaac's affine joint action into a measured-position hold action.

    ``JointPositionAction`` applies ``offset + scale * raw_action``. Replaying
    the actor's last raw action preserves its old target, not the robot's
    current position, so a moving arm continues through the goal. This inverse
    transform commands the measured joint position at latch time.
    """

    zero_scale = scale == 0
    if hasattr(zero_scale, "any"):
        zero_scale = zero_scale.any()
    if bool(zero_scale):
        raise ValueError("settle-hold joint action scale must be nonzero")
    return (joint_position - offset) / scale
