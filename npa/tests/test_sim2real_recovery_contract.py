from __future__ import annotations

import pytest

from npa.workflows.sim2real.recovery_contract import (
    RecoveryAction,
    RecoveryKind,
    RecoveryRequest,
    RecoverySample,
    RecoveryThresholds,
    classify_recovery,
    validate_recovery_action,
)


def _thresholds() -> RecoveryThresholds:
    return RecoveryThresholds(
        window_decisions=3,
        commanded_motion_min=0.3,
        measured_motion_min=0.03,
        task_progress_min_m=0.01,
        target_pixels_min=16,
        finger_pixels_min=16,
    )


def _sample(**changes: object) -> RecoverySample:
    values = {
        "commanded_motion": 0.2,
        "measured_motion": 0.04,
        "task_progress_m": 0.02,
        "workspace_target_pixels": 32,
        "wrist_target_pixels": 32,
        "wrist_left_finger_pixels": 24,
        "wrist_right_finger_pixels": 24,
    }
    values.update(changes)
    return RecoverySample(**values)


def test_recovery_classifier_separates_blockage_from_task_stagnation() -> None:
    blocked = [_sample(measured_motion=0.001) for _ in range(3)]
    moving = [_sample(task_progress_m=0.001) for _ in range(3)]

    assert (
        classify_recovery(blocked, _thresholds()).kind == RecoveryKind.COMMAND_BLOCKED
    )
    assert (
        classify_recovery(moving, _thresholds()).kind
        == RecoveryKind.MOVING_WITHOUT_TASK_PROGRESS
    )


def test_recovery_classifier_requires_both_policy_views_and_fingers() -> None:
    occluded = [_sample(wrist_right_finger_pixels=0) for _ in range(3)]

    result = classify_recovery(occluded, _thresholds())

    assert result.kind == RecoveryKind.POOR_VISIBILITY
    assert result.evidence == ("workspace_wrist_geometry",)


def test_recovery_classifier_treats_edge_clipping_as_poor_visibility() -> None:
    clipped = [_sample(wrist_target_clipped=True) for _ in range(3)]

    assert (
        classify_recovery(clipped, _thresholds()).kind == RecoveryKind.POOR_VISIBILITY
    )


def _request(*, grasp_active: bool = True) -> RecoveryRequest:
    return RecoveryRequest(
        classification=classify_recovery(
            [_sample(wrist_target_pixels=0) for _ in range(3)], _thresholds()
        ),
        robot_identity="robot-profile-sha256",
        joint_names=("joint_a", "joint_b"),
        joint_positions=(0.0, 0.0),
        joint_limits=((-1.0, 1.0), (-2.0, 2.0)),
        camera_roles=("workspace", "wrist"),
        camera_extrinsics_sha256="a" * 64,
        wrist_camera_world_pose=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
        collision_model_identity="robot-urdf-sha256",
        grasp_active=grasp_active,
    )


def test_recovery_action_is_separately_owned_and_constraint_checked() -> None:
    action = RecoveryAction(
        owner="recovery_planner",
        representation="absolute_joint_position",
        joint_names=("joint_a", "joint_b"),
        values=(0.1, -0.2),
        units="rad",
        collision_checked=True,
        preserves_grasp=True,
    )

    validate_recovery_action(_request(), action)


@pytest.mark.parametrize("failure", ["owner", "limit", "grasp"])
def test_recovery_action_rejects_hidden_or_unsafe_fallbacks(failure: str) -> None:
    values = {
        "owner": "recovery_planner",
        "representation": "absolute_joint_position",
        "joint_names": ("joint_a", "joint_b"),
        "values": (0.1, -0.2),
        "units": "rad",
        "collision_checked": True,
        "preserves_grasp": True,
    }
    if failure == "owner":
        values["owner"] = "policy"
    elif failure == "limit":
        values["values"] = (3.0, -0.2)
    else:
        values["preserves_grasp"] = False

    with pytest.raises(ValueError):
        validate_recovery_action(_request(), RecoveryAction(**values))
