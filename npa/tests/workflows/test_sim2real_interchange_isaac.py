from __future__ import annotations

import math

import pytest

from npa.workflows.sim2real.interchange_contract import (
    CameraView,
    GripperMapping,
    InterchangeContractError,
    RobotSensorProfile,
    Transform,
)
from npa.workflows.sim2real.interchange_isaac import (
    IsaacActionLayout,
    assert_live_robot_matches_profile,
    camera_world_pose,
    render_absolute_joint_action,
)
from npa.workflows.sim2real.interchange_isaac_runtime import _update_events


def _profile(dof: int = 6) -> RobotSensorProfile:
    arm = tuple(f"axis_{index}" for index in range(dof))
    return RobotSensorProfile(
        identity=f"sealed-arm-{dof}",
        bundle_sha256="a" * 64,
        source_sha256="b" * 64,
        base_link="base_link",
        ee_link="tool0",
        arm_joint_names=arm,
        measured_state_names=arm + ("gripper",),
        measured_state_units=("rad",) * dof + ("normalized_closed",),
        joint_limits_rad=tuple((-2.0, 2.0) for _ in arm),
        home_positions_rad=tuple(0.0 for _ in arm),
        controller_representation="absolute_joint_position",
        gripper=GripperMapping(
            joint_names=("finger_driver",),
            joint_units=("rad",),
            finger_links=("left_pad", "right_pad"),
            open_joint_positions=(0.0,),
            closed_joint_positions=(0.8,),
            maximum_width_m=0.085,
            model_semantics="normalized_closed",
            model_open=0.0,
            model_closed=1.0,
        ),
        cameras=(
            _camera("scene", "exterior", "world"),
            _camera("tool", "wrist", "tool0"),
        ),
    )


def _camera(key: str, role: str, mount: str) -> CameraView:
    return CameraView(
        key=key,
        role=role,  # type: ignore[arg-type]
        mount_link=mount,
        mount_from_camera=Transform(
            (0.1, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=(190.0, 190.0, 112.0, 112.0),
        image_size=(224, 224),
        layout="HWC",
        mask_key=f"{key}_valid",
        capture_timestamp_key=f"{key}_capture_s",
        state_timestamp_key="state_timestamp_s",
        maximum_state_skew_s=0.02,
    )


def _layout(profile: RobotSensorProfile) -> IsaacActionLayout:
    return IsaacActionLayout(
        total_dimension=len(profile.arm_joint_names) + 1,
        arm_start=0,
        gripper_start=len(profile.arm_joint_names),
        arm_joint_names=profile.arm_joint_names,
        gripper_joint_names=profile.gripper.joint_names,
        default_arm_positions_rad=tuple(0.25 for _ in profile.arm_joint_names),
        arm_scales_rad=tuple(0.5 for _ in profile.arm_joint_names),
    )


def test_six_dof_absolute_targets_render_without_seven_dof_padding() -> None:
    profile = _profile(6)
    action, receipt = render_absolute_joint_action(
        [0.5, 0.25, 0.0, -0.25, -0.5, 0.75, 0.9],
        profile=profile,
        layout=_layout(profile),
    )
    assert action == pytest.approx((0.5, 0.0, -0.5, -1.0, -1.5, 1.0, -1.0))
    assert len(action) == 7
    assert receipt["binary_gripper_command"] == "close"
    assert receipt["physical_gripper_joint_targets"] == {"finger_driver": 0.8}


def test_seven_dof_profile_uses_its_own_exact_dimension() -> None:
    profile = _profile(7)
    action, receipt = render_absolute_joint_action(
        [0.25] * 7 + [0.1], profile=profile, layout=_layout(profile)
    )
    assert action == pytest.approx((0.0,) * 7 + (1.0,))
    assert receipt["binary_gripper_command"] == "open"


def test_reordered_live_action_names_fail_before_control() -> None:
    profile = _profile()
    layout = _layout(profile)
    reordered = IsaacActionLayout(
        **{
            **layout.__dict__,
            "arm_joint_names": (
                layout.arm_joint_names[1],
                layout.arm_joint_names[0],
                *layout.arm_joint_names[2:],
            ),
        }
    )
    with pytest.raises(InterchangeContractError, match="action order"):
        assert_live_robot_matches_profile(
            robot_joint_names=(
                *profile.arm_joint_names,
                *profile.gripper.joint_names,
            ),
            robot_body_names=(
                profile.base_link,
                profile.ee_link,
                *profile.gripper.finger_links,
            ),
            layout=reordered,
            profile=profile,
        )


def test_joint_limit_and_dimension_mismatch_are_rejected() -> None:
    profile = _profile()
    with pytest.raises(InterchangeContractError, match="exactly 7"):
        render_absolute_joint_action(
            [0.0] * 8, profile=profile, layout=_layout(profile)
        )
    with pytest.raises(InterchangeContractError, match="outside physical limits"):
        render_absolute_joint_action(
            [3.0] + [0.0] * 5 + [0.0],
            profile=profile,
            layout=_layout(profile),
        )


def test_declared_wrist_mount_composes_with_different_ee_pose() -> None:
    half = math.sqrt(0.5)
    pose = camera_world_pose(
        link_position_m=(1.0, 2.0, 3.0),
        link_quaternion_xyzw=(0.0, 0.0, half, half),
        mount_from_camera=Transform(
            (0.1, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    assert pose.translation_m == pytest.approx((1.0, 2.1, 3.0))
    assert pose.quaternion_xyzw == pytest.approx((0.0, 0.0, half, half))


def test_surface_success_requires_ordered_physics_and_three_stable_steps() -> None:
    context = {
        "ordered_event_steps": {},
        "required_events": (
            "contact",
            "grasp",
            "lift",
            "transport",
            "release",
            "support_contact",
            "stable",
        ),
        "retreat_distance_m": 0.1,
        "stable_steps": 0,
    }
    sample = {
        "bilateral_finger_contact": True,
        "gripper_closed": True,
        "lifted": True,
        "target_xy_distance_m": 0.04,
        "hand_object_distance_m": 0.02,
        "object_support_force_n": 0.0,
        "object_speed_m_s": 0.01,
    }
    _update_events(context, sample, 4)
    assert list(context["ordered_event_steps"]) == [
        "contact",
        "grasp",
        "lift",
        "transport",
    ]
    sample.update(
        bilateral_finger_contact=False,
        gripper_closed=False,
        hand_object_distance_m=0.11,
        object_support_force_n=1.0,
    )
    for step in range(5, 9):
        _update_events(context, sample, step)
    assert list(context["ordered_event_steps"]) == list(context["required_events"])
    assert context["ordered_event_steps"]["stable"] == 8
