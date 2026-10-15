from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from npa.workflows.sim2real.interchange_contract import (
    CameraView,
    GripperMapping,
    InterchangeContractError,
    RigidObjectProfile,
    RobotSensorProfile,
    SurfacePlacementTask,
    Transform,
)
from npa.workflows.sim2real.interchange_isaac import (
    IsaacActionLayout,
    assert_live_robot_matches_profile,
    camera_world_pose,
    render_absolute_joint_action,
    surface_teacher_waypoints,
)
from npa.workflows.sim2real.interchange_isaac_runtime import (
    _axis_angle_error,
    _update_events,
    _visual_evidence_decisions,
    _write_teacher_episode,
    _write_png_rgb8,
)
from npa.workflows.sim2real.interchange_teacher_runtime import _scenario_offset
from npa.workflows.franka_rl_environment import policy_interchange_timing


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


def test_teacher_uses_offset_object_reference_and_grasp_frames() -> None:
    prop = RigidObjectProfile(
        identity="offset-spool",
        source_uri="s3://example/assets/spool.usda",
        source_sha256="a" * 64,
        bundle_sha256="b" * 64,
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        world_from_root=Transform((0.48, -0.14, 0.0), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.012, -0.006, 0.0), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((0.012, -0.006, 0.026), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.054, 0.054, 0.052),
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=0.12,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
        grasp_axis="y",
    )
    task = SurfacePlacementTask(
        identity="surface-placement-v1",
        target_world_xyz_m=(0.55, 0.15, 0.0),
        approach_clearance_m=0.08,
        lift_delta_m=0.08,
        retreat_distance_m=0.1,
    )

    waypoints = surface_teacher_waypoints(prop, task)

    assert [item.phase for item in waypoints] == [
        "approach",
        "contact",
        "grasp",
        "lift",
        "transport",
        "lower",
        "release",
        "retreat",
    ]
    assert waypoints[1].world_from_ee.translation_m == pytest.approx(
        (0.492, -0.146, 0.026)
    )
    # The destination root compensates for the non-centered reference; the
    # grasp/reference relative offset leaves the tool 26 mm above the target.
    assert waypoints[5].world_from_ee.translation_m == pytest.approx(
        (0.55, 0.15, 0.026)
    )
    assert waypoints[-1].world_from_ee.translation_m == pytest.approx(
        (0.55, 0.15, 0.126)
    )
    assert waypoints[2].gripper_closed is True
    assert waypoints[6].gripper_closed is False


def test_successful_teacher_episode_preserves_dynamic_six_axis_policy_state(
    tmp_path: Path,
) -> None:
    robot = _profile(6)
    prop = RigidObjectProfile(
        identity="offset-spool",
        source_uri="s3://example/assets/spool.usda",
        source_sha256="a" * 64,
        bundle_sha256="b" * 64,
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        world_from_root=Transform((0.48, -0.14, 0.0), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.012, -0.006, 0.0), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((0.012, -0.006, 0.026), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.054, 0.054, 0.052),
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=0.12,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
        grasp_axis="y",
    )
    task = SurfacePlacementTask(
        identity="surface-placement-v1",
        target_world_xyz_m=(0.55, 0.15, 0.0),
        approach_clearance_m=0.08,
        lift_delta_m=0.08,
        retreat_distance_m=0.1,
    )
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3).tobytes()
    frames = [
        {
            "scene": (pixels, 2, 2, "exterior"),
            "tool": (pixels[::-1], 2, 2, "wrist"),
        }
        for _ in range(2)
    ]
    states = [[0.0] * 7, [0.1] * 7]
    actions = [[0.1] * 7, [0.2] * 7]
    teacher = {
        "task_success": True,
        "ordered_event_steps": {
            name: index for index, name in enumerate(task.success_events)
        },
    }

    metadata = _write_teacher_episode(
        directory=tmp_path / "episode_0000",
        robot=robot,
        task=task,
        prop=prop,
        states=states,
        actions=actions,
        camera_frames=frames,
        timestamps=[0.0, 1.0 / 15.0],
        teacher=teacher,
    )

    episode = tmp_path / "episode_0000"
    assert np.load(episode / "state.npy").shape == (2, 7)
    assert np.load(episode / "actions.npy").shape == (2, 7)
    assert np.load(episode / "obs_workspace.npy").shape == (2, 2, 2, 3)
    assert np.load(episode / "obs_wrist.npy").shape == (2, 2, 2, 3)
    assert metadata["action_names"] == [*robot.arm_joint_names, "gripper"]
    assert metadata["privileged_object_state_policy_input"] is False


def test_teacher_scenario_layouts_are_deterministic_and_split_disjoint() -> None:
    train = [_scenario_offset("train", index) for index in range(32)]
    validation = [_scenario_offset("validation", index) for index in range(32)]
    gold = [_scenario_offset("gold", index) for index in range(32)]
    assert train == [_scenario_offset("train", index) for index in range(32)]
    assert max(value[0] for value in train) < min(value[0] for value in validation)
    assert max(value[0] for value in validation) < min(value[0] for value in gold)
    assert len({*train, *validation, *gold}) == 96


def test_failed_teacher_episode_cannot_enter_training(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="only successful"):
        _write_teacher_episode(
            directory=tmp_path / "episode_0000",
            robot=_profile(6),
            task=type("Task", (), {"identity": "task"})(),
            prop=type("Prop", (), {"identity": "prop"})(),
            states=[],
            actions=[],
            camera_frames=[],
            timestamps=[],
            teacher={"task_success": False},
        )


def test_teacher_orientation_error_is_sign_invariant() -> None:
    assert _axis_angle_error(
        (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)
    ) == pytest.approx((0.0, 0.0, 0.0))
    assert _axis_angle_error(
        (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)
    ) == pytest.approx((math.pi, 0.0, 0.0))


@pytest.mark.parametrize(
    ("semantics", "model_open", "model_closed", "open_value", "closed_value"),
    [
        ("normalized_open", 1.0, 0.0, 0.9, 0.1),
        ("normalized_closed", 0.0, 1.0, 0.1, 0.9),
        ("binary_open", 1.0, 0.0, 1.0, 0.0),
        ("width_m", 0.085, 0.0, 0.08, 0.005),
    ],
)
def test_gripper_closed_classification_uses_declared_endpoints(
    semantics: str,
    model_open: float,
    model_closed: float,
    open_value: float,
    closed_value: float,
) -> None:
    mapping = GripperMapping(
        joint_names=("finger_joint",),
        joint_units=("rad",),
        finger_links=("left_finger", "right_finger"),
        open_joint_positions=(0.0,),
        closed_joint_positions=(0.8,),
        maximum_width_m=0.085,
        model_semantics=semantics,
        model_open=model_open,
        model_closed=model_closed,
    )

    assert mapping.model_value_is_closed(open_value) is False
    assert mapping.model_value_is_closed(closed_value) is True


def test_policy_interchange_cadence_is_explicit_and_exact() -> None:
    assert policy_interchange_timing(control_hz=15.0) == {
        "simulation_hz": 60.0,
        "simulation_dt_s": 1.0 / 60.0,
        "control_hz": 15.0,
        "decimation": 4,
        "episode_length_s": 240.0,
    }
    with pytest.raises(ValueError, match="must divide"):
        policy_interchange_timing(control_hz=14.0)


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


def test_native_visual_evidence_preserves_exact_rgb_bytes(tmp_path: Path) -> None:
    raw = bytes((255, 0, 0, 0, 255, 0))
    path = tmp_path / "frame.png"
    receipt = _write_png_rgb8(path, raw, 2, 1)
    encoded = path.read_bytes()
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    length = struct.unpack(">I", encoded[33:37])[0]
    assert encoded[37:41] == b"IDAT"
    assert zlib.decompress(encoded[41 : 41 + length]) == b"\x00" + raw
    assert receipt["raw_rgb_sha256"] and receipt["bytes"] == len(encoded)
    assert _visual_evidence_decisions(192) == {0, 96, 191}
