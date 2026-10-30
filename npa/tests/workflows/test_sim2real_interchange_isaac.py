from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
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
    gripper_position_tolerances,
    measured_policy_state,
    render_absolute_joint_action,
    surface_teacher_waypoints,
)
from npa.workflows.sim2real.interchange_isaac_runtime import (
    _axis_angle_error,
    _run_with_app,
    _update_events,
    _visual_evidence_decisions,
    _write_teacher_episode,
    _write_png_rgb8,
)
from npa.workflows.sim2real.interchange_isaac_entry import (
    _child_command,
    _cohort_report,
    _run_child,
)
from npa.workflows.sim2real import (
    interchange_teacher_collect,
    interchange_teacher_runtime,
)
from npa.workflows.sim2real.interchange_teacher_runtime import _scenario_offset
from npa.workflows.sim2real.interchange_teacher_export import (
    EPISODE_FILES,
    _validated_episodes,
)
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


def test_gripper_tolerance_uses_actual_velocity_limit_and_physics_period() -> None:
    profile = _profile()
    tolerances = gripper_position_tolerances(
        velocity_limits=(0.2,), physics_dt=1.0 / 120.0, profile=profile
    )
    representation = abs(float(np.spacing(np.float32(0.8))))
    assert tolerances == pytest.approx((0.2 / 120.0 + representation,))
    with pytest.raises(InterchangeContractError, match="velocity limits"):
        gripper_position_tolerances(
            velocity_limits=(-0.2,), physics_dt=1.0 / 120.0, profile=profile
        )
    with pytest.raises(InterchangeContractError, match="physics period"):
        gripper_position_tolerances(
            velocity_limits=(0.2,), physics_dt=0.0, profile=profile
        )


def test_measured_policy_state_bounds_one_physics_step_of_joint_overshoot() -> None:
    torch = pytest.importorskip("torch")
    profile = _profile()
    names = [*profile.arm_joint_names, *profile.gripper.joint_names]
    native = SimpleNamespace(
        physics_dt=1.0 / 120.0,
        scene={
            "robot": SimpleNamespace(
                joint_names=names,
                data=SimpleNamespace(
                    joint_pos=torch.tensor([[0.0] * 6 + [-0.001]]),
                    joint_vel_limits=torch.tensor([[2.0] * 6 + [0.2]]),
                ),
            )
        },
    )
    state = measured_policy_state(native, profile)
    assert state["gripper"] == pytest.approx(profile.gripper.model_open)


def test_teacher_runtime_turns_premature_zero_exit_into_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class App:
        closed = False

        def close(self) -> None:
            self.closed = True

    app = App()
    args = argparse.Namespace(output_path=tmp_path)
    monkeypatch.setattr(
        interchange_teacher_runtime,
        "run",
        lambda _args: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(RuntimeError, match="SystemExit code=0") as failure:
        interchange_teacher_runtime._run_with_app(args, app)

    assert isinstance(failure.value.__cause__, SystemExit)
    assert app.closed is True
    assert not (tmp_path / "report.json").exists()
    receipt = json.loads((tmp_path / "runtime_failure.json").read_text())
    assert receipt == {
        "error": "0",
        "error_type": "SystemExit",
        "report_written": False,
        "schema": "npa.sim2real.policy_interchange.teacher_runtime_failure.v1",
        "status": "failed",
        "system_exit_code": 0,
    }


def test_teacher_runtime_preserves_original_failure_when_close_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class App:
        def close(self) -> None:
            raise SystemExit(0)

    args = argparse.Namespace(output_path=tmp_path)
    monkeypatch.setattr(
        interchange_teacher_runtime,
        "run",
        lambda _args: (_ for _ in ()).throw(ValueError("live action layout mismatch")),
    )

    with pytest.raises(ValueError, match="live action layout mismatch"):
        interchange_teacher_runtime._run_with_app(args, App())

    receipt = json.loads((tmp_path / "runtime_failure.json").read_text())
    assert receipt["error_type"] == "ValueError"
    assert receipt["error"] == "live action layout mismatch"
    assert receipt["system_exit_code"] is None


def test_teacher_parent_surfaces_structured_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    log = tmp_path / "teacher.log"
    failure = {
        "schema": "npa.sim2real.policy_interchange.teacher_runtime_failure.v1",
        "status": "failed",
        "error_type": "ValueError",
        "error": "live action layout mismatch",
    }
    (output / "runtime_failure.json").write_text(json.dumps(failure))
    monkeypatch.setattr(
        interchange_teacher_collect.subprocess,
        "run",
        lambda *_args, **_kwargs: (
            interchange_teacher_collect.subprocess.CompletedProcess([], 0)
        ),
    )

    with pytest.raises(RuntimeError, match="live action layout mismatch"):
        interchange_teacher_collect._execute(["native"], output, log)


def test_policy_runtime_preserves_original_failure_when_close_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class App:
        def close(self) -> None:
            raise SystemExit(0)

    args = argparse.Namespace(output_path=tmp_path)
    monkeypatch.setattr(
        "npa.workflows.sim2real.interchange_isaac_runtime.run",
        lambda _args: (_ for _ in ()).throw(ValueError("camera role mismatch")),
    )

    with pytest.raises(ValueError, match="camera role mismatch"):
        _run_with_app(args, App())

    receipt = json.loads((tmp_path / "runtime_failure.json").read_text())
    assert receipt["error_type"] == "ValueError"
    assert receipt["error"] == "camera role mismatch"
    assert receipt["system_exit_code"] is None
    assert receipt["report_written"] is False


def test_policy_runtime_turns_premature_zero_exit_into_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class App:
        closed = False

        def close(self) -> None:
            self.closed = True

    app = App()
    args = argparse.Namespace(output_path=tmp_path)
    monkeypatch.setattr(
        "npa.workflows.sim2real.interchange_isaac_runtime.run",
        lambda _args: (_ for _ in ()).throw(SystemExit(0)),
    )

    with pytest.raises(RuntimeError, match="SystemExit code=0") as failure:
        _run_with_app(args, app)

    assert isinstance(failure.value.__cause__, SystemExit)
    assert app.closed is True
    receipt = json.loads((tmp_path / "runtime_failure.json").read_text())
    assert receipt["error_type"] == "SystemExit"
    assert receipt["system_exit_code"] == 0
    assert receipt["report_written"] is False


def test_policy_parent_surfaces_structured_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    log = tmp_path / "policy.log"
    failure = {
        "schema": "npa.sim2real.policy_interchange.isaac_runtime_failure.v1",
        "status": "failed",
        "error_type": "ValueError",
        "error": "camera role mismatch",
    }
    (output / "runtime_failure.json").write_text(json.dumps(failure))
    monkeypatch.setattr(
        "npa.workflows.sim2real.interchange_isaac_entry.subprocess.run",
        lambda *_args, **_kwargs: (
            interchange_teacher_collect.subprocess.CompletedProcess([], 0)
        ),
    )

    with pytest.raises(RuntimeError, match="camera role mismatch"):
        _run_child(["native"], output, log)


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
    persisted_teacher = json.loads((episode / "native_teacher.json").read_text())
    assert persisted_teacher["task_success"] is True
    assert persisted_teacher["ordered_event_steps"] == teacher["ordered_event_steps"]
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


def test_teacher_export_requires_the_complete_native_trace(tmp_path: Path) -> None:
    robot = _profile(6)
    task = type(
        "Task",
        (),
        {
            "identity": "task",
            "success_events": ("contact", "stable"),
        },
    )()
    prop = type("Prop", (), {"identity": "prop"})()
    frame = bytes(range(12))
    frames = [
        {
            "scene": (frame, 2, 2, "exterior"),
            "tool": (frame[::-1], 2, 2, "wrist"),
        }
    ]
    episode = tmp_path / "splits" / "train" / "episode_0000"
    _write_teacher_episode(
        directory=episode,
        robot=robot,
        task=task,
        prop=prop,
        states=[[0.0] * 7],
        actions=[[0.1] * 7],
        camera_frames=frames,
        timestamps=[0.0],
        teacher={
            "schema": "npa.sim2real.policy_interchange.native_teacher.v1",
            "task_success": True,
            "ordered_event_steps": {"contact": 0, "stable": 1},
            "controls": [{"step": 0}],
            "camera_hashes": [{"scene": "a", "tool": "b"}],
            "camera_clocks": [
                {
                    "capture_timestamps_s": {"scene": 0.0, "tool": 0.0},
                    "camera_world_poses": {
                        "scene": {"translation_m": [0.0, 0.0, 0.0]},
                        "tool": {"translation_m": [0.0, 0.0, 0.0]},
                    },
                    "command_application_time_s": 0.0,
                }
            ],
        },
    )
    artifacts = {
        name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
        for name in EPISODE_FILES
    }
    report = {
        "schema": "npa.sim2real.policy_interchange.teacher_collection.v1",
        "status": "passed",
        "all_teacher_episodes_successful": True,
        "counts": {"train": 1, "validation": 0, "gold": 0},
        "episodes": [
            {
                "split": "train",
                "episode_index": 0,
                "task_success": True,
                "artifacts": artifacts,
            }
        ],
    }
    assert _validated_episodes(tmp_path, report)["train"] == [episode]
    (episode / "native_teacher.json").unlink()
    with pytest.raises(RuntimeError, match="teacher episode bytes differ"):
        _validated_episodes(tmp_path, report)


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


def test_native_policy_chunk_is_retained_while_only_action_zero_is_selected() -> None:
    from types import SimpleNamespace

    from npa.workflows.sim2real.interchange_isaac_runtime import (
        _current_policy_action,
    )

    chunk = tuple(
        tuple(float(row * 8 + column) for column in range(8)) for row in range(15)
    )
    output = SimpleNamespace(
        action=SimpleNamespace(
            execute_prefix=1,
            horizon=15,
            names=tuple(f"action_{index}" for index in range(8)),
        ),
        action_chunk=chunk,
    )

    selected, receipt = _current_policy_action(output)

    assert selected == chunk[0]
    assert receipt["shape"] == [15, 8]
    assert receipt["actions"] == [list(row) for row in chunk]
    assert receipt["selection_method"] == "native_fresh_chunk_first_action"
    assert receipt["selected_action_index"] == 0


def test_native_policy_uniform_temporal_ensemble_is_causal_and_time_aligned() -> None:
    from types import SimpleNamespace

    from npa.workflows.sim2real.interchange_isaac_runtime import (
        _select_policy_action,
    )

    history: list[dict] = []
    chunks = (
        ((0.1,) * 8, (0.3,) * 8, (0.5,) * 8),
        ((0.7,) * 8, (0.9,) * 8),
        ((1.1,) * 8,),
    )
    expected = (0.1, 0.5, 0.8333333333333334)
    for decision, (chunk, scalar) in enumerate(zip(chunks, expected, strict=True)):
        output = SimpleNamespace(
            action=SimpleNamespace(
                execute_prefix=1,
                horizon=3,
                names=tuple(f"action_{index}" for index in range(8)),
            ),
            action_chunk=chunk,
        )
        selected, receipt = _select_policy_action(
            output,
            decision=decision,
            strategy="uniform_temporal_ensemble",
            history=history,
        )
        assert selected == pytest.approx((scalar,) * 8)
        assert receipt["selection_method"] == (
            "native_fresh_chunk_uniform_temporal_mean"
        )
        assert receipt["contributor_count"] == decision + 1
        assert receipt["contributors"][-1]["action_index"] == 0


def test_native_child_command_uses_every_explicit_binding(tmp_path: Path) -> None:
    paths = [
        tmp_path / name
        for name in (
            "prepared",
            "preparation.json",
            "service.json",
            "manifest.json",
            "robot.usd",
            "output",
        )
    ]
    command = _child_command(
        argparse.Namespace(
            decisions=12,
            seed=7,
            action_selection="uniform_temporal_ensemble",
        ),
        *paths,
    )
    assert command == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.interchange_isaac_runtime",
        "--input-path",
        str(paths[0]),
        "--preparation-path",
        str(paths[1]),
        "--service-receipt-path",
        str(paths[2]),
        "--robot-usd-path",
        str(paths[4]),
        "--robot-asset-manifest-path",
        str(paths[3]),
        "--output-path",
        str(paths[5]),
        "--decisions",
        "12",
        "--seed",
        "7",
        "--action-selection",
        "uniform_temporal_ensemble",
    ]


def test_native_child_command_accepts_split_disjoint_episode_seed(
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / name
        for name in ("prepared", "prep", "service", "manifest", "robot", "output")
    ]
    command = _child_command(
        argparse.Namespace(decisions=12, seed=7, action_selection="first_action"),
        *paths,
        seed=11,
    )
    assert command[-4:] == [
        "--seed",
        "11",
        "--action-selection",
        "first_action",
    ]


def test_native_object_cohort_preserves_each_result_and_identity() -> None:
    reports = [
        {
            "execution_valid": True,
            "task_success": success,
            "seed": 41 + index,
            "ordered_event_steps": {"contact": 3} if success else {},
            "robot_identity": "franka",
            "object_identity": "offset-spool",
            "task_identity": "surface-placement",
            "checkpoint_sha256": "a" * 64,
            "normalization_identity": "b" * 64,
            "action_selection_strategy": "uniform_temporal_ensemble",
        }
        for index, success in enumerate((True, False, True))
    ]
    cohort = _cohort_report(reports, seed=41)
    assert cohort["execution_valid"] is True
    assert cohort["task_success_measured"] is True
    assert cohort["episode_count"] == 3
    assert cohort["success_count"] == 2
    assert cohort["success_rate"] == pytest.approx(2 / 3)
    assert [episode["seed"] for episode in cohort["episodes"]] == [41, 42, 43]
    assert {episode["object_identity"] for episode in cohort["episodes"]} == {
        "offset-spool"
    }

    reports[1]["object_identity"] = "different-object"
    with pytest.raises(RuntimeError, match="identities or seeds"):
        _cohort_report(reports, seed=41)
