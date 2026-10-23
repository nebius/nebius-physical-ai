"""Behavioral tests for the model-neutral Sim2Real interchange boundary."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from npa.workflows.sim2real.interchange_contract import (
    CameraView,
    GripperMapping,
    InterchangeContractError,
    PolicyAction,
    PolicyCapabilities,
    PolicyObservation,
    PolicyOutput,
    PolicyProfile,
    RigidObjectProfile,
    RobotSensorProfile,
    SurfacePlacementTask,
    TimestampedCameraObservation,
    Transform,
    assert_policy_compatible,
    assert_surface_placement_compatible,
    policy_observation_from_wire,
    policy_output_from_wire,
    policy_profile_from_dict,
    rigid_object_profile_from_asset_manifest,
    rigid_object_profile_from_dict,
    robot_sensor_profile_from_dict,
    robot_sensor_profile_from_resolved,
    surface_placement_task_from_dict,
    validate_observation,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _camera(key: str, role: str, mount: str) -> CameraView:
    return CameraView(
        key=key,
        role=role,  # type: ignore[arg-type]
        mount_link=mount,
        mount_from_camera=Transform(
            (0.09, 0.0, 0.015) if role == "wrist" else (1.2, 0.0, 0.9),
            (0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=(192.43, 192.43, 111.5, 111.5),
        image_size=(224, 224),
        layout="HWC",
        mask_key=f"{key}_valid",
        capture_timestamp_key=f"{key}_capture_s",
        state_timestamp_key="state_timestamp_s",
        maximum_state_skew_s=1.0 / 60.0,
    )


def _robot(dof: int = 7) -> RobotSensorProfile:
    arm = tuple(f"axis_{index}" for index in range(1, dof + 1))
    fingers = ("left_finger_joint", "right_finger_joint")
    return RobotSensorProfile(
        identity=f"arm-{dof}",
        bundle_sha256=DIGEST_A,
        source_sha256=DIGEST_B,
        base_link="base",
        ee_link="tool0" if dof == 6 else "hand",
        arm_joint_names=arm,
        measured_state_names=arm + ("gripper",),
        measured_state_units=("rad",) * len(arm) + ("normalized_closed",),
        joint_limits_rad=tuple((-2.0, 2.0) for _ in arm),
        home_positions_rad=tuple(0.0 for _ in arm),
        controller_representation="absolute_joint_position",
        gripper=GripperMapping(
            joint_names=fingers,
            joint_units=("m", "m"),
            finger_links=("left_finger", "right_finger"),
            open_joint_positions=(0.03, 0.03),
            closed_joint_positions=(0.0, 0.0),
            maximum_width_m=0.06,
            model_semantics="normalized_closed",
            model_open=0.0,
            model_closed=1.0,
        ),
        cameras=(
            _camera("scene", "exterior", "world"),
            _camera("tool", "wrist", "tool0" if dof == 6 else "hand"),
        ),
    )


def _task() -> SurfacePlacementTask:
    return SurfacePlacementTask(
        identity="released-surface-v1",
        target_world_xyz_m=(0.5, 0.2, 0.05),
        approach_clearance_m=0.08,
        lift_delta_m=0.05,
        retreat_distance_m=0.10,
    )


def _policy(robot: RobotSensorProfile) -> PolicyProfile:
    return PolicyProfile(
        backend="lerobot_smolvla",
        identity=f"smolvla-{robot.identity}",
        checkpoint_sha256=DIGEST_A,
        robot_identity=robot.identity,
        task_identity=_task().identity,
        camera_keys=tuple(camera.key for camera in robot.cameras),
        measured_state_names=robot.measured_state_names,
        measured_state_units=robot.measured_state_units,
        action=PolicyAction(
            representation="absolute_joint_position",
            names=robot.action_names,
            units=("rad",) * len(robot.arm_joint_names) + ("normalized_closed",),
            frame="joint",
            gripper_semantics="normalized_closed",
            horizon=50,
            control_hz=15.0,
            execute_prefix=1,
        ),
        capabilities=PolicyCapabilities(
            inference=True,
            native_training=True,
            checkpoint_reload=True,
            model_loss=True,
            action_metrics=True,
            deterministic_noise_coupling=False,
            repeated_paired_trials=True,
            hardware_topologies=("single-gpu",),
        ),
        normalization_identity=DIGEST_B,
    )


def test_different_joint_count_and_end_effector_are_supported_when_exact() -> None:
    robot = _robot(6)
    receipt = assert_policy_compatible(robot, _task(), _policy(robot))
    assert receipt["arm_dof"] == 6
    assert receipt["ee_link"] == "tool0"


def test_profile_consumes_canonical_resolved_robot_without_franka_defaults() -> None:
    resolved = {
        "embodiment_digest": DIGEST_A,
        "embodiment": {
            "base_link": "base",
            "ee_link": "tool0",
            "arm_joint_names": ["a1", "a2", "a3", "a4", "a5", "a6"],
            "gripper_joint_names": ["finger_a", "finger_b"],
            "finger_links": ["left_finger", "right_finger"],
            "home_qpos": [0.0] * 6 + [0.025, 0.025],
            "gripper_commands": {"open": 0.025, "close": 0.0},
            "source": {
                "source_sha256": DIGEST_B,
                "source_tree_sha256": DIGEST_A,
            },
        },
    }
    sensor_config = {
        "fixed_base": True,
        "controller_representation": "absolute_joint_position",
        "joint_limits_rad": [[-2.0, 2.0]] * 6,
        "measured_state": {
            "names": ["a1", "a2", "a3", "a4", "a5", "a6", "gripper"],
            "units": ["rad"] * 6 + ["normalized_closed"],
        },
        "gripper": {
            "joint_units": ["m", "m"],
            "maximum_width_m": 0.05,
            "model_semantics": "normalized_closed",
            "model_open": 0.0,
            "model_closed": 1.0,
        },
        "cameras": [camera.to_dict() for camera in _robot(6).cameras],
    }
    profile = robot_sensor_profile_from_resolved(resolved, sensor_config=sensor_config)
    assert profile.arm_joint_names == ("a1", "a2", "a3", "a4", "a5", "a6")
    assert profile.ee_link == "tool0"
    assert profile.gripper.maximum_width_m == 0.05
    assert profile.measured_state_names[-1] == "gripper"
    assert profile.measured_state_units[-1] == "normalized_closed"


def test_policy_state_keeps_physical_fingers_out_of_model_dimensions() -> None:
    robot = _robot(7)
    assert robot.gripper.joint_names == (
        "left_finger_joint",
        "right_finger_joint",
    )
    assert robot.measured_state_names == (*robot.arm_joint_names, "gripper")
    assert len(robot.measured_state_names) == 8
    assert len(robot.arm_joint_names + robot.gripper.joint_names) == 9


def test_resolved_robot_requires_explicit_actual_joint_limits() -> None:
    resolved = {
        "embodiment_digest": DIGEST_A,
        "embodiment": {
            "base_link": "base",
            "ee_link": "tool0",
            "arm_joint_names": ["a1"] * 6,
            "gripper_joint_names": ["finger_a", "finger_b"],
            "finger_links": ["left_finger", "right_finger"],
            "home_qpos": [0.0] * 8,
            "gripper_commands": {"open": 0.025, "close": 0.0},
            "source": {
                "source_sha256": DIGEST_B,
                "source_tree_sha256": DIGEST_A,
            },
        },
    }
    with pytest.raises(InterchangeContractError, match="joint_limits_rad"):
        robot_sensor_profile_from_resolved(resolved, sensor_config={})


def test_six_dof_policy_cannot_be_padded_into_seven_dof_robot() -> None:
    six = _robot(6)
    seven = _robot(7)
    policy = replace(_policy(six), robot_identity=seven.identity)
    with pytest.raises(InterchangeContractError, match="state names/order"):
        assert_policy_compatible(seven, _task(), policy)


def test_reordered_joint_names_fail_before_controller_use() -> None:
    robot = _robot(7)
    policy = _policy(robot)
    reordered = replace(
        policy,
        measured_state_names=(
            policy.measured_state_names[1],
            policy.measured_state_names[0],
            *policy.measured_state_names[2:],
        ),
    )
    with pytest.raises(InterchangeContractError, match="state names/order"):
        assert_policy_compatible(robot, _task(), reordered)


def test_normalized_closed_gripper_mapping_uses_declared_non_franka_aperture() -> None:
    gripper = _robot(6).gripper
    assert gripper.width_to_model(0.06) == pytest.approx(0.0)
    assert gripper.width_to_model(0.0) == pytest.approx(1.0)
    assert gripper.model_to_joint_positions(0.0) == pytest.approx((0.03, 0.03))
    with pytest.raises(InterchangeContractError, match="out of range"):
        gripper.width_to_model(0.08)


def test_normalized_open_gripper_endpoints_are_not_silently_reversed() -> None:
    gripper = GripperMapping(
        joint_names=("finger_driver",),
        joint_units=("rad",),
        finger_links=("left_pad", "right_pad"),
        open_joint_positions=(0.0,),
        closed_joint_positions=(0.8,),
        maximum_width_m=0.085,
        model_semantics="normalized_open",
        model_open=1.0,
        model_closed=0.0,
    )
    assert gripper.width_to_model(0.085) == pytest.approx(1.0)
    assert gripper.model_to_joint_positions(0.0) == pytest.approx((0.8,))
    with pytest.raises(InterchangeContractError, match="open=1, closed=0"):
        replace(gripper, model_open=0.0, model_closed=1.0)


def test_durable_robot_and_policy_profiles_round_trip_without_defaults() -> None:
    robot = _robot(6)
    policy = _policy(robot)
    assert robot_sensor_profile_from_dict(robot.to_dict()) == robot
    assert policy_profile_from_dict(policy.to_dict()) == policy


def test_durable_object_and_task_profiles_round_trip_without_defaults() -> None:
    prop = RigidObjectProfile(
        identity="offset-spool",
        source_uri="s3://example/assets/spool.usda",
        source_sha256=DIGEST_A,
        bundle_sha256=DIGEST_B,
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        world_from_root=Transform((0.4, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.0, 0.0, -0.02), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((0.01, 0.0, 0.02), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.06, 0.05, 0.08),
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=0.12,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
        grasp_axis="y",
    )
    task = _task()
    task_payload = {"identity": task.identity, **task.teacher_frames(prop)}
    assert rigid_object_profile_from_dict(prop.to_dict()) == prop
    assert surface_placement_task_from_dict(task_payload) == task


def test_one_driver_parallel_jaw_gripper_is_supported() -> None:
    gripper = GripperMapping(
        joint_names=("finger_driver",),
        joint_units=("rad",),
        finger_links=("left_pad", "right_pad"),
        open_joint_positions=(0.0,),
        closed_joint_positions=(0.8,),
        maximum_width_m=0.085,
        model_semantics="normalized_closed",
        model_open=0.0,
        model_closed=1.0,
    )
    assert gripper.model_to_joint_positions(1.0) == pytest.approx((0.8,))
    assert gripper.joint_positions_to_width((0.4,)) == pytest.approx(0.0425)


def test_unsupported_gripper_is_rejected_during_profile_construction() -> None:
    with pytest.raises(InterchangeContractError, match="two physical"):
        GripperMapping(
            joint_names=("finger_driver",),
            joint_units=("rad",),
            finger_links=("finger_a", "finger_b", "finger_c"),
            open_joint_positions=(0.0,),
            closed_joint_positions=(0.8,),
            maximum_width_m=0.04,
            model_semantics="normalized_closed",
            model_open=0.0,
            model_closed=1.0,
        )


def test_camera_mount_calibration_and_staleness_are_enforced() -> None:
    with pytest.raises(InterchangeContractError, match="moving link"):
        _camera("wrist", "wrist", "world")
    robot = _robot(6)
    policy = _policy(robot)
    state = {name: 0.0 for name in policy.measured_state_names}
    payloads = {
        key: {
            "layout": "HWC",
            "image_size_wh": [224, 224],
            "mask": True,
            "capture_timestamp_s": 1.0,
            "state_timestamp_s": 1.0,
            "rgb8": bytes(224 * 224 * 3),
            "image_sha256": hashlib.sha256(bytes(224 * 224 * 3)).hexdigest(),
        }
        for key in policy.camera_keys
    }
    payloads["tool"]["capture_timestamp_s"] = 0.9
    with pytest.raises(InterchangeContractError, match="stale"):
        validate_observation(
            robot=robot,
            policy=policy,
            camera_payloads=payloads,
            measured_state=state,
            prompt="place the object",
            input_identity=DIGEST_B,
            task_identity=_task().identity,
        )


def test_validated_observation_is_typed_and_binds_exact_camera_bytes() -> None:
    robot = _robot(6)
    policy = _policy(robot)
    rgb8 = bytes([17]) * (224 * 224 * 3)
    payloads = {
        key: {
            "layout": "HWC",
            "image_size_wh": [224, 224],
            "mask": True,
            "capture_timestamp_s": 2.0,
            "state_timestamp_s": 2.0,
            "rgb8": rgb8,
            "image_sha256": hashlib.sha256(rgb8).hexdigest(),
        }
        for key in policy.camera_keys
    }
    observation = validate_observation(
        robot=robot,
        policy=policy,
        camera_payloads=payloads,
        measured_state={name: 0.0 for name in policy.measured_state_names},
        prompt="place the object",
        input_identity=DIGEST_B,
        task_identity=_task().identity,
    )
    assert isinstance(observation, PolicyObservation)
    assert all(
        isinstance(view, TimestampedCameraObservation) for view in observation.views
    )
    wire = observation.to_wire_dict()
    assert wire["privileged_object_state"] is False
    assert wire["views"][1]["role"] == "wrist"
    assert wire["views"][1]["image_sha256"] == hashlib.sha256(rgb8).hexdigest()
    decoded = policy_observation_from_wire(wire)
    assert decoded == observation
    wire["privileged_object_state"] = {"object_xyz": [0.0, 0.0, 0.0]}
    with pytest.raises(InterchangeContractError, match="privileged"):
        policy_observation_from_wire(wire)


def test_gripper_inverse_uses_independent_measured_finger_travel() -> None:
    gripper = _robot(6).gripper
    assert gripper.joint_positions_to_width((0.03, 0.03)) == pytest.approx(0.06)
    assert gripper.joint_positions_to_width((0.03, 0.0)) == pytest.approx(0.03)
    assert gripper.joint_positions_to_width((0.029, 0.028)) == pytest.approx(0.057)
    assert gripper.joint_positions_to_width((0.0301, 0.0298)) == pytest.approx(0.0599)


def test_gripper_inverse_handles_changed_spans_and_directions() -> None:
    gripper = GripperMapping(
        joint_names=("left_driver", "right_driver"),
        joint_units=("m", "rad"),
        finger_links=("left_pad", "right_pad"),
        open_joint_positions=(0.02, -0.04),
        closed_joint_positions=(-0.01, 0.02),
        maximum_width_m=0.09,
        model_semantics="normalized_closed",
        model_open=0.0,
        model_closed=1.0,
    )

    assert gripper.joint_positions_to_width((0.005, -0.01)) == pytest.approx(0.045)
    assert gripper.joint_positions_to_width((-0.0025, -0.025)) == pytest.approx(0.045)


@pytest.mark.parametrize("positions", [(-0.001, -0.001), (0.031, 0.031)])
def test_gripper_inverse_rejects_aggregate_aperture_outside_declared_range(
    positions: tuple[float, float],
) -> None:
    with pytest.raises(InterchangeContractError, match="aperture is outside"):
        _robot(6).gripper.joint_positions_to_width(positions)


def test_gripper_inverse_saturates_only_with_declared_measurement_tolerance() -> None:
    gripper = _robot(6).gripper
    assert gripper.joint_positions_to_width(
        (0.03002, 0.03001), position_tolerances=(0.001, 0.001)
    ) == pytest.approx(0.06)
    isaac_gripper = replace(
        gripper,
        open_joint_positions=(0.04, 0.04),
        maximum_width_m=0.08,
    )
    assert isaac_gripper.joint_positions_to_width(
        (0.0400162, 0.0400162),
        position_tolerances=(0.2 / 120.0, 0.2 / 120.0),
    ) == pytest.approx(0.08)
    with pytest.raises(InterchangeContractError, match="aperture is outside"):
        gripper.joint_positions_to_width(
            (0.032, 0.032), position_tolerances=(0.001, 0.001)
        )
    with pytest.raises(InterchangeContractError, match="must be nonnegative"):
        gripper.joint_positions_to_width(
            (0.03, 0.03), position_tolerances=(-0.001, 0.001)
        )


def test_policy_output_binds_action_semantics_and_finite_chunk() -> None:
    policy = _policy(_robot(6))
    output = PolicyOutput(
        observation_input_identity="input-1",
        policy_identity=policy.identity,
        checkpoint_sha256=policy.checkpoint_sha256,
        action=policy.action,
        action_chunk=((0.0,) * len(policy.action.names),),
        generated_timestamp_s=3.0,
    )
    assert output.to_dict()["action"]["horizon"] == 50
    assert output.to_dict()["action_chunk"][0][-1] == 0.0
    with pytest.raises(InterchangeContractError, match="finite"):
        PolicyOutput(
            observation_input_identity="input-1",
            policy_identity=policy.identity,
            checkpoint_sha256=policy.checkpoint_sha256,
            action=policy.action,
            action_chunk=((float("nan"),) * len(policy.action.names),),
            generated_timestamp_s=3.0,
        )


def test_offset_non_cube_object_uses_declared_grasp_and_reference_frames() -> None:
    prop = RigidObjectProfile(
        identity="mug-family",
        source_uri="s3://example/object.usd",
        source_sha256=DIGEST_A,
        bundle_sha256=DIGEST_B,
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        world_from_root=Transform((0.5, -0.1, 0.02), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.02, 0.0, 0.03), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((-0.01, 0.015, 0.04), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.08, 0.05, 0.10),
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=0.12,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
    )
    frames = _task().teacher_frames(prop)
    assert frames["object_root_from_grasp"]["translation_m"] == [-0.01, 0.015, 0.04]
    assert frames["object_root_from_reference"]["translation_m"] == [0.02, 0.0, 0.03]


def test_sealed_non_cube_asset_requires_explicit_offset_frames() -> None:
    manifest = {
        "schema": "npa.franka-rl.assets.v1",
        "license": "Apache-2.0",
        "units": "metres",
        "physics": {
            "collision": "declared",
            "mass_kg": 0.08,
            "material": {
                "static_friction": 0.8,
                "dynamic_friction": 0.6,
                "restitution": 0.0,
            },
        },
        "files": {
            "assets/spool.usda": DIGEST_A,
            "assets/bottle.usda": DIGEST_B,
        },
    }
    prop = rigid_object_profile_from_asset_manifest(
        manifest,
        target="bottle",
        source_root_uri="s3://example/sealed",
        world_from_root=Transform((0.5, 0.0, 0.038), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.0, 0.0, -0.038), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((0.0, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.044, 0.044, 0.056),
    )
    assert prop.source_sha256 == DIGEST_B
    assert prop.root_from_reference.translation_m == (0.0, 0.0, -0.038)
    compatibility = assert_surface_placement_compatible(_robot(6), prop, _task())
    assert compatibility["required_gripper_width_m"] == 0.044
    with pytest.raises(InterchangeContractError, match="not an exact sealed"):
        rigid_object_profile_from_asset_manifest(
            manifest,
            target="cube",
            source_root_uri="s3://example/sealed",
            world_from_root=prop.world_from_root,
            root_from_reference=prop.root_from_reference,
            root_from_grasp=prop.root_from_grasp,
            grasp_extents_m=prop.grasp_extents_m,
        )


def test_asset_manifest_cannot_fabricate_missing_physics_metadata() -> None:
    manifest = {
        "schema": "npa.franka-rl.assets.v1",
        "license": "Apache-2.0",
        "units": "metres",
        "files": {"assets/spool.usda": DIGEST_A},
    }
    with pytest.raises(InterchangeContractError, match="physics metadata"):
        rigid_object_profile_from_asset_manifest(
            manifest,
            target="spool",
            source_root_uri="s3://example/sealed",
            world_from_root=Transform((0.5, 0.0, 0.026), (0.0, 0.0, 0.0, 1.0)),
            root_from_reference=Transform((0.0, 0.0, -0.026), (0.0, 0.0, 0.0, 1.0)),
            root_from_grasp=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            grasp_extents_m=(0.054, 0.054, 0.052),
        )


def test_object_grasp_width_must_fit_declared_gripper_aperture() -> None:
    prop = RigidObjectProfile(
        identity="wide-spool",
        source_uri="s3://example/spool.usda",
        source_sha256=DIGEST_A,
        bundle_sha256=DIGEST_B,
        kind="rigid",
        units_m_per_unit=1.0,
        scale_xyz=(1.0, 1.0, 1.0),
        world_from_root=Transform((0.5, 0.0, 0.026), (0.0, 0.0, 0.0, 1.0)),
        root_from_reference=Transform((0.0, 0.0, -0.026), (0.0, 0.0, 0.0, 1.0)),
        root_from_grasp=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        grasp_extents_m=(0.054, 0.09, 0.052),
        collision_declared=True,
        inertia_declared=True,
        material_declared=True,
        mass_kg=0.12,
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.05,
        grasp_axis="y",
    )
    with pytest.raises(InterchangeContractError, match="gripper aperture"):
        assert_surface_placement_compatible(_robot(6), prop, _task())


@pytest.mark.parametrize("kind", ["articulated", "deformable"])
def test_non_rigid_object_kinds_are_explicitly_rejected(kind: str) -> None:
    with pytest.raises(InterchangeContractError, match="only rigid"):
        RigidObjectProfile(
            identity="unsupported",
            source_uri="s3://example/object.usd",
            source_sha256=DIGEST_A,
            bundle_sha256=DIGEST_B,
            kind=kind,
            units_m_per_unit=1.0,
            scale_xyz=(1.0, 1.0, 1.0),
            world_from_root=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            root_from_reference=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            root_from_grasp=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            grasp_extents_m=(0.1, 0.1, 0.1),
            collision_declared=True,
            inertia_declared=True,
            material_declared=True,
            mass_kg=0.12,
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.05,
        )


def test_action_units_horizon_and_cartesian_controller_fail_closed() -> None:
    robot = _robot(6)
    policy = _policy(robot)
    wrong_units = replace(
        policy,
        action=replace(policy.action, units=("degree",) * 6 + ("normalized_closed",)),
    )
    with pytest.raises(InterchangeContractError, match="units"):
        assert_policy_compatible(robot, _task(), wrong_units)
    with pytest.raises(InterchangeContractError, match="horizon"):
        replace(policy.action, horizon=2, execute_prefix=3)
    cartesian_robot = replace(robot, controller_representation="ee_pose")
    cartesian = replace(
        policy,
        action=replace(policy.action, representation="ee_pose", frame="base"),
    )
    with pytest.raises(InterchangeContractError, match="IK/controller"):
        assert_policy_compatible(cartesian_robot, _task(), cartesian)


def test_policy_output_wire_cannot_change_action_semantics() -> None:
    action = _policy(_robot(6)).action
    output = PolicyOutput(
        observation_input_identity="observation-1",
        policy_identity="policy-1",
        checkpoint_sha256=DIGEST_A,
        action=action,
        action_chunk=(tuple(0.0 for _ in action.names),),
        generated_timestamp_s=2.0,
    )
    payload = output.to_dict()
    assert policy_output_from_wire(payload, expected_action=action) == output
    payload["action"]["units"][0] = "degree"
    with pytest.raises(InterchangeContractError, match="changed"):
        policy_output_from_wire(payload, expected_action=action)
