"""Tests for the explicit, bounded Sim2Real backend catalog."""

from dataclasses import replace

import pytest

from npa.workflows.sim2real.interchange_backends import (
    BACKENDS,
    backend_contract,
    openpi_pi05_policy_profile,
    smolvla_policy_profile,
)
from npa.workflows.sim2real.interchange_contract import (
    CameraView,
    GripperMapping,
    InterchangeContractError,
    RobotSensorProfile,
    SurfacePlacementTask,
    Transform,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _robot(dof: int) -> RobotSensorProfile:
    arm = tuple(f"axis_{index}" for index in range(dof))
    fingers = ("left_finger_joint", "right_finger_joint")
    return RobotSensorProfile(
        identity=f"arm-{dof}",
        bundle_sha256=DIGEST_A,
        source_sha256=DIGEST_B,
        base_link="base",
        ee_link="tool0" if dof == 6 else "hand",
        arm_joint_names=arm,
        measured_joint_names=arm + fingers,
        joint_limits_rad=tuple((-2.0, 2.0) for _ in arm),
        home_positions_rad=tuple(0.0 for _ in arm),
        controller_representation="absolute_joint_position",
        gripper=GripperMapping(
            joint_names=fingers,
            joint_units=("m", "m"),
            finger_links=("left_pad", "right_pad"),
            open_joint_positions=(0.04, 0.04),
            closed_joint_positions=(0.0, 0.0),
            maximum_width_m=0.08,
            model_semantics="normalized_closed",
            model_open=0.0,
            model_closed=1.0,
        ),
        cameras=(
            CameraView(
                key="scene",
                role="exterior",
                mount_link="world",
                mount_from_camera=Transform(
                    (1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)
                ),
                intrinsics=(192.0, 192.0, 111.5, 111.5),
                image_size=(224, 224),
                layout="HWC",
                mask_key="scene_valid",
                capture_timestamp_key="scene_capture_s",
                state_timestamp_key="state_timestamp_s",
                maximum_state_skew_s=1.0 / 60.0,
            ),
            CameraView(
                key="tool",
                role="wrist",
                mount_link="tool0" if dof == 6 else "hand",
                mount_from_camera=Transform(
                    (0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
                ),
                intrinsics=(192.0, 192.0, 111.5, 111.5),
                image_size=(224, 224),
                layout="HWC",
                mask_key="tool_valid",
                capture_timestamp_key="tool_capture_s",
                state_timestamp_key="state_timestamp_s",
                maximum_state_skew_s=1.0 / 60.0,
            ),
        ),
    )


def _task() -> SurfacePlacementTask:
    return SurfacePlacementTask(
        identity="released-surface-v1",
        target_world_xyz_m=(0.5, 0.1, 0.05),
        approach_clearance_m=0.08,
        lift_delta_m=0.05,
        retreat_distance_m=0.10,
    )


def test_backend_catalog_is_closed_and_has_distinct_native_protocols() -> None:
    assert set(BACKENDS) == {"openpi_pi05", "lerobot_smolvla"}
    openpi = backend_contract("openpi_pi05")
    smolvla = backend_contract("lerobot_smolvla")
    assert openpi.transport != smolvla.transport
    assert openpi.training_adapter != smolvla.training_adapter
    assert openpi.deterministic_noise_protocol is not None
    assert smolvla.deterministic_noise_protocol is None
    assert smolvla.repeated_paired_trials_required is True
    with pytest.raises(InterchangeContractError, match="unsupported"):
        backend_contract("random_actor")


def test_pi05_profile_rejects_six_dof_and_non_franka_aperture() -> None:
    with pytest.raises(InterchangeContractError, match="seven arm joints"):
        openpi_pi05_policy_profile(
            robot=_robot(6),
            task=_task(),
            policy_identity="pi05",
            checkpoint_sha256=DIGEST_A,
            normalization_identity=DIGEST_B,
            hardware_topologies=("single-b200",),
        )
    robot = replace(
        _robot(7),
        gripper=replace(_robot(7).gripper, maximum_width_m=0.06),
    )
    with pytest.raises(InterchangeContractError, match="80 mm"):
        openpi_pi05_policy_profile(
            robot=robot,
            task=_task(),
            policy_identity="pi05",
            checkpoint_sha256=DIGEST_A,
            normalization_identity=DIGEST_B,
            hardware_topologies=("single-b200",),
        )


def test_smolvla_manifest_preserves_six_dof_names_and_repeated_trials() -> None:
    robot = _robot(6)
    manifest = {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "policy_identity": "smolvla-six-axis",
        "checkpoint_sha256": DIGEST_A,
        "robot_identity": robot.identity,
        "task_identity": _task().identity,
        "normalization_identity": DIGEST_B,
        "measured_state_names": list(robot.measured_joint_names),
        "cameras": [
            {"key": camera.key, "feature_key": f"observation.images.{camera.key}"}
            for camera in robot.cameras
        ],
        "action": {
            "representation": "absolute_joint_position",
            "names": list(robot.action_names),
            "units": ["rad"] * 6 + ["normalized_closed"],
            "frame": "joint",
            "gripper_semantics": "normalized_closed",
            "horizon": 50,
            "control_hz": 15.0,
            "execute_prefix": 1,
        },
    }
    profile = smolvla_policy_profile(
        manifest,
        robot=robot,
        task=_task(),
        hardware_topologies=("single-gpu",),
    )
    assert len(profile.action.names) == 7
    assert profile.measured_state_names == robot.measured_joint_names
    assert profile.capabilities.deterministic_noise_coupling is False
    assert profile.capabilities.repeated_paired_trials is True
