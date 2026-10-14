from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.lerobot_transfer_data import (
    materialize,
    publish,
    tree_hashes,
    write_json,
)
from npa.workflows.sim2real.interchange_contract import InterchangeContractError
from npa.workflows.sim2real.interchange_stage import (
    _artifact_preflight,
    build_preparation,
    training_adapter,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _camera(key: str, role: str, mount_link: str) -> dict:
    return {
        "key": key,
        "role": role,
        "mount_link": mount_link,
        "mount_from_camera": {
            "translation_m": [0.0, 0.0, 0.1],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "intrinsics_fx_fy_cx_cy": [190.0, 190.0, 112.0, 112.0],
        "image_size_wh": [224, 224],
        "layout": "HWC",
        "mask_key": f"{key}_valid",
        "capture_timestamp_key": f"{key}_capture_s",
        "state_timestamp_key": "state_timestamp_s",
        "maximum_state_skew_s": 0.02,
    }


def _write_profile(root: Path, *, backend: str = "lerobot_smolvla") -> None:
    arm = [f"ur_axis_{index}" for index in range(1, 7)]
    fingers = ["left_driver", "right_follower"]
    embodiment = {
        "base_link": "base_link",
        "ee_link": "tool0",
        "arm_joint_names": arm,
        "gripper_joint_names": fingers,
        "finger_links": ["left_pad", "right_pad"],
        "home_qpos": [0.0] * 8,
        "gripper_commands": {"open": 0.03, "close": 0.0},
        "source": {
            "source_sha256": DIGEST_A,
            "source_tree_sha256": DIGEST_B,
        },
    }
    robot_contract = {
        "schema": "npa.sim2real.robot_contract.v1",
        "embodiment_digest": _digest(embodiment),
        "embodiment": embodiment,
    }
    robot_contract["contract_digest"] = _digest(robot_contract)
    task_identity = "surface-placement-v1"
    adapter = {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "policy_identity": "smolvla-ur6",
        "checkpoint_sha256": DIGEST_A,
        "robot_identity": robot_contract["embodiment_digest"],
        "task_identity": task_identity,
        "normalization_identity": DIGEST_B,
        "measured_state_names": arm + fingers,
        "cameras": [{"key": "scene"}, {"key": "tool"}],
        "action": {
            "representation": "absolute_joint_position",
            "names": arm + ["gripper"],
            "units": ["rad"] * 6 + ["normalized_closed"],
            "frame": "joint",
            "gripper_semantics": "normalized_closed",
            "horizon": 50,
            "control_hz": 15.0,
            "execute_prefix": 1,
        },
    }
    values = {
        "profile.json": {
            "schema": "npa.sim2real.policy_interchange.profile.v1",
            "backend": backend,
            "hardware_topologies": ["single-gpu"],
            "cartesian_controller_qualified": False,
        },
        "robot-contract.json": robot_contract,
        "sensor-config.json": {
            "fixed_base": True,
            "controller_representation": "absolute_joint_position",
            "joint_limits_rad": [[-3.14, 3.14]] * 6,
            "gripper": {
                "joint_units": ["rad", "rad"],
                "open_joint_positions": [0.03, 0.03],
                "closed_joint_positions": [0.0, 0.0],
                "maximum_width_m": 0.085,
                "model_semantics": "normalized_closed",
                "model_open": 0.0,
                "model_closed": 1.0,
            },
            "cameras": [
                _camera("scene", "exterior", "world"),
                _camera("tool", "wrist", "tool0"),
            ],
        },
        "object-manifest.json": {
            "schema": "npa.workbench.franka-rl.assets.v1",
            "license": "Apache-2.0",
            "units": "metres",
            "files": {"assets/spool.usda": DIGEST_A},
            "physics": {
                "collision": "declared",
                "mass_kg": 0.12,
                "material": {
                    "static_friction": 0.8,
                    "dynamic_friction": 0.6,
                    "restitution": 0.05,
                },
            },
        },
        "object-config.json": {
            "target": "spool",
            "source_root_uri": "s3://example/assets",
            "world_from_root": {
                "translation_m": [0.4, 0.0, 0.04],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "root_from_reference": {
                "translation_m": [0.0, 0.0, -0.02],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "root_from_grasp": {
                "translation_m": [0.01, 0.0, 0.02],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "grasp_extents_m": [0.06, 0.05, 0.08],
            "grasp_axis": "y",
            "scale_xyz": [1.0, 1.0, 1.0],
        },
        "task.json": {
            "schema": "npa.sim2real.surface_placement_task.v1",
            "identity": task_identity,
            "target_world_xyz_m": [0.55, 0.15, 0.04],
            "approach_clearance_m": 0.08,
            "lift_delta_m": 0.08,
            "retreat_distance_m": 0.1,
            "success_events": [
                "contact",
                "grasp",
                "lift",
                "transport",
                "release",
                "support_contact",
                "stable",
            ],
            "privileged_object_state": False,
        },
        "policy-adapter.json": adapter,
    }
    root.mkdir()
    for name, value in values.items():
        write_json(root / name, value)


def _sealed(tmp_path: Path, *, backend: str = "lerobot_smolvla") -> Path:
    source = tmp_path / "source"
    _write_profile(source, backend=backend)
    sealed = tmp_path / "sealed"
    publish(source, str(sealed))
    return materialize(str(sealed), tmp_path / "readback")


def _reseal(root: Path) -> None:
    write_json(root / "checksums.json", tree_hashes(root))


def test_preparation_binds_six_dof_non_franka_and_offset_spool(tmp_path: Path) -> None:
    prepared = build_preparation(_sealed(tmp_path), run_id="run-1")
    assert prepared["launch_authorized"] is True
    assert prepared["robot"]["ee_link"] == "tool0"
    assert len(prepared["robot"]["arm_joint_names"]) == 6
    assert prepared["object"]["root_from_grasp"]["translation_m"] == [0.01, 0.0, 0.02]
    assert prepared["backend"]["name"] == "lerobot_smolvla"
    assert prepared["capabilities"]["deterministic_noise_coupling"] is False
    assert prepared["capabilities"]["repeated_paired_trials"] is True
    adapter = training_adapter(prepared)
    assert adapter["normalization_identity"] == ""
    assert adapter["measured_state_names"] == prepared["policy"][
        "measured_state_names"
    ]
    assert [camera["feature_key"] for camera in adapter["cameras"]] == [
        "observation.images.workspace",
        "observation.images.wrist",
    ]


def test_preparation_rejects_six_dof_openpi_padding(tmp_path: Path) -> None:
    with pytest.raises(InterchangeContractError, match="seven arm joints"):
        build_preparation(
            _sealed(tmp_path, backend="openpi_pi05"), run_id="run-openpi"
        )


def test_preparation_rejects_object_wider_than_gripper(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    config = json.loads((root / "object-config.json").read_text())
    config["grasp_extents_m"][1] = 0.10
    write_json(root / "object-config.json", config)
    _reseal(root)
    with pytest.raises(InterchangeContractError, match="exceeds"):
        build_preparation(root, run_id="run-wide")


def test_preparation_rejects_reordered_policy_joints(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    adapter = json.loads((root / "policy-adapter.json").read_text())
    adapter["measured_state_names"][0:2] = reversed(
        adapter["measured_state_names"][0:2]
    )
    write_json(root / "policy-adapter.json", adapter)
    _reseal(root)
    with pytest.raises(InterchangeContractError, match="state names/order"):
        build_preparation(root, run_id="run-reordered")


def test_artifact_preflight_requires_exact_huggingface_revision() -> None:
    from argparse import Namespace

    with pytest.raises(InterchangeContractError, match="40-hex"):
        _artifact_preflight(
            Namespace(
                checkpoint_uri="hf://lerobot/smolvla_base@main",
                output_uri="unused",
                component_root_uri="unused",
            )
        )
