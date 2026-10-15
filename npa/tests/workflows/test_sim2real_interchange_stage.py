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
from npa.workflows.sim2real import interchange_object_asset
from npa.workflows.sim2real.interchange_object_asset import (
    assert_collection_object_binding,
    materialize_prepared_object,
)
from npa.workflows.sim2real.interchange_stage import (
    _artifact_preflight,
    assert_execution_profile,
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
        "robot_spec": {
            "schema": "npa.robot-spec/v0.0.1",
            "name": "sealed_ur6",
            "robot_source": "byo_urdf",
            "base_link": "base_link",
            "ee_link": "tool0",
            "joint_names": arm + fingers,
            "gripper_joint_names": fingers,
            "finger_links": ["left_pad", "right_pad"],
            "n_arm_joints": 6,
            "n_gripper_joints": 2,
            "home_qpos": [0.0] * 8,
            "kp": [100.0] * 8,
            "kv": [10.0] * 8,
            "force_lower": [-100.0] * 8,
            "force_upper": [100.0] * 8,
            "source_sha256": DIGEST_A,
            "source_tree_sha256": DIGEST_B,
            "source_format": "urdf",
            "source_relative_path": "robot.urdf",
            "asset_root_uri": "s3://example/robot/source/",
            "embodiment_digest": _digest(embodiment),
            "expected_action_dim": 7,
            "expected_observation_dim": None,
            "resolved_usd_uri": "s3://example/robot/resolved/robot.usd",
            "resolved_manifest_uri": "s3://example/robot/resolved/manifest.json",
        },
        "isaac_asset": {
            "source_format": "urdf",
            "resolved_usd_uri": "s3://example/robot/resolved/robot.usd",
            "conversion": "isaaclab.sim.converters.UrdfConverter",
        },
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
        "measured_state_names": arm + ["gripper"],
        "measured_state_units": ["rad"] * 6 + ["normalized_closed"],
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
            "measured_state": {
                "names": arm + ["gripper"],
                "units": ["rad"] * 6 + ["normalized_closed"],
            },
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
    assert prepared["robot_asset"]["robot_spec"]["joint_names"] == [
        *[f"ur_axis_{index}" for index in range(1, 7)],
        "left_driver",
        "right_follower",
    ]
    assert prepared["robot_asset"]["isaac_asset"]["conversion"].endswith(
        "UrdfConverter"
    )
    assert prepared["capabilities"]["deterministic_noise_coupling"] is False
    assert prepared["capabilities"]["repeated_paired_trials"] is True
    adapter = training_adapter(prepared)
    assert adapter["normalization_identity"] == ""
    assert adapter["measured_state_names"] == prepared["policy"]["measured_state_names"]
    assert [camera["feature_key"] for camera in adapter["cameras"]] == [
        "observation.images.workspace",
        "observation.images.wrist",
    ]


def test_preparation_rejects_six_dof_openpi_padding(tmp_path: Path) -> None:
    with pytest.raises(InterchangeContractError, match="seven arm joints"):
        build_preparation(_sealed(tmp_path, backend="openpi_pi05"), run_id="run-openpi")


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


def test_preparation_rejects_incomplete_robot_actuator_arrays(tmp_path: Path) -> None:
    root = _sealed(tmp_path)
    contract = json.loads((root / "robot-contract.json").read_text())
    contract["robot_spec"]["kp"] = contract["robot_spec"]["kp"][:-1]
    contract["contract_digest"] = _digest(
        {
            key: value
            for key, value in contract.items()
            if key not in {"contract_digest", "content_addressed_uri"}
        }
    )
    write_json(root / "robot-contract.json", contract)
    _reseal(root)
    with pytest.raises(InterchangeContractError, match="kp must cover every"):
        build_preparation(root, run_id="run-incomplete-actuators")


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


def _prepared_object(source_sha256: str) -> dict:
    return {
        "object": {
            "identity": "sealed-cube",
            "source_uri": "s3://private-run/assets/cube.usda",
            "source_sha256": source_sha256,
            "bundle_sha256": DIGEST_B,
            "kind": "rigid",
            "units_m_per_unit": 1.0,
            "scale_xyz": [1.2, 1.2, 1.2],
            "world_from_root": {
                "translation_m": [0.4, 0.0, 0.04],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "root_from_reference": {
                "translation_m": [0.0, 0.0, -0.03],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "root_from_grasp": {
                "translation_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "grasp_extents_m": [0.072, 0.072, 0.072],
            "collision_declared": True,
            "inertia_declared": True,
            "material_declared": True,
            "mass_kg": 0.12,
            "static_friction": 0.8,
            "dynamic_friction": 0.6,
            "restitution": 0.05,
            "grasp_axis": "y",
        }
    }


def test_prepared_object_bytes_and_collection_binding_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b'#usda 1.0\ndef Xform "Cube" {}\n'
    digest = hashlib.sha256(payload).hexdigest()

    class Storage:
        def download_file(self, uri: str, destination: str) -> None:
            assert uri == "s3://private-run/assets/cube.usda"
            Path(destination).write_bytes(payload)

    monkeypatch.setattr(interchange_object_asset, "storage", lambda: Storage())
    materialized = materialize_prepared_object(
        _prepared_object(digest), tmp_path / "asset"
    )
    receipt = materialized.receipt()
    assert materialized.local_path.read_bytes() == payload
    assert_collection_object_binding(receipt, materialized.profile)

    changed = {**receipt, "source_sha256": DIGEST_A}
    with pytest.raises(InterchangeContractError, match="differs from prepared"):
        assert_collection_object_binding(changed, materialized.profile)


def test_prepared_object_rejects_downloaded_byte_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Storage:
        def download_file(self, _uri: str, destination: str) -> None:
            Path(destination).write_bytes(b"wrong")

    monkeypatch.setattr(interchange_object_asset, "storage", lambda: Storage())
    with pytest.raises(InterchangeContractError, match="downloaded object bytes"):
        materialize_prepared_object(_prepared_object(DIGEST_A), tmp_path / "asset")


def test_franka_execution_profile_rejects_six_axis_robot_before_gpu() -> None:
    preparation = {
        "robot": {
            "base_link": "base_link",
            "ee_link": "tool0",
            "arm_joint_names": [f"ur_axis_{index}" for index in range(1, 7)],
            "measured_state_names": [
                *[f"ur_axis_{index}" for index in range(1, 7)],
                "gripper",
            ],
            "controller_representation": "absolute_joint_position",
            "gripper": {
                "joint_names": ["left_driver", "right_follower"],
                "finger_links": ["left_pad", "right_pad"],
                "maximum_width_m": 0.085,
            },
            "cameras": [
                {"role": "exterior", "mount_link": "world"},
                {"role": "wrist", "mount_link": "tool0"},
            ],
        },
        "policy": {"backend": "lerobot_smolvla"},
    }

    with pytest.raises(
        InterchangeContractError, match="incompatible before GPU submission"
    ):
        assert_execution_profile(preparation, "pi05_franka_surface_v1")


def test_franka_execution_profile_accepts_exact_robot_and_cameras() -> None:
    arm = [f"panda_joint{index}" for index in range(1, 8)]
    preparation = {
        "robot": {
            "base_link": "panda_link0",
            "ee_link": "panda_hand",
            "arm_joint_names": arm,
            "measured_state_names": [*arm, "gripper"],
            "controller_representation": "absolute_joint_position",
            "gripper": {
                "joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
                "finger_links": ["panda_leftfinger", "panda_rightfinger"],
                "maximum_width_m": 0.08,
            },
            "cameras": [
                {"role": "exterior", "mount_link": "world"},
                {"role": "wrist", "mount_link": "panda_hand"},
            ],
        },
        "policy": {"backend": "lerobot_smolvla"},
    }

    assert_execution_profile(preparation, "pi05_franka_surface_v1")
