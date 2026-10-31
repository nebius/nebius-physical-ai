from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.sim2real import interchange_stage
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
    _experimental_finalize,
    _non_franka_finalize,
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


def test_offline_viewer_embeds_four_panel_native_evidence_without_network() -> None:
    report = {
        "terminal_success_rate": 0.25,
        "wrist_ablation": {"qualification": "matched_effectiveness_eligible"},
        "checkpoint_provenance": {"checkpoint_sha256": DIGEST_A},
        "episodes": [{"queries": [{"action_selection": {"action_owner": "policy"}}]}],
    }

    document = interchange_stage._offline_viewer_html(report, b"native-mp4")

    assert "model-consumed workspace" in document
    assert "model-consumed wrist" in document
    assert "goal witness" in document
    assert "release witness" in document
    assert "data:video/mp4;base64," in document
    assert "default-src 'none'" in document
    assert "https://" not in document
    assert "checkpoint_sha256" in document


def test_experimental_finalize_binds_complete_uniform_ensemble_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace

    per_mode = {
        name: {
            "episodes": 8,
            "terminal_released_surface_placements": successes,
            "terminal_success_rate": successes / 8,
        }
        for name, successes in (
            ("fresh", 5),
            ("frozen", 0),
            ("pixel_channel_shuffle", 0),
        )
    }
    report = {
        "schema": "npa.sim2real.policy_interchange.closed_loop_eval.v1",
        "split": "gold",
        "checkpoint_role": "adapted",
        "checkpoint_provenance": {"checkpoint_sha256": DIGEST_A},
        "protocol": {"action_selection": {"strategy": "uniform_temporal_ensemble"}},
        "episode_count": 24,
        "episodes": [{"queries": [{}]} for _ in range(24)],
        "runtime_boundary_smoke": {
            "schema": "npa.sim2real.native_policy_boundary_smoke.v1",
            "policy_backend": "lerobot_smolvla",
            "input_lineage": {"workspace": DIGEST_A, "wrist": DIGEST_B},
            "controller_proof": {"applied": True},
        },
        "wrist_ablation": {"per_mode": per_mode},
    }
    written: dict = {}
    component: dict = {}
    monkeypatch.setattr(interchange_stage, "read_json", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        interchange_stage,
        "_review_artifacts",
        lambda *args, **kwargs: {"mp4": {"sha256": DIGEST_A}},
    )
    monkeypatch.setattr(
        interchange_stage,
        "_write_offline_viewer",
        lambda *args, **kwargs: {"sha256": DIGEST_B},
    )
    monkeypatch.setattr(
        interchange_stage,
        "write_json",
        lambda uri, value, *, directory: written.update(value),
    )
    monkeypatch.setattr(
        interchange_stage,
        "publish_component_record",
        lambda **kwargs: component.update(kwargs),
    )
    args = Namespace(
        run_id="experimental-eval",
        adapted_uri="adapted",
        checkpoint_uri="s3://qualified/checkpoint/",
        episodes="8",
        adapted_artifact_root_uri="s3://run/eval/",
        final_mp4_uri="s3://run/report.mp4",
        final_witness_mp4_uri="s3://run/witness.mp4",
        final_rrd_uri="s3://run/report.rrd",
        final_policy_inputs_uri="s3://run/policy-inputs.npz",
        final_html_uri="s3://run/report.html",
        output_uri=str(tmp_path / "report.json"),
        component_root_uri="s3://run/components",
    )

    result = _experimental_finalize(args)

    assert result == written
    assert result["scope"] == "franka_cube_simulation_evaluation_only"
    assert result["per_mode"]["fresh"]["terminal_success_rate"] == 0.625
    assert result["causal_wrist_benefit_claimed"] is False
    assert result["clean_start_training_claimed"] is False
    assert component["name"] == "policy_interchange_experimental_eval_finalize"

    report["protocol"]["action_selection"]["strategy"] = "first_action"
    with pytest.raises(InterchangeContractError, match="uniform temporal"):
        _experimental_finalize(args)


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
    assert prepared["capabilities"]["deterministic_noise_coupling"] is True
    assert prepared["capabilities"]["repeated_paired_trials"] is False
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


def test_non_franka_finalize_binds_protocol_reload_and_withholds_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace

    checkpoint_uri = "s3://private-run/checkpoints/step-10000"
    protocol = {
        "schema": "npa.sim2real.policy_interchange.non_franka_execution.v1",
        "robot_identity": DIGEST_A,
        "object_identity": DIGEST_B,
        "task_identity": "surface-placement-v1",
        "seed": 17,
        "decisions": [{"step": 0, "action": [0.0] * 7}],
        "success_contract": [
            "contact",
            "grasp",
            "lift",
            "transport",
            "release",
            "support_contact",
            "stable",
        ],
        "execution_valid": True,
        "task_success": False,
    }
    baseline = {
        **protocol,
        "service_provenance": {
            "source_commit": "c" * 40,
            "image_digest": f"sha256:{DIGEST_A}",
            "checkpoint_sha256": DIGEST_A,
        },
    }
    adapted = {
        **protocol,
        "task_success": True,
        "service_provenance": {
            "source_commit": "c" * 40,
            "image_digest": f"sha256:{DIGEST_A}",
            "checkpoint_sha256": DIGEST_B,
        },
        "visual_evidence": ["s3://private-run/evidence/adapted.mp4"],
        "teacher": {"visual_evidence": ["s3://private-run/evidence/teacher.mp4"]},
    }
    payloads = {
        "baseline": baseline,
        "adapted": adapted,
        "training": {"selected_checkpoint_uri": checkpoint_uri},
        "reload": {
            "schema": "npa.sim2real.smolvla_reload_validation.v1",
            "status": "native_checkpoint_reloaded",
            "interchange": {"checkpoint_sha256": DIGEST_B},
        },
        "external": {"status": "SEAM", "executed": False},
    }
    written: dict = {}
    component: dict = {}
    monkeypatch.setattr(
        interchange_stage,
        "read_json",
        lambda uri, *, directory: payloads[uri],
    )
    monkeypatch.setattr(
        interchange_stage,
        "write_json",
        lambda uri, value, *, directory: written.update(value),
    )
    monkeypatch.setattr(
        interchange_stage,
        "publish_component_record",
        lambda **kwargs: component.update(kwargs),
    )

    result = _non_franka_finalize(
        Namespace(
            run_id="run-non-franka",
            baseline_uri="baseline",
            adapted_uri="adapted",
            training_uri="training",
            reload_uri="reload",
            checkpoint_uri=checkpoint_uri,
            external_seam_uri="external",
            output_uri=str(tmp_path / "report.json"),
            component_root_uri="s3://private-run/components",
        )
    )

    assert result == written
    assert result["baseline_task_success"] is False
    assert result["adapted_task_success"] is True
    assert result["improvement_claimed"] is False
    assert result["checkpoint"]["sha256"] == DIGEST_B
    assert component["stage"] == 14
    assert component["name"] == "policy_interchange_non_franka_finalize"

    payloads["adapted"] = {
        **adapted,
        "service_provenance": {
            **adapted["service_provenance"],
            "source_commit": "d" * 40,
        },
    }
    with pytest.raises(InterchangeContractError, match="sealed protocol/runtime"):
        _non_franka_finalize(
            Namespace(
                run_id="run-non-franka",
                baseline_uri="baseline",
                adapted_uri="adapted",
                training_uri="training",
                reload_uri="reload",
                checkpoint_uri=checkpoint_uri,
                external_seam_uri="external",
                output_uri=str(tmp_path / "report.json"),
                component_root_uri="s3://private-run/components",
            )
        )
