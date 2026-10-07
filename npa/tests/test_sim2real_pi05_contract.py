from __future__ import annotations

import copy

import pytest

from npa.workflows.sim2real.pi05_contract import (
    DATASET_SCHEMA,
    EPISODE_SCHEMA,
    JOINT_NAMES,
    Pi05ContractError,
    absolute_to_isaac_action,
    assert_contract,
    build_contract,
    droid_gripper_to_width,
    validate_dense_episode,
    validate_split_manifest,
    verify_controller_roundtrip,
    width_to_droid_gripper,
)


DEFAULTS = [0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.7]
SCALES = [0.5] * 7
TARGET = [0.1, -0.5, 0.2, -2.1, 0.3, 2.1, 0.8, 0.73]


def _episode() -> dict:
    steps = []
    for index in range(15):
        timestamp = (index + 1) / 15
        steps.append(
            {
                "step": index,
                "timestamp_s": timestamp,
                "exterior_timestamp_s": timestamp,
                "wrist_timestamp_s": timestamp,
                "command_application_timestamp_s": timestamp,
                "resulting_state_timestamp_s": timestamp + 1 / 15,
                "joint_position": TARGET[:7],
                "gripper_position": [0.0],
                "action": TARGET,
                "exterior_image_1_left": {
                    "path": f"exterior/{index:06d}.png",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
                "wrist_image_left": {
                    "path": f"wrist/{index:06d}.png",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
            }
        )
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": "episode-001",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "object_identity": "cube-red-v1",
        "object_width_m": 0.05,
        "initial_object_position_m": [0.45, -0.15, 0.025],
        "target_position_m": [0.50, 0.18, 0.025],
        "scene_configuration_digest": "scene-a",
        "scene_seed": 41,
        "steps": steps,
        "success_evidence": {
            "ordered_events": [
                "contact",
                "grasp",
                "lift",
                "transport",
                "release",
                "support_contact",
                "stable",
            ],
            "released_supported_surface_placement": True,
            "stable_steps": 3,
        },
    }


def test_contract_is_final_byte_bound_and_distinct_from_ppo() -> None:
    contract = build_contract()
    assert_contract(contract)
    assert contract["task_id"] == "Isaac-Surface-PickPlace-Franka-OpenPI-v0"
    assert contract["comparison_scope"]["canonical_ppo"] == "historical_diagnostic_only"
    assert contract["observation"]["wrist_image_left"]["mount"] == "panda_hand"
    assert contract["action"]["joint_order"] == list(JOINT_NAMES)

    modified = copy.deepcopy(contract)
    modified["action"]["control_hz"] = 30
    with pytest.raises(Pi05ContractError, match="digest"):
        assert_contract(modified)


def test_absolute_bridge_and_physical_gripper_roundtrip() -> None:
    rendered = absolute_to_isaac_action(TARGET, defaults=DEFAULTS, scales=SCALES)
    assert rendered[:7] == pytest.approx([0.2, -0.2, 0.4, -0.2, 0.6, 0.2, 0.2])
    assert rendered[7] == -1.0
    proof = verify_controller_roundtrip(
        TARGET,
        defaults=DEFAULTS,
        scales=SCALES,
        controller_target=TARGET[:7],
        finger_controller_target_m=[0.0, 0.0],
    )
    assert proof["max_controller_target_error_rad"] == 0
    assert proof["thresholded_gripper_target"] == 1.0
    assert proof["expected_observed_gripper_width_m"] == 0.0
    assert width_to_droid_gripper(droid_gripper_to_width(0.25)) == pytest.approx(0.25)
    assert droid_gripper_to_width(0.0) == pytest.approx(0.08)
    assert droid_gripper_to_width(1.0) == pytest.approx(0.0)

    open_target = [*TARGET[:7], 0.42]
    open_rendered = absolute_to_isaac_action(
        open_target, defaults=DEFAULTS, scales=SCALES
    )
    assert open_rendered[7] == 1.0
    open_proof = verify_controller_roundtrip(
        open_target,
        defaults=DEFAULTS,
        scales=SCALES,
        controller_target=TARGET[:7],
        finger_controller_target_m=[0.04, 0.04],
    )
    assert open_proof["thresholded_gripper_target"] == 0.0


@pytest.mark.parametrize(
    "target,error",
    [
        ([0.0] * 7, "exactly 8"),
        ([9.0, *TARGET[1:]], "physical limits"),
        ([*TARGET[:7], 1.5], r"in \[0, 1\]"),
        ([1.0, *TARGET[1:]], "configured Isaac action range"),
    ],
)
def test_absolute_bridge_rejects_padding_clipping_and_ambiguous_gripper(
    target, error
) -> None:
    with pytest.raises(Pi05ContractError, match=error):
        absolute_to_isaac_action(target, defaults=DEFAULTS, scales=SCALES)


def test_dense_episode_requires_real_aligned_wrist_and_ordered_physics_evidence() -> (
    None
):
    assert validate_dense_episode(_episode())["step_count"] == 15

    sparse = _episode()
    sparse["steps"][4]["wrist_timestamp_s"] = sparse["steps"][3]["timestamp_s"]
    with pytest.raises(Pi05ContractError, match="causal"):
        validate_dense_episode(sparse)

    incompatible = _episode()
    incompatible["action_semantics"] = "relative_7d_lift_only"
    incompatible["steps"][0]["action"] = [0.0] * 7
    with pytest.raises(Pi05ContractError, match="7D"):
        validate_dense_episode(incompatible)

    teleported = _episode()
    teleported["object_motion"] = "scripted_object_pose"
    with pytest.raises(Pi05ContractError, match="physics"):
        validate_dense_episode(teleported)


def test_split_manifest_rejects_object_or_scene_leakage() -> None:
    manifest = {
        "schema": DATASET_SCHEMA,
        "splits": {
            "train": [{"object_identity": "red", "scene_configuration_digest": "a"}],
            "validation": [
                {"object_identity": "green", "scene_configuration_digest": "b"}
            ],
            "gold": [{"object_identity": "blue", "scene_configuration_digest": "c"}],
        },
        "normalization": {"source_split": "train", "sha256": "a" * 64},
    }
    assert validate_split_manifest(manifest) == {"train": 1, "validation": 1, "gold": 1}
    manifest["splits"]["gold"][0]["object_identity"] = "red"
    with pytest.raises(Pi05ContractError, match="disjoint"):
        validate_split_manifest(manifest)
