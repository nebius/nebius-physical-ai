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
    release_width_threshold,
    validate_dense_episode,
    validate_split_manifest,
    verify_controller_roundtrip,
    width_to_droid_gripper,
)


DEFAULTS = [0.0, -0.4, 0.0, -2.0, 0.0, 2.0, 0.7]
SCALES = [0.5] * 7
TARGET = [0.1, -0.5, 0.2, -2.1, 0.3, 2.1, 0.8, 0.73]
EVENTS = [
    "contact",
    "grasp",
    "lift",
    "transport",
    "release",
    "support_contact",
    "stable",
]


def test_release_width_threshold_tracks_measured_asset_width() -> None:
    assert release_width_threshold(0.048) == pytest.approx(0.053)
    assert release_width_threshold(0.072) == pytest.approx(0.077)
    assert release_width_threshold(0.079) == pytest.approx(0.079)
    with pytest.raises(Pi05ContractError, match="object grasp width"):
        release_width_threshold(0.081)


def _episode_step(index: int) -> dict:
    timestamp = (index + 1) / 15
    image = {"shape": [224, 224, 3], "dtype": "uint8"}
    return {
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
            **image,
            "path": f"exterior/{index:06d}.png",
        },
        "wrist_image_left": {**image, "path": f"wrist/{index:06d}.png"},
    }


def _reset_settle() -> dict:
    return {
        "control_steps": 2,
        "physics_steps": 8,
        "training_row": False,
        "samples": [
            {
                "command_time_s": 0.0,
                "result_time_s": 1.0 / 15.0,
                "object_speed_m_s": 0.08,
                "object_support_force_n": 0.0,
            },
            {
                "command_time_s": 1.0 / 15.0,
                "result_time_s": 2.0 / 15.0,
                "object_speed_m_s": 0.0,
                "object_support_force_n": 0.6,
            },
        ],
    }


def _episode() -> dict:
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": "episode-001",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "object_identity": "cube-red-v1",
        "object_width_m": 0.05,
        "object_height_m": 0.05,
        "initial_object_position_m": [0.45, -0.15, 0.025],
        "target_position_m": [0.50, 0.18, 0.025],
        "active_surface_command_m": [0.50, 0.18, 0.025],
        "reset_settle": _reset_settle(),
        "destination_marker": {
            "semantic_label": "pi05_destination_marker",
            "physics_collision": False,
            "minimum_exterior_semantic_pixels": 16,
        },
        "scene_configuration_digest": "scene-a",
        "scene_seed": 41,
        "steps": [_episode_step(index) for index in range(15)],
        "success_evidence": {
            "ordered_events": EVENTS,
            "released_supported_surface_placement": True,
            "stable_steps": 3,
        },
    }


def test_contract_is_final_byte_bound_and_distinct_from_ppo() -> None:
    contract = build_contract()
    assert_contract(contract)
    assert contract["task_id"] == "Isaac-Surface-PickPlace-Franka-OpenPI-v0"
    assert contract["comparison_scope"]["canonical_ppo"] == "historical_diagnostic_only"
    wrist = contract["observation"]["wrist_image_left"]
    assert wrist["mount"] == "calibrated_live_panda_hand_world_pose"
    assert wrist["renderer_bridge"] == "world_usd_and_fabric"
    assert wrist["mount_translation_m"] == [0.09, 0.0, 0.015]
    assert contract["action"]["joint_order"] == list(JOINT_NAMES)
    expert = contract["action"]["expert_adapter"]
    assert expert["waypoint_frame"] == "panda_hand_origin"
    assert expert["differential_ik_body_offset"].startswith("zero_translation")

    modified = copy.deepcopy(contract)
    modified["action"]["control_hz"] = 30
    with pytest.raises(Pi05ContractError, match="digest"):
        assert_contract(modified)


def test_dense_episode_rejects_initial_pose_outside_configured_reset() -> None:
    episode = _episode()
    episode["initial_object_position_m"] = [0.5908, 0.0677, 0.024]

    with pytest.raises(Pi05ContractError, match="configured range"):
        validate_dense_episode(episode)


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
    assert width_to_droid_gripper(-0.01) == 1.0
    assert width_to_droid_gripper(0.080001) == 0.0
    assert width_to_droid_gripper(0.09) == 0.0
    with pytest.raises(Pi05ContractError, match="measurement tolerance"):
        width_to_droid_gripper(-0.010001)
    with pytest.raises(Pi05ContractError, match="measurement tolerance"):
        width_to_droid_gripper(0.090001)
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
        ([*TARGET[:7], float("inf")], "finite"),
        ([float("nan"), *TARGET[1:]], "finite"),
    ],
)
def test_absolute_bridge_rejects_malformed_or_unsafe_targets(target, error) -> None:
    with pytest.raises(Pi05ContractError, match=error):
        absolute_to_isaac_action(target, defaults=DEFAULTS, scales=SCALES)


def test_absolute_bridge_supports_unclipped_teacher_affine_domain() -> None:
    defaults = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    target = [0.1, 0.632708, 0.2, -2.1, 0.3, 2.1, 0.8, 0.0]

    rendered = absolute_to_isaac_action(target, defaults=defaults, scales=[0.5] * 7)

    assert rendered[1] == pytest.approx(2.403416)
    assert [rendered[i] * 0.5 + defaults[i] for i in range(7)] == pytest.approx(
        target[:7]
    )


def test_gripper_score_is_thresholded_without_clamping() -> None:
    open_target = [*TARGET[:7], -0.2]
    opened = absolute_to_isaac_action(open_target, defaults=DEFAULTS, scales=SCALES)
    closed = absolute_to_isaac_action(
        [*TARGET[:7], 1.2], defaults=DEFAULTS, scales=SCALES
    )
    proof = verify_controller_roundtrip(
        open_target,
        defaults=DEFAULTS,
        scales=SCALES,
        controller_target=TARGET[:7],
        finger_controller_target_m=[0.04, 0.04],
    )

    assert opened[7] == 1.0
    assert closed[7] == -1.0
    assert proof["gripper_prediction_score"] == -0.2


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

    invisible_destination = _episode()
    invisible_destination["destination_marker"]["minimum_exterior_semantic_pixels"] = 0
    with pytest.raises(Pi05ContractError, match="exterior-visible"):
        validate_dense_episode(invisible_destination)


@pytest.mark.parametrize("value", [-0.001, 1.001, float("nan")])
def test_dense_episode_rejects_invalid_gripper_state(value: float) -> None:
    episode = _episode()
    episode["steps"][0]["gripper_position"] = [value]

    with pytest.raises(Pi05ContractError, match="gripper"):
        validate_dense_episode(episode)


def test_dense_episode_rejects_unmeasured_reset_settle_cadence() -> None:
    missing_clock = _episode()
    del missing_clock["reset_settle"]["samples"][0]["command_time_s"]
    with pytest.raises(Pi05ContractError, match="clocks are not finite"):
        validate_dense_episode(missing_clock)

    wrong_cadence = _episode()
    wrong_cadence["reset_settle"]["samples"][1]["result_time_s"] = 0.3
    with pytest.raises(Pi05ContractError, match="cadence is not 15 Hz"):
        validate_dense_episode(wrong_cadence)


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
