"""Contracts for the no-code GR00T LeRobot input declaration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.groot.standard_inputs import (
    SCHEMA_VERSION,
    STANDARD_PROFILE,
    StandardSingleWristConfig,
    load_standard_config,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": STANDARD_PROFILE,
        "dataset": {
            "uri": "s3://operator-input/lerobot-v2",
            "tree_sha256": "a" * 64,
            "source_kind": "private-object-store",
            "source_identifier": "operator/recorded-rollouts",
            "source_revision": "2026-11-10",
            "license": "operator-declared-private",
            "privacy_classification": "private",
            "redistributable_media": False,
            "label_provenance": "operator-observed outcomes",
        },
        "feedback": {
            "uri": "s3://operator-input/labels.json",
            "sha256": "b" * 64,
            "mode": "hybrid",
            "label_protocol": "operator observed task outcome",
            "episode_index_field": "episode_index",
            "split_field": "split",
            "group_field": "group",
            "outcome_field": "success",
            "collection_kind_field": "collection_kind",
            "tags_field": "tags",
            "progress_field": "progress",
            "progress_source": "measured",
            "progress_contract": {
                "units": "normalized progress",
                "direction": "higher is further along",
                "alignment": "one value per recorded frame",
            },
        },
        "mapping": {
            "camera_key": "observation.images.wrist",
            "state_key": "observation.state",
            "action_key": "action",
            "instruction_key": "task_index",
            "state_dimension": 7,
            "action_dimension": 7,
        },
        "action": {
            "kind": "cartesian_relative",
            "frame": "tool",
            "translation_unit": "millimeter",
            "rotation_representation": "axis_angle",
            "rotation_unit": "radian",
            "gripper_fields": ["gripper"],
            "transform_identity": "operator-calibration-2026-11-10",
        },
        "split": {
            "train_episode_count": 3,
            "eval_episode_count": 2,
            "grouping": "collection",
            "strategy": "adapter",
        },
        "evidence": {
            "episode_index": 3,
            "start_frame": 0,
            "frame_count": 12,
            "fps": 15,
        },
        "model": {
            "model_repository": "nvidia/GR00T-N1.5-3B",
            "model_revision": "c" * 40,
            "groot_repository": "https://github.com/NVIDIA/Isaac-GR00T.git",
            "groot_revision": "d" * 40,
        },
    }


def test_standard_config_binds_data_labels_mapping_action_and_models(
    tmp_path: Path,
) -> None:
    path = tmp_path / "standard.json"
    path.write_text(json.dumps(_payload()))

    config = load_standard_config(path)

    assert config.profile == STANDARD_PROFILE
    assert config.mapping.camera_key == "observation.images.wrist"
    assert config.action.translation_unit == "millimeter"
    assert config.feedback.mode == "hybrid"


def test_standard_config_rejects_executable_metadata(tmp_path: Path) -> None:
    payload = _payload()
    payload["adapter_callable"] = "untrusted:run"
    path = tmp_path / "standard.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="adapter_callable"):
        load_standard_config(path)


def test_standard_config_requires_progress_mapping_for_hybrid(tmp_path: Path) -> None:
    payload = _payload()
    feedback = payload["feedback"]
    assert isinstance(feedback, dict)
    feedback.pop("progress_field")
    path = tmp_path / "standard.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="progress feedback"):
        load_standard_config(path)


def test_shipped_standard_profile_has_one_camera_and_one_action_vector() -> None:
    profile = StandardSingleWristConfig()

    assert profile.video_keys == ["video.wrist"]
    assert profile.state_keys == ["state.robot"]
    assert profile.action_keys == ["action.robot"]
    assert profile.action_indices == list(range(16))


def test_checked_in_standard_input_example_matches_the_strict_schema() -> None:
    root = Path(__file__).resolve().parents[3]

    config = load_standard_config(
        root / "workflows/testing/groot-n15-standard-input.example.json"
    )

    assert config.profile == STANDARD_PROFILE
    assert config.dataset.uri.startswith("s3://")
