from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.workflows.bellboy_wan import (
    BellboyManifestError,
    evaluate_heldout_boundary,
    validate_episode_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SCHEMA_PATH = ROOT / "docs" / "workbench" / "bellboy-episode-manifest-v1.schema.json"
WAN_SPEC = WORKFLOW_DIR / "byof-wan2.2.yaml"
BELLBOY_SPEC = WORKFLOW_DIR / "bellboy-wan2.2-e2e.yaml"


def _manifest(split: str) -> dict[str, object]:
    prefix = f"s3://robot-data/episodes/{split}-0001"
    return {
        "schema": "npa.bellboy.episode_manifest.v1",
        "dataset_id": "hotel-robot-episodes",
        "version": "v1",
        "quality_stats": {
            "record_count": 1,
            "modalities": ["gripper_rgb"],
            "events": ["open-door"],
            "locations": ["hotel-room"],
            "mean_completeness": 1.0,
            "corrupt_count": 0,
            "per_modality_counts": {"gripper_rgb": 1},
        },
        "camera": {
            "modality": "rgb",
            "mount": "gripper",
            "projection": "very-wide-angle",
        },
        "action_schema": {
            "uri": "s3://robot-data/contracts/actions-v1.json",
            "version": "actions-v1",
        },
        "episodes": [
            {
                "episode_id": f"{split}-0001",
                "split": split,
                "task": "open-door",
                "outcome": "failure" if split == "train" else "success",
                "observation": {
                    "gripper_rgb_uri": f"{prefix}/gripper.mp4",
                    "timestamps_uri": f"{prefix}/rgb-time.jsonl",
                },
                "actions": {
                    "uri": f"{prefix}/actions.parquet",
                    "timestamps_uri": f"{prefix}/action-time.jsonl",
                },
                "joint_state": {
                    "uri": f"{prefix}/joints.parquet",
                    "timestamps_uri": f"{prefix}/joint-time.jsonl",
                },
                "timing": {
                    "clock": "monotonic-nanoseconds",
                    "start_ns": 100,
                    "end_ns": 200,
                },
                "recovery": {
                    "attempt": 2,
                    "parent_episode_id": f"{split}-0000",
                    "correction": "regrasp and retry",
                },
            }
        ],
        "records": [
            {
                "record_id": f"{split}-0001-gripper-rgb",
                "modality": "gripper_rgb",
                "uri": f"{prefix}/gripper.mp4",
            }
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_episode_validation_and_heldout_boundary_are_real_and_honest(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.json"
    dataset_validation_path = tmp_path / "dataset-validation.json"
    validation_path = tmp_path / "episode-validation.json"
    heldout_path = tmp_path / "heldout.json"
    wan_path = tmp_path / "wan.json"
    boundary_path = tmp_path / "boundary.json"
    _write_json(train_path, _manifest("train"))
    _write_json(
        dataset_validation_path,
        {
            "schema": "npa.dataset.validation_report.v1",
            "source_manifest_uri": str(train_path),
            "passed": True,
        },
    )
    _write_json(heldout_path, _manifest("heldout"))
    _write_json(
        wan_path,
        {
            "schema": "npa.workbench.byof.wan2_2_ti2v_5b.v1",
            "solution": "wan2.2",
            "capability": "wan2.2_ti2v_5b_text_to_video",
            "output_filename": "wan2_2_ti2v_5b.mp4",
            "output_size_bytes": 8192,
            "observed": {
                "width": 1280,
                "height": 704,
                "frame_count": 17,
                "fps": 24.0,
                "max_spatial_std": 31.0,
                "pixel_range": 255,
                "mean_temporal_abs_delta": 2.5,
            },
            "capabilities_exercised": [
                "wan2.2_ti2v_5b_text_to_video",
                "wan2.2_decoded_mp4_validation",
            ],
            "deferred": [],
        },
    )

    validation = validate_episode_manifest(
        str(train_path), str(validation_path), str(dataset_validation_path), "train"
    )
    assert validation["status"] == "validated"
    assert validation["episode_count"] == 1
    assert validation["recovery_episode_count"] == 1
    assert validation["split_counts"] == {"train": 1}

    boundary = evaluate_heldout_boundary(
        str(heldout_path),
        str(validation_path),
        str(wan_path),
        str(boundary_path),
    )
    assert boundary["status"] == "boundary_verified"
    assert boundary["release_gate"]["satisfied"] is False
    assert (
        boundary["real_robot_boundary"][
            "synthetic_or_generated_video_replaces_heldout_evaluation"
        ]
        is False
    )
    assert {item["capability"] for item in boundary["deferred"]} == {
        "bellboy_action_conditioned_training",
        "bellboy_action_prediction_inference_and_evaluation",
    }
    assert json.loads(boundary_path.read_text(encoding="utf-8")) == boundary


def test_heldout_boundary_rejects_split_leakage_and_fake_action_claim(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    validation_path = tmp_path / "validation.json"
    wan_path = tmp_path / "wan.json"
    output_path = tmp_path / "boundary.json"
    _write_json(manifest_path, _manifest("train"))
    validation = {
        "schema": "npa.bellboy.episode_validation.v1",
        "status": "validated",
    }
    _write_json(validation_path, validation)
    _write_json(
        wan_path,
        {
            "schema": "npa.workbench.byof.wan2_2_ti2v_5b.v1",
            "output_size_bytes": 8192,
            "capabilities_exercised": [
                "bellboy_private_action_prediction",
                "wan2.2_decoded_mp4_validation",
            ],
            "deferred": [],
        },
    )

    with pytest.raises(BellboyManifestError, match="split must equal heldout"):
        evaluate_heldout_boundary(
            str(manifest_path),
            str(validation_path),
            str(wan_path),
            str(output_path),
        )

    _write_json(manifest_path, _manifest("heldout"))
    with pytest.raises(BellboyManifestError, match="stock generation mode"):
        evaluate_heldout_boundary(
            str(manifest_path),
            str(validation_path),
            str(wan_path),
            str(output_path),
        )


def test_episode_validation_requires_the_real_dataset_validator_to_pass(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    dataset_validation_path = tmp_path / "dataset-validation.json"
    output_path = tmp_path / "episode-validation.json"
    _write_json(manifest_path, _manifest("train"))
    _write_json(
        dataset_validation_path,
        {
            "schema": "npa.dataset.validation_report.v1",
            "source_manifest_uri": str(manifest_path),
            "passed": False,
        },
    )

    with pytest.raises(BellboyManifestError, match="validation did not pass"):
        validate_episode_manifest(
            str(manifest_path), str(output_path), str(dataset_validation_path), "train"
        )


def test_manifest_schema_and_workflow_share_the_versioned_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    bellboy = yaml.safe_load(BELLBOY_SPEC.read_text(encoding="utf-8"))
    states = bellboy["states"]

    assert schema["properties"]["schema"]["const"] == "npa.bellboy.episode_manifest.v1"
    assert "quality_stats" in schema["required"]
    assert set(states) == {
        "validate-dataset",
        "validate-episode-contract",
        "generate-stock-wan",
        "record-heldout-boundary",
    }
    assert states["validate-dataset"]["toolRef"] == "workbench.dataset.validate"
    assert states["generate-stock-wan"]["toolRef"] == "workbench.byof.repo"
    assert (
        "evaluate_heldout_boundary" in states["record-heldout-boundary"]["run"]["shell"]
    )
    assert (
        "action-prediction or real-world"
        in states["record-heldout-boundary"]["description"]
    )


def test_bellboy_and_standalone_use_the_same_pinned_wan_workload() -> None:
    standalone = yaml.safe_load(WAN_SPEC.read_text(encoding="utf-8"))["config"]
    bellboy = yaml.safe_load(BELLBOY_SPEC.read_text(encoding="utf-8"))["config"]

    for key in (
        "repo_url",
        "repo_ref",
        "base_profile",
        "base_image",
        "build_command",
        "resource_profile_yaml",
        "capability_name",
        "smoke_artifact_name",
    ):
        assert bellboy[key] == standalone[key], key

    smoke = standalone["smoke_command"]
    build = standalone["build_command"]
    assert "wan.WanTI2V(" in smoke
    assert "generator.generate(" in smoke
    assert "save_video(" in smoke
    assert "cv2.VideoCapture" in smoke
    assert '"1280"' in smoke and '"704"' in smoke
    assert "flash_attn" not in build
    assert "snapshot_download" not in build
    assert "snapshot_download" in smoke
    assert "bellboy_private_action_prediction" in smoke
    assert 'devices[0]["compute_capability"] != [12, 0]' in smoke
    assert '"sm_120" not in torch_cuda_arch_list' in smoke
    assert '"driver_versions": driver_versions' in smoke
    assert "from .attention import attention as flash_attention" in build
    assert "wan_model.flash_attention is wan_attention.attention" in smoke
    assert "{{run.id}}" not in standalone["output_root"]
    for marker in (
        "wan.WanTI2V(",
        "generator.generate(",
        "Wan-AI/Wan2.2-TI2V-5B",
        "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        "wan2.2_ti2v_5b_text_to_video",
        "wan2.2_decoded_mp4_validation",
        "wan2_2_runtime_inventory.json",
        "cv2.VideoCapture",
    ):
        assert marker in bellboy["smoke_command"]
