"""Unit tests for workbench-hosted Cosmos Reason2/3 helpers."""

from __future__ import annotations

from npa.workbench.cosmos.reason import (
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_MODEL,
    merge_dual_reason_evaluations,
    resolve_cosmos_reason_model_id,
    task_description_from_manifest,
)


def test_resolve_cosmos_reason_alias_defaults_to_reason2() -> None:
    assert (
        resolve_cosmos_reason_model_id("npa-cosmos3-reason")
        == DEFAULT_REASON2_MODEL
    )


def test_task_description_from_manifest_prefers_task_description() -> None:
    manifest = {"task_description": "Pick up the cube.", "task": "ignored"}
    assert task_description_from_manifest(manifest) == "Pick up the cube."


def test_merge_dual_reason_evaluations_averages_scores_and_requires_both_success() -> None:
    reason2 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_REASON2_MODEL,
        "success": True,
        "score": 0.8,
        "per_step": [
            {
                "step": 0,
                "critique_text": "aligned",
                "error_tags": ["ok"],
                "action": [0.0, 0.0, 0.0],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "reason2 ok",
    }
    reason3 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_REASON3_MODEL,
        "success": False,
        "score": 0.4,
        "per_step": [
            {
                "step": 0,
                "critique_text": "late grasp",
                "error_tags": ["late_grasp"],
                "action": [0.0, 0.0, 0.0],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "reason3 miss",
    }

    merged = merge_dual_reason_evaluations(reason2, reason3, threshold=0.75)

    assert merged["dual_reason"] is True
    assert merged["component_source"] == "cosmos_dual_reason_vlm"
    assert merged["score"] == 0.6
    assert merged["success"] is False
    assert merged["per_step"][0]["error_tags"] == ["ok", "late_grasp"]
    assert "reason2_critique" in merged["per_step"][0]
    assert "reason3_critique" in merged["per_step"][0]
