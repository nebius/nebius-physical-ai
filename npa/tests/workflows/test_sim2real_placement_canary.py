from __future__ import annotations

import pytest

from npa.workflows.sim2real.placement_canary import assess_placement_report


def _report(*, stable: bool, split: str = "validation") -> dict:
    checkpoint = "s3://bucket/checkpoints/model_500.pt"
    rows = []
    for index in range(2):
        row_stable = stable and index == 0
        rows.append(
            {
                "env_id": f"validation-{index}",
                "success": row_stable,
                "details": {
                    "object_goal_distance_m": 0.02 if row_stable else 0.2,
                    "placement_stable": row_stable,
                    "reach": True,
                    "contact": True,
                    "stable_grasp": True,
                    "lift": True,
                    "place": row_stable,
                    "scenario_config_digest": f"digest-{index}",
                },
            }
        )
    return {
        "evaluation_split": split,
        "policy_checkpoint": checkpoint,
        "policy_inference_provenance": {
            "checkpoint_uri": checkpoint,
            "checkpoint_sha256": "a" * 64,
            "loaded_for_inference": True,
            "actor_is_learned": True,
            "scripted_post_actor_controller": False,
            "policy_composition": "learned_actor_only",
            "post_actor_controller": None,
        },
        "component_invocation": {
            "gpu_provenance": {"image_digests": ["registry/isaac@sha256:" + "b" * 64]}
        },
        "scenario_input_provenance": {
            "uri": "s3://bucket/scenario-input/" + "c" * 64 + ".jsonl",
            "sha256": "c" * 64,
            "size_bytes": 4096,
            "scenario_count": 2,
            "transport": "s3_sha256",
            "content_addressed": True,
        },
        "per_env": rows,
    }


def test_canary_accepts_real_validation_stable_placement() -> None:
    report = _report(stable=True)
    result = assess_placement_report(
        report,
        checkpoint_uri=report["policy_checkpoint"],
        expected_scenarios=2,
    )
    assert result["strict_stable_placements"] == 1
    assert result["credible_placement_signal"] is True
    assert result["strict_distance_m"] == 0.05


def test_canary_reports_zero_signal_without_weakening_threshold() -> None:
    report = _report(stable=False)
    result = assess_placement_report(
        report,
        checkpoint_uri=report["policy_checkpoint"],
        expected_scenarios=2,
    )
    assert result["strict_stable_placements"] == 0
    assert result["credible_placement_signal"] is False


def test_canary_rejects_gold_or_distance_only_success() -> None:
    report = _report(stable=True, split="gold_heldout")
    with pytest.raises(ValueError, match="validation"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )
    report = _report(stable=False)
    report["per_env"][0]["success"] = True
    report["per_env"][0]["details"]["object_goal_distance_m"] = 0.01
    with pytest.raises(ValueError, match="strict success"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )


def test_canary_rejects_unpinned_scenario_transport() -> None:
    report = _report(stable=True)
    report["scenario_input_provenance"]["sha256"] = ""
    with pytest.raises(ValueError, match="content-addressed scenario"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )


def test_canary_rejects_any_scripted_post_actor_controller() -> None:
    report = _report(stable=True)
    inference = report["policy_inference_provenance"]
    inference["scripted_post_actor_controller"] = True
    inference["post_actor_controller"] = {"declares_success": False}
    with pytest.raises(ValueError, match="learned-actor-only provenance"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )
