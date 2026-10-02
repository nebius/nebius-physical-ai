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


@pytest.mark.parametrize("value", ["false", "true"])
def test_canary_rejects_mutually_truthy_success_evidence(value: str) -> None:
    report = _report(stable=True)
    report["per_env"][0]["success"] = value
    report["per_env"][0]["details"]["placement_stable"] = value
    with pytest.raises(ValueError, match="row success must be a literal boolean"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )


def test_canary_rejects_non_boolean_placement_stability() -> None:
    report = _report(stable=True)
    report["per_env"][0]["details"]["placement_stable"] = "true"
    with pytest.raises(ValueError, match="placement_stable must be a literal boolean"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("policy_inference_provenance", "loaded_for_inference"),
        ("policy_inference_provenance", "actor_is_learned"),
        ("policy_inference_provenance", "scripted_post_actor_controller"),
        ("scenario_input_provenance", "content_addressed"),
    ],
)
def test_canary_rejects_non_boolean_provenance(section: str, field: str) -> None:
    report = _report(stable=True)
    report[section][field] = "false"
    with pytest.raises(ValueError, match=f"{field} must be a literal boolean"):
        assess_placement_report(
            report,
            checkpoint_uri=report["policy_checkpoint"],
            expected_scenarios=2,
        )


@pytest.mark.parametrize("field", ["reach", "contact", "stable_grasp", "lift", "place"])
def test_canary_rejects_non_boolean_decomposed_stage(field: str) -> None:
    report = _report(stable=True)
    report["per_env"][0]["details"][field] = "false"
    with pytest.raises(ValueError, match=f"decomposed stage {field}"):
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


@pytest.mark.parametrize(
    "value",
    [None, True, False, "0.01", -0.01, float("nan"), float("inf"), -float("inf")],
)
def test_canary_rejects_invalid_verdict_distance(value):
    report = _report(stable=True)
    report["per_env"][0]["details"]["object_goal_distance_m"] = value
    with pytest.raises(ValueError, match="object_goal_distance_m"):
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
        )


@pytest.mark.parametrize("field", ["scenario_count", "size_bytes"])
@pytest.mark.parametrize(
    "value", [None, True, False, "2", 2.0, 2.5, 0, -1, float("nan"), float("inf")]
)
def test_canary_rejects_invalid_provenance_counts(field, value):
    report = _report(stable=True)
    report["scenario_input_provenance"][field] = value
    with pytest.raises(ValueError, match=field):
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
        )


@pytest.mark.parametrize("value", [None, True, "2", 2.0, 0, -1])
def test_canary_rejects_invalid_expected_count(value):
    report = _report(stable=True)
    with pytest.raises(ValueError, match="expected_scenarios"):
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=value
        )


def test_canary_fallback_names_actual_key_and_preserves_null():
    report = _report(stable=True)
    details = report["per_env"][0]["details"]
    del details["placement_stable"]
    assert (
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
        )["strict_stable_placements"]
        == 1
    )
    details["place"] = "true"
    with pytest.raises(ValueError, match="canary place must"):
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
        )
    details["place"] = True
    details["placement_stable"] = None
    with pytest.raises(ValueError, match="canary placement_stable must"):
        assess_placement_report(
            report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
        )


@pytest.mark.parametrize(
    "distance,success", [(0, True), (0.049, True), (0.05, False), (1, False)]
)
def test_canary_preserves_strict_distance_boundary(distance, success):
    report = _report(stable=True)
    report["per_env"][0]["details"]["object_goal_distance_m"] = distance
    report["per_env"][0]["success"] = success
    result = assess_placement_report(
        report, checkpoint_uri=report["policy_checkpoint"], expected_scenarios=2
    )
    assert result["credible_placement_signal"] is success


@pytest.mark.parametrize("value", [True, "2", 2.0, 0, -1])
def test_canary_rejects_bad_count_before_storage_or_gpu(monkeypatch, tmp_path, value):
    from npa.workflows.sim2real import placement_canary

    def unexpected(**kwargs):
        pytest.fail("invalid ingress reached storage")

    monkeypatch.setattr(placement_canary, "_validation_rows", unexpected)
    with pytest.raises(ValueError, match="scenario_count"):
        placement_canary.run_validation_canary(
            run_id="test",
            checkpoint_uri="s3://bucket/model.pt",
            validation_envs_uri="s3://bucket/validation",
            gold_envs_uri="s3://bucket/gold",
            output_json=tmp_path / "report.json",
            scenario_count=value,
        )
    assert not (tmp_path / "report.json").exists()
