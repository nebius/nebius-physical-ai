"""Verify that visual coverage, disagreement, and ambiguity remain meaningful qualification gates."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from npa.workbench.vlm_eval.temporal import RUBRIC
from npa.workflows.franka_rl import _recipe
from npa.workflows.franka_rl_vlm import summarize_visual


@pytest.fixture
def visual_case():
    recipe = _recipe(SimpleNamespace(run_id="test-visual", seed=42, iterations=1500, num_envs=4096,
                                     eval_episodes=128, minimum_success=0.7))
    recipe["visual_eval"] = {"arms": ["initial", "trained"], "minimum_lift_agreement": 0.8,
        "model": "MiniMaxAI/MiniMax-M3", "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest()}
    rows = []
    for arm in recipe["visual_eval"]["arms"]:
        for condition in recipe["conditions"]:
            for index in range(recipe["capture_episodes"]):
                positive = arm == "trained"
                verdict = {name: {"verdict": "yes" if positive else "no"}
                           for name in ("lifted", "held_at_end")}
                verdict.update(scene_disturbed={"verdict": "no"}, failure_modes=["none"])
                rows.append({"arm": arm, "condition": condition, "capture_index": index,
                    "reference_lifted": positive, "visual": {"backend": "token_factory", "verdict": verdict,
                    "model": recipe["visual_eval"]["model"], "rubric_sha256": recipe["visual_eval"]["rubric_sha256"]}})
    return recipe, rows


def test_perfect_visual_agreement_is_separate_from_robot_readiness(visual_case):
    recipe, rows = visual_case
    summary = summarize_visual(rows, recipe)
    assert summary["episodes"] == 32
    assert summary["lift_balanced_accuracy"] == 1.0
    assert summary["visual_audit_passed"] and summary["visual_task_passed"]
    assert not summary["calibrated_on_independent_human_labels"]
    assert "simulation_qualified" not in summary and "ready_for_robot_deployment" not in summary


def test_false_positive_lifts_cannot_be_hidden_in_mean_score(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["visual"]["verdict"]["lifted"]["verdict"] = "yes"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_sensitivity"] == 1.0 and summary["lift_specificity"] == 0.0
    assert summary["lift_confusion"]["negative_yes"] == 16
    assert not summary["visual_audit_passed"]


def test_uncertain_lifts_count_against_agreement(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["visual"]["verdict"]["lifted"]["verdict"] = "uncertain"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_balanced_accuracy"] == 0.0
    assert not summary["visual_audit_passed"]


@pytest.mark.parametrize("verdict", ["yes", "uncertain"])
def test_scene_disturbance_or_ambiguity_keeps_visual_task_gate_closed(visual_case, verdict):
    recipe, rows = visual_case
    rows[-1]["visual"]["verdict"]["scene_disturbed"]["verdict"] = verdict
    summary = summarize_visual(rows, recipe)
    assert summary["visual_audit_passed"]
    assert not summary["visual_task_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "model", "rubric", "stub", "reference"])
def test_incomplete_or_mixed_visual_evidence_is_rejected(visual_case, mutation):
    recipe, rows = deepcopy(visual_case)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "reference":
        rows[-1]["reference_lifted"] = "true"
    else:
        key, value = {"model": ("model", "substitute"), "rubric": ("rubric_sha256", "x" * 64),
                      "stub": ("backend", "stub")}[mutation]
        rows[-1]["visual"][key] = value
    with pytest.raises(ValueError):
        summarize_visual(rows, recipe)


def test_audit_without_both_reference_classes_is_unqualified(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["reference_lifted"] = False
        row["visual"]["verdict"]["lifted"]["verdict"] = "no"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_balanced_accuracy"] is None
    assert not summary["visual_audit_passed"]
