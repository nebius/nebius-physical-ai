"""Test complete-panel anchored checkpoint selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-anchored-training"
)
SPEC = importlib.util.spec_from_file_location(
    "behavior_anchored_selection", IMPLEMENTATION / "selection.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RULE = MODULE.SelectionRule(
    task_stages=((0, 3),),
    candidate_steps=(1, 2),
    action_dimensions=2,
)


def rows(candidate_loss: float = 0.8):
    result = []
    for step, loss in ((-1, 1.0), (1, candidate_loss), (2, candidate_loss)):
        for frame, stage in ((10, 0), (11, 2)):
            result.append(
                {
                    "step": step,
                    "task_id": 0,
                    "episode_index": 4,
                    "episode_relative_frame": frame,
                    "teacher_stage": stage,
                    "action_loss": loss,
                    "stage_cross_entropy": 0.5,
                    "action_dimension_losses": [loss, loss],
                }
            )
    return result


def test_complete_tied_candidates_choose_earlier_step():
    receipt = MODULE.select(rows(), RULE)
    assert receipt["selected_step"] == 1
    assert receipt["fallback_used"] is False
    assert receipt["status"] == "offline_checkpoint_selected_not_rollout_evaluated"


def test_no_eligible_candidate_falls_back_to_stock():
    receipt = MODULE.select(rows(candidate_loss=2.0), RULE)
    assert receipt["selected_step"] == -1
    assert receipt["fallback_used"] is True


def test_duplicate_or_incomplete_sample_panel_fails_closed():
    complete = rows()
    with pytest.raises(ValueError, match="duplicated"):
        MODULE.select([*complete, complete[0]], RULE)
    with pytest.raises(ValueError, match="sample panel differs"):
        MODULE.select(complete[:-1], RULE)


def test_changed_teacher_stage_fails_closed():
    changed = rows()
    changed[-1]["teacher_stage"] = 1
    with pytest.raises(ValueError, match="teacher stage differs"):
        MODULE.select(changed, RULE)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", True, "step or task"),
        ("task_id", 0.0, "step or task"),
        ("episode_index", -1, "sample identity"),
        ("episode_relative_frame", 10.0, "sample identity"),
        ("teacher_stage", False, "teacher stage"),
        ("action_loss", -0.1, "scalar loss"),
        ("stage_cross_entropy", float("inf"), "scalar loss"),
    ],
)
def test_row_types_and_nonnegative_losses_fail_closed(field, value, message):
    changed = rows()
    changed[0][field] = value
    with pytest.raises(ValueError, match=message):
        MODULE.select(changed, RULE)


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rule_ratios_fail_closed(ratio):
    rule = MODULE.SelectionRule(
        task_stages=((0, 3),),
        candidate_steps=(1, 2),
        action_dimensions=2,
        uniform_flow_ratio=ratio,
    )
    with pytest.raises(ValueError, match="thresholds"):
        MODULE.select(rows(), rule)


def test_historical_rule_does_not_require_positive_late_improvement():
    receipt = MODULE.select(rows(candidate_loss=1.001), RULE)
    assert receipt["selected_step"] == 1
    assert receipt["positive_late_improvement_required"] is False
