from copy import deepcopy
from types import SimpleNamespace

import pytest

from npa.workflows.behavior_challenge.train_official_q import (
    OFFICIAL_Q_SOURCE,
    OfficialQObserver,
    recompute_official_q,
    validate_official_q_rows,
    validate_official_q_terminal,
)


def _compute(*, success, now_satisfied_options, initial_satisfied_options):
    if success:
        return 1.0
    scores = []
    for current, initial in zip(
        now_satisfied_options, initial_satisfied_options, strict=True
    ):
        scores.append(
            sum(
                not before and after
                for before, after in zip(initial, current, strict=True)
            )
            / len(current)
        )
    return max(scores)


def _observer(current, success=False):
    accessor = SimpleNamespace(
        success=success,
        get_goal_option_satisfaction=lambda: deepcopy(current),
    )
    metric = SimpleNamespace(
        initial_predicate_states=[[False, False, False]],
        env_accessor=accessor,
    )
    return OfficialQObserver(metric, _compute, OFFICIAL_Q_SOURCE)


def test_records_exact_initial_relative_progress_and_regression():
    rows = []
    for frame, current in enumerate(
        (
            [[False, False, False]],
            [[True, False, False]],
            [[True, True, False]],
            [[False, True, False]],
        )
    ):
        rows.append(_observer(current).observe(frame))
    assert [row["q_score"] for row in rows] == [0.0, 1 / 3, 2 / 3, 1 / 3]
    assert validate_official_q_rows(rows, _compute, OFFICIAL_Q_SOURCE) == rows


def test_success_forces_one_and_terminal_joins_without_mutation():
    row = _observer([[False, False, False]], success=True).observe(0)
    result = {200: {"q_score": {"final": 1.0}, "success": True}}
    before = deepcopy(result)
    validate_official_q_terminal([row], result, 200)
    assert result == before


def test_shape_source_score_and_terminal_mutations_reject():
    row = _observer([[True, False, False]]).observe(0)
    cases = []
    changed = deepcopy(row)
    changed["current_satisfied_options"] = [[True, False]]
    cases.append(changed)
    changed = deepcopy(row)
    changed["source"] = {}
    cases.append(changed)
    changed = deepcopy(row)
    changed["q_score"] = 1.0
    cases.append(changed)
    for changed in cases:
        with pytest.raises(ValueError):
            validate_official_q_rows([changed], _compute, OFFICIAL_Q_SOURCE)
    with pytest.raises(ValueError, match="terminal score"):
        validate_official_q_terminal(
            [row], {200: {"q_score": {"final": 0.0}, "success": False}}, 200
        )


def test_snapshot_copies_masks_and_never_exposes_live_state():
    current = [[True, False, False]]
    observer = _observer(current)
    initial_before = deepcopy(observer.metric.initial_predicate_states)
    current_before = deepcopy(current)
    row = observer.observe(0)
    assert observer.metric.initial_predicate_states == initial_before
    assert current == current_before
    current[0][0] = False
    observer.metric.initial_predicate_states[0][0] = True
    assert row["current_satisfied_options"] == [[True, False, False]]
    assert row["initial_satisfied_options"] == [[False, False, False]]


def test_offline_recompute_matches_pinned_formula_without_simulator_imports():
    inputs = {
        "success": False,
        "now_satisfied_options": [[True, False, True]],
        "initial_satisfied_options": [[False, False, True]],
    }
    assert recompute_official_q(**inputs) == _compute(**inputs) == 1 / 3
    assert recompute_official_q(**{**inputs, "success": True}) == 1.0
