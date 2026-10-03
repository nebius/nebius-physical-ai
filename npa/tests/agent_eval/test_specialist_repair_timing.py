"""Prevent overlapping activity, missing receipts and uncertain effects from misleading timing claims."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def timing():
    path = Path(__file__).parents[2] / "examples/specialists/repair_benchmark/timing.py"
    spec = importlib.util.spec_from_file_location("repair_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_model_intervals_are_not_added_as_elapsed_time(timing):
    receipts = {
        "first": {
            "events": [{"type": "context_view", "at": 10}, {"type": "model", "at": 20}]
        },
        "second": {
            "events": [{"type": "context_view", "at": 12}, {"type": "model", "at": 18}]
        },
        "third": {
            "events": [
                {"type": "context_view", "at": 19},
                {"type": "provider_failure", "at": 23},
            ]
        },
    }
    result = timing._model_intervals(receipts)
    assert result["complete"] is True
    assert result["sum_seconds"] == 20
    assert result["occupied_wall_seconds"] == 13


@pytest.mark.parametrize(
    "events,missing",
    [
        ([{"type": "model", "at": 20}], 1),
        ([{"type": "context_view", "at": 10}], 1),
        (
            [
                {"type": "context_view", "at": 10},
                {"type": "context_view", "at": 11},
                {"type": "model", "at": 20},
            ],
            1,
        ),
    ],
)
def test_missing_model_boundaries_remain_incomplete(timing, events, missing):
    result = timing._model_intervals({"task": {"events": events}})
    assert result["complete"] is False
    assert result["unmatched_events"] == missing


def _delegation(call, at, result, goal="Repair"):
    return [
        {
            "type": "tool_started",
            "call_id": call,
            "at": at,
            "name": "delegate",
            "arguments": {"specialist": "physics", "task_id": "stable", "goal": goal},
        },
        {
            "type": "tool",
            "call_id": call,
            "at": at + 1,
            "name": "delegate",
            "result": result,
        },
    ]


@pytest.mark.parametrize("retry_goal,recovered", [("Repair", 1), ("Different task", 0)])
def test_only_the_exact_refused_assignment_counts_as_recovered(
    timing, retry_goal, recovered
):
    busy = {"status": "busy", "submission_attempted": False, "safe_to_retry": True}
    events = _delegation("first", 10, busy)
    events += _delegation("second", 20, {"status": "queued"}, goal=retry_goal)
    result = timing._delegations({"supervisor": {"events": events}})
    assert result["safe_busy_refusals"] == 1
    assert result["same_assignment_retries_accepted"] == recovered
    assert result["unresolved_busy_assignments"] == 1 - recovered


def test_post_submission_failure_is_not_a_safe_lock_refusal(timing):
    events = _delegation("failed", 10, {"ok": False, "error_type": "BlockingIOError"})
    result = timing._delegations({"supervisor": {"events": events}})
    assert result["safe_busy_refusals"] == 0
    assert result["uncertain_or_rejected"] == 1


def test_missing_tool_completion_remains_unknown(timing):
    event = _delegation("unfinished", 10, {})[0]
    result = timing._tool_durations({"task": {"events": [event]}})
    assert result["complete"] is False
    assert result["unmatched_events"] == 1


def test_missing_router_latency_is_not_free(timing):
    result = timing._routing({"router": {"responses": [{"api_call_attempted": True}]}})
    assert result["complete"] is False
    assert result["sum_request_seconds"] is None


@pytest.mark.parametrize("start,end", [(20, 10), (True, 20), (10, float("nan"))])
def test_invalid_clocks_cannot_produce_timings(timing, start, end):
    with pytest.raises(ValueError):
        timing._interval(start, end)


def test_host_wait_cannot_overlap_an_astra_turn(timing, tmp_path):
    (tmp_path / "coordinator-config.json").write_text(
        json.dumps(
            {"turns": [{"phase": "delegate", "started_epoch": 10, "ended_epoch": 20}]}
        )
    )
    (tmp_path / "coordination.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "phase": "host_wait",
                        "model_calls": 0,
                        "started_epoch": 19,
                        "ended_epoch": 30,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        timing._coordinator(tmp_path)


def test_no_receipts_preserves_unknown_timing_without_losing_failed_arm(
    timing, tmp_path
):
    result = timing._timing(tmp_path, {})
    assert result["complete"] is False
    assert "seconds" not in result
    assert str(tmp_path) not in json.dumps(result)


def test_lane_breakdown_distinguishes_unsubmitted_work_from_fast_completion(timing):
    task = {
        "profile": "physics",
        "created": 2,
        "events": [
            {"type": "context_view", "at": 5},
            {"type": "model", "at": 15},
        ],
    }
    lanes = timing._lanes({"task": task})
    assert lanes["physics"]["queue_to_first_model_seconds"] == [3]
    assert lanes["physics"]["observed_task_spans"]["occupied_wall_seconds"] == 13
    assert lanes["publication"]["submitted_tasks"] == 0
    assert lanes["publication"]["queue_to_first_model_seconds"] == []
