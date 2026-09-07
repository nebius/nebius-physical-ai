"""A complete empty read answers only the standalone lookup it represents."""

import json

import pytest

from npa.agent_backend import actions as A


def _empty_lookup(goal, tool, observation, *, max_steps=1):
    calls = []

    def model(messages, *, tier):
        calls.append((messages, tier))
        args = {"run_id": "example-run"} if tool == "artifacts_run" else {}
        return {
            "choices": [
                {"message": {"content": json.dumps({"tool": tool, "args": args})}}
            ]
        }

    result = A.run_action_loop(
        goal,
        tools={tool: lambda _args: observation},
        model_call=model,
        max_steps=max_steps,
    )
    return result, calls


@pytest.mark.parametrize(
    "goal,tool,observation",
    [
        ("List runs.", "artifacts_runs", {"runs": []}),
        ("Please show recent runs?", "artifacts_runs", {"runs": [], "count": 0}),
        (
            "Which runs match workflow missing?",
            "insights_query",
            {"records": [], "count": 0},
        ),
        ("Which runs used 4 GPUs?", "insights_query", {"records": [], "count": 0}),
        (
            'Find records for workflow "training and evaluation" and stage train.',
            "insights_query",
            {"records": [], "count": 0},
        ),
        ("Query recorded metrics.", "insights_query", {"records": [], "count": 0}),
        ("Show artifacts for run example-run.", "artifacts_run", {"artifacts": []}),
        ("List runs after 2026-01-01.", "artifacts_runs", {"runs": []}),
        (
            "Can you list runs with prefix example/ or prefix second/?",
            "artifacts_runs",
            {"runs": []},
        ),
    ],
)
def test_empty_standalone_lookup_finishes_without_another_planner_call(
    goal, tool, observation
):
    result, calls = _empty_lookup(goal, tool, observation)
    assert len(calls) == 1
    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["steps"][0]["terminal_observation"] is True
    assert result["replans"] == 0


@pytest.mark.parametrize(
    "goal,tool,observation",
    [
        (
            "List the available tools and report service health.",
            "artifacts_runs",
            {"runs": []},
        ),
        ("What tools can inspect runs?", "artifacts_runs", {"runs": []}),
        ("Show metrics for run example-run.", "artifacts_runs", {"runs": []}),
        ("Find saved artifacts.", "insights_query", {"records": [], "count": 0}),
        ("List runs and report their status.", "artifacts_runs", {"runs": []}),
        ("List runs with service health status.", "artifacts_runs", {"runs": []}),
        (
            "Show artifacts, then explain their formats.",
            "artifacts_run",
            {"artifacts": []},
        ),
        ("List runs; list available metrics.", "artifacts_runs", {"runs": []}),
        (
            "Find runs or try a different data source.",
            "insights_query",
            {"records": []},
        ),
    ],
)
def test_empty_intermediate_or_unrelated_lookup_does_not_complete_goal(
    goal, tool, observation
):
    result, calls = _empty_lookup(goal, tool, observation)
    assert len(calls) == 1
    assert result["ok"] is False
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    assert not result["steps"][0].get("terminal_observation")
    assert result["steps"][0]["replan_reason"] == "empty_observation"


@pytest.mark.parametrize(
    "metadata",
    [
        {"query_complete": False},
        {"pagination_complete": False},
        {"next_cursor": "next-page"},
        {"source_errors": [{"error": "source unavailable"}]},
        {"truncated": True},
        {"partial": True},
        {"has_more": True},
        {"observed_match_count": 1},
        {"count": False},
        {"count": 0.0},
        {"records": [{"run_id": "example-run"}]},
    ],
)
def test_empty_discovery_requires_consistent_complete_result_metadata(metadata):
    result, _calls = _empty_lookup(
        "List runs.", "artifacts_runs", {"runs": [], **metadata}
    )
    assert result["ok"] is False
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    assert not result["steps"][0].get("terminal_observation")


def test_complete_filtered_discovery_can_have_nonmatching_source_runs():
    observation = {
        "runs": [],
        "count": 0,
        "observed_match_count": 0,
        "observed_run_count": 8,
        "total_runs": 0,
        "query_complete": True,
        "pagination_complete": True,
        "next_cursor": "",
        "source_errors": [],
        "truncated": False,
    }
    result, _calls = _empty_lookup(
        "List runs with prefix missing/.", "artifacts_runs", observation
    )
    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE


def test_composite_request_continues_to_factual_tool_observations():
    plans = iter(
        [
            {"tool": "artifacts_runs", "args": {}},
            {"tool": "health", "args": {}},
            {"final": "No stored runs were found. The service health check succeeded."},
        ]
    )
    messages_seen = []

    def model(messages, *, tier):
        messages_seen.append(messages)
        return {"choices": [{"message": {"content": json.dumps(next(plans))}}]}

    result = A.run_action_loop(
        "List stored runs and check service health.",
        tools={
            "artifacts_runs": lambda _args: {"runs": []},
            "health": lambda _args: {"ok": True},
        },
        model_call=model,
        max_steps=3,
    )
    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == ["artifacts_runs", "health"]
    assert result["steps"][0]["replan_reason"] == "empty_observation"
    assert not result["steps"][0].get("terminal_observation")
    assert "health check succeeded" in result["reply"]
    assert len(messages_seen) == 3


@pytest.mark.parametrize(
    "goal,tool,observation,expected",
    [
        (
            "List artifacts for run example-run.",
            "artifacts_run",
            {"artifacts": [], "count": 0},
            "No artifacts found for the requested run.",
        ),
        (
            "Query recorded metrics.",
            "insights_query",
            {"records": [], "count": 0},
            "No matching metrics found (0 matching records in the store).",
        ),
        (
            "Find records for workflow missing.",
            "insights_query",
            {"records": [], "count": 0},
            "No matching records found.",
        ),
    ],
)
def test_terminal_reply_names_the_requested_empty_subject(
    goal, tool, observation, expected
):
    result, calls = _empty_lookup(goal, tool, observation)
    assert len(calls) == 1
    assert result["ok"] is True
    assert result["reply"] == expected
