"""Synthetic planner protocol regressions, not live model convergence evidence."""

import json
import pytest

from npa.agent_backend import actions as A


def completion(plan):
    return {
        "choices": [{"message": {"content": json.dumps(plan)}}],
        "usage": {"total_tokens": 7},
    }


def catalog():
    return {"tool_refs": [f"synthetic-platform.tool-{i:03d}" for i in range(116)]}


def parsed_observations(messages):
    parsed = []
    malformed = []
    for line in messages[1]["content"].splitlines():
        if not line.startswith("{"):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            malformed.append(line)
    return parsed, malformed


def test_bounded_catalog_survives_prompt_as_complete_valid_json():
    observation = catalog()
    assert 1200 < len(json.dumps(observation, sort_keys=True)) <= 4000
    observed = A._observe(observation)
    assert observed == observation
    messages = A._planner_messages(
        "Enumerate supported capabilities.",
        A.TOOL_ALLOWLIST,
        [{"tool": "tools_catalog", "result": observed}],
    )
    parsed, malformed = parsed_observations(messages)
    assert malformed == []
    assert parsed == [{"tool": "tools_catalog", "result": observation}]
    assert len(parsed[0]["result"]["tool_refs"]) == 116


def test_successful_duplicate_does_not_claim_failed_or_missing_observation():
    observations = [
        {"tool": "health", "result": {"ok": True}},
        {
            "tool": "health",
            "rejected": "already completed",
            "replan_required": True,
            "replan_reason": "already_completed",
        },
    ]
    system = A._planner_messages(
        "Inspect service state.", A.TOOL_ALLOWLIST, observations
    )[0]["content"]
    assert "already completed successfully" in system
    assert "returned no usable observation" not in system
    assert "previous tool call failed" not in system
    assert "If more facts are needed" in system


def test_old_rejection_does_not_describe_later_success_as_failed():
    observations = [
        {
            "tool": "health",
            "error": "synthetic unavailable",
            "replan_required": True,
            "replan_reason": "tool_error",
        },
        {"tool": "sim_viz_status", "result": {"ready": True}},
    ]
    system = A._planner_messages(
        "Inspect service state.", A.TOOL_ALLOWLIST, observations
    )[0]["content"]
    assert "previous tool call failed" not in system
    assert "previous proposal did not advance" not in system
    assert "Its earlier observation remains available" not in system


@pytest.mark.parametrize(
    "goal",
    [
        "Check whether the service is available and enumerate its supported capabilities.",
        "Report endpoint availability together with the feature inventory.",
        "What checks can this installation perform, and is the status probe successful?",
    ],
)
def test_complete_observations_allow_synthetic_planner_to_finish_without_extra_budget(
    goal,
):
    history = []
    tools_called = []
    expected = catalog()

    def planner(messages, *, tier):
        history.append(messages)
        parsed, _malformed = parsed_observations(messages)
        results = {row["tool"]: row["result"] for row in parsed if "result" in row}
        if "health" not in results:
            return completion({"tool": "health", "args": {}})
        if "tools_catalog" not in results:
            return completion({"tool": "tools_catalog", "args": {}})
        # A deterministic protocol consumer can answer only from the complete JSON.
        return completion(
            {
                "final": "ok="
                + json.dumps(results["health"]["ok"])
                + "; tool_refs="
                + json.dumps(results["tools_catalog"]["tool_refs"])
            }
        )

    def health(_args):
        tools_called.append("health")
        return {"ok": True}

    def inventory(_args):
        tools_called.append("tools_catalog")
        return expected

    result = A.run_action_loop(
        goal, tools={"health": health, "tools_catalog": inventory}, model_call=planner
    )
    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE
    assert len(history) == 3 <= A.DEFAULT_MAX_STEPS
    assert tools_called == ["health", "tools_catalog"]
    assert result["reply"] == "ok=true; tool_refs=" + json.dumps(expected["tool_refs"])
    assert result["tokens"] == 21


def test_general_non_catalog_structured_observation_is_preserved():
    data = {
        "available": True,
        "samples": [{"name": f"sensor-{i:03d}", "unit": "cm"} for i in range(70)],
    }
    assert 1200 < len(json.dumps(data, sort_keys=True)) <= 4000
    messages = A._planner_messages(
        "Read a sensor summary.",
        A.TOOL_ALLOWLIST,
        [{"tool": "independent_inventory", "result": A._observe(data)}],
    )
    parsed, malformed = parsed_observations(messages)
    assert malformed == []
    assert parsed[0]["result"] == data


def test_real_failure_has_replanning_instruction_and_no_automatic_success():
    calls = []

    def planner(messages, *, tier):
        calls.append(messages)
        return completion({"tool": "health", "args": {}})

    result = A.run_action_loop(
        "Inspect availability.",
        tools={"health": lambda args: {"error": "synthetic failure"}},
        model_call=planner,
    )
    assert result["ok"] is False
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    assert result["steps"][0]["status"] == "error"
    assert len(calls) == A.DEFAULT_MAX_STEPS
    assert "failure" in calls[1][0]["content"]


def test_additional_requested_fact_is_not_satisfied_by_health_and_catalog():
    script = [
        {"tool": "health", "args": {}},
        {"tool": "tools_catalog", "args": {}},
        {"final": "premature"},
        {"tool": "sim2real_status", "args": {"run_id": "synthetic-run"}},
        {"final": "state=running"},
    ]
    actual = []

    def planner(messages, *, tier):
        return completion(script.pop(0))

    def status(args):
        actual.append(args)
        return {"state": "running"}

    result = A.run_action_loop(
        "Use health, tools_catalog and sim2real_status to report the run state.",
        tools={
            "health": lambda _: {"ok": True},
            "tools_catalog": lambda _: catalog(),
            "sim2real_status": status,
        },
        model_call=planner,
    )
    assert result["ok"] is True and result["reply"] == "state=running"
    assert actual == [{"run_id": "synthetic-run"}]
    assert any(
        step.get("replan_reason") == "required_tools_remaining"
        for step in result["steps"]
    )


def test_unrelated_extra_goal_does_not_trigger_automatic_completion():
    plans = [
        {"tool": "health", "args": {}},
        {"tool": "tools_catalog", "args": {}},
        {"tool": "sim2real_submit", "args": {"run_id": "synthetic-run"}},
    ]

    def planner(messages, *, tier):
        return completion(plans.pop(0))

    def forbidden(_args):
        pytest.fail("confirmation gate bypassed")

    result = A.run_action_loop(
        "Inspect capabilities, then submit the requested workload.",
        tools={
            "health": lambda _: {"ok": True},
            "tools_catalog": lambda _: catalog(),
            "sim2real_submit": forbidden,
        },
        model_call=planner,
    )
    assert result["needs_confirmation"] is True
    assert result["stopped_reason"] == A.STOP_NEEDS_CONFIRMATION
    assert result["proposed_action"]["tool"] == "sim2real_submit"


def test_filtered_observation_preserves_only_returned_subset():
    result = {"tool_refs": ["synthetic-filtered-tool"]}
    parsed, malformed = parsed_observations(
        A._planner_messages(
            "Show the filtered subset.",
            A.TOOL_ALLOWLIST,
            [{"tool": "tools_catalog", "result": A._observe(result)}],
        )
    )
    assert malformed == [] and parsed[0]["result"] == result


def test_existing_observation_preview_remains_bounded_and_valid():
    observed = A._observe({"oversized": "x" * 9000})
    assert observed["truncated"] is True
    assert len(observed["preview"]) == 4000
    messages = A._planner_messages(
        "Report observed data.",
        A.TOOL_ALLOWLIST,
        [{"tool": "synthetic", "result": observed}],
    )
    parsed, malformed = parsed_observations(messages)
    assert malformed == [] and parsed[0]["result"] == observed


def test_same_repeated_choices_still_stop_at_original_limit():
    calls = []

    def planner(messages, *, tier):
        calls.append(messages)
        return completion({"tool": "health", "args": {}})

    result = A.run_action_loop(
        "Provide a factual report.",
        tools={"health": lambda _: {"ok": True}},
        model_call=planner,
    )
    assert result["ok"] is False and result["stopped_reason"] == A.STOP_MAX_STEPS
    assert len(calls) == A.DEFAULT_MAX_STEPS
    assert len([step for step in result["steps"] if step["phase"] == "call"]) == 1


def test_escaped_structured_observation_survives_json_serialization():
    data = {"message": 'quoted "value"\n\\path\t' * 90, "unicode": "snowman: ☃"}
    observed = A._observe(data)
    assert observed == data
    parsed, malformed = parsed_observations(
        A._planner_messages(
            "Report the returned message.",
            A.TOOL_ALLOWLIST,
            [{"tool": "health", "result": observed}],
        )
    )
    assert malformed == []
    assert parsed == [{"tool": "health", "result": data}]


def test_catalog_has_zero_arguments_and_is_read_only():
    spec = A.TOOL_ALLOWLIST["tools_catalog"]
    assert spec.read_only is True
    assert spec.requires_confirmation is False
    assert spec.params == ()
    assert (
        A.normalize_action_args("tools_catalog", {"limit": 1}, A.TOOL_ALLOWLIST) == {}
    )


def test_latest_real_failure_after_successful_duplicate_keeps_failure_guidance():
    observations = [
        {"tool": "health", "result": {"ok": True}},
        {
            "tool": "health",
            "replan_required": True,
            "replan_reason": "already_completed",
        },
        {
            "tool": "sim_viz_status",
            "error": "synthetic unavailable",
            "replan_required": True,
            "replan_reason": "tool_error",
        },
    ]
    system = A._planner_messages(
        "Report observed state.", A.TOOL_ALLOWLIST, observations
    )[0]["content"]
    assert "rejection or failure" in system
    assert "identifies any unresolved facts" in system
    assert "already completed successfully" not in system
