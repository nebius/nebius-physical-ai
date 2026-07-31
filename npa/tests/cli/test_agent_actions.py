"""Tier-0/1 tests for the bounded agentic tool-calling loop (agent_actions).

All tests inject a deterministic fake planner and fake tool executors, so they
spend zero tokens and touch no infra/GPU.
"""

from __future__ import annotations

import json

from npa.cli import agent_actions as A


def _completion(obj: dict) -> dict:
    """Wrap a planner decision in a chat-completion-shaped response."""
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(obj)}}],
        "usage": {"total_tokens": 7},
    }


def _scripted_planner(script):
    """Return a model_call that yields successive scripted planner decisions."""
    calls = {"n": 0}

    def _call(messages, *, tier="cheap"):
        idx = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        return _completion(script[idx])

    return _call


def test_allowlist_contains_readonly_and_gated_tools():
    assert A.is_allowed("sim_viz_status")
    assert A.is_allowed("workflow_validate_spec")
    assert A.is_allowed("sim2real_submit")
    assert not A.is_allowed("rm_rf_everything")
    # sim2real_submit is the GPU-spending gated tool; status tools are not.
    assert A.requires_confirmation("sim2real_submit")
    assert not A.requires_confirmation("sim_viz_status")


def test_readonly_tool_runs_and_produces_final_answer():
    planner = _scripted_planner(
        [
            {"thought": "check status", "tool": "sim_viz_status", "args": {}},
            {"thought": "done", "final": "**stage**: `demo`"},
        ]
    )
    tools = {"sim_viz_status": lambda args: {"run_id": "r1", "stage": "demo"}}
    result = A.run_action_loop(
        "what is the current status", tools=tools, model_call=planner
    )
    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == ["sim_viz_status"]
    assert "demo" in result["reply"]
    # step trace shape
    assert result["steps"][0]["tool"] == "sim_viz_status"
    assert result["steps"][0]["status"] == "ok"
    assert result["steps"][-1]["phase"] == "final"
    assert result["tokens"] == 14  # two planner calls x 7 tokens


def test_allowlist_enforcement_rejects_unknown_tool_without_executing():
    executed = {"count": 0}

    def _boom(args):  # pragma: no cover - must never run
        executed["count"] += 1
        return {"ok": True}

    planner = _scripted_planner(
        [
            {"tool": "danger_delete", "args": {}},
            {"final": "recovered without the disallowed tool"},
        ]
    )
    # danger_delete is not wired and not allowed; sim_viz_status is available.
    tools = {"sim_viz_status": _boom}
    result = A.run_action_loop("do a thing", tools=tools, model_call=planner)
    assert executed["count"] == 0
    rejected = [s for s in result["steps"] if s.get("status") == "rejected"]
    assert rejected and rejected[0]["tool"] == "danger_delete"
    assert result["stopped_reason"] == A.STOP_DONE


def test_confirmation_gate_blocks_gpu_action_without_token():
    submitted = {"count": 0}

    def _submit(args):  # pragma: no cover - must never run without token
        submitted["count"] += 1
        return {"run_id": "x"}

    planner = _scripted_planner([{ "tool": "sim2real_submit", "args": {"run_id": "x"}}])
    tools = {"sim2real_submit": _submit}
    result = A.run_action_loop(
        "launch a sim2real run", tools=tools, model_call=planner
    )
    assert submitted["count"] == 0
    assert result["needs_confirmation"] is True
    assert result["stopped_reason"] == A.STOP_NEEDS_CONFIRMATION
    assert result["proposed_action"]["tool"] == "sim2real_submit"
    assert result["proposed_action"]["args"] == {"run_id": "x"}
    # The proposal carries the action digest the confirmation token binds to.
    assert result["proposed_action"]["digest"] == A.action_digest(
        {"tool": "sim2real_submit", "args": {"run_id": "x"}}
    )


def test_confirmation_gate_executes_with_matching_token():
    submitted = {"count": 0}

    def _submit(args):
        submitted["count"] += 1
        return {"run_id": args.get("run_id"), "submit_mode": "agent-local"}

    planner = _scripted_planner(
        [
            {"tool": "sim2real_submit", "args": {"run_id": "x"}},
            {"final": "submitted run x"},
        ]
    )
    tools = {"sim2real_submit": _submit}
    result = A.run_action_loop(
        "launch a sim2real run",
        tools=tools,
        model_call=planner,
        confirm_token="tok-123",
        session_token="tok-123",
    )
    assert submitted["count"] == 1
    assert result["stopped_reason"] == A.STOP_DONE
    assert "sim2real_submit" in result["tools_used"]


def test_confirmation_gate_rejects_mismatched_token():
    assert not A.confirmation_ok("a", "b")
    assert not A.confirmation_ok("", "b")
    assert not A.confirmation_ok("a", "")
    assert A.confirmation_ok("same", "same")


def test_confirm_token_bound_to_action_digest():
    submitted = {"count": 0}

    def _submit(args):
        submitted["count"] += 1
        return {"run_id": args.get("run_id")}

    # Token is valid, but the digest was issued for a *different* action, so the
    # gated tool must not execute — it re-proposes instead.
    planner = _scripted_planner([{ "tool": "sim2real_submit", "args": {"run_id": "x"}}])
    tools = {"sim2real_submit": _submit}
    result = A.run_action_loop(
        "launch run x",
        tools=tools,
        model_call=planner,
        confirm_token="tok",
        session_token="tok",
        confirm_digest="mismatch-digest",
    )
    assert submitted["count"] == 0
    assert result["needs_confirmation"] is True

    # Matching digest executes.
    good_digest = A.action_digest({"tool": "sim2real_submit", "args": {"run_id": "x"}})
    planner2 = _scripted_planner(
        [
            {"tool": "sim2real_submit", "args": {"run_id": "x"}},
            {"final": "done"},
        ]
    )
    result2 = A.run_action_loop(
        "launch run x",
        tools={"sim2real_submit": _submit},
        model_call=planner2,
        confirm_token="tok",
        session_token="tok",
        confirm_digest=good_digest,
    )
    assert submitted["count"] == 1
    assert result2["stopped_reason"] == A.STOP_DONE


def test_max_steps_guard_stops_loop():
    # Planner keeps calling a read-only tool forever; guard must stop it.
    planner = _scripted_planner([{ "tool": "health", "args": {}}])
    tools = {"health": lambda args: {"ok": True}}
    result = A.run_action_loop(
        "loop forever", tools=tools, model_call=planner, max_steps=3
    )
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    call_steps = [s for s in result["steps"] if s.get("phase") == "call"]
    assert len(call_steps) == 3


def test_planner_non_json_output_stops_gracefully():
    def _call(messages, *, tier="cheap"):
        return {"choices": [{"message": {"content": "I cannot help"}}], "usage": {}}

    result = A.run_action_loop("x", tools={}, model_call=_call)
    assert result["stopped_reason"] == A.STOP_NO_PLAN
    assert result["ok"] is False


def test_empty_goal_short_circuits():
    result = A.run_action_loop("   ", tools={}, model_call=lambda *a, **k: {})
    assert result["stopped_reason"] == A.STOP_NO_PLAN
    assert result["tokens"] == 0


def test_tool_error_is_recorded_as_observation_and_loop_continues():
    def _flaky(args):
        raise RuntimeError("transient boom")

    planner = _scripted_planner(
        [
            {"tool": "artifacts_runs", "args": {}},
            {"final": "handled the tool error"},
        ]
    )
    tools = {"artifacts_runs": _flaky}
    result = A.run_action_loop("list runs", tools=tools, model_call=planner)
    call_step = result["steps"][0]
    assert call_step["status"] == "error"
    assert "boom" in json.dumps(call_step["observation"])
    assert result["stopped_reason"] == A.STOP_DONE


def test_extract_json_object_handles_fenced_and_embedded():
    assert A._extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert A._extract_json_object('prefix {"tool": "x"} suffix') == {"tool": "x"}
    assert A._extract_json_object("not json") is None


def test_allowlist_contains_readonly_insights_tools():
    for name in ("insights_query", "insights_compare", "insights_lineage", "insights_dashboard"):
        assert A.is_allowed(name)
        # Insights tools observe recorded metrics — read-only, no confirmation gate.
        assert not A.requires_confirmation(name)


def test_loop_uses_insights_query_to_answer_gpu_question():
    captured = {"args": None}

    def _insights_query(args):
        captured["args"] = args
        return {
            "backend": "jsonl",
            "count": 1,
            "records": [{"run_id": "insights-4gpu-viz", "metric_name": "gpus", "value": 4.0}],
        }

    planner = _scripted_planner(
        [
            {
                "thought": "filter runs by gpu count",
                "tool": "insights_query",
                "args": {"metric_name": "gpus", "threshold_metric": "gpus", "threshold_op": "ge", "threshold_value": 4},
            },
            {"thought": "answer", "final": "Runs using >=4 GPUs: `insights-4gpu-viz`."},
        ]
    )
    result = A.run_action_loop(
        "which runs use 4 gpus", tools={"insights_query": _insights_query}, model_call=planner
    )
    assert result["ok"] is True
    assert result["tools_used"] == ["insights_query"]
    assert captured["args"]["threshold_value"] == 4
    assert "insights-4gpu-viz" in result["reply"]


def test_loop_uses_insights_compare_to_answer_regression_question():
    def _insights_compare(args):
        assert args["base_run"] == "r1"
        assert args["candidate_run"] == "r2"
        return {
            "base_run": "r1",
            "candidate_run": "r2",
            "regressed": ["collision_rate"],
            "improved": [],
        }

    planner = _scripted_planner(
        [
            {"tool": "insights_compare", "args": {"base_run": "r1", "candidate_run": "r2"}},
            {"final": "Run `r2` regressed on **collision_rate** vs `r1`."},
        ]
    )
    result = A.run_action_loop(
        "which runs regressed on collision rate",
        tools={"insights_compare": _insights_compare},
        model_call=planner,
    )
    assert result["stopped_reason"] == A.STOP_DONE
    assert "insights_compare" in result["tools_used"]
    assert "collision_rate" in result["reply"]


def test_run_chat_action_loop_shapes_readonly_result():
    planner = _scripted_planner(
        [
            {"tool": "insights_query", "args": {"metric_name": "gpus"}},
            {"final": "grounded answer"},
        ]
    )
    tools = {"insights_query": lambda args: {"count": 1, "records": [{"run_id": "r1"}]}}
    result = A.run_chat_action_loop("which runs use gpus", tools=tools, model_call=planner)
    assert result["mode"] == A.CHAT_ACTION_MODE
    assert result["grounded"] is False
    assert result["tools_used"] == ["insights_query"]
    assert result["steps"], "chat action result must carry a step trace"
    assert result["needs_confirmation"] is False
    assert result["reply"] == "grounded answer"


def test_allowlist_contains_workflow_author_readonly():
    assert A.is_allowed("workflow_author")
    assert not A.requires_confirmation("workflow_author")


def test_loop_authors_workflow_with_repair_then_pass():
    calls = {"n": 0}

    def _author(args):
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt not runnable -> planner repairs and retries.
            return {"ok": False, "runnable": False, "yaml": "", "error": "authored spec did not pass validate+plan"}
        return {
            "ok": True,
            "runnable": True,
            "yaml": "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nstates: {}",
            "tool_refs": ["workbench.cosmos2.transfer"],
            "states": ["augment", "envgen"],
        }

    planner = _scripted_planner(
        [
            {"tool": "workflow_author", "args": {"goal": "2 step cosmos"}},
            {"tool": "workflow_author", "args": {"goal": "2 step cosmos"}},
            {"final": "Here is your workflow:\n```yaml\napiVersion: npa.workflow/v0.0.1\n```"},
        ]
    )
    result = A.run_action_loop(
        "write me a 2 step npa yaml that uses cosmos",
        tools={"workflow_author": _author},
        model_call=planner,
    )
    assert calls["n"] == 2
    assert "workflow_author" in result["tools_used"]
    assert result["stopped_reason"] == A.STOP_DONE


def test_loop_insights_empty_store_reports_no_fabrication():
    def _query(args):
        # Real (empty) store observation — never a canned example fallback.
        return {"backend": "jsonl", "count": 0, "records": []}

    planner = _scripted_planner(
        [
            {"tool": "insights_query", "args": {"metric_name": "gpus", "threshold_op": "ge", "threshold_value": 4}},
            {"final": "No matching runs were found in the insights store."},
        ]
    )
    result = A.run_action_loop(
        "which runs used 4 gpus", tools={"insights_query": _query}, model_call=planner
    )
    assert result["tools_used"] == ["insights_query"]
    blob = json.dumps(result["steps"])
    assert "candidate-4gpu" not in blob and "hardened-4gpu" not in blob
    assert "no matching runs" in result["reply"].lower()


def test_normalize_threshold_op_accepts_common_aliases():
    assert A.normalize_threshold_op(">=") == "ge"
    assert A.normalize_threshold_op(">") == "gt"
    assert A.normalize_threshold_op("at least") == ""  # space form not aliased
    assert A.normalize_threshold_op("at_least") == "ge"
    assert A.normalize_threshold_op("<=") == "le"
    assert A.normalize_threshold_op("==") == "eq"
    assert A.normalize_threshold_op("GT") == "gt"
    assert A.normalize_threshold_op("nonsense") == ""
    assert A.normalize_threshold_op("") == ""


def test_normalize_group_by_clamps_to_allowed_facets():
    assert A.normalize_group_by("tool") == "tool"
    assert A.normalize_group_by("STAGE") == "stage"
    assert A.normalize_group_by("run_id") == "metric_name"  # not a valid facet
    assert A.normalize_group_by("") == "metric_name"


def test_run_chat_action_loop_executes_gated_tool_with_matching_token():
    # F1: a chat turn carrying the minted confirm token (bound to the action
    # digest) executes the gated tool — confirmation symmetry with /api/agent/act.
    submitted = {"n": 0}

    def _submit(args):
        submitted["n"] += 1
        return {"run_id": args.get("run_id"), "submit_mode": "agent-local"}

    digest = A.action_digest({"tool": "sim2real_submit", "args": {"run_id": "x"}})
    planner = _scripted_planner(
        [
            {"tool": "sim2real_submit", "args": {"run_id": "x"}},
            {"final": "submitted run x"},
        ]
    )
    result = A.run_chat_action_loop(
        "launch a sim2real run",
        tools={"sim2real_submit": _submit},
        model_call=planner,
        confirm_token="tok",
        session_token="tok",
        confirm_digest=digest,
    )
    assert submitted["n"] == 1
    assert result["needs_confirmation"] is False
    assert "sim2real_submit" in result["tools_used"]


def test_run_chat_action_loop_gpu_tool_needs_confirmation_without_token():
    submitted = {"count": 0}

    def _submit(args):  # pragma: no cover - must never run from a chat turn
        submitted["count"] += 1
        return {"run_id": "x"}

    planner = _scripted_planner([{ "tool": "sim2real_submit", "args": {"run_id": "x"}}])
    result = A.run_chat_action_loop(
        "launch a sim2real run", tools={"sim2real_submit": _submit}, model_call=planner
    )
    assert submitted["count"] == 0
    assert result["needs_confirmation"] is True
    assert result["stopped_reason"] == A.STOP_NEEDS_CONFIRMATION
    assert result["proposed_action"]["tool"] == "sim2real_submit"
