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
    assert result["proposed_action"] == {"tool": "sim2real_submit", "args": {"run_id": "x"}}


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
