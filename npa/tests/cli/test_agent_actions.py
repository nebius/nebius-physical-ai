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


def test_matching_confirmation_authorizes_only_one_attempt_per_loop():
    submitted = {"count": 0}

    def _submit(args):
        submitted["count"] += 1
        return {"run_id": args.get("run_id"), "submit_mode": "agent-local"}

    planner = _scripted_planner(
        [{"tool": "sim2real_submit", "args": {"run_id": "only-once"}}]
    )
    digest = A.action_digest(
        {"tool": "sim2real_submit", "args": {"run_id": "only-once"}}
    )
    result = A.run_action_loop(
        "Use sim2real_submit exactly once for only-once.",
        tools={"sim2real_submit": _submit},
        model_call=planner,
        confirm_token="tok",
        session_token="tok",
        confirm_digest=digest,
        max_steps=6,
    )

    assert submitted["count"] == 1
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    assert all(step.get("phase") != "call" for step in result["steps"][1:])


def test_confirmed_pending_action_executes_directly_without_replanning():
    submitted = []
    action = {"tool": "sim2real_submit", "args": {"run_id": "bound-run"}}

    def _planner_must_not_run(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("a confirmed action must not be replanned")

    result = A.run_action_loop(
        "Use sim2real_submit once for bound-run.",
        tools={
            "sim2real_submit": lambda args: submitted.append(args)
            or {"run_id": args["run_id"], "submit_mode": "agent-local"}
        },
        model_call=_planner_must_not_run,
        confirm_token="token",
        session_token="token",
        confirm_digest=A.action_digest(action),
        confirmed_action={**action, "digest": A.action_digest(action)},
    )

    assert submitted == [{"run_id": "bound-run"}]
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == ["sim2real_submit"]
    assert len(result["steps"]) == 1


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
    # Planner keeps calling a read-only tool forever; the first call succeeds and
    # repeats are rejected until the hard guard stops the loop.
    planner = _scripted_planner([{ "tool": "health", "args": {}}])
    tools = {"health": lambda args: {"ok": True}}
    result = A.run_action_loop(
        "loop forever", tools=tools, model_call=planner, max_steps=3
    )
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    call_steps = [s for s in result["steps"] if s.get("phase") == "call"]
    assert len(call_steps) == 1
    assert [s.get("replan_reason") for s in result["steps"][1:]] == [
        "already_completed",
        "already_completed",
    ]


def test_explicitly_requested_tools_must_run_before_final_answer():
    planner = _scripted_planner(
        [
            {"final": "too early"},
            {"tool": "insights_query", "args": {}},
            {"final": "still too early"},
            {"tool": "artifacts_run", "args": {"run_id": "run-a"}},
            {"final": "Both requested observations are complete for run-a."},
        ]
    )
    result = A.run_action_loop(
        "Call insights_query and artifacts_run for run-a, then answer.",
        tools={
            "insights_query": lambda args: {"count": 1, "records": [{"run_id": "run-a"}]},
            "artifacts_run": lambda args: {"run_id": args["run_id"], "artifacts": ["report"]},
        },
        model_call=planner,
    )

    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == ["insights_query", "artifacts_run"]
    assert [
        step.get("replan_reason")
        for step in result["steps"]
        if step.get("phase") == "replan"
    ] == ["required_tools_remaining", "required_tools_remaining"]


def test_action_args_are_allowlisted_before_digest_and_execution():
    captured = []
    planner = _scripted_planner(
        [
            {
                "tool": "artifacts_runs",
                "args": {"q": "run-a", "limit": 50, "invented": "digest-evasion"},
            },
            {"final": "found run-a"},
        ]
    )
    result = A.run_action_loop(
        "Use artifacts_runs for run-a.",
        tools={"artifacts_runs": lambda args: captured.append(args) or {"runs": ["run-a"]}},
        model_call=planner,
    )

    assert captured == [{"limit": 50, "q": "run-a"}]
    assert result["steps"][0]["args"] == captured[0]
    assert A.normalize_action_args(
        "artifacts_runs", {"q": "run-a", "unknown": True}
    ) == {"q": "run-a"}


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


def test_tool_error_replans_with_changed_arguments_and_recovers():
    calls = []
    planner_prompts = []

    def _flaky(args):
        calls.append(args)
        if not args.get("prefix"):
            raise RuntimeError("transient boom")
        return {"count": 1, "runs": ["run-recovered"]}

    decisions = iter(
        [
            {"tool": "artifacts_runs", "args": {}},
            {"tool": "artifacts_runs", "args": {"prefix": "runs/"}},
            {"final": "Recovered run: `run-recovered`."},
        ]
    )

    def planner(messages, *, tier="cheap"):
        planner_prompts.append(messages)
        return _completion(next(decisions))

    tools = {"artifacts_runs": _flaky}
    result = A.run_action_loop("list runs", tools=tools, model_call=planner)
    call_step = result["steps"][0]
    assert call_step["status"] == "error"
    assert "boom" in json.dumps(call_step["observation"])
    assert calls == [{}, {"prefix": "runs/"}]
    assert "changed strategy" in planner_prompts[1][0]["content"]
    assert result["replans"] == 1
    assert result["stopped_reason"] == A.STOP_DONE


def test_replan_rejects_exact_failed_action_repeat():
    calls = {"n": 0}

    def _broken(args):
        calls["n"] += 1
        return {"ok": False, "error": "bad prefix"}

    planner = _scripted_planner(
        [
            {"tool": "artifacts_runs", "args": {"prefix": "bad"}},
            {"tool": "artifacts_runs", "args": {"prefix": "bad"}},
            {"final": "Which artifact prefix should I search?"},
        ]
    )
    result = A.run_action_loop(
        "find the run", tools={"artifacts_runs": _broken}, model_call=planner
    )

    assert calls["n"] == 1
    rejected = [step for step in result["steps"] if step.get("phase") == "replan"]
    assert rejected[0]["replan_reason"] == "unchanged_strategy"
    assert result["replans"] == 2
    assert result["stopped_reason"] == A.STOP_DONE


def test_empty_observation_replans_with_an_alternate_tool():
    planner = _scripted_planner(
        [
            {"tool": "insights_query", "args": {"workflow": "missing"}},
            {"tool": "insights_dashboard", "args": {"group_by": "workflow"}},
            {"final": "No runs matched `missing`; the dashboard has 3 records."},
        ]
    )
    result = A.run_action_loop(
        "find runs or summarize what is available",
        tools={
            "insights_query": lambda args: {"count": 0, "records": []},
            "insights_dashboard": lambda args: {
                "total_records": 3,
                "runs": ["run-a"],
            },
        },
        model_call=planner,
    )

    assert result["steps"][0]["status"] == "empty"
    assert result["steps"][0]["replan_reason"] == "empty_observation"
    assert result["tools_used"] == ["insights_query", "insights_dashboard"]
    assert result["replans"] == 1
    assert result["stopped_reason"] == A.STOP_DONE


def test_standalone_empty_query_is_an_honest_terminal_observation():
    planner_calls = {"n": 0}

    def planner(messages, *, tier="cheap"):
        planner_calls["n"] += 1
        if planner_calls["n"] > 1:  # pragma: no cover - terminal empty must stop
            raise AssertionError("terminal empty result spent another planner step")
        return _completion(
            {"tool": "insights_query", "args": {"workflow": "missing"}}
        )

    result = A.run_action_loop(
        "Which runs match workflow missing?",
        tools={"insights_query": lambda args: {"count": 0, "records": []}},
        model_call=planner,
    )

    assert result["ok"] is True
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["reply"] == "No runs found (0 matching records in the store)."
    assert result["replans"] == 0
    assert result["steps"] == [
        {
            "step": 1,
            "phase": "call",
            "tool": "insights_query",
            "args": {"workflow": "missing"},
            "status": "empty",
            "thought": "",
            "observation": {"count": 0, "records": []},
            "terminal_observation": True,
        }
    ]


def test_intermediate_empty_query_recovers_with_changed_arguments():
    calls = []

    def query(args):
        calls.append(args)
        if not args.get("workflow"):
            return {"count": 0, "records": []}
        return {"count": 1, "records": [{"run_id": "run-a"}]}

    planner = _scripted_planner(
        [
            {"tool": "insights_query", "args": {}},
            {"tool": "insights_query", "args": {"workflow": "wf"}},
            {"final": "Found `run-a`; it is ready to compare."},
        ]
    )
    result = A.run_action_loop(
        "Discover a run, then compare it later.",
        tools={"insights_query": query},
        model_call=planner,
    )

    assert calls == [{}, {"workflow": "wf"}]
    assert result["steps"][0]["replan_reason"] == "empty_observation"
    assert result["steps"][1]["status"] == "ok"
    assert result["replans"] == 1
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


def test_loop_uses_readonly_run_memory_to_explain_regression():
    assert A.is_allowed("memory_explain_regression")
    assert not A.requires_confirmation("memory_explain_regression")

    def _explain(args):
        assert args == {"baseline_run": "run-a", "candidate_run": "run-b"}
        return {
            "ok": True,
            "baseline_run": "run-a",
            "candidate_run": "run-b",
            "verdict": "regression",
            "metric_evidence": [
                {
                    "field": "metrics.success_rate",
                    "baseline": 0.85,
                    "candidate": 0.55,
                    "delta": -0.3,
                }
            ],
        }

    planner = _scripted_planner(
        [
            {
                "tool": "memory_explain_regression",
                "args": {"baseline_run": "run-a", "candidate_run": "run-b"},
            },
            {
                "final": (
                    "Stored metrics.success_rate was 0.85 for run-a and 0.55 for "
                    "run-b; observed delta was -0.3."
                )
            },
        ]
    )
    result = A.run_action_loop(
        "why did run-b regress vs run-a",
        tools={"memory_explain_regression": _explain},
        model_call=planner,
    )

    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == ["memory_explain_regression"]
    assert "-0.3" in result["reply"]


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


def test_loop_composes_discover_compare_and_dashboard_in_one_goal():
    def _query(args):
        assert args == {}
        return {
            "count": 2,
            "records": [
                {"run_id": "run-a", "metric_name": "loss", "value": 0.2},
                {"run_id": "run-b", "metric_name": "loss", "value": 0.4},
            ],
        }

    def _compare(args):
        assert args == {
            "base_run": "run-a",
            "candidate_run": "run-b",
            "metric_names": ["loss"],
        }
        return {"base_run": "run-a", "candidate_run": "run-b", "regressed": ["loss"]}

    def _dashboard(args):
        assert args == {"group_by": "metric_name", "latest_run": True}
        return {"total_records": 2, "runs": ["run-a", "run-b"], "groups": {"loss": 2}}

    planner = _scripted_planner(
        [
            {"tool": "insights_query", "args": {}},
            {
                "tool": "insights_compare",
                "args": {
                    "base_run": "run-a",
                    "candidate_run": "run-b",
                    "metric_names": ["loss"],
                },
            },
            {
                "tool": "insights_dashboard",
                "args": {"group_by": "metric_name", "latest_run": True},
            },
            {"final": "`run-b` regressed on `loss`; dashboard total_records is 2."},
        ]
    )
    result = A.run_action_loop(
        "discover two runs, compare loss, and summarize the dashboard",
        tools={
            "insights_query": _query,
            "insights_compare": _compare,
            "insights_dashboard": _dashboard,
        },
        model_call=planner,
    )

    assert result["stopped_reason"] == A.STOP_DONE
    assert result["tools_used"] == [
        "insights_query",
        "insights_compare",
        "insights_dashboard",
    ]
    assert result["replans"] == 0
    assert "run-b" in result["reply"]


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
            {"tool": "workflow_author", "args": {"goal": "2 step cosmos", "steps": 2}},
            {
                "tool": "workflow_author",
                "args": {"goal": "2 step cosmos with validated tool refs", "steps": 3},
            },
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


def test_workflow_author_receives_the_complete_operator_goal() -> None:
    captured = []
    operator_goal = (
        "Use workflow_author for three stages: curate a dataset, train a policy, "
        "then evaluate the policy."
    )
    planner = _scripted_planner(
        [
            {"tool": "workflow_author", "args": {"goal": "train then evaluate"}},
            {"final": "authored all three stages"},
        ]
    )
    result = A.run_action_loop(
        operator_goal,
        tools={
            "workflow_author": lambda args: captured.append(args)
            or {"ok": True, "runnable": True, "states": ["curate", "train", "evaluate"]}
        },
        model_call=planner,
    )

    assert captured[0]["goal"] == operator_goal
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
    assert "no runs found" in result["reply"].lower()


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


# ── Planner robustness against reasoning-model output ────────────────────────
# Regression coverage for a live failure observed against Token Factory
# ``Qwen/Qwen3-32B`` (the cheap planner tier): the model emits a ``<think>`` block
# that contains a JSON-looking snippet, then the real decision object. A greedy
# ``{.*}`` match spanned both and failed to parse, so the loop aborted with
# ``no_plan`` and discarded the observations it had already gathered.

_REASONING_REPLY_WITH_BRACES = """<think>
Okay, the operator wants to know which runs used 4 GPUs. So the parameters would be:

{
  "metric_name": "gpus",
  "threshold_op": "eq",
  "threshold_value": 4
}

Let's proceed with that.
</think>

{"thought": "Query the gpus metric.", "tool": "insights_query", "args": {"metric_name": "gpus", "threshold_op": "eq", "threshold_value": 4}}"""


def test_extract_json_object_ignores_braces_inside_reasoning_trace():
    parsed = A._extract_json_object(_REASONING_REPLY_WITH_BRACES)
    assert parsed is not None
    assert parsed["tool"] == "insights_query"
    assert parsed["args"]["threshold_value"] == 4


def test_extract_json_object_prefers_the_decision_object():
    text = '{"metric_name": "gpus"}\nfinal answer:\n{"thought": "t", "final": "done"}'
    assert A._extract_json_object(text) == {"thought": "t", "final": "done"}


def test_strip_reasoning_trace_handles_truncated_think():
    assert A.strip_reasoning_trace("<think>still thinking about {x}") == ""
    assert A.strip_reasoning_trace("<think>t</think> {\"a\": 1}") == '{"a": 1}'


def test_balanced_json_spans_respects_strings_and_escapes():
    spans = A._balanced_json_spans('{"a": "}"} tail {"b": 2}')
    assert spans == ['{"a": "}"}', '{"b": 2}']


def test_planner_retries_once_on_unparseable_reply():
    calls = {"n": 0}

    def _call(messages, *, tier="cheap"):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {"content": "sorry, prose only"}}], "usage": {}}
        return _completion({"final": "recovered after the nudge"})

    result = A.run_action_loop("x", tools={}, model_call=_call)
    assert calls["n"] == 2, "planner must be re-asked exactly once"
    assert result["stopped_reason"] == A.STOP_DONE
    assert result["reply"] == "recovered after the nudge"


def test_unparseable_plan_reports_empty_store_as_no_runs_found():
    """A terminal empty result must not spend a retry on unusable planner prose."""
    calls = {"n": 0}

    def _call(messages, *, tier="cheap"):
        calls["n"] += 1
        if calls["n"] == 1:
            return _completion({"tool": "insights_query", "args": {"metric_name": "gpus"}})
        return {"choices": [{"message": {"content": "prose, not json"}}], "usage": {}}

    tools = {"insights_query": lambda args: {"backend": "jsonl", "count": 0, "records": []}}
    result = A.run_action_loop("which runs used 4 gpus", tools=tools, model_call=_call)
    assert result["stopped_reason"] == A.STOP_DONE
    assert "no runs found" in result["reply"].lower()
    assert "insights_query" in result["tools_used"]
    assert calls["n"] == 1
    assert not [s for s in result["steps"] if s.get("phase") == "plan"]


def test_summarize_observations_reports_only_what_tools_returned():
    summary = A.summarize_observations(
        [
            {"tool": "insights_query", "result": {"count": 2, "records": [{"run_id": "run-a"}, {"run_id": "run-b"}]}},
            {"tool": "insights_dashboard", "result": {"total_records": 0, "runs": []}},
            {"tool": "health", "result": {"error": "boom"}},
        ]
    )
    assert "run-a" in summary and "run-b" in summary
    assert "no runs found" in summary
    assert "boom" in summary
    assert A.summarize_observations([]) == ""


def test_max_steps_reply_includes_observation_summary():
    planner = _scripted_planner([{"tool": "insights_query", "args": {}}])
    tools = {"insights_query": lambda args: {"count": 0, "records": []}}
    result = A.run_action_loop("x", tools=tools, model_call=planner, max_steps=2)
    assert result["stopped_reason"] == A.STOP_MAX_STEPS
    assert "no runs found" in result["reply"]


# ── Observation bounding must not destroy grounding ──────────────────────────
# Live failure: "which runs regressed on corruption_rate" queried 10 records, the
# observation exceeded the size budget, and the whole result collapsed into a
# {"truncated": true, "preview": "<json prefix>"} string. The planner could no
# longer read any run id, invented the placeholder "<second_run_id>", and spun to
# max_steps. The tool correctly refused the placeholder -- but the turn was lost.


def _fat_record(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "metric_name": "corruption_rate",
        "value": 0.3,
        "unit": "",
        "workflow": "insights-smoke",
        "stage": "validate",
        "tool": "dataset",
        "labels": {},
        "artifact_uri": "s3://bucket/" + "x" * 400,
        "lineage": {"input_uris": ["s3://bucket/" + "y" * 400]},
        "timestamp": "2026-07-31T19:01:21Z",
    }


def test_oversized_record_observation_keeps_run_ids_visible():
    observation = {
        "backend": "jsonl",
        "count": 10,
        "records": [_fat_record(f"run-{i}") for i in range(10)],
    }
    observed = A._observe(observation, limit=1200)
    assert observed.get("records_summarized") is True
    assert observed["count"] == 10
    assert observed["records"], "at least one record must survive"
    kept = [r["run_id"] for r in observed["records"]]
    assert kept[0] == "run-0"
    # Identifying fields survive; the bulky provenance blobs do not.
    assert "artifact_uri" not in observed["records"][0]
    assert observed["records"][0]["metric_name"] == "corruption_rate"
    assert len(json.dumps(observed)) <= 1200


def test_oversized_non_record_observation_still_falls_back_to_preview():
    observed = A._observe({"blob": "z" * 9000}, limit=500)
    assert observed["truncated"] is True
    assert observed["preview"].startswith('{"blob"')


def test_small_observation_is_passed_through_unchanged():
    observation = {"backend": "jsonl", "count": 1, "records": [{"run_id": "r"}]}
    assert A._observe(observation) is observation


def test_planner_prompt_carries_the_grounding_rule():
    """The final answer must be told to copy values verbatim from observations."""
    messages = A._planner_messages("goal", A.TOOL_ALLOWLIST, [])
    system = messages[0]["content"]
    assert "copied verbatim from an observation field" in system
    assert "never sum, average, recompute, or estimate" in system.lower()


def test_single_oversized_record_still_yields_a_run_id():
    """One record with a huge labels blob must not collapse to a text preview."""
    observation = {
        "backend": "jsonl",
        "count": 1,
        "records": [
            {
                "run_id": "run-huge",
                "metric_name": "corruption_rate",
                "value": 0.3,
                "unit": "",
                "workflow": "insights-smoke",
                "stage": "validate",
                "tool": "dataset",
                "labels": {"blob": "z" * 5000},
            }
        ],
    }
    observed = A._observe(observation, limit=600)
    assert observed.get("records_summarized") is True
    assert observed["records"][0]["run_id"] == "run-huge"
    assert "labels" not in observed["records"][0], "bulky fields drop before grounding does"
    assert len(json.dumps(observed)) <= 600


def test_strip_reasoning_trace_matches_token_factory_split_reasoning():
    """Parity guard: the embedded copy must not drift from the shared helper.

    ``agent_actions`` is embedded verbatim into the agent-VM backend and cannot
    import from the wider package, so the logic is duplicated on purpose. This
    pins the two implementations to the same behavior on the shapes that matter.
    """
    from npa.clients.token_factory import split_reasoning

    cases = [
        "<think>reasoning here</think>\n{\"tool\": \"health\"}",
        "<think>braces {\"a\": 1} inside</think> {\"final\": \"done\"}",
        "plain answer with no trace",
    ]
    for content in cases:
        visible, _ = split_reasoning({"content": content})
        assert A.strip_reasoning_trace(content) == visible.strip()

    # Truncated mid-thought: both drop the unusable partial trace.
    truncated = "<think>still thinking about {x}"
    visible, _ = split_reasoning({"content": truncated})
    assert visible == ""
    assert A.strip_reasoning_trace(truncated) == ""
