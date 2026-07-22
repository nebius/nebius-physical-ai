"""Mocked task-eval harness: run scenarios against the real agent modules.

Every scenario runs against the actual pure modules (agent_chat, agent_actions,
agent_sim2real_loop, agent_semantic_router) with deterministic fake
collaborators, so the suite spends zero real tokens and touches no infra. Each
result records success, step count, and token usage; the harness aggregates a
scorecard (success_rate / avg_steps / avg_tokens).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from npa.cli import agent_actions
from npa.cli import agent_chat
from npa.cli import agent_semantic_router
from npa.cli import agent_sim2real_loop
from npa.cli import agent_workflow

from .scenarios import SCENARIOS, Scenario

KNOWN_INTENTS = frozenset(agent_chat.INTENT_APIS.keys())

# Representative workbench toolRefs so workflow drafts validate offline.
_EVAL_TOOL_REFS = frozenset(
    {
        "workbench.rl.policy_train",
        "workbench.rl.evaluate_policy",
        "workbench.cosmos2.transfer",
        "workbench.token_factory.vlm_judge",
        "workbench.lerobot.eval",
        "workbench.sonic.train",
    }
)


@dataclass
class EvalResult:
    id: str
    kind: str
    success: bool
    steps: int
    tokens: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "success": self.success,
            "steps": self.steps,
            "tokens": self.tokens,
            "detail": self.detail,
        }


def _completion(obj: dict, tokens: int = 6) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(obj)}}],
        "usage": {"total_tokens": tokens},
    }


def _scripted(script: list[dict]):
    state = {"n": 0}

    def _call(messages, *, tier="cheap"):
        idx = min(state["n"], len(script) - 1)
        state["n"] += 1
        return _completion(script[idx])

    return _call


def _run_grounded(sc: Scenario) -> EvalResult:
    intent = agent_chat.match_chat_intent(sc.goal)
    reply = agent_chat.build_grounded_reply(intent or "", {}, ["workbench.cosmos.train"]) if intent else ""
    success = intent == sc.expected.get("intent") and bool(reply)
    return EvalResult(sc.id, sc.kind, success, steps=1, tokens=0, detail=f"intent={intent}")


def _run_workflow(sc: Scenario) -> EvalResult:
    # End-state: the intent is recognized AND a runnable spec is drafted+validated.
    intent = agent_chat.match_chat_intent(sc.goal)
    if intent != sc.expected.get("intent"):
        return EvalResult(sc.id, sc.kind, False, steps=1, tokens=0, detail=f"intent={intent}")
    draft = agent_workflow.generate_workflow_draft(
        intent=intent, user_text=sc.goal, tool_refs=_EVAL_TOOL_REFS
    )
    validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {}
    success = bool(draft.get("runnable")) and bool(validation.get("ok")) and bool(draft.get("yaml"))
    return EvalResult(
        sc.id, sc.kind, success, steps=2, tokens=0, detail=f"template={draft.get('template')}"
    )


def _run_action_loop(sc: Scenario) -> EvalResult:
    expected_tool = sc.expected.get("tool")
    if sc.expected.get("needs_confirmation"):
        planner = _scripted([{ "tool": expected_tool, "args": {"run_id": "eval"}}])
        tools = {expected_tool: lambda args: {"run_id": "eval"}}
        result = agent_actions.run_action_loop(sc.goal, tools=tools, model_call=planner)
        success = bool(result.get("needs_confirmation")) and result.get(
            "proposed_action", {}
        ).get("tool") == expected_tool
    else:
        planner = _scripted(
            [
                {"tool": expected_tool, "args": {}},
                {"final": "summarized the status"},
            ]
        )
        tools = {expected_tool: lambda args: {"run_id": "r", "stage": "demo"}}
        result = agent_actions.run_action_loop(sc.goal, tools=tools, model_call=planner)
        success = (
            result.get("stopped_reason") == sc.expected.get("stopped_reason")
            and expected_tool in result.get("tools_used", [])
        )
    steps = len([s for s in result.get("steps", []) if s.get("phase") in {"call", "confirm"}])
    return EvalResult(sc.id, sc.kind, success, steps=steps, tokens=int(result.get("tokens") or 0))


def _run_sim2real_loop(sc: Scenario) -> EvalResult:
    def _status(run_id):
        return {"ok": True, "sim_viz": {"run_id": run_id}, "run": {"run_id": run_id}}

    if sc.expected.get("needs_confirmation"):
        result = agent_sim2real_loop.drive_sim2real_loop(
            sc.goal,
            config={"run_id": "eval", "threshold": 0.8},
            launch=lambda cfg: {"ok": True, "run_id": "eval"},
            status=_status,
            gate=lambda rid, it: {"success_rate": 1.0, "threshold": 0.8},
        )
        success = bool(result.get("needs_confirmation"))
    else:
        result = agent_sim2real_loop.drive_sim2real_loop(
            sc.goal,
            config={"run_id": "eval", "threshold": 0.8},
            launch=lambda cfg: {"ok": True, "run_id": "eval"},
            status=_status,
            gate=lambda rid, it: {"success_rate": 0.95, "threshold": 0.8},
            confirm_token="t",
            session_token="t",
        )
        success = (
            result.get("decision") == sc.expected.get("decision")
            and result.get("stopped_reason") == sc.expected.get("stopped_reason")
        )
    steps = len(result.get("iterations", [])) or 1
    return EvalResult(sc.id, sc.kind, success, steps=steps, tokens=0)


def _run_semantic(sc: Scenario) -> EvalResult:
    expected = sc.expected.get("intent")
    # End-state: the turn resolves to the EXPECTED intent, whether the regex
    # already grounds it or the semantic fallthrough maps the paraphrase.
    regex_intent = agent_chat.match_chat_intent(sc.goal)
    if regex_intent is not None:
        return EvalResult(
            sc.id, sc.kind, regex_intent == expected, steps=1, tokens=0, detail="regex-hit"
        )
    result = agent_semantic_router.classify_intent_semantic(
        sc.goal,
        known_intents=KNOWN_INTENTS,
        model_call=lambda *a, **k: _completion({"intent": "none"}),
    )
    success = result.get("intent") == expected
    return EvalResult(
        sc.id, sc.kind, success, steps=1, tokens=int(result.get("tokens") or 0),
        detail=result.get("source", ""),
    )


_RUNNERS = {
    "grounded": _run_grounded,
    "workflow": _run_workflow,
    "action_loop": _run_action_loop,
    "sim2real_loop": _run_sim2real_loop,
    "semantic": _run_semantic,
}


def run_scenario(sc: Scenario) -> EvalResult:
    runner = _RUNNERS.get(sc.kind)
    if runner is None:
        return EvalResult(sc.id, sc.kind, False, steps=0, tokens=0, detail="unknown kind")
    try:
        return runner(sc)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed task, not a suite error
        return EvalResult(sc.id, sc.kind, False, steps=0, tokens=0, detail=f"error: {exc}")


def run_suite(scenarios: list[Scenario] | None = None) -> dict[str, Any]:
    cases = scenarios if scenarios is not None else SCENARIOS
    results = [run_scenario(sc) for sc in cases]
    total = len(results) or 1
    passed = sum(1 for r in results if r.success)
    scorecard = {
        "total": len(results),
        "passed": passed,
        "success_rate": round(passed / float(total), 4),
        "avg_steps": round(sum(r.steps for r in results) / float(total), 4),
        "avg_tokens": round(sum(r.tokens for r in results) / float(total), 4),
    }
    return {"results": [r.to_dict() for r in results], "scorecard": scorecard}
