"""Mocked task-eval harness: run scenarios against the real agent modules.

Every scenario runs against the actual pure modules (agent_chat, agent_actions,
agent_sim2real_loop, agent_semantic_router) with deterministic fake
collaborators, so the suite spends zero real tokens and touches no infra. Each
result records success, step count, and token usage; the harness aggregates a
scorecard (success_rate / avg_steps / avg_tokens).
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from npa.cli import agent_actions
from npa.cli import agent_chat
from npa.cli import agent_retrieval
from npa.cli import agent_semantic_router
from npa.cli import agent_sim2real_loop
from npa.cli import agent_workflow

from .scenarios import SCENARIOS, Scenario
from .policy import scorecard_policy_violations

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


def _module(overrides: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    return (overrides or {}).get(name, default)


def _run_grounded(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    chat = _module(overrides, "agent_chat", agent_chat)
    intent = chat.match_chat_intent(sc.goal)
    reply = chat.build_grounded_reply(intent or "", {}, ["workbench.cosmos.train"]) if intent else ""
    success = intent == sc.expected.get("intent") and bool(reply)
    return EvalResult(sc.id, sc.kind, success, steps=1, tokens=0, detail=f"intent={intent}")


def _run_workflow(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    # End-state: the intent is recognized AND a runnable spec is drafted+validated.
    chat = _module(overrides, "agent_chat", agent_chat)
    workflow = _module(overrides, "agent_workflow", agent_workflow)
    intent = chat.match_chat_intent(sc.goal)
    if intent != sc.expected.get("intent"):
        return EvalResult(sc.id, sc.kind, False, steps=1, tokens=0, detail=f"intent={intent}")
    draft = workflow.generate_workflow_draft(
        intent=intent, user_text=sc.goal, tool_refs=_EVAL_TOOL_REFS
    )
    validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {}
    success = bool(draft.get("runnable")) and bool(validation.get("ok")) and bool(draft.get("yaml"))
    return EvalResult(
        sc.id, sc.kind, success, steps=2, tokens=0, detail=f"template={draft.get('template')}"
    )


def _run_action_loop(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    actions = _module(overrides, "agent_actions", agent_actions)
    expected_tool = sc.expected.get("tool")
    if sc.expected.get("needs_confirmation"):
        planner = _scripted([{ "tool": expected_tool, "args": {"run_id": "eval"}}])
        tools = {expected_tool: lambda args: {"run_id": "eval"}}
        result = actions.run_action_loop(sc.goal, tools=tools, model_call=planner)
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
        result = actions.run_action_loop(sc.goal, tools=tools, model_call=planner)
        success = (
            result.get("stopped_reason") == sc.expected.get("stopped_reason")
            and expected_tool in result.get("tools_used", [])
        )
    steps = len([s for s in result.get("steps", []) if s.get("phase") in {"call", "confirm"}])
    return EvalResult(sc.id, sc.kind, success, steps=steps, tokens=int(result.get("tokens") or 0))


def _run_sim2real_loop(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    sim2real = _module(overrides, "agent_sim2real_loop", agent_sim2real_loop)
    def _status(run_id):
        return {"ok": True, "sim_viz": {"run_id": run_id}, "run": {"run_id": run_id}}

    if sc.expected.get("needs_confirmation"):
        result = sim2real.drive_sim2real_loop(
            sc.goal,
            config={"run_id": "eval", "threshold": 0.8},
            launch=lambda cfg: {"ok": True, "run_id": "eval"},
            status=_status,
            gate=lambda rid, it: {"success_rate": 1.0, "threshold": 0.8},
        )
        success = bool(result.get("needs_confirmation"))
    else:
        result = sim2real.drive_sim2real_loop(
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


def _run_semantic(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    chat = _module(overrides, "agent_chat", agent_chat)
    semantic_router = _module(overrides, "agent_semantic_router", agent_semantic_router)
    expected = sc.expected.get("intent")
    # End-state: the turn resolves to the EXPECTED intent, whether the regex
    # already grounds it or the semantic fallthrough maps the paraphrase.
    regex_intent = chat.match_chat_intent(sc.goal)
    if regex_intent is not None:
        return EvalResult(
            sc.id, sc.kind, regex_intent == expected, steps=1, tokens=0, detail="regex-hit"
        )
    result = semantic_router.classify_intent_semantic(
        sc.goal,
        known_intents=KNOWN_INTENTS,
        model_call=lambda *a, **k: _completion({"intent": "none"}),
    )
    success = result.get("intent") == expected
    return EvalResult(
        sc.id, sc.kind, success, steps=1, tokens=int(result.get("tokens") or 0),
        detail=result.get("source", ""),
    )


def _fake_embed(texts, dim: int = 32):
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for token in str(text).lower().split():
            vec[hash(token) % dim] += 1.0
        vectors.append(vec)
    return vectors


def _run_retrieval(sc: Scenario, overrides: Mapping[str, Any] | None = None) -> EvalResult:
    # End-state: indexing the corpus then retrieving returns a citation whose uri
    # matches the expected source. Fully mocked embedder -> 0 tokens.
    retrieval = _module(overrides, "agent_retrieval", agent_retrieval)
    store = retrieval.InMemoryVectorStore()
    corpus = sc.expected.get("corpus") or []
    documents = [(uri, title, text) for uri, title, text in corpus]
    retrieval.index_corpus(documents, embed=_fake_embed, store=store, source="repo")
    result = retrieval.retrieve(sc.goal, embed=_fake_embed, store=store, k=3, min_score=0.0)
    citations = result.get("citations") or []
    expected_uri = sc.expected.get("uri")
    success = bool(result.get("ok")) and bool(citations) and citations[0].get("uri") == expected_uri
    return EvalResult(sc.id, sc.kind, success, steps=1, tokens=0, detail=f"count={result.get('count')}")


_RUNNERS = {
    "grounded": _run_grounded,
    "workflow": _run_workflow,
    "action_loop": _run_action_loop,
    "sim2real_loop": _run_sim2real_loop,
    "semantic": _run_semantic,
    "retrieval": _run_retrieval,
}


def run_scenario(
    sc: Scenario, *, module_overrides: Mapping[str, Any] | None = None
) -> EvalResult:
    runner = _RUNNERS.get(sc.kind)
    if runner is None:
        return EvalResult(sc.id, sc.kind, False, steps=0, tokens=0, detail="unknown kind")
    try:
        return runner(sc, module_overrides)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed task, not a suite error
        return EvalResult(sc.id, sc.kind, False, steps=0, tokens=0, detail=f"error: {exc}")


def _scenario_identity(scenarios: list[Scenario]) -> dict[str, Any]:
    payload = [
        {"id": sc.id, "kind": sc.kind, "goal": sc.goal, "expected": sc.expected}
        for sc in scenarios
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "scenario_count": len(payload),
        "scenario_ids": [item["id"] for item in payload],
        "scenario_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def run_suite(
    scenarios: list[Scenario] | None = None,
    *,
    module_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = scenarios if scenarios is not None else SCENARIOS
    results = [run_scenario(sc, module_overrides=module_overrides) for sc in cases]
    total = len(results) or 1
    passed = sum(1 for r in results if r.success)
    scorecard = {
        "total": len(results),
        "passed": passed,
        "success_rate": round(passed / float(total), 4),
        "avg_steps": round(sum(r.steps for r in results) / float(total), 4),
        "avg_tokens": round(sum(r.tokens for r in results) / float(total), 4),
        **_scenario_identity(cases),
    }
    return {"results": [r.to_dict() for r in results], "scorecard": scorecard}


_SCORECARD_DIRECTIONS = {
    "success_rate": "higher",
    "avg_steps": "lower",
    "avg_tokens": "lower",
}


def scorecard_regressions(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[str]:
    """Describe competitive-scorecard regressions relative to a committed baseline."""
    regressions = scorecard_policy_violations(current, role="current")
    regressions.extend(scorecard_policy_violations(baseline, role="baseline"))
    for metric, direction in _SCORECARD_DIRECTIONS.items():
        if metric not in current or metric not in baseline:
            regressions.append(f"{metric} is missing from current or baseline scorecard")
            continue
        current_value = float(current[metric])
        baseline_value = float(baseline[metric])
        regressed = (
            current_value < baseline_value
            if direction == "higher"
            else current_value > baseline_value
        )
        if regressed:
            comparator = "below" if direction == "higher" else "above"
            regressions.append(
                f"{metric}={current_value} is {comparator} baseline={baseline_value}"
            )
    return regressions


def assert_scorecard_not_regressed(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    regressions = scorecard_regressions(current, baseline)
    if regressions:
        raise AssertionError("agent eval scorecard regressed: " + "; ".join(regressions))


def _reply_and_tools(response: Mapping[str, Any] | str) -> tuple[str, list[str], int]:
    if isinstance(response, str):
        return response, [], 0
    reply = str(response.get("reply") or response.get("answer") or "")
    tools = [str(tool) for tool in response.get("tools_used", []) if tool]
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return reply, tools, int(usage.get("total_tokens") or 0)


def _observed_run_ids(observation: Mapping[str, Any]) -> list[str]:
    run_ids: list[str] = []
    for record in observation.get("records", []):
        if not isinstance(record, Mapping):
            continue
        run_id = str(record.get("run_id") or "").strip()
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids


def run_operate_eval(
    *,
    run_id: str,
    empty_store_uri: str,
    store_uri: str,
    submit: Callable[[str], Mapping[str, Any]],
    ingest: Callable[[Mapping[str, Any], str, str], Mapping[str, Any]],
    observe: Callable[[str], Mapping[str, Any]],
    ask: Callable[[str], Mapping[str, Any] | str],
) -> dict[str, Any]:
    """Round-trip submit → ingest → ask with every asserted value observed.

    All side effects are injected. CI uses local fixtures and deterministic agent
    summaries; the gated live adapter submits the CPU-only Insights smoke workflow.
    """
    empty_answer = ask(empty_store_uri)
    empty_reply, empty_tools, empty_tokens = _reply_and_tools(empty_answer)
    empty_ok = "no runs found" in empty_reply.lower() and run_id not in empty_reply

    submission = submit(run_id)
    ingestion = ingest(submission, store_uri, run_id)
    observation = observe(store_uri)
    observed_ids = _observed_run_ids(observation)

    populated_answer = ask(store_uri)
    populated_reply, populated_tools, populated_tokens = _reply_and_tools(populated_answer)
    populated_ok = (
        run_id in observed_ids
        and run_id in populated_reply
        and "insights_query" in populated_tools
    )
    return {
        "success": empty_ok and populated_ok,
        "empty": {
            "ok": empty_ok,
            "reply": empty_reply,
            "tools_used": empty_tools,
        },
        "submission": dict(submission),
        "ingestion": dict(ingestion),
        "observation": dict(observation),
        "observed_run_ids": observed_ids,
        "populated": {
            "ok": populated_ok,
            "reply": populated_reply,
            "tools_used": populated_tools,
        },
        "scorecard": {
            "success_rate": 1.0 if empty_ok and populated_ok else 0.0,
            "avg_steps": 4.0,
            "avg_tokens": float(empty_tokens + populated_tokens) / 2.0,
        },
    }
