"""Bounded agentic tool-calling loop for the NPA agent VM backend.

This module implements the *fallthrough* agent loop that runs only after the
grounded intent router (`agent_chat.match_chat_intent` /
`build_grounded_reply`) misses. Grounded, high-frequency turns never enter this
loop, so the zero-token default path is preserved.

Design contract (see `docs/architecture/agent-competitive-plan.md`):

- A small, explicit **tool allowlist** the model may call. Read-only tools run
  freely; state-changing / GPU-spending tools require a confirmation-gate token.
- The loop is: classify -> plan -> call tool -> observe -> decide -> stop, with
  a hard ``max_steps`` guard and a full step trace in the response.
- All side effects (model calls, tool execution) are **injected callables** so
  the pure loop logic unit-tests with zero network/model/GPU access. The VM
  backend wires the real Token Factory client and route handlers; tests inject
  deterministic fakes.

Every function here is pure/deterministic given its injected collaborators. The
module source is embedded verbatim into the agent VM backend by ``agent.py``
(same mechanism as ``agent_chat`` / ``agent_routing``), so it must not import
anything unavailable on the deployed VM.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

# ── Tool allowlist ───────────────────────────────────────────────────────────
# read_only tools observe state and never spend GPU or mutate infra. Tools with
# requires_confirmation are state-changing / GPU-spending and only execute when
# a valid confirmation-gate token accompanies the request.


class ToolSpec:
    """Declarative description of an allowlisted tool.

    ``read_only`` tools can run inside the loop unconditionally. Tools with
    ``requires_confirmation`` propose an action that the operator must confirm
    with a matching gate token before it executes.
    """

    __slots__ = ("name", "read_only", "requires_confirmation", "summary", "params")

    def __init__(
        self,
        name: str,
        *,
        read_only: bool,
        requires_confirmation: bool = False,
        summary: str = "",
        params: Sequence[str] = (),
    ) -> None:
        self.name = str(name)
        self.read_only = bool(read_only)
        self.requires_confirmation = bool(requires_confirmation)
        self.summary = str(summary)
        self.params = tuple(str(p) for p in params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "read_only": self.read_only,
            "requires_confirmation": self.requires_confirmation,
            "summary": self.summary,
            "params": list(self.params),
        }


def _build_allowlist() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec(
            "health",
            read_only=True,
            summary="Backend health + tool_refs count.",
        ),
        ToolSpec(
            "sim_viz_status",
            read_only=True,
            summary="Live sim-viz/status: run_id, stage, rerun_ready, rrd_uri.",
        ),
        ToolSpec(
            "sim2real_status",
            read_only=True,
            summary="Staged Sim2Real monitor status for the active run.",
            params=("run_id",),
        ),
        ToolSpec(
            "artifacts_runs",
            read_only=True,
            summary="Discover S3-backed run prefixes.",
            params=("prefix", "limit"),
        ),
        ToolSpec(
            "artifacts_run",
            read_only=True,
            summary="List artifacts for a specific run_id with render hints.",
            params=("run_id",),
        ),
        ToolSpec(
            "workflow_validate_spec",
            read_only=True,
            summary="Validate an npa.workflow YAML spec (no execution).",
            params=("spec_yaml",),
        ),
        ToolSpec(
            "workflow_plan_spec",
            read_only=True,
            summary="Plan an npa.workflow YAML spec (scheduler plan only).",
            params=("spec_yaml", "run_id"),
        ),
        ToolSpec(
            "retrieval_search",
            read_only=True,
            summary="Retrieve grounded citations from the indexed docs/skills corpus.",
            params=("query", "k"),
        ),
        ToolSpec(
            "sim2real_submit",
            read_only=False,
            requires_confirmation=True,
            summary="Submit/launch a Sim2Real run. GPU-spending — needs confirmation.",
            params=("run_id",),
        ),
    ]
    return {spec.name: spec for spec in specs}


TOOL_ALLOWLIST: dict[str, ToolSpec] = _build_allowlist()

DEFAULT_MAX_STEPS = 6

STOP_DONE = "done"
STOP_MAX_STEPS = "max_steps"
STOP_NEEDS_CONFIRMATION = "needs_confirmation"
STOP_ERROR = "error"
STOP_NO_PLAN = "no_plan"


def allowlist_specs(allowlist: Mapping[str, ToolSpec] | None = None) -> list[dict[str, Any]]:
    """Return the allowlist as JSON-serializable specs (for prompts/inspection)."""
    resolved = allowlist if allowlist is not None else TOOL_ALLOWLIST
    return [spec.to_dict() for spec in resolved.values()]


def is_allowed(tool: str, allowlist: Mapping[str, ToolSpec] | None = None) -> bool:
    resolved = allowlist if allowlist is not None else TOOL_ALLOWLIST
    return str(tool or "") in resolved


def requires_confirmation(tool: str, allowlist: Mapping[str, ToolSpec] | None = None) -> bool:
    resolved = allowlist if allowlist is not None else TOOL_ALLOWLIST
    spec = resolved.get(str(tool or ""))
    return bool(spec and spec.requires_confirmation)


def confirmation_ok(confirm_token: str, session_token: str) -> bool:
    """A confirmation gate opens only on a non-empty exact token match."""
    token = str(confirm_token or "").strip()
    expected = str(session_token or "").strip()
    return bool(token) and bool(expected) and token == expected


def action_digest(action: Any) -> str:
    """Stable short digest binding a confirmation token to a specific action.

    A confirmation token is only valid for the exact tool+args it was issued
    for; if the planner later proposes a *different* gated action, the digest
    will not match and the operator must confirm again. This prevents a token
    issued for one action from authorizing a different (or repeated) one.
    """
    try:
        payload = json.dumps(action or {}, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = str(action)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model output.

    Accepts raw JSON, fenced ```json blocks, or a JSON object embedded in prose.
    Returns ``None`` when nothing parseable is found.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(raw)
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _message_content(data: Any) -> str:
    """Pull assistant text out of a chat-completion-shaped response."""
    if not isinstance(data, dict):
        return ""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _tokens_from(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0
    return int(total)


def _planner_messages(
    goal: str,
    allowlist: Mapping[str, ToolSpec],
    observations: Sequence[dict[str, Any]],
    *,
    live_context: str = "",
) -> list[dict[str, str]]:
    """Assemble the small structured prompt used to pick the next tool."""
    catalog_lines = []
    for spec in allowlist.values():
        gate = " [needs-confirmation]" if spec.requires_confirmation else ""
        ro = "read-only" if spec.read_only else "state-changing"
        params = f" params={list(spec.params)}" if spec.params else ""
        catalog_lines.append(f"- {spec.name} ({ro}){gate}: {spec.summary}{params}")
    system = (
        "You are the NPA workbench action planner. Pick ONE next tool call to make "
        "progress on the operator goal, or finish.\n"
        "Respond with a SINGLE JSON object and nothing else. To call a tool:\n"
        '{\"thought\": \"...\", \"tool\": \"<name>\", \"args\": {...}}\n'
        "To finish with the answer:\n"
        '{\"thought\": \"...\", \"final\": \"<markdown answer grounded in observations>\"}\n'
        "Rules: only call tools from the catalog; prefer read-only tools first; "
        "never claim a run/stage is complete unless an observation confirms it; "
        "state-changing tools will require operator confirmation.\n\n"
        "Tool catalog:\n" + "\n".join(catalog_lines)
    )
    if live_context:
        system += "\n\n" + live_context
    lines = [f"Operator goal: {goal}"]
    if observations:
        lines.append("\nObservations so far:")
        for obs in observations:
            lines.append(json.dumps(obs, sort_keys=True)[:1200])
    else:
        lines.append("\nNo observations yet.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _observe(observation: Any, *, limit: int = 4000) -> Any:
    """Bound the size of a tool observation fed back into the planner."""
    try:
        text = json.dumps(observation, sort_keys=True)
    except (TypeError, ValueError):
        text = str(observation)
    if len(text) <= limit:
        return observation
    return {"truncated": True, "preview": text[:limit]}


def run_action_loop(
    goal: str,
    *,
    tools: Mapping[str, Callable[[dict[str, Any]], Any]],
    model_call: Callable[..., Any],
    confirm_token: str = "",
    session_token: str = "",
    confirm_digest: str = "",
    tier: str = "cheap",
    max_steps: int = DEFAULT_MAX_STEPS,
    allowlist: Mapping[str, ToolSpec] | None = None,
    live_context: str = "",
) -> dict[str, Any]:
    """Run the bounded classify->plan->call->observe->decide->stop loop.

    Parameters
    ----------
    goal:
        The operator goal (last user turn) that fell through the grounded router.
    tools:
        Mapping of tool name -> executor. Executors take an ``args`` dict and
        return any JSON-serializable observation. Only tools present in both
        ``tools`` and the allowlist can run.
    model_call:
        Callable invoked as ``model_call(messages, tier=...)`` returning a
        chat-completion-shaped dict. Injected so tests spend zero tokens.
    confirm_token / session_token:
        Confirmation-gate tokens. A state-changing tool only executes when
        ``confirmation_ok(confirm_token, session_token)`` is True; otherwise the
        loop stops and returns the proposed action for operator confirmation.
    tier:
        Cost tier passed to ``model_call`` (cheap by default; the caller may
        escalate via ``agent_routing.classify_tier``).
    max_steps:
        Hard guard on planner/tool iterations.
    """
    resolved_allow = allowlist if allowlist is not None else TOOL_ALLOWLIST
    steps: list[dict[str, Any]] = []
    tools_used: list[str] = []
    observations: list[dict[str, Any]] = []
    total_tokens = 0
    reply = ""
    stopped_reason = STOP_MAX_STEPS
    needs_confirmation = False
    proposed_action: dict[str, Any] | None = None

    hard_cap = max(1, int(max_steps))
    goal_text = str(goal or "").strip()
    if not goal_text:
        return {
            "ok": False,
            "goal": "",
            "reply": "No goal provided.",
            "steps": [],
            "tools_used": [],
            "stopped_reason": STOP_NO_PLAN,
            "needs_confirmation": False,
            "proposed_action": None,
            "tokens": 0,
            "tier": tier,
        }

    for step_index in range(hard_cap):
        messages = _planner_messages(
            goal_text, resolved_allow, observations, live_context=live_context
        )
        try:
            data = model_call(messages, tier=tier)
        except Exception as exc:  # noqa: BLE001 - surface planner failure as a step
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "plan",
                    "status": "error",
                    "error": f"planner call failed: {exc}",
                }
            )
            stopped_reason = STOP_ERROR
            reply = "Planning failed — the model planner was unavailable."
            break
        total_tokens += _tokens_from(data)
        plan = _extract_json_object(_message_content(data))
        if not isinstance(plan, dict):
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "plan",
                    "status": "error",
                    "error": "planner did not return a JSON object",
                }
            )
            stopped_reason = STOP_NO_PLAN
            reply = "Could not determine a next action from the planner."
            break

        if plan.get("final") is not None and not plan.get("tool"):
            reply = str(plan.get("final") or "").strip()
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "final",
                    "status": "ok",
                    "thought": str(plan.get("thought") or ""),
                }
            )
            stopped_reason = STOP_DONE
            break

        tool = str(plan.get("tool") or "").strip()
        args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
        thought = str(plan.get("thought") or "")

        if not is_allowed(tool, resolved_allow):
            observation = {"error": f"tool '{tool}' is not in the allowlist"}
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "call",
                    "tool": tool,
                    "args": args,
                    "status": "rejected",
                    "thought": thought,
                    "observation": observation,
                }
            )
            observations.append({"tool": tool, "rejected": observation["error"]})
            continue

        if requires_confirmation(tool, resolved_allow):
            proposed = {"tool": tool, "args": args}
            digest = action_digest(proposed)
            token_ok = confirmation_ok(confirm_token, session_token)
            # The token is bound to a specific action digest; a token issued for
            # one action can never authorize a different (or repeated) one.
            digest_ok = (not confirm_digest) or confirm_digest == digest
            if not (token_ok and digest_ok):
                proposed_action = dict(proposed)
                proposed_action["digest"] = digest
                steps.append(
                    {
                        "step": step_index + 1,
                        "phase": "confirm",
                        "tool": tool,
                        "args": args,
                        "status": "needs_confirmation",
                        "thought": thought,
                        "digest": digest,
                    }
                )
                needs_confirmation = True
                stopped_reason = STOP_NEEDS_CONFIRMATION
                reply = (
                    f"Action **{tool}** is GPU-spending / state-changing and needs "
                    "explicit confirmation. Re-send with the confirmation token issued "
                    "for this exact action to execute."
                )
                break

        executor = tools.get(tool)
        if executor is None:
            observation = {"error": f"tool '{tool}' has no executor wired"}
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "call",
                    "tool": tool,
                    "args": args,
                    "status": "error",
                    "thought": thought,
                    "observation": observation,
                }
            )
            observations.append({"tool": tool, "error": observation["error"]})
            continue

        try:
            result = executor(args)
            status = "ok"
            observation = result
        except Exception as exc:  # noqa: BLE001 - tool errors are observations
            status = "error"
            observation = {"error": str(exc)}
        steps.append(
            {
                "step": step_index + 1,
                "phase": "call",
                "tool": tool,
                "args": args,
                "status": status,
                "thought": thought,
                "observation": _observe(observation),
            }
        )
        if tool not in tools_used:
            tools_used.append(tool)
        observations.append({"tool": tool, "result": _observe(observation)})
    else:
        stopped_reason = STOP_MAX_STEPS
        if not reply:
            reply = (
                "Reached the maximum number of steps without a final answer. "
                "Observations gathered are in the step trace."
            )

    ok = stopped_reason in {STOP_DONE, STOP_NEEDS_CONFIRMATION}
    return {
        "ok": ok,
        "goal": goal_text,
        "reply": reply,
        "steps": steps,
        "tools_used": tools_used,
        "stopped_reason": stopped_reason,
        "needs_confirmation": needs_confirmation,
        "proposed_action": proposed_action,
        "tokens": total_tokens,
        "tier": tier,
    }
