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
module is shipped as ``agent_backend.actions`` on the agent VM; the historical
``npa.cli.agent_actions`` path is a compatibility re-export.
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
            summary=(
                "Discover S3-backed run prefixes. When a run id is known, pass it "
                "as q so discovery searches all runs before applying limit."
            ),
            params=("prefix", "limit", "q"),
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
            "insights_query",
            read_only=True,
            summary=(
                "Query recorded run metrics by facet. When the operator provides a run_id, "
                "query that run_id without metric_name to retrieve all of its metrics. Only "
                "call with NO args when a run must be discovered. For GPU counts set "
                "metric_name='gpus' with threshold_metric='gpus', threshold_op='ge', "
                "threshold_value=N; filter an accelerator type with accelerator='RTXPRO6000'."
            ),
            params=(
                "run_id",
                "workflow",
                "tool",
                "stage",
                "metric_name",
                "accelerator",
                "metric_kind",
                "currency",
                "cost_basis",
                "score_name",
                "threshold_metric",
                "threshold_op",
                "threshold_value",
                "limit",
                "input_uri",
            ),
        ),
        ToolSpec(
            "insights_compare",
            read_only=True,
            summary=(
                "Compare recorded metrics between two runs; flags improved/regressed. Use "
                "for 'which run regressed on <metric>' — set base_run and candidate_run to "
                "run_ids (discover them first via insights_query with no args)."
            ),
            params=("base_run", "candidate_run", "metric_names"),
        ),
        ToolSpec(
            "insights_lineage",
            read_only=True,
            summary="Traverse the provenance graph (ancestors/descendants) of an artifact URI.",
            params=("uri", "version", "direction", "depth"),
        ),
        ToolSpec(
            "insights_dashboard",
            read_only=True,
            summary="Roll up recorded metrics into a grouped dashboard summary.",
            params=("workflow", "group_by", "latest_run"),
        ),
        ToolSpec(
            "memory_explain_regression",
            read_only=True,
            summary=(
                "Explain why a candidate run regressed against a baseline using only "
                "the two records already stored in run memory."
            ),
            params=("baseline_run", "candidate_run"),
        ),
        ToolSpec(
            "workflow_author",
            read_only=True,
            summary=(
                "Author a runnable npa.workflow/v0.0.1 YAML from a goal by composing real "
                "toolRefs from the live catalog, then self-validate + plan (returns yaml only "
                "when runnable). Use for 'write/generate an N-step npa yaml that uses <tool>'."
            ),
            params=("goal", "steps"),
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


def normalize_action_args(
    tool: str,
    args: Mapping[str, Any] | None,
    allowlist: Mapping[str, ToolSpec] | None = None,
) -> dict[str, Any]:
    """Return only non-empty arguments declared by the tool's allowlist spec.

    The planner is untrusted input. Unknown keys must not change an action digest
    or make a repeated action look novel when the executor silently ignores them.
    Keeping normalization here also makes the digest, trace, and executed action
    describe the same operation.
    """
    resolved = allowlist if allowlist is not None else TOOL_ALLOWLIST
    spec = resolved.get(str(tool or ""))
    if spec is None or not isinstance(args, Mapping):
        return {}
    normalized: dict[str, Any] = {}
    for key in spec.params:
        if key not in args:
            continue
        value = args[key]
        if value is None or value == "" or value == [] or value == {}:
            continue
        normalized[key] = value
    return normalized


def requested_tools_from_goal(
    goal: str, allowlist: Mapping[str, ToolSpec] | None = None
) -> list[str]:
    """Find tool names explicitly requested by the operator, in catalog order."""
    resolved = allowlist if allowlist is not None else TOOL_ALLOWLIST
    text = str(goal or "").lower()
    return [
        name
        for name in resolved
        if re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", text)
    ]


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


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def strip_reasoning_trace(text: str) -> str:
    """Drop ``<think>`` reasoning traces from model output.

    Reasoning models on Token Factory (Cosmos 3, Qwen 3 — the cheap planner tier)
    emit a ``<think>...</think>`` block before the answer, and that block routinely
    contains JSON-looking snippets while the model deliberates over tool args. Those
    stray braces must never be mistaken for the planner's actual decision.

    Mirrors ``npa.clients.token_factory.split_reasoning`` semantics; reimplemented
    locally because this module is embedded verbatim into the agent-VM backend and
    cannot import from the wider package.
    """
    raw = str(text or "")
    stripped = _THINK_BLOCK_RE.sub(" ", raw)
    if "<think>" in stripped and "</think>" not in stripped:
        # Truncated mid-thought (finish_reason=length): nothing after it is usable.
        stripped = stripped.split("<think>", 1)[0]
    return stripped.strip()


def _balanced_json_spans(text: str) -> list[str]:
    """Return balanced ``{...}`` spans in order of appearance.

    A greedy ``\\{.*\\}`` match spans from the first brace anywhere in the text to
    the last one, which silently corrupts output that mixes prose (or a reasoning
    trace) with JSON. Brace matching that honors string literals and escapes is the
    only way to recover the real object.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
                start = -1
    return spans


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the planner's JSON object from model output.

    Accepts raw JSON, fenced ```json blocks, or an object embedded in prose or
    trailing a ``<think>`` trace. Candidates are tried reasoning-stripped first and
    last-object-first (the decision is emitted last), and an object carrying
    ``tool``/``final`` wins over an incidental one (e.g. an args dict the model
    echoed while reasoning). Returns ``None`` when nothing parseable is found.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    visible = strip_reasoning_trace(raw)
    candidates: list[str] = []
    for source in (visible, raw):
        if not source:
            continue
        candidates.extend(match.group(1) for match in _FENCED_JSON_RE.finditer(source))
        candidates.append(source)
        candidates.extend(reversed(_balanced_json_spans(source)))
    fallback: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("tool") or parsed.get("final") is not None:
            return parsed
        if fallback is None:
            fallback = parsed
    return fallback


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
    required_tools: Sequence[str] = (),
    completed_tools: Sequence[str] = (),
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
        "state-changing tools will require operator confirmation.\n"
        # Faithfulness rule: measured 4/5 unfaithful scalar answers without it --
        # the planner reported "40" (a metric value that appeared elsewhere in the
        # same observation) when the observation's total_records was 73.
        "Grounding: every number, run id, and URI in your final answer must be "
        "copied verbatim from an observation field. Name the field you took it "
        "from. Never sum, average, recompute, or estimate a value the tools "
        "already returned, and never fill a gap with a plausible-looking value: "
        "if an observation does not contain the answer, say so.\n\n"
        "Argument grounding: when the operator gives an explicit value for a tool "
        "parameter (especially run_id, base_run, or candidate_run), copy that exact "
        "value into the tool arguments; do not substitute a session-context value.\n\n"
        "Tool catalog:\n" + "\n".join(catalog_lines)
    )
    if live_context:
        system += (
            "\n\nThe live session context is a convenience snapshot, not an "
            "exhaustive view of object-storage runs or Insights records. Use the "
            "requested discovery tools when the goal asks for those records.\n\n"
            + live_context
        )
    pending_tools = [name for name in required_tools if name not in completed_tools]
    if required_tools:
        completed = ", ".join(completed_tools) or "none"
        pending = ", ".join(pending_tools) or "none"
        system += (
            "\n\nOperator tool contract: the goal explicitly names tools. Call every "
            f"named tool once with useful arguments before finishing. Completed: {completed}. "
            f"Still required: {pending}. Do not repeat a successfully completed tool; "
            "finish as soon as all named tools have useful observations."
        )
    if any(obs.get("replan_required") for obs in observations):
        system += (
            "\n\nReplanning is required because the previous tool call failed or "
            "returned no usable observation. Choose a changed strategy: adjust "
            "the tool arguments, call a different tool, or finish by asking for a "
            "specific clarifying sub-goal. Do not repeat the same tool with the "
            "same arguments."
        )
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


PLAN_RETRY_NUDGE = (
    "Your previous reply could not be parsed. Reply with ONE JSON object and nothing "
    "else — no prose, no reasoning trace, no code fence. Either "
    '{"thought": "...", "tool": "<name>", "args": {...}} or '
    '{"thought": "...", "final": "<answer grounded in the observations>"}.'
)

NO_PLAN_REPLY = "Could not determine a next action from the planner."


def summarize_observations(observations: Sequence[Mapping[str, Any]]) -> str:
    """Deterministically report what the tools actually returned (no model call).

    Used when the planner cannot produce a final answer: throwing away real tool
    observations and replying with a bare "no plan" is both unhelpful and unsafe —
    it hides the very evidence the answer must be grounded in. This states only what
    the observations contain (an empty result set is reported as "no runs found"),
    so it can never invent a run, metric, or value.
    """
    lines: list[str] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        tool = str(observation.get("tool") or "tool")
        if observation.get("rejected"):
            lines.append(f"- `{tool}`: rejected ({observation['rejected']}).")
            continue
        result = observation.get("result", observation.get("error"))
        if not isinstance(result, Mapping):
            lines.append(f"- `{tool}`: {str(result)[:200]}")
            continue
        if result.get("error"):
            lines.append(f"- `{tool}`: error — {str(result['error'])[:200]}")
            continue
        if "count" in result:
            count = result.get("count")
            raw_records = result.get("records")
            records = raw_records if isinstance(raw_records, list) else []
            run_ids: list[str] = []
            for record in records:
                if isinstance(record, Mapping):
                    run_id = str(record.get("run_id") or "")
                    if run_id and run_id not in run_ids:
                        run_ids.append(run_id)
            if not count:
                lines.append(f"- `{tool}`: no runs found (0 matching records in the store).")
            else:
                detail = f" across runs: {', '.join(run_ids[:10])}" if run_ids else ""
                lines.append(f"- `{tool}`: {count} matching record(s){detail}.")
            continue
        if "total_records" in result:
            total = result.get("total_records")
            runs = result.get("runs") if isinstance(result.get("runs"), list) else []
            if not total:
                lines.append(f"- `{tool}`: store is empty (0 records) — no runs found.")
            else:
                detail = f"; runs: {', '.join(str(r) for r in runs[:10])}" if runs else ""
                lines.append(f"- `{tool}`: {total} record(s) in the store{detail}.")
            continue
        try:
            compact = json.dumps(result, sort_keys=True)
        except (TypeError, ValueError):
            compact = str(result)
        lines.append(f"- `{tool}`: {compact[:400]}")
    if not lines:
        return ""
    return (
        "The planner did not return a usable next step. Here is exactly what the "
        "read-only tools returned:\n" + "\n".join(lines)
    )


# Fields that identify a metric record well enough for the planner to act on it
# (pick a run, compare a pair) without carrying every URI/lineage blob.
_RECORD_SUMMARY_FIELDS = (
    "run_id",
    "metric_name",
    "value",
    "unit",
    "workflow",
    "stage",
    "tool",
    "labels",
)

#: Smallest set that still lets the planner act on a record (pick a run, compare a
#: pair). Used when even one fully-summarized record exceeds the size budget.
_RECORD_IDENTITY_FIELDS = ("run_id", "metric_name", "value")


def _summarize_records(observation: Mapping[str, Any], *, limit: int) -> dict[str, Any] | None:
    """Shrink a record-bearing observation while keeping its structure intact.

    Dropping to a flat text preview is what breaks the planner: it can no longer
    read run ids out of the result, so it either stalls or invents a placeholder
    id. Keeping the identifying fields of as many records as fit preserves the
    grounding the next tool call needs.
    """
    records = observation.get("records")
    if not isinstance(records, list) or not records:
        return None
    base = {key: value for key, value in observation.items() if key != "records"}
    # Widest field set first; the identity-only set is the last line of defence so
    # that even a single record carrying a huge labels blob still yields a readable
    # run id instead of collapsing to a text preview -- the exact failure this
    # summarizer exists to prevent.
    for fields in (_RECORD_SUMMARY_FIELDS, _RECORD_IDENTITY_FIELDS):
        summarized = [
            {field: record.get(field) for field in fields if field in record}
            for record in records
            if isinstance(record, Mapping)
        ]
        if not summarized:
            continue
        kept = len(summarized)
        while kept > 0:
            candidate = dict(base)
            candidate["records"] = summarized[:kept]
            if kept < len(summarized):
                candidate["records_omitted"] = len(summarized) - kept
            candidate["records_summarized"] = True
            try:
                if len(json.dumps(candidate, sort_keys=True)) <= limit:
                    return candidate
            except (TypeError, ValueError):
                return None
            # Halve while there is room to, then try a single record before
            # falling through to the narrower field set.
            kept = kept // 2 if kept > 1 else 0
    return None


def _observe(observation: Any, *, limit: int = 4000) -> Any:
    """Bound the size of a tool observation fed back into the planner."""
    try:
        text = json.dumps(observation, sort_keys=True)
    except (TypeError, ValueError):
        text = str(observation)
    if len(text) <= limit:
        return observation
    if isinstance(observation, Mapping):
        summarized = _summarize_records(observation, limit=limit)
        if summarized is not None:
            return summarized
    return {"truncated": True, "preview": text[:limit]}


def _replan_reason(observation: Any, *, raised: bool = False) -> str:
    """Classify tool output that cannot advance the current plan.

    Empty result sets are observations rather than fabricated successes: the next
    planner step may broaden the query, try another tool, or truthfully finish by
    asking for a narrower sub-goal. Numeric zero and ``False`` remain valid scalar
    observations; only structurally empty containers and explicit empty-store
    shapes trigger replanning.
    """
    if raised:
        return "tool_error"
    if observation is None:
        return "empty_observation"
    if isinstance(observation, str) and not observation.strip():
        return "empty_observation"
    if isinstance(observation, Mapping):
        if observation.get("error"):
            return "tool_error"
        if observation.get("ok") is False:
            return "tool_error"
        if not observation:
            return "empty_observation"
        if observation.get("count") == 0 or observation.get("total_records") == 0:
            return "empty_observation"
        for field in ("records", "runs", "items", "artifacts"):
            values = observation.get(field)
            if field in observation and isinstance(values, Sequence) and not values:
                return "empty_observation"
        return ""
    if isinstance(observation, Sequence) and not observation:
        return "empty_observation"
    return ""


_TERMINAL_EMPTY_TOOLS = frozenset(
    {"artifacts_runs", "artifacts_run", "insights_query", "insights_lineage"}
)
_EMPTY_DOWNSTREAM_RE = re.compile(
    r"\b(?:compare|comparison|dashboard|lineage|then|next|after(?:wards)?|"
    r"recover|retry|broaden|alternate|alternative)\b|"
    r"\b(?:or|and)\s+(?:summari[sz]e|show|find|try)\b",
    re.IGNORECASE,
)
_EMPTY_LOOKUP_GOAL_RE = re.compile(
    r"\b(?:which|what|list|find|show|query|search|match|matching|runs?|records?|"
    r"artifacts?|metrics?|lineage)\b",
    re.IGNORECASE,
)


def _empty_result_is_terminal(
    goal: str,
    tool: str,
    *,
    required_tools: Sequence[str] = (),
    completed_tools: Sequence[str] = (),
) -> bool:
    """Whether an empty read is itself the honest answer to this goal.

    Discovery that feeds a later comparison/dashboard remains an intermediate
    failure and must trigger a changed strategy. A standalone lookup is already
    complete when it truthfully finds zero matches, so it must not spend another
    planner step or blacklist the correct query merely for returning no rows.
    """
    if tool not in _TERMINAL_EMPTY_TOOLS:
        return False
    pending_other_tools = [
        name
        for name in required_tools
        if name != tool and name not in completed_tools
    ]
    if pending_other_tools:
        return False
    goal_text = str(goal or "")
    return bool(_EMPTY_LOOKUP_GOAL_RE.search(goal_text)) and not _EMPTY_DOWNSTREAM_RE.search(
        goal_text
    )


def _terminal_empty_reply(tool: str, observation: Any) -> str:
    """Return a concise, deterministic answer for a terminal empty lookup."""
    if isinstance(observation, Mapping):
        if observation.get("count") == 0 or "records" in observation or "runs" in observation:
            return "No runs found (0 matching records in the store)."
        if observation.get("total_records") == 0:
            return "No runs found (the store contains 0 records)."
        if "artifacts" in observation:
            return "No artifacts found for the requested run."
        if "items" in observation:
            return "No matching items found."
    if tool == "artifacts_run":
        return "No artifacts found for the requested run."
    return "No matching results found."


def run_action_loop(
    goal: str,
    *,
    tools: Mapping[str, Callable[[dict[str, Any]], Any]],
    model_call: Callable[..., Any],
    confirm_token: str = "",
    session_token: str = "",
    confirm_digest: str = "",
    confirmed_action: Mapping[str, Any] | None = None,
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
    failed_actions: dict[str, str] = {}
    successful_actions: set[str] = set()
    required_tools = requested_tools_from_goal(goal, resolved_allow)
    completed_tools: list[str] = []
    confirmation_consumed = False
    replans = 0

    hard_cap = max(1, int(max_steps))
    plan_retry_used = False
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

    # A confirming request executes the exact previously proposed action, not a
    # fresh model plan. Re-planning after confirmation lets the model substitute
    # a different first tool (or never reach the approved action), while one token
    # must authorize exactly one digest-bound attempt.
    if isinstance(confirmed_action, Mapping):
        confirmed_tool = str(confirmed_action.get("tool") or "").strip()
        confirmed_args = normalize_action_args(
            confirmed_tool,
            confirmed_action.get("args")
            if isinstance(confirmed_action.get("args"), Mapping)
            else {},
            resolved_allow,
        )
        normalized_confirmed = {"tool": confirmed_tool, "args": confirmed_args}
        confirmed_digest = action_digest(normalized_confirmed)
        approved = bool(
            confirmation_ok(confirm_token, session_token)
            and confirm_digest
            and confirm_digest == confirmed_digest
            and requires_confirmation(confirmed_tool, resolved_allow)
        )
        executor = tools.get(confirmed_tool)
        if approved and executor is not None:
            raised = False
            try:
                observation = executor(confirmed_args)
            except Exception as exc:  # noqa: BLE001 - preserve tool failure in trace
                raised = True
                observation = {"error": str(exc)}
            replan_reason = _replan_reason(observation, raised=raised)
            status = "error" if replan_reason else "ok"
            observed = _observe(observation)
            return {
                "ok": not replan_reason,
                "goal": goal_text,
                "reply": (
                    f"Confirmed action `{confirmed_tool}` executed exactly once."
                    if not replan_reason
                    else f"Confirmed action `{confirmed_tool}` failed during its single attempt."
                ),
                "steps": [
                    {
                        "step": 1,
                        "phase": "call",
                        "tool": confirmed_tool,
                        "args": confirmed_args,
                        "status": status,
                        "thought": "execute the exact digest-bound confirmed action",
                        "observation": observed,
                        **({"replan_reason": replan_reason} if replan_reason else {}),
                    }
                ],
                "tools_used": [confirmed_tool],
                "stopped_reason": STOP_ERROR if replan_reason else STOP_DONE,
                "needs_confirmation": False,
                "proposed_action": None,
                "tokens": 0,
                "tier": tier,
                "replans": 0,
            }

    for step_index in range(hard_cap):
        messages = _planner_messages(
            goal_text,
            resolved_allow,
            observations,
            live_context=live_context,
            required_tools=required_tools,
            completed_tools=completed_tools,
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
        if not isinstance(plan, dict) and not plan_retry_used:
            # One bounded corrective re-ask before abandoning the turn: a single
            # unparseable planner reply must not discard the observations already
            # gathered. Costs at most one extra cheap call per loop.
            plan_retry_used = True
            try:
                retry_data = model_call(
                    list(messages) + [{"role": "user", "content": PLAN_RETRY_NUDGE}],
                    tier=tier,
                )
            except Exception:  # noqa: BLE001 - fall through to the no-plan handling
                retry_data = None
            if retry_data is not None:
                total_tokens += _tokens_from(retry_data)
                plan = _extract_json_object(_message_content(retry_data))
        if not isinstance(plan, dict):
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "plan",
                    "status": "error",
                    "error": "planner did not return a JSON object",
                    "retried": plan_retry_used,
                }
            )
            stopped_reason = STOP_NO_PLAN
            # Prefer a truthful, observation-grounded answer over a bare failure.
            reply = summarize_observations(observations) or NO_PLAN_REPLY
            break

        if plan.get("final") is not None and not plan.get("tool"):
            pending_tools = [name for name in required_tools if name not in completed_tools]
            if pending_tools:
                observation = {
                    "error": "explicitly requested tools remain: " + ", ".join(pending_tools)
                }
                steps.append(
                    {
                        "step": step_index + 1,
                        "phase": "replan",
                        "status": "rejected",
                        "thought": str(plan.get("thought") or ""),
                        "observation": observation,
                        "replan_reason": "required_tools_remaining",
                    }
                )
                observations.append(
                    {
                        "rejected": observation["error"],
                        "replan_required": True,
                        "replan_reason": "required_tools_remaining",
                    }
                )
                replans += 1
                continue
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
        raw_args = plan.get("args") if isinstance(plan.get("args"), dict) else {}
        args = normalize_action_args(tool, raw_args, resolved_allow)
        if tool == "workflow_author":
            # Preserve every operator-requested semantic stage. A lossy planner
            # paraphrase must not silently drop curation/training/evaluation steps.
            args["goal"] = goal_text
        thought = str(plan.get("thought") or "")
        planned_action = {"tool": tool, "args": args}
        planned_digest = action_digest(planned_action)

        if planned_digest in failed_actions:
            reason = "unchanged_strategy"
            observation = {
                "error": (
                    "replan repeated the same failed tool and arguments; change "
                    "arguments, choose another tool, or ask a clarifying sub-goal"
                )
            }
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "replan",
                    "tool": tool,
                    "args": args,
                    "status": "rejected",
                    "thought": thought,
                    "observation": observation,
                    "replan_reason": reason,
                }
            )
            observations.append(
                {
                    "tool": tool,
                    "rejected": observation["error"],
                    "replan_required": True,
                    "replan_reason": reason,
                }
            )
            replans += 1
            continue

        if planned_digest in successful_actions or (
            tool in completed_tools and tool in required_tools
        ):
            reason = "already_completed"
            observation = {
                "error": (
                    "this tool action already completed successfully; use the remaining "
                    "requested tool or finish from existing observations"
                )
            }
            steps.append(
                {
                    "step": step_index + 1,
                    "phase": "replan",
                    "tool": tool,
                    "args": args,
                    "status": "rejected",
                    "thought": thought,
                    "observation": observation,
                    "replan_reason": reason,
                }
            )
            observations.append(
                {
                    "tool": tool,
                    "rejected": observation["error"],
                    "replan_required": True,
                    "replan_reason": reason,
                }
            )
            replans += 1
            continue

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
            failed_actions[planned_digest] = "tool_not_allowed"
            observations.append(
                {
                    "tool": tool,
                    "rejected": observation["error"],
                    "replan_required": True,
                    "replan_reason": "tool_not_allowed",
                }
            )
            replans += 1
            continue

        if requires_confirmation(tool, resolved_allow):
            proposed = {"tool": tool, "args": args}
            digest = action_digest(proposed)
            token_ok = confirmation_ok(confirm_token, session_token) and not confirmation_consumed
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
            # One matching token authorizes one attempt inside this loop. Consume
            # it before invoking the executor so errors and planner repeats cannot
            # turn one confirmation into multiple state changes.
            confirmation_consumed = True

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
            failed_actions[planned_digest] = "executor_unavailable"
            observations.append(
                {
                    "tool": tool,
                    "error": observation["error"],
                    "replan_required": True,
                    "replan_reason": "executor_unavailable",
                }
            )
            replans += 1
            continue

        raised = False
        try:
            result = executor(args)
            observation = result
        except Exception as exc:  # noqa: BLE001 - tool errors are observations
            raised = True
            observation = {"error": str(exc)}
        replan_reason = _replan_reason(observation, raised=raised)
        empty_result = replan_reason == "empty_observation"
        terminal_empty = empty_result and _empty_result_is_terminal(
            goal_text,
            tool,
            required_tools=required_tools,
            completed_tools=completed_tools,
        )
        if terminal_empty:
            replan_reason = ""
        status = "error" if replan_reason == "tool_error" else "empty" if empty_result else "ok"
        observed = _observe(observation)
        steps.append(
            {
                "step": step_index + 1,
                "phase": "call",
                "tool": tool,
                "args": args,
                "status": status,
                "thought": thought,
                "observation": observed,
                **({"terminal_observation": True} if terminal_empty else {}),
                **({"replan_reason": replan_reason} if replan_reason else {}),
            }
        )
        if tool not in tools_used:
            tools_used.append(tool)
        observation_entry = {"tool": tool, "result": observed}
        if replan_reason:
            failed_actions[planned_digest] = replan_reason
            observation_entry.update(
                {"replan_required": True, "replan_reason": replan_reason}
            )
            replans += 1
        else:
            successful_actions.add(planned_digest)
            if tool in required_tools and tool not in completed_tools:
                completed_tools.append(tool)
        observations.append(observation_entry)
        if terminal_empty:
            reply = _terminal_empty_reply(tool, observation)
            stopped_reason = STOP_DONE
            break
    else:
        stopped_reason = STOP_MAX_STEPS
        if not reply:
            summary = summarize_observations(observations)
            reply = (
                "Reached the maximum number of steps without a final answer. "
                "Observations gathered are in the step trace."
            )
            if summary:
                reply = f"{reply}\n\n{summary}"

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
        "replans": replans,
    }


# Common ways a planner (or operator) spells a threshold operator, mapped to the
# canonical insights QueryRequest tokens. Keeps the read-only insights_query tool
# robust to LLM arg drift (e.g. ">=", "at least") instead of failing validation.
_THRESHOLD_OP_ALIASES: dict[str, str] = {
    ">": "gt",
    ">=": "ge",
    "=>": "ge",
    "<": "lt",
    "<=": "le",
    "=<": "le",
    "==": "eq",
    "=": "eq",
    "gt": "gt",
    "ge": "ge",
    "gte": "ge",
    "lt": "lt",
    "le": "le",
    "lte": "le",
    "eq": "eq",
    "greater": "gt",
    "greater_than": "gt",
    "at_least": "ge",
    "min": "ge",
    "less": "lt",
    "less_than": "lt",
    "at_most": "le",
    "max": "le",
    "equal": "eq",
    "equals": "eq",
}

_DASHBOARD_GROUP_BY = (
    "metric_name",
    "tool",
    "stage",
    "workflow",
    "metric_kind",
    "step",
    "currency",
    "cost_basis",
    "score_name",
)


def normalize_threshold_op(op: str) -> str:
    """Map a loose threshold operator spelling to a canonical token (or "")."""
    return _THRESHOLD_OP_ALIASES.get(str(op or "").strip().lower(), "")


def normalize_group_by(value: str) -> str:
    """Clamp a dashboard group_by to an allowed facet (default metric_name)."""
    resolved = str(value or "").strip().lower()
    return resolved if resolved in _DASHBOARD_GROUP_BY else "metric_name"


CHAT_ACTION_MODE = "chat-action"


def run_chat_action_loop(
    goal: str,
    *,
    tools: Mapping[str, Callable[[dict[str, Any]], Any]],
    model_call: Callable[..., Any],
    allowlist: Mapping[str, ToolSpec] | None = None,
    tier: str = "cheap",
    confirm_token: str = "",
    session_token: str = "",
    confirm_digest: str = "",
    max_steps: int = DEFAULT_MAX_STEPS,
    live_context: str = "",
) -> dict[str, Any]:
    """Drive the bounded tool loop for a ``/chat`` "action" turn and shape the reply.

    This is the fallthrough that lets a chat turn actually *use* read-only tools
    (e.g. the insights backbone) instead of describing an endpoint to call.
    Read-only tools execute inside the loop; a state-changing tool
    (``requires_confirmation``) never auto-runs from a chat turn — with no
    confirmation token the loop stops at ``needs_confirmation`` and the caller
    issues a gate token, preserving the existing safety contract.

    Returns a JSON-serializable chat-response fragment carrying the loop's
    ``reply`` plus a compact ``steps``/``tools_used`` trace. All side effects
    (model + tool calls) are injected, so it unit-tests with zero tokens/infra.
    """
    result = run_action_loop(
        goal,
        tools=tools,
        model_call=model_call,
        confirm_token=confirm_token,
        session_token=session_token,
        confirm_digest=confirm_digest,
        tier=tier,
        max_steps=max_steps,
        allowlist=allowlist,
        live_context=live_context,
    )
    return {
        "ok": bool(result.get("ok")),
        "reply": str(result.get("reply") or "").strip(),
        "grounded": False,
        "mode": CHAT_ACTION_MODE,
        "tier": result.get("tier", tier),
        "steps": result.get("steps", []),
        "tools_used": result.get("tools_used", []),
        "stopped_reason": result.get("stopped_reason"),
        "needs_confirmation": bool(result.get("needs_confirmation")),
        "proposed_action": result.get("proposed_action"),
        "usage": {"total_tokens": int(result.get("tokens") or 0)},
        "replans": int(result.get("replans") or 0),
    }
