"""Observability for the NPA agent backend (Blueprint Phase I).

Open-source replacement for the Blueprint reference agent's LangSmith tracing:
structured spans wrapping the existing step/iteration traces, emitted through an
**injected tracer** so the transport is swappable (a no-op ``NullTracer`` by
default, an ``InMemoryTracer`` for tests/analysis, or a guarded-import Langfuse /
OpenTelemetry adapter to a self-hosted OSS backend). Plus an offline analyzer
that clusters traces and flags *silent failures* — truncated observations, empty
tool results, and unsurfaced tool/planner errors.

Invariants:

- **No PII/secrets in spans.** ``redact_attributes`` strips secret-like keys and
  long high-entropy values before any attribute reaches a tracer.
- **Dependency injection.** Span emission goes through the injected tracer;
  span construction is pure and unit-tests at 0 tokens / no network.

Phase G: shipped as an importable file (``agent_backend/trace.py``); the backend
imports it via ``from agent_backend.trace import ...``. The
``npa/src/npa/cli/agent_trace.py`` shim re-exports it for callers/tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

SPAN_OK = "ok"
SPAN_ERROR = "error"
SPAN_WARN = "warn"

KIND_ROOT = "root"
KIND_PLAN = "plan"
KIND_TOOL = "tool"
KIND_FINAL = "final"
KIND_CONFIRM = "confirm"
KIND_ITERATION = "iteration"

# Attribute keys / substrings that must never be emitted into a span verbatim.
_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|auth|credential|"
    r"access[_-]?key|private[_-]?key|bearer)",
    re.IGNORECASE,
)
# A value that looks like a long high-entropy secret even under a benign key.
_SECRET_VALUE_RE = re.compile(r"^[A-Za-z0-9_\-+/=]{32,}$")
_REDACTED = "«redacted»"


@dataclass
class Span:
    """A structured span over an agent step/iteration (transport-agnostic)."""

    name: str
    kind: str = KIND_ROOT
    status: str = SPAN_OK
    duration_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    parent: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "events": list(self.events),
            "parent": self.parent,
        }


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if _SECRET_VALUE_RE.match(value.strip()):
            return _REDACTED
        return value if len(value) <= 500 else value[:500] + "…"
    if isinstance(value, dict):
        return redact_attributes(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


def redact_attributes(attrs: Any) -> dict[str, Any]:
    """Return a copy of ``attrs`` with secret-like keys/values redacted.

    Keys matching common credential names are dropped entirely; string values
    that look like long high-entropy tokens are replaced. This runs before any
    attribute is handed to a tracer so no PII/secret can enter a span.
    """
    if not isinstance(attrs, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in attrs.items():
        if _SECRET_KEY_RE.search(str(key)):
            clean[str(key)] = _REDACTED
            continue
        clean[str(key)] = _redact_value(value)
    return clean


class NullTracer:
    """Default no-op tracer: tracing is inert unless a backend is wired."""

    def emit(self, span: dict[str, Any]) -> None:  # noqa: D401 - protocol method
        return None


class InMemoryTracer:
    """Collects emitted spans in memory (tests + offline analysis)."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    def emit(self, span: dict[str, Any]) -> None:
        self.spans.append(dict(span))


def record_spans(tracer: Any, spans: Sequence[Span | dict[str, Any]]) -> int:
    """Emit ``spans`` through the injected tracer; never raises on tracer error."""
    if tracer is None:
        return 0
    emitted = 0
    for span in spans:
        payload = span.to_dict() if isinstance(span, Span) else dict(span)
        payload["attributes"] = redact_attributes(payload.get("attributes") or {})
        try:
            tracer.emit(payload)
        except Exception:  # noqa: BLE001 - observability must never break the request
            continue
        emitted += 1
    return emitted


def _observation_is_empty(observation: Any) -> bool:
    if observation is None:
        return True
    if isinstance(observation, dict):
        return len(observation) == 0
    if isinstance(observation, (list, str)):
        return len(observation) == 0
    return False


def spans_from_action_loop(result: dict[str, Any]) -> list[Span]:
    """Turn a ``run_action_loop`` result into structured spans (root + steps)."""
    data = result if isinstance(result, dict) else {}
    stopped = str(data.get("stopped_reason") or "")
    root_status = SPAN_OK if data.get("ok") else SPAN_ERROR
    root = Span(
        name="agent.act",
        kind=KIND_ROOT,
        status=root_status,
        attributes=redact_attributes(
            {
                "goal_len": len(str(data.get("goal") or "")),
                "stopped_reason": stopped,
                "tools_used": list(data.get("tools_used") or []),
                "tokens": int(data.get("tokens") or 0),
                "tier": str(data.get("tier") or ""),
                "needs_confirmation": bool(data.get("needs_confirmation")),
            }
        ),
    )
    spans = [root]
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        phase = str(step.get("phase") or "step")
        kind = {
            "plan": KIND_PLAN,
            "call": KIND_TOOL,
            "final": KIND_FINAL,
            "confirm": KIND_CONFIRM,
        }.get(phase, KIND_TOOL)
        status = str(step.get("status") or SPAN_OK)
        span_status = SPAN_ERROR if status in {"error", "rejected"} else SPAN_OK
        observation = step.get("observation")
        attrs = {
            "phase": phase,
            "step": step.get("step"),
            "tool": step.get("tool"),
            "arg_keys": sorted((step.get("args") or {}).keys()) if isinstance(step.get("args"), dict) else [],
            "status": status,
        }
        events: list[dict[str, Any]] = []
        if isinstance(observation, dict) and observation.get("truncated"):
            events.append({"name": "observation_truncated"})
        if _observation_is_empty(observation) and phase == "call" and status == "ok":
            events.append({"name": "empty_tool_result"})
        spans.append(
            Span(
                name=f"{phase}.{step.get('tool') or step.get('step')}",
                kind=kind,
                status=span_status,
                attributes=redact_attributes(attrs),
                events=events,
                parent=root.name,
            )
        )
    return spans


def spans_from_drive(result: dict[str, Any]) -> list[Span]:
    """Turn a ``drive_sim2real_loop`` result into structured spans."""
    data = result if isinstance(result, dict) else {}
    root = Span(
        name="agent.sim2real.drive",
        kind=KIND_ROOT,
        status=SPAN_OK if data.get("ok") else SPAN_ERROR,
        attributes=redact_attributes(
            {
                "stopped_reason": str(data.get("stopped_reason") or ""),
                "decision": data.get("decision"),
                "final_run_id": str(data.get("final_run_id") or ""),
                "needs_confirmation": bool(data.get("needs_confirmation")),
                "iterations": len(data.get("iterations") or []),
            }
        ),
    )
    spans = [root]
    for record in data.get("iterations") or []:
        if not isinstance(record, dict):
            continue
        status = SPAN_ERROR if record.get("error") else SPAN_OK
        spans.append(
            Span(
                name=f"iteration.{record.get('iteration')}",
                kind=KIND_ITERATION,
                status=status,
                attributes=redact_attributes(
                    {
                        "iteration": record.get("iteration"),
                        "run_id": str(record.get("run_id") or ""),
                        "decision": record.get("decision"),
                        "status_confirmed": bool(record.get("status_confirmed")),
                    }
                ),
                parent=root.name,
            )
        )
    return spans


# ── offline analyzer ─────────────────────────────────────────────────────────


def _trace_signature(trace: dict[str, Any]) -> str:
    stopped = str(trace.get("stopped_reason") or "")
    tools = ",".join(sorted(str(t) for t in (trace.get("tools_used") or [])))
    return f"{stopped}|{tools}"


def _silent_failures_for(trace: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    stopped = str(trace.get("stopped_reason") or "")
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []

    if stopped == "max_steps":
        findings.append(
            {"trace_index": idx, "kind": "max_steps_exhausted", "detail": "loop hit max_steps without a final answer"}
        )

    tool_error = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        observation = step.get("observation")
        status = str(step.get("status") or "")
        phase = str(step.get("phase") or "")
        if isinstance(observation, dict) and observation.get("truncated"):
            findings.append(
                {"trace_index": idx, "kind": "truncated_observation", "detail": f"step {step.get('step')} observation truncated"}
            )
        if phase == "call" and status == "ok" and _observation_is_empty(observation):
            findings.append(
                {"trace_index": idx, "kind": "empty_tool_result", "detail": f"tool {step.get('tool')} returned empty result"}
            )
        if status in {"error", "rejected"}:
            tool_error = True

    # A tool/planner error that never surfaced: the loop still "succeeded" but a
    # step errored — a classic silent failure the operator would miss.
    if tool_error and trace.get("ok") and stopped == "done":
        findings.append(
            {"trace_index": idx, "kind": "unsurfaced_tool_error", "detail": "a step errored but the run reported done/ok"}
        )
    return findings


def analyze_traces(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cluster traces and flag silent failures for offline review.

    ``traces`` is a list of action-loop / drive result dicts. Returns
    ``{clusters, silent_failures, totals}``. Pure/deterministic — no network.
    """
    items = [t for t in (traces or []) if isinstance(t, dict)]
    clusters_map: dict[str, dict[str, Any]] = {}
    silent_failures: list[dict[str, Any]] = []
    for idx, trace in enumerate(items):
        sig = _trace_signature(trace)
        cluster = clusters_map.setdefault(
            sig,
            {
                "signature": sig,
                "stopped_reason": str(trace.get("stopped_reason") or ""),
                "tools": sorted(str(t) for t in (trace.get("tools_used") or [])),
                "count": 0,
            },
        )
        cluster["count"] += 1
        silent_failures.extend(_silent_failures_for(trace, idx))
    clusters = sorted(clusters_map.values(), key=lambda c: c["count"], reverse=True)
    return {
        "clusters": clusters,
        "silent_failures": silent_failures,
        "totals": {
            "traces": len(items),
            "clusters": len(clusters),
            "silent_failures": len(silent_failures),
        },
    }


# ── guarded-import tracer adapters (self-hosted OSS backends) ─────────────────


def build_langfuse_tracer(*, public_key: str = "", secret_key: str = "", host: str = "") -> Any:
    """Build a Langfuse-backed tracer (guarded import; live/VM path only).

    Endpoint + keys are passed in (operator/config-resolved), never hardcoded.
    Returns a tracer with an ``emit`` method; raises if ``langfuse`` is absent.
    """
    from langfuse import Langfuse  # local import: optional extra

    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host or None)

    class _LangfuseTracer:
        def emit(self, span: dict[str, Any]) -> None:
            client.trace(name=str(span.get("name") or "agent"), metadata=span.get("attributes") or {})

    return _LangfuseTracer()


def build_otel_tracer(tracer_provider: Any = None, *, service_name: str = "npa-agent") -> Any:
    """Build an OpenTelemetry-backed tracer (guarded import; live/VM path only)."""
    from opentelemetry import trace as _otel_trace  # local import: optional extra

    provider = tracer_provider or _otel_trace.get_tracer_provider()
    otel = provider.get_tracer(service_name)

    class _OtelTracer:
        def emit(self, span: dict[str, Any]) -> None:
            with otel.start_as_current_span(str(span.get("name") or "agent")) as otel_span:
                for key, value in (span.get("attributes") or {}).items():
                    try:
                        otel_span.set_attribute(str(key), value)
                    except Exception:  # noqa: BLE001 - attribute typing is best-effort
                        otel_span.set_attribute(str(key), str(value))

    return _OtelTracer()
