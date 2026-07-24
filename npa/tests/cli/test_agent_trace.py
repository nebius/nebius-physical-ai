"""Tier-0 tests for observability / tracing (Blueprint Phase I).

Span construction and the offline analyzer are pure; the tracer is injected
(``InMemoryTracer``) so nothing touches a network backend.
"""

from __future__ import annotations

from npa.cli import agent_trace as T


# ── redaction (no PII/secrets in spans) ──────────────────────────────────────


def test_redact_drops_secret_like_keys():
    clean = T.redact_attributes(
        {"api_key": "abc", "password": "p", "authorization": "Bearer x", "goal": "hi"}
    )
    assert clean["api_key"] == "«redacted»"
    assert clean["password"] == "«redacted»"
    assert clean["authorization"] == "«redacted»"
    assert clean["goal"] == "hi"


def test_redact_masks_high_entropy_values():
    token = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"  # 32-char high-entropy string
    clean = T.redact_attributes({"note": token})
    assert clean["note"] == "«redacted»"


def test_redact_recurses_into_nested_dicts():
    clean = T.redact_attributes({"outer": {"secret": "s", "ok": "v"}})
    assert clean["outer"]["secret"] == "«redacted»"
    assert clean["outer"]["ok"] == "v"


# ── spans from loop/drive results ────────────────────────────────────────────


def _action_result(**overrides):
    base = {
        "ok": True,
        "goal": "check status",
        "stopped_reason": "done",
        "tools_used": ["sim_viz_status"],
        "tokens": 12,
        "tier": "cheap",
        "needs_confirmation": False,
        "steps": [
            {"step": 1, "phase": "call", "tool": "sim_viz_status", "args": {}, "status": "ok", "observation": {"run_id": "r"}},
            {"step": 2, "phase": "final", "status": "ok"},
        ],
    }
    base.update(overrides)
    return base


def test_spans_from_action_loop_has_root_and_steps():
    spans = T.spans_from_action_loop(_action_result())
    assert spans[0].kind == T.KIND_ROOT
    assert spans[0].name == "agent.act"
    assert any(s.kind == T.KIND_TOOL for s in spans)
    assert any(s.kind == T.KIND_FINAL for s in spans)


def test_spans_flag_empty_tool_result_event():
    result = _action_result(
        steps=[{"step": 1, "phase": "call", "tool": "artifacts_runs", "args": {}, "status": "ok", "observation": {}}]
    )
    spans = T.spans_from_action_loop(result)
    tool_span = next(s for s in spans if s.kind == T.KIND_TOOL)
    assert any(e["name"] == "empty_tool_result" for e in tool_span.events)


def test_spans_never_leak_secret_args():
    result = _action_result(
        steps=[{"step": 1, "phase": "call", "tool": "x", "args": {"api_key": "sekret"}, "status": "ok", "observation": {"a": 1}}]
    )
    spans = T.spans_from_action_loop(result)
    # arg_keys are sorted key names only; the value never enters the span.
    for span in spans:
        assert "sekret" not in str(span.to_dict())


def test_spans_from_drive():
    drive = {
        "ok": True,
        "stopped_reason": "promoted",
        "decision": "promote_checkpoint",
        "final_run_id": "run-1",
        "iterations": [{"iteration": 1, "run_id": "run-1", "decision": "promote_checkpoint", "status_confirmed": True}],
    }
    spans = T.spans_from_drive(drive)
    assert spans[0].name == "agent.sim2real.drive"
    assert any(s.kind == T.KIND_ITERATION for s in spans)


# ── injected tracer ──────────────────────────────────────────────────────────


def test_record_spans_emits_through_injected_tracer():
    tracer = T.InMemoryTracer()
    emitted = T.record_spans(tracer, T.spans_from_action_loop(_action_result()))
    assert emitted == len(tracer.spans)
    assert tracer.spans[0]["name"] == "agent.act"


def test_record_spans_tolerates_tracer_failure():
    class _Boom:
        def emit(self, span):
            raise RuntimeError("tracer down")

    # A failing tracer must never break the request path.
    assert T.record_spans(_Boom(), T.spans_from_action_loop(_action_result())) == 0


def test_null_tracer_is_noop():
    assert T.NullTracer().emit({"name": "x"}) is None


# ── offline analyzer: clustering + silent failures ───────────────────────────


def test_analyze_flags_max_steps_exhaustion():
    trace = _action_result(ok=False, stopped_reason="max_steps")
    report = T.analyze_traces([trace])
    kinds = {f["kind"] for f in report["silent_failures"]}
    assert "max_steps_exhausted" in kinds


def test_analyze_flags_unsurfaced_tool_error():
    trace = _action_result(
        steps=[
            {"step": 1, "phase": "call", "tool": "x", "args": {}, "status": "error", "observation": {"error": "boom"}},
            {"step": 2, "phase": "final", "status": "ok"},
        ]
    )
    report = T.analyze_traces([trace])
    kinds = {f["kind"] for f in report["silent_failures"]}
    assert "unsurfaced_tool_error" in kinds


def test_analyze_flags_truncated_observation():
    trace = _action_result(
        steps=[{"step": 1, "phase": "call", "tool": "x", "args": {}, "status": "ok", "observation": {"truncated": True, "preview": "..."}}]
    )
    report = T.analyze_traces([trace])
    kinds = {f["kind"] for f in report["silent_failures"]}
    assert "truncated_observation" in kinds


def test_analyze_clusters_by_signature():
    traces = [_action_result(), _action_result(), _action_result(stopped_reason="max_steps", ok=False)]
    report = T.analyze_traces(traces)
    assert report["totals"]["traces"] == 3
    # Two distinct signatures: done|sim_viz_status and max_steps|sim_viz_status.
    assert report["totals"]["clusters"] == 2
    assert report["clusters"][0]["count"] == 2


def test_analyze_empty_input():
    report = T.analyze_traces([])
    assert report["totals"]["traces"] == 0
    assert report["silent_failures"] == []
