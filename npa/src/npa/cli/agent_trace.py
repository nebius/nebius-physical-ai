"""Compatibility shim: observability/tracing now lives in the shipped package.

The real implementation lives in ``npa/src/npa/agent_backend/trace.py``
(Blueprint Phase I: shipped importable package instead of embed). This shim
preserves the ``npa.cli.agent_trace`` import path for callers and tests.
"""

from __future__ import annotations

from npa.agent_backend.trace import (  # noqa: F401
    KIND_CONFIRM,
    KIND_FINAL,
    KIND_ITERATION,
    KIND_PLAN,
    KIND_ROOT,
    KIND_TOOL,
    SPAN_ERROR,
    SPAN_OK,
    SPAN_WARN,
    InMemoryTracer,
    NullTracer,
    Span,
    analyze_traces,
    build_langfuse_tracer,
    build_otel_tracer,
    record_spans,
    redact_attributes,
    spans_from_action_loop,
    spans_from_drive,
)

__all__ = [
    "KIND_CONFIRM",
    "KIND_FINAL",
    "KIND_ITERATION",
    "KIND_PLAN",
    "KIND_ROOT",
    "KIND_TOOL",
    "SPAN_ERROR",
    "SPAN_OK",
    "SPAN_WARN",
    "InMemoryTracer",
    "NullTracer",
    "Span",
    "analyze_traces",
    "build_langfuse_tracer",
    "build_otel_tracer",
    "record_spans",
    "redact_attributes",
    "spans_from_action_loop",
    "spans_from_drive",
]
