"""Compatibility shim for the shipped agent action-loop implementation."""

from __future__ import annotations

from npa.agent_backend import actions as _impl
from npa.agent_backend.actions import (
    CHAT_ACTION_MODE,
    DEFAULT_MAX_STEPS,
    NO_PLAN_REPLY,
    PLAN_RETRY_NUDGE,
    STOP_DONE,
    STOP_ERROR,
    STOP_MAX_STEPS,
    STOP_NEEDS_CONFIRMATION,
    STOP_NO_PLAN,
    TOOL_ALLOWLIST,
    ToolSpec,
    action_digest,
    allowlist_specs,
    confirmation_ok,
    is_allowed,
    normalize_group_by,
    normalize_threshold_op,
    requires_confirmation,
    run_action_loop,
    run_chat_action_loop,
    strip_reasoning_trace,
    summarize_observations,
)

__all__ = [
    "CHAT_ACTION_MODE",
    "DEFAULT_MAX_STEPS",
    "NO_PLAN_REPLY",
    "PLAN_RETRY_NUDGE",
    "STOP_DONE",
    "STOP_ERROR",
    "STOP_MAX_STEPS",
    "STOP_NEEDS_CONFIRMATION",
    "STOP_NO_PLAN",
    "TOOL_ALLOWLIST",
    "ToolSpec",
    "action_digest",
    "allowlist_specs",
    "confirmation_ok",
    "is_allowed",
    "normalize_group_by",
    "normalize_threshold_op",
    "requires_confirmation",
    "run_action_loop",
    "run_chat_action_loop",
    "strip_reasoning_trace",
    "summarize_observations",
]


def __getattr__(name: str):
    """Preserve access to historical private helpers without duplicating logic."""
    return getattr(_impl, name)
