"""Compatibility shim for the shipped Sim2Real-loop implementation."""

from __future__ import annotations

from npa.agent_backend import sim2real_loop as _impl
from npa.agent_backend.sim2real_loop import (
    DECISION_LOOP_BACK,
    DECISION_PROMOTE,
    DEFAULT_MAX_ITERATIONS,
    STOP_ERROR,
    STOP_EXHAUSTED,
    STOP_INSUFFICIENT_SIGNAL,
    STOP_NEEDS_CONFIRMATION,
    STOP_NO_ADJUSTMENT,
    STOP_PROMOTED,
    STOP_UNCONFIRMED_STATUS,
    drive_action_digest,
    drive_sim2real_loop,
    evaluate_gate,
    gate_with_config_threshold,
    resolve_drive_config,
)

__all__ = [
    "DECISION_LOOP_BACK",
    "DECISION_PROMOTE",
    "DEFAULT_MAX_ITERATIONS",
    "STOP_ERROR",
    "STOP_EXHAUSTED",
    "STOP_INSUFFICIENT_SIGNAL",
    "STOP_NEEDS_CONFIRMATION",
    "STOP_NO_ADJUSTMENT",
    "STOP_PROMOTED",
    "STOP_UNCONFIRMED_STATUS",
    "drive_action_digest",
    "drive_sim2real_loop",
    "evaluate_gate",
    "gate_with_config_threshold",
    "resolve_drive_config",
]


def __getattr__(name: str):
    """Preserve access to historical private helpers without duplicating logic."""
    return getattr(_impl, name)
