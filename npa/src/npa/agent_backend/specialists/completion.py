"""Finish opted-in bounded tasks from authoritative verification receipts."""

from __future__ import annotations

from .reports import _report
from .waiting import _tool_step


def tool_step(state, profile, tools):
    """Execute a tool and optionally finish from current, resolved verification.

    Args: state: Persisted graph state. profile: Operator-owned completion policy.
        tools: Journaled tool executor and task store.
    Returns: Tool-node update, with a host-generated answer only after all gates pass.
    Raises: Tool/storage errors propagate without inventing a completion receipt.
    """
    update = {**_tool_step(state, tools), "verified_completion": False}
    if (
        not profile.finish_on_verified_operations
        or update.get("pending", state.get("pending"))
        or update.get("operation_failure", state.get("operation_failure"))
    ):
        return update
    task = {
        **tools.store._get(tools.task_id),
        "events": tools.store._events(tools.task_id),
        "calls": tools.store._calls(tools.task_id),
    }
    report = _report(tools.store, profile, task)
    if (
        task["status"] != "running"
        or report["paused"]
        or not report["required_operations_passed"]
    ):
        return update
    tools.store._event(
        tools.task_id,
        {"type": "verified_completion", "required_operations": report["required_operations"]},
    )
    return {
        **update,
        "verified_completion": True,
        "answer": "Runtime verified the operator-declared completion checks against current source; no unresolved tool effects remain.",
    }
