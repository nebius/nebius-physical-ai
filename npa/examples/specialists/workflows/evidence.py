"""Freeze coordinator and specialist receipts without promoting missing usage to zero."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

DIRECT_TOOLS = (
    "read_file",
    "list_files",
    "edit_file",
    "run_operation",
    "run_operations",
)
DELEGATION_TOOLS = (
    "delegate",
    "specialist_status",
    "wait_specialist",
    "wait_specialists",
    "take_over",
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _receipts(store):
    return {
        task["id"]: {
            **task,
            "events": store._events(task["id"]),
            "calls": store._calls(task["id"]),
        }
        for task in store._list()
    }


def _astra_events(directory):
    path = directory / "codex.jsonl"
    events, malformed = [], []
    for number, line in enumerate(
        path.read_text().splitlines() if path.exists() else [], 1
    ):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            events.append(event)
        except ValueError:
            malformed.append(number)
    return events, malformed


def _outside_tool_scope(event, allowed):
    item = event.get("item")
    if item is None:
        return str(event.get("type", "")).startswith("item.")
    if not isinstance(item, dict):
        return True
    if item.get("type") in {"agent_message", "reasoning"}:
        return False
    return not (
        item.get("type") == "mcp_tool_call"
        and item.get("server") == "workbench"
        and item.get("tool") in allowed
    )


def _astra_usage(directory, arm="astra-only"):
    events, malformed = _astra_events(directory)
    turns = [
        event["usage"] if isinstance(event.get("usage"), dict) else {}
        for event in events
        if event.get("type") == "turn.completed"
    ]
    allowed = set(DIRECT_TOOLS)
    if arm == "astra-tofa":
        allowed.update(DELEGATION_TOOLS)
    forbidden = [event for event in events if _outside_tool_scope(event, allowed)]
    failed = any(event.get("type") in {"turn.failed", "error"} for event in events)
    required = {"input_tokens", "output_tokens"}
    return {
        "turns": turns,
        "malformed_lines": malformed,
        "out_of_scope_events": forbidden,
        "usage_complete": bool(turns)
        and not malformed
        and not failed
        and all(required <= turn.keys() for turn in turns),
        "matched_tool_scope": bool(events) and not malformed and not forbidden,
    }


def _specialist_usage(receipts):
    responses, complete = [], True
    for task_id, task in receipts.items():
        previous = {}
        for event in task["events"]:
            if event["type"] == "needs_attention" and not (
                previous.get("type") == "model" and previous.get("accepted") is False
            ):
                complete = False
            if event["type"] == "model":
                responses.append({"task_id": task_id, **event})
                if (
                    not {"prompt_tokens", "completion_tokens"}
                    <= event.get("usage", {}).keys()
                ):
                    complete = False
            previous = event
        if task["status"] not in {"completed", "cancelled"} or any(
            call["status"] != "completed" for call in task["calls"]
        ):
            complete = False
    return {"responses": responses, "usage_complete": complete}


def _snapshot(team, coordinator, directory, execution):
    errors = {}
    receipts = {}
    for name, store in (
        ("task-receipts", team.store),
        ("coordinator-receipts", coordinator),
    ):
        try:
            payload = _receipts(store)
            _write_json(directory / (name + ".json"), payload)
            if name == "task-receipts":
                receipts = payload
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            errors[name] = type(error).__name__
    try:
        usage = {
            "astra": _astra_usage(directory, execution["arm"]),
            "specialists": _specialist_usage(receipts),
        }
        usage["usage_complete"] = not errors and all(
            part["usage_complete"] for part in usage.values()
        )
        _write_json(directory / "usage.json", usage)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        errors["usage"] = type(error).__name__
    execution["snapshot_errors"] = errors
    execution["unfinished_tasks"] = [
        task["id"] for task in receipts.values() if task["status"] != "completed"
    ]
    _write_json(directory / "execution.json", execution)
