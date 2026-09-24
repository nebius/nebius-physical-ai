"""Freeze coordinator and specialist receipts without promoting missing usage to zero."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from npa.agent_backend.specialists.config import TeamConfig

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
OBSERVATION_TOOLS = ("dismiss_interrupted_observation",)


def _granted_tools(arm, config=None):
    names = DIRECT_TOOLS
    if arm == "astra-tofa":
        names += DELEGATION_TOOLS
        if config and any(
            operation.observation_only
            for profile in config.profiles
            for operation in profile.operations.values()
        ):
            names += OBSERVATION_TOOLS
    return names


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


def _explicit_zero_invocations(directory, config, completed):
    path = directory / "protocol.json"
    protocol = json.loads(path.read_text()) if path.exists() else {}
    return (
        config.get("strategy") == "specialists-first"
        and config.get("turns") == []
        and completed == 0
        and protocol.get("arm") == "astra-tofa"
        and protocol.get("coordination") == "specialists-first"
    )


def _recorded_invocations(config, completed):
    if "turns" not in config:
        return {
            "verification": "legacy_single_invocation",
            "expected_turns": 1,
            "usage_complete": completed == 1 and bool(config.get("argv")),
        }
    turns = config["turns"]
    if not isinstance(turns, list) or not turns:
        return {"verification": "invalid", "usage_complete": False}
    unfinished = [
        index + 1
        for index, turn in enumerate(turns)
        if not isinstance(turn, dict)
        or type(turn.get("exit_code")) is not int
        or turn["exit_code"] != 0
        or turn.get("ended_epoch") is None
    ]
    return {
        "verification": "invocation_records",
        "expected_turns": len(turns),
        "completed_turns": completed,
        "failed_or_unfinished_invocations": unfinished,
        "usage_complete": not unfinished and completed == len(turns),
    }


def _coordinator_invocations(directory, completed):
    path = directory / "coordinator-config.json"
    if not path.exists():
        protocol = directory / "protocol.json"
        modern = protocol.exists() and "coordination" in json.loads(
            protocol.read_text()
        )
        return {
            "verification": "missing" if modern else "legacy_unavailable",
            "usage_complete": not modern,
        }
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        return {"verification": "invalid", "usage_complete": False}
    if _explicit_zero_invocations(directory, config, completed):
        return {
            "verification": "explicit_zero_invocations",
            "expected_turns": 0,
            "completed_turns": 0,
            "usage_complete": True,
        }
    return _recorded_invocations(config, completed)


def _astra_usage(directory, arm="astra-only"):
    events, malformed = _astra_events(directory)
    turns = [
        event["usage"] if isinstance(event.get("usage"), dict) else {}
        for event in events
        if event.get("type") == "turn.completed"
    ]
    path = directory / "team.json"
    config = TeamConfig.model_validate_json(path.read_text()) if path.exists() else None
    allowed = set(_granted_tools(arm, config))
    forbidden = [event for event in events if _outside_tool_scope(event, allowed)]
    failed = any(event.get("type") in {"turn.failed", "error"} for event in events)
    required = {"input_tokens", "output_tokens"}
    invocations = _coordinator_invocations(directory, len(turns))
    zero = invocations["verification"] == "explicit_zero_invocations" and not events
    return {
        "turns": turns,
        "invocations": invocations,
        "malformed_lines": malformed,
        "out_of_scope_events": forbidden,
        "usage_complete": (bool(turns) or zero)
        and not malformed
        and not failed
        and invocations["usage_complete"]
        and all(required <= turn.keys() for turn in turns),
        "matched_tool_scope": (bool(events) or zero)
        and not malformed
        and not forbidden,
    }


def _specialist_usage(receipts):
    responses, failures, complete = [], [], True
    for task_id, task in receipts.items():
        previous = {}
        for event in task["events"]:
            if event["type"] == "provider_failure":
                failures.append({"task_id": task_id, **event})
                complete = False
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
    return {"responses": responses, "failures": failures, "usage_complete": complete}


def _router_attempted(route):
    attempted = route.get("api_call_attempted")
    if type(attempted) is bool or attempted == "unknown":
        return attempted
    if "api_call_attempted" in route:
        return "unknown"
    if route.get("reason") in {"missing_credential", "invalid_configuration"}:
        return False
    if route.get("status") in {"accepted", "abstained"} or route.get("reason") in {
        "provider_http_error",
        "provider_transport_error",
        "invalid_response",
    }:
        return True
    return "unknown"


def _router_decisions(receipts):
    for task_id, task in receipts.items():
        route = task.get("route", {})
        if route.get("provider") == "typesafe":
            yield task_id, "profile", route
        selection = route.get("model_selection")
        if selection is not None:
            yield task_id, "model", selection


def _router_usage(receipts):
    responses, complete = [], True
    for task_id, kind, selection in _router_decisions(receipts):
        attempted = _router_attempted(selection)
        responses.append(
            {
                **{
                    key: value
                    for key, value in selection.items()
                    if key != "model_selection"
                },
                "task_id": task_id,
                "route_kind": kind,
                "api_call_attempted": attempted,
            }
        )
        counters = selection.get("usage", {})
        if attempted is not False and (
            attempted is not True
            or not isinstance(counters, dict)
            or any(
                type(counters.get(key)) is not int or counters[key] < 0
                for key in ("input_tokens", "output_tokens")
            )
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
            "router": _router_usage(receipts),
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
