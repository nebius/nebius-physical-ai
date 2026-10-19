"""Detect repeated failed mutations with no intervening source progress."""

from __future__ import annotations

import json

from .json_edit import _SOURCE_EDIT_TOOLS


def _result(message):
    if message.get("role") != "tool":
        return None
    try:
        value = json.loads(message.get("content", ""))
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _mutation_identity(call):
    arguments = call.get("arguments")
    try:
        arguments = json.loads(arguments)
    except (ValueError, TypeError):
        pass
    return call.get("name"), arguments


def _failed_edit_cycle(messages):
    last = _result(messages[-1]) if messages else None
    if not last or last.get("ok") is not False:
        return False
    calls = {
        call.get("id"): call.get("function", {})
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    target = calls.get(messages[-1].get("tool_call_id"), {})
    if target.get("name") not in _SOURCE_EDIT_TOOLS:
        return False
    for message in reversed(messages[:-1]):
        if message.get("role") == "user":
            break
        result = _result(message)
        call = calls.get(message.get("tool_call_id"), {})
        if result is None or call.get("name") not in _SOURCE_EDIT_TOOLS:
            continue
        if result.get("ok") is True:
            break
        if result == last and _mutation_identity(call) == _mutation_identity(target):
            return True
    return False


def _failed_verification_cycle(messages):
    latest = _result(messages[-1]) if messages else None
    if (
        not latest
        or latest.get("ok") is not False
        or not latest.get("deterministic_inputs")
        or latest.get("inputs_unchanged") is not True
    ):
        return False
    for message in reversed(messages[:-1]):
        previous = _result(message)
        if (
            previous
            and previous.get("ok") is False
            and previous.get("inputs_unchanged") is True
        ):
            if (
                previous.get("operation") == latest.get("operation")
                and previous.get("deterministic_inputs")
                == latest["deterministic_inputs"]
            ):
                return True
    return False
