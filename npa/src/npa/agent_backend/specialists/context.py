"""Omit superseded successful observations from requests while preserving durable history."""

from __future__ import annotations

import hashlib
import json


def _calls(messages):
    return {
        call.get("id"): call.get("function", {}).get("name")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", []) or []
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    }


def _result(message):
    if message.get("role") != "tool" or not isinstance(message.get("content"), str):
        return None
    try:
        result = json.loads(message["content"])
    except ValueError:
        return None
    return result if isinstance(result, dict) and result.get("ok") is True else None


def _omitted(result, fields):
    result = dict(result)
    for field in fields:
        value = result.get(field)
        if not isinstance(value, str):
            continue
        del result[field]
        result[field + "_omitted"] = (
            "superseded successful observation; full receipt retained"
        )
        result[field + "_bytes"] = len(value.encode())
        result[field + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
    return result


def _read(result, files):
    path, digest = result.get("path"), result.get("sha256")
    start, end = result.get("start_line"), result.get("end_line")
    if not isinstance(path, str) or not isinstance(digest, str):
        return result
    known = files.setdefault(path, {"sha256": digest, "ranges": [], "edited": False})
    if known["edited"] or known["sha256"] != digest:
        return _omitted(result, ("content",))
    if type(start) is not int or type(end) is not int:
        return result
    if any(left <= start and end <= right for left, right in known["ranges"]):
        return _omitted(result, ("content",))
    known["ranges"].append((start, end))
    return result


def _observation(name, result, files, operations, observation_operations):
    if name == "edit_file" and isinstance(result.get("path"), str):
        known = files.setdefault(
            result["path"], {"sha256": result.get("sha256"), "ranges": []}
        )
        known["edited"] = True
    if name == "read_file":
        return _read(result, files)
    if name == "run_operation" and result.get("returncode") == 0:
        operation = result.get("operation")
        if not isinstance(operation, str) or operation not in observation_operations:
            return result
        if operation in operations:
            return _omitted(result, ("stdout", "stderr"))
        operations.add(operation)
    return result


def model_messages(messages, *, observation_operations=()):
    """Build a request view with stale source and repeated successful logs omitted.

    Args:
        messages: Complete persisted conversation, including native tool-call pairs.
        observation_operations: Explicitly declared read-only operation names whose
            earlier successful observations may be replaced by newer ones.
    Returns:
        New list retaining every message and tool-call identifier. Failed observations,
        latest source ranges, latest operation logs and non-tool messages stay intact.
    Raises:
        None for valid persisted message lists.
    """
    calls, files, operations = _calls(messages), {}, set()
    rendered = list(messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        result = _result(message)
        if result is None:
            continue
        name = calls.get(message.get("tool_call_id"))
        compact = _observation(name, result, files, operations, observation_operations)
        if compact != result:
            rendered[index] = {**message, "content": json.dumps(compact)}
    return rendered
