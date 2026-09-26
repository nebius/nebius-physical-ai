"""Poll explicit observation contracts at durable boundaries without model turns."""

from __future__ import annotations

import hashlib
import json
import time


def _pointer(document, pointer):
    current = document
    for raw in pointer.split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif (
            isinstance(current, list)
            and key.isascii()
            and key.isdecimal()
            and str(int(key)) == key
            and int(key) < len(current)
        ):
            current = current[int(key)]
        else:
            raise ValueError("observation state field is missing")
    if not isinstance(current, str):
        raise ValueError("observation state must be a string")
    return current


def _observation_result(result, policy):
    if policy is None:
        return result
    if result.get("returncode") != 0:
        return {**result, "observation": {"status": "command_failed"}}
    try:
        value = _pointer(json.loads(result["stdout"]), policy.field)
        status = _state_label(policy, value)
    except ValueError as error:
        return {
            **result,
            "ok": False,
            "error": str(error),
            "observation": {"status": "invalid"},
        }
    return {
        **result,
        "ok": status == "succeeded",
        "observation": {"status": status, "value": value},
    }


def _state_label(policy, value):
    if value in policy.pending_values:
        return "pending"
    if value in policy.success_values:
        return "succeeded"
    if value in policy.failure_values:
        return "failed"
    raise ValueError("observation returned an unconfigured state")


def _policy(tools, call):
    if call["function"]["name"] != "run_operation":
        return None
    arguments = json.loads(call["function"]["arguments"])
    if set(arguments) != {"name"} or not isinstance(arguments["name"], str):
        return None
    operation = tools.profile.operations.get(arguments["name"])
    return operation.wait_for if operation else None


def _poll_call(call, ordinal):
    identity = json.dumps([call["id"], ordinal], separators=(",", ":"))
    return {
        **call,
        "id": "observation-" + hashlib.sha256(identity.encode()).hexdigest(),
    }


def _tool_step(state, tools):
    call = state["pending"][0]
    policy = _policy(tools, call)
    ordinal = state.get("poll_ordinal", 0)
    actual = _poll_call(call, ordinal) if policy is not None else call
    result = tools.execute(actual)
    if policy is not None and result.get("observation", {}).get("status") == "pending":
        return {
            "poll_ordinal": ordinal + 1,
            "poll_after": time.time() + policy.poll_interval,
        }
    message = {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(result),
    }
    return {
        "messages": [*state["messages"], message],
        "pending": state["pending"][1:],
        "poll_ordinal": 0,
        "poll_after": 0.0,
        "operation_failure": state.get("operation_failure", "")
        or _failed_operation(tools, result),
    }


def _failed_operation(tools, result):
    name = result.get("operation")
    operation = tools.profile.operations.get(name)
    if operation is None or not operation.handoff_on_failure:
        return ""
    code = result.get("returncode")
    if result.get("ok") is not False or type(code) is not int or code < 0:
        return ""
    if code > 0 or (
        operation.wait_for is not None
        and result.get("observation", {}).get("status") == "failed"
    ):
        return name
    return ""
