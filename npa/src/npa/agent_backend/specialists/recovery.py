"""Recover rejected generations without replaying tools or expanding their grants."""

from __future__ import annotations

import json

from .routing import _endpoints


class _RejectedGeneration(ValueError):
    """An unusable generation that is safe to hand to an authorized backup."""


def _require_operations(messages, required):
    passed, seen, calls = set(), set(), {}
    for message in messages:
        if message.get("role") == "assistant":
            calls.update({call["id"]: call for call in message.get("tool_calls") or []})
        if message.get("role") != "tool":
            continue
        identifier = message["tool_call_id"]
        if identifier in seen:
            continue  # Reusing a journal receipt does not execute a fresh check.
        seen.add(identifier)
        call = calls[identifier]["function"]
        result = json.loads(message["content"])
        if call["name"] == "edit_file" and result.get("ok") is True:
            passed.clear()
        if call["name"] == "run_operation":
            name = json.loads(call["arguments"]).get("name")
            if not isinstance(name, str):
                continue
            passed.discard(name)
            if result.get("ok") is True and result.get("returncode") == 0:
                passed.add(name)
    missing = sorted(set(required) - passed)
    if missing:
        raise _RejectedGeneration(
            "completion requires successful operations after the latest edit: "
            + ", ".join(missing)
        )


def _handoff(state, profile, tools, error):
    index = state.get("model_index", 0)
    if index >= len(profile.fallback_models):
        raise error
    endpoints = _endpoints(profile, state)
    tools.store._event(
        tools.task_id,
        {
            "type": "model_fallback",
            "from_model": endpoints[index].model,
            "to_model": endpoints[index + 1].model,
            "reason": str(error),
        },
    )
    messages = [
        {
            key: value
            for key, value in message.items()
            if key not in {"reasoning", "reasoning_content"}
        }
        for message in state["messages"]
    ]
    messages.append(_recovery_instruction(error))
    return {
        "messages": messages,
        "pending": [],
        "answer": "",
        "model_index": index + 1,
        "recovering": True,
    }


def _recovery_instruction(error):
    return {
        "role": "user",
        "content": (
            "Workbench runtime recovery: the previous request did not yield an accepted "
            "response and none of its proposed tools executed. Reason: "
            + str(error)
            + ". Continue from the "
            "recorded tool results and current files; preserve completed work. Use native "
            "tool calls, inspect results, and satisfy the configured completion checks."
        ),
    }
