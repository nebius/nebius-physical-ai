"""Queue explicitly authorized deterministic checks after an opted-in model completion."""

from __future__ import annotations

import json


def _completion_checks(profile, tools, attempt_id, messages, pending):
    if pending or not profile.verify_on_completion:
        return messages, pending
    calls = [
        {
            "id": f"host-verify-{attempt_id[:20]}-{index}",
            "type": "function",
            "function": {
                "name": "run_operation",
                "arguments": json.dumps({"name": name}),
            },
        }
        for index, name in enumerate(profile.required_operations)
    ]
    tools.store._event(
        tools.task_id,
        {
            "type": "host_verification",
            "operations": profile.required_operations,
            "cause": "model_completion",
            "attempt_id": attempt_id,
        },
    )
    message = {
        "role": "assistant",
        "content": "Runtime verification of the operator-declared deterministic checks.",
        "tool_calls": calls,
    }
    return [*messages, message], calls
