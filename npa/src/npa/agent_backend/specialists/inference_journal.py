"""Journal configuration-worker inference before payment and reject silent boundary replay."""

from __future__ import annotations

import hashlib
import json

from .store import UncertainOperation


def _generation_intent(candidate, tools, messages, extra):
    body = [candidate.name, candidate.model, candidate.base_url, messages, extra]
    identifier = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if any(
        event.get("attempt_id") == identifier
        for event in tools.store._events(tools.task_id)
    ):
        raise UncertainOperation(
            "Generation boundary has an uncommitted inference attempt; inspect its receipts, do not repeat the paid request"
        )
    tools.store._event(
        tools.task_id,
        {
            "type": "inference_attempt",
            "attempt_id": identifier,
            "model": candidate.model,
            "status": "started",
            "usage": {},
            "api_call_attempted": "unknown",
        },
    )
    return identifier
