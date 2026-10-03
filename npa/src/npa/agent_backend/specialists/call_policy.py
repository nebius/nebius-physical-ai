"""Bind an operation's observation declaration to its original durable invocation."""

from __future__ import annotations

import hashlib
import json

from .config import fingerprint


def _digest(invocation):
    return hashlib.sha256(json.dumps(invocation, sort_keys=True).encode()).hexdigest()


def _classification(profile, invocation):
    operation = None
    if invocation["name"] == "run_operation":
        try:
            arguments = json.loads(invocation["arguments"])
            if isinstance(arguments, dict) and set(arguments) == {"name"}:
                operation = (
                    arguments["name"] if isinstance(arguments["name"], str) else None
                )
        except (ValueError, TypeError):
            pass
    configured = profile.operations.get(operation)
    return {
        "version": 1,
        "policy": fingerprint(profile),
        "tool": invocation["name"],
        "operation": operation,
        "digest": _digest(invocation),
        "observation_only": configured is not None and configured.observation_only,
    }


def _require_observation(call, profile):
    classification = call.get("classification") or {}
    operation = profile.operations.get(classification.get("operation"))
    if not (
        classification.get("version") == 1
        and classification.get("observation_only") is True
        and classification.get("tool") == "run_operation"
        and classification.get("policy") == fingerprint(profile)
        and classification.get("digest") == call["digest"]
        and operation is not None
        and operation.observation_only
    ):
        raise ValueError(
            "interrupted call is not an observation under its original policy"
        )
