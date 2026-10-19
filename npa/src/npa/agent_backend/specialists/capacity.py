"""Admit text/tool requests conservatively against declared served context capacity."""

from __future__ import annotations

import json

from .context import model_messages


def _visible_messages(messages):
    return [
        {
            key: value
            for key, value in message.items()
            if key not in {"reasoning", "reasoning_content"}
        }
        for message in messages
    ]


def _context_view(messages, candidate, profile, previous=None):
    if previous and (previous.model, previous.base_url) != (
        candidate.model,
        candidate.base_url,
    ):
        messages = _visible_messages(messages)
    if not candidate.compact_context:
        return messages
    observations = {
        name
        for name, operation in profile.operations.items()
        if operation.observation_only
    }
    return model_messages(messages, observation_operations=observations)


def _admission(endpoint, messages, **payload):
    if any(
        message.get("content") is not None and not isinstance(message["content"], str)
        for message in messages
    ):
        raise ValueError("configuration routing supports text and native tools only")
    body = {"messages": messages, **payload}
    serialized = json.dumps(
        body, ensure_ascii=True, sort_keys=True, allow_nan=False
    ).encode("ascii")
    estimate = len(serialized) + endpoint.framing_tokens + 128 * len(messages)
    reserved = endpoint.model_options["max_tokens"]
    return {
        "admitted": estimate + reserved <= endpoint.context_window,
        "input_token_estimate": estimate,
        "output_tokens_reserved": reserved,
        "context_window": endpoint.context_window,
        "estimator": "ascii-json-bytes-plus-framing-v1",
        "exact_tokenization": False,
    }
