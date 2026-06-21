"""Resolve ``{{config.*}}``, ``{{run.*}}``, and ``{{state.*}}`` tokens in workflow specs."""

from __future__ import annotations

import re
from typing import Any, Mapping

_TOKEN_RE = re.compile(
    r"\{\{\s*(config|run|state)\.([a-zA-Z0-9_.-]+)\s*\}\}"
)


class TokenError(ValueError):
    """Raised when a token cannot be resolved."""


def resolve_tokens(
    value: str,
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Substitute supported tokens in ``value``."""

    outputs = state_outputs or {}

    def _replace(match: re.Match[str]) -> str:
        scope, key = match.group(1), match.group(2)
        if scope == "config":
            if key not in config:
                raise TokenError(f"unknown config token: config.{key}")
            return str(config[key])
        if scope == "run":
            if key not in run:
                raise TokenError(f"unknown run token: run.{key}")
            return str(run[key])
        if scope == "state":
            state_name, _, output_key = key.partition(".")
            state_map = outputs.get(state_name)
            if not state_map or output_key not in state_map:
                raise TokenError(f"unknown state token: state.{key}")
            return str(state_map[output_key])
        raise TokenError(f"unsupported token scope: {scope}")

    return _TOKEN_RE.sub(_replace, value)


def resolve_mapping(
    data: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Deep-resolve string tokens in a shallow mapping (one level of values)."""

    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            resolved[key] = resolve_tokens(
                value, config=config, run=run, state_outputs=state_outputs
            )
        else:
            resolved[key] = value
    return resolved
