"""Resolve config, run, state-output, and loop tokens in workflow specs."""

from __future__ import annotations

import base64
import re
from typing import Any, Mapping

_TOKEN_RE = re.compile(
    r"\{\{\s*(config|run|state|loop)\.([a-zA-Z0-9_.-]+)"
    r"(?:\|([a-zA-Z0-9_-]+))?\s*\}\}"
)


class TokenError(ValueError):
    """Raised when a token cannot be resolved."""


def resolve_tokens(
    value: str,
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
    loop_iterations: Mapping[str, int] | None = None,
) -> str:
    """Substitute supported tokens in ``value``."""

    outputs = state_outputs or {}
    loops = loop_iterations or {}

    def _replace(match: re.Match[str]) -> str:
        scope, key, transform = match.group(1), match.group(2), match.group(3)
        if scope == "config":
            if key not in config:
                raise TokenError(f"unknown config token: config.{key}")
            resolved = str(config[key])
        elif scope == "run":
            if key not in run:
                raise TokenError(f"unknown run token: run.{key}")
            resolved = str(run[key])
        elif scope == "state":
            state_name, _, output_key = key.partition(".")
            state_map = outputs.get(state_name)
            if not state_map or output_key not in state_map:
                raise TokenError(f"unknown state token: state.{key}")
            resolved = str(state_map[output_key])
        elif scope == "loop":
            if key not in loops:
                raise TokenError(f"unknown loop token: loop.{key}")
            resolved = str(loops[key])
        else:  # pragma: no cover - constrained by _TOKEN_RE
            raise TokenError(f"unsupported token scope: {scope}")
        if transform is None:
            return resolved
        if transform == "base64":
            return base64.b64encode(resolved.encode("utf-8")).decode("ascii")
        raise TokenError(f"unsupported token transform: {transform}")

    return _TOKEN_RE.sub(_replace, value)


def resolve_mapping(
    data: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
    loop_iterations: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Deep-resolve string tokens in a shallow mapping (one level of values)."""

    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            resolved[key] = resolve_tokens(
                value,
                config=config,
                run=run,
                state_outputs=state_outputs,
                loop_iterations=loop_iterations,
            )
        else:
            resolved[key] = value
    return resolved


def resolve_value(
    value: Any,
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    state_outputs: Mapping[str, Mapping[str, str]] | None = None,
    loop_iterations: Mapping[str, int] | None = None,
) -> Any:
    """Recursively resolve tokens in nested resource and pod configuration."""

    if isinstance(value, str):
        return resolve_tokens(
            value,
            config=config,
            run=run,
            state_outputs=state_outputs,
            loop_iterations=loop_iterations,
        )
    if isinstance(value, dict):
        return {
            key: resolve_value(
                item,
                config=config,
                run=run,
                state_outputs=state_outputs,
                loop_iterations=loop_iterations,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_value(
                item,
                config=config,
                run=run,
                state_outputs=state_outputs,
                loop_iterations=loop_iterations,
            )
            for item in value
        ]
    return value
