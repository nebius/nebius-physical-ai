"""Resolve config, run, state-output, and loop tokens in workflow specs."""

from __future__ import annotations

import base64
import re
from typing import Any, Mapping

_TOKEN_RE = re.compile(
    r"\{\{\s*(config|run|state|loop)\.([a-zA-Z0-9_.-]+)"
    r"(?:\|([a-zA-Z0-9_-]+))?\s*\}\}"
)
_BRACED_EXPRESSION_RE = re.compile(r"\{\{[^{}]*\}\}")
_CODE_EVALUATOR_NAMES = frozenset(
    {
        "bash",
        "dash",
        "deno",
        "ksh",
        "node",
        "perl",
        "python",
        "python3",
        "ruby",
        "sh",
        "zsh",
    }
)


class TokenError(ValueError):
    """Raised when a token cannot be resolved."""


def validate_shell_token_safety(value: str) -> None:
    """Reject token interpolation that could alter a ``bash -c`` program.

    ``run.shell`` is intentionally an escape hatch for a small set of stateful
    workflows.  Its source is ultimately supplied to ``bash -c`` locally and on
    SkyPilot, so resolving an arbitrary config value directly into that source
    makes quoting in a YAML template a security boundary.  It is not one.

    The existing ``base64`` token transform produces only the base64 alphabet;
    callers decode it into a quoted shell variable before use.  That is the one
    token spelling permitted in shell source.  Values that are naturally argv
    elements belong in ``run.argv`` instead.
    """

    for expression in _BRACED_EXPRESSION_RE.findall(value):
        token = _TOKEN_RE.fullmatch(expression)
        if token is None or token.group(3) != "base64":
            raise TokenError(
                "run.shell tokens must use the |base64 transform and be decoded "
                "into a quoted shell variable; use run.argv for direct arguments"
            )


def validate_argv_code_token_safety(argv: list[str]) -> None:
    """Reject tokens interpolated into an interpreter's evaluated source string.

    ``run.argv`` keeps normal resolved values out of a command parser, but
    ``python -c`` and shell ``-c`` reintroduce a source-code boundary.  Tokens
    in that source can change the program just as directly as ``run.shell``
    interpolation.  Keep the program literal and pass resolved data in later
    argv elements instead.
    """

    if len(argv) < 3:
        return
    executable = argv[0].rsplit("/", 1)[-1].lower()
    is_python = executable.startswith("python")
    is_shell_or_script = executable in _CODE_EVALUATOR_NAMES
    if not (is_python or is_shell_or_script):
        return
    for index, argument in enumerate(argv[:-1]):
        if argument in {"-c", "--command"} or (
            argument in {"-e", "--eval"}
            and executable in {"deno", "node", "perl", "ruby"}
        ):
            if _BRACED_EXPRESSION_RE.search(argv[index + 1]):
                raise TokenError(
                    "run.argv interpreter code must not interpolate tokens; "
                    "pass resolved values as later argv elements"
                )


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
