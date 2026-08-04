"""Finite-time HTTP adapter for the opt-in deployed-agent evaluation."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping

import httpx

REQUEST_TIMEOUT_ENV = "NPA_AGENT_EVAL_REQUEST_TIMEOUT_SECONDS"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0


def request_timeout_seconds(env: Mapping[str, str] | None = None) -> float:
    """Resolve a positive finite request timeout from the live-eval environment."""
    values = env if env is not None else os.environ
    raw = str(values.get(REQUEST_TIMEOUT_ENV, DEFAULT_REQUEST_TIMEOUT_SECONDS)).strip()
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{REQUEST_TIMEOUT_ENV} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{REQUEST_TIMEOUT_ENV} must be a positive finite number")
    return timeout


def post_agent_action(
    *,
    url: str,
    payload: Mapping[str, Any],
    auth: tuple[str, str] | None,
    verify: bool,
    post: Callable[..., Any] = httpx.post,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST one live action with a finite timeout and actionable timeout error."""
    timeout = request_timeout_seconds(env)
    try:
        response = post(
            url,
            json=dict(payload),
            auth=auth,
            timeout=timeout,
            verify=verify,
        )
    except httpx.TimeoutException as exc:
        raise AssertionError(
            f"live agent request timed out after {timeout:g} seconds"
        ) from exc
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise AssertionError("live agent response was not a JSON object")
    return result
