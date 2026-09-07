"""Shared helpers for the RoboCasa workbench CLI."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import httpx
import typer

from npa.workbench.robocasa.schemas import DEFAULT_TOKEN_ENV


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text if text is not None else "\n".join(f"{key}: {value}" for key, value in payload.items()))


def resolve_endpoint(endpoint: str) -> str:
    resolved = endpoint.strip() or os.environ.get("NPA_ROBOCASA_ENDPOINT", "")
    if not resolved:
        fail("--endpoint is required")
    if not resolved.startswith(("http://", "https://")):
        fail("--endpoint must be an http:// or https:// URL")
    return resolved.rstrip("/")


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str = DEFAULT_TOKEN_ENV,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{endpoint}{path}",
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fail(f"RoboCasa request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach RoboCasa endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("RoboCasa endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("RoboCasa endpoint returned an unexpected response")
    return data
