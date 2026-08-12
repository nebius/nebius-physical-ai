"""One fail-closed confirmation contract for destructive CLI commands."""

from __future__ import annotations

import json
import sys
from typing import Any

import typer


def require_destructive_confirmation(
    *,
    yes: bool,
    prompt: str,
    output_json: bool = False,
    payload: dict[str, Any] | None = None,
) -> None:
    """Require ``--yes`` without a TTY, or an affirmative TTY confirmation."""

    if yes:
        return
    if not sys.stdin.isatty():
        message = f"{prompt} Re-run with --yes; no TTY is available to confirm."
        _emit_refusal("confirmation_required", message, output_json, payload)
        raise typer.Exit(code=1)
    if not typer.confirm(prompt, default=False):
        message = "Aborted. No destructive action was attempted."
        _emit_refusal("cancelled", message, output_json, payload)
        raise typer.Exit(code=1)


def _emit_refusal(
    result: str,
    message: str,
    output_json: bool,
    payload: dict[str, Any] | None,
) -> None:
    if output_json:
        document = dict(payload or {})
        document.update({"result": result, "message": message, "mutated": False})
        typer.echo(json.dumps(document, indent=2, sort_keys=True))
    else:
        typer.echo(message)
