"""Expose evidence-bound, absence-only recovery for an original failed Sky submit."""

from enum import Enum
import json
from pathlib import Path
import sqlite3

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.skypilot.absence_recovery import reconcile_absent


class OutputFormat(str, Enum):
    """Select machine-readable JSON or a short text status."""

    json = "json"
    text = "text"


@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def reconcile_absent_cmd(
    evidence_file: Path = typer.Option(
        ..., help="Private pinned original evidence manifest."
    ),
    apply: bool = typer.Option(
        False, help="Release the original lease after fresh verified absence."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, help="Output representation."
    ),
) -> None:
    """Verify original native absence without inferring historical job success.

    Args:
        evidence_file: Original producing records and exact reader bindings.
        apply: Commit the local journal/lease transition; default retains it.
        output_format: Emit one JSON object or a short status.
    Returns:
        None.
    Raises:
        typer.Exit: Evidence, reads, or exclusive ownership could not be verified.
    """
    try:
        result = reconcile_absent(evidence_file, apply=apply)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error):
        typer.echo(
            json.dumps({"status": "verification-unavailable", "reconciled": False})
            if output_format == OutputFormat.json
            else "Original absence could not be verified."
        )
        raise typer.Exit(1) from None
    typer.echo(
        json.dumps(result) if output_format == OutputFormat.json else result["status"]
    )


def register(app: typer.Typer) -> None:
    """Register the absence-only workflow recovery command.

    Args:
        app: Existing workflow command group.
    Returns:
        None.
    Raises:
        None.
    """
    app.command("reconcile-absent")(reconcile_absent_cmd)
