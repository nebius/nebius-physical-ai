"""Expose exact-controller recovery when the original local SkyPilot receipt is lost."""

from pathlib import Path
import json

import typer
from botocore.exceptions import BotoCoreError, ClientError

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.skypilot.controller_recovery import reconcile_controller


@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def reconcile_controller_cmd(
    record: Path = typer.Argument(help="Owner-only original controller recovery JSON."),
    cancel: bool = typer.Option(False, help="Request exact-ID native cancellation after verification."),
    json_output: bool = typer.Option(False, "--json", help="Emit sanitized JSON."),
) -> None:
    """Reconcile an orphaned workflow against its original exclusive controller.

    Args:
        record: Private execution identity retained outside the repository.
        cancel: Request cancellation; omitted means read-only reconciliation.
        json_output: Emit one JSON document.
    Returns:
        None.
    Raises:
        typer.Exit: Verification failed; no terminal cleanup claim is made.
    """
    try:
        result = reconcile_controller(record, cancel=cancel)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, BotoCoreError, ClientError):
        result = {"status": "VERIFICATION_UNAVAILABLE", "cleanup_verified": False}
        typer.echo(json.dumps(result) if json_output else "Exact controller verification failed.")
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result) if json_output else str(result))


def register(app: typer.Typer) -> None:
    """Register the original-controller recovery command.

    Args:
        app: Existing workflow command group.
    Returns:
        None.
    Raises:
        None.
    """
    app.command("reconcile-controller")(reconcile_controller_cmd)
