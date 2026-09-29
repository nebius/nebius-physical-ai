"""Explicit, read-only cloud verification and local terminal reconciliation."""

import json
from pathlib import Path

import typer

from npa.cluster.reconcile_absent import reconcile_absent
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract


@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def reconcile_absent_cmd(
    evidence_file: Path = typer.Option(
        ...,
        "--evidence-file",
        help="Private pinned original lifecycle evidence manifest.",
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Output format: json."
    ),
) -> None:
    """Reconcile an absent legacy cluster operation; never delete or relaunch."""
    if output_format != "json":
        raise typer.BadParameter("Only JSON output is supported")
    typer.echo(json.dumps(reconcile_absent(evidence_file)))
