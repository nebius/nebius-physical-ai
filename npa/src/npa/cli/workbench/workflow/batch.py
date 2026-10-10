"""Expose workflow batch planning, submission, and local progress."""

from enum import Enum
import json
from pathlib import Path

import typer
import yaml

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.orchestration.npa_workflow.batch import batch_status, run_batch
from npa.orchestration.npa_workflow.batch_plan import plan_batch

app = typer.Typer(help="Submit dataset episodes as independent durable workflow runs.")


class OutputFormat(str, Enum):
    """Batch result representation."""

    json = "json"
    text = "text"


def _emit(result, output_format, *, require_success=False):
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    elif isinstance(result.get("runs"), list):
        typer.echo(f"{result['batch_id']}: {len(result['runs'])} planned workflow runs")
    else:
        counts = {}
        for record in result.get("runs", {}).values():
            status = record["status"]
            counts[status] = counts.get(status, 0) + 1
        typer.echo(f"{result.get('batch_id', 'batch')}: {counts or result}")
    if require_success and any(
        row["status"] != "succeeded" for row in result["runs"].values()
    ):
        raise typer.Exit(1)


def _fail(exc, output_format):
    _emit({"status": "failed", "error": str(exc)}, output_format)
    raise typer.Exit(1) from exc


@app.command("plan")
@json_stdout_contract
def plan_cmd(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Expand and validate a batch without submitting jobs.

    Args:
        manifest: Private YAML batch manifest.
        output_format: JSON or text.
    Returns:
        None.
    Raises:
        typer.Exit: The manifest or workflow is invalid.
    """
    try:
        result = plan_batch(manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        _fail(exc, output_format)
    _emit(result, output_format)


@app.command("submit")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def submit_cmd(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    state_dir: Path = typer.Option(
        ..., "--state-dir", help="Private durable local batch ledger directory."
    ),
    max_concurrent_runs: int = typer.Option(..., min=1, help="Maximum active runs."),
    resume: bool = typer.Option(
        False, "--resume", help="Reconcile started runs before admitting pending runs."
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Run a batch in the foreground through standard workflow submission.

    Args:
        manifest: Private YAML batch manifest.
        state_dir: Retained local ledger and client logs.
        max_concurrent_runs: Explicit concurrency ceiling.
        resume: Continue using original run identities.
        output_format: JSON or text.
    Returns:
        None.
    Raises:
        typer.Exit: Invalid plan, local failure, or any unresolved workflow.
    """
    try:
        result = run_batch(
            manifest,
            state_dir=state_dir,
            max_concurrent_runs=max_concurrent_runs,
            resume=resume,
            reporter=lambda message: typer.echo(message, err=True),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        _fail(exc, output_format)
    _emit(result, output_format, require_success=True)


@app.command("status")
@json_stdout_contract
def status_cmd(
    batch_id: str = typer.Argument(...),
    state_dir: Path = typer.Option(..., "--state-dir"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Read local observations; use workflow status for live reconciliation.

    Args:
        batch_id: Exact batch identifier.
        state_dir: Original local ledger directory.
        output_format: JSON or text.
    Returns:
        None.
    Raises:
        typer.Exit: The local batch state cannot be read.
    """
    try:
        result = batch_status(state_dir, batch_id)
    except (OSError, ValueError) as exc:
        _fail(exc, output_format)
    _emit(result, output_format)
