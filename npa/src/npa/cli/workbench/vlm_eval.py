"""npa workbench vlm-eval - stub VLM evaluation commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from typing import Any

import typer
from rich.console import Console

from npa.workbench.vlm_eval import VlmEvalError, evaluate_stub, write_result

app = typer.Typer(
    name="vlm-eval",
    help="Stub VLM evaluation for sim-to-real pipeline gating.",
    no_args_is_help=True,
)
console = Console(stderr=True)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.command("run")
def run_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 or local artifact path to score."),
    output_path: str = typer.Option(..., "--output-path", help="S3 or local path for eval JSON."),
    task: str = typer.Option("sim-to-real", "--task", help="Evaluation task label."),
    model: str = typer.Option("vlm-eval-stub", "--model", help="Stub model label."),
    success_threshold: float = typer.Option(
        0.8,
        "--success-threshold",
        help="Score threshold required to pass.",
    ),
    score: float = typer.Option(
        -1.0,
        "--score",
        help="Override stub score for tests and controller loops.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Produce deterministic stub VLM metrics for a pipeline artifact."""
    try:
        result = evaluate_stub(
            input_path=input_path,
            output_path=output_path,
            task=task,
            model=model,
            success_threshold=success_threshold,
            score=None if score < 0 else score,
        )
        payload = asdict(result)
        effective_dry_run = dry_run or _env_dry_run()
        payload["dry_run"] = effective_dry_run
        if not effective_dry_run:
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show stub backend status."""
    _emit(
        {
            "backend": "stub",
            "status": "available",
            "real_vlm_backend": False,
            "message": "VLM eval is a schema-compatible stub.",
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List available VLM eval backends."""
    _emit({"backends": [{"name": "stub", "real_backend": False}]}, output)


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN", ""
    ).lower() in {"1", "true", "yes"}


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
