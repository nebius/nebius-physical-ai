"""npa workbench mjlab - locomotion policy evaluation commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.deploy.images import DEFAULT_WORKBENCH_IMAGE_ENV, default_workbench_image
from npa.workbench.mjlab import MjlabEvalError, evaluate_locomotion, write_result

app = typer.Typer(
    name="mjlab",
    help="MJLab locomotion policy evaluation for SONIC workflows.",
    no_args_is_help=True,
)
console = Console(stderr=True)

WORKFLOW_PATH = Path("npa/src/npa/workflows/skypilot/mjlab-eval.yaml")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.command("eval")
def eval_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="S3 or local rollout/motion artifact path to evaluate.",
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "--checkpoint-path",
        help="SONIC checkpoint path or URI to score.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="S3 or local path for MJLab evaluation JSON.",
    ),
    suite: str = typer.Option("locomotion", "--suite", help="MJLab suite name."),
    embodiment: str = typer.Option("unitree-g1", "--embodiment", help="Robot embodiment."),
    episodes: int = typer.Option(8, "--episodes", help="Evaluation episode count."),
    success_threshold: float = typer.Option(
        0.75,
        "--success-threshold",
        help="Score threshold required to pass.",
    ),
    score: float = typer.Option(
        -1.0,
        "--score",
        help="Override deterministic score for tests and dry validation.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Evaluate a SONIC locomotion checkpoint against MJLab metrics."""

    try:
        result = evaluate_locomotion(
            input_path=input_path,
            checkpoint=checkpoint,
            output_path=output_path,
            suite=suite,
            embodiment=embodiment,
            episodes=episodes,
            success_threshold=success_threshold,
            score=None if score < 0 else score,
        )
        payload = asdict(result)
        effective_dry_run = dry_run or _env_dry_run()
        payload["dry_run"] = effective_dry_run
        if not effective_dry_run:
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except MjlabEvalError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("workflow")
def workflow_cmd(
    image: str = typer.Option(
        "",
        "--image",
        envvar=DEFAULT_WORKBENCH_IMAGE_ENV,
        help="Workbench workflow image. Also settable with NPA_WORKBENCH_IMAGE.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the SkyPilot YAML template for MJLab evaluation."""

    _emit(
        {
            "workflow": str(WORKFLOW_PATH),
            "image_env": DEFAULT_WORKBENCH_IMAGE_ENV,
            "image": image.strip() or default_workbench_image(),
        },
        output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show MJLab tool status."""

    _emit(
        {
            "backend": "mjlab",
            "status": "available",
            "workflow": str(WORKFLOW_PATH),
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List supported MJLab evaluation suites."""

    _emit(
        {
            "suites": [
                {"name": "locomotion", "embodiments": ["unitree-g1"]},
                {"name": "stability", "embodiments": ["unitree-g1"]},
            ]
        },
        output,
    )


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN",
        "",
    ).lower() in {"1", "true", "yes"}


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
