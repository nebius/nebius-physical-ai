"""Thin cuRobo V2 CLI; all capability behavior lives in the workbench module."""

from __future__ import annotations

import json
from enum import Enum

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.curobo import runtime
from npa.workbench.curobo.schemas import PrepareRequest, RunRequest

app = typer.Typer(
    name="curobo",
    help="NVIDIA cuRobo V2 motion planning and complete benchmark evaluation.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    json = "json"
    text = "text"


def _call(operation, request, output_format):
    try:
        result = operation(request)
    except (runtime.CuroboError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, sort_keys=True, indent=2))


@app.command("prepare")
@json_stdout_contract
def prepare_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    mode: str = typer.Option("both", "--mode"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
):
    """Write a complete benchmark recipe (both full datasets; no problem cap)."""
    _call(
        runtime.prepare,
        PrepareRequest(output_path=output_path, mode=mode),
        output_format,
    )


@app.command("benchmark")
@json_stdout_contract
def benchmark_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
):
    """Run every benchmark problem in each selected dynamics configuration."""
    _call(
        runtime.benchmark,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
        output_format,
    )


@app.command("plan")
@json_stdout_contract
def plan_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
):
    """Plan each Franka start/goal and cuboid scene from an S3 input manifest."""
    _call(
        runtime.plan,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
        output_format,
    )


@app.command("validate")
@json_stdout_contract
def validate_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
):
    """Verify journal hashes, identities, finite trajectories and all denominators."""
    _call(
        runtime.validate,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
        output_format,
    )


@app.command("visualize")
@json_stdout_contract
def visualize_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
):
    """Build and verify RRD joint timelines and actual forward-kinematics paths."""
    _call(
        runtime.visualize,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
        output_format,
    )
