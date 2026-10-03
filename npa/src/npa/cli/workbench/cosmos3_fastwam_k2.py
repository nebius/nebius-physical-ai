"""CLI bindings for closed-loop Cosmos3 Edge FastWAM-K2 evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.cosmos.fastwam_k2 import (
    compare_variants,
    emit_visualization,
    prepare_evaluation_inputs,
    run_variant,
)


def _invoke(function: Callable[..., dict[str, Any]], **kwargs: str) -> None:
    """Validate common S3 handoffs, emit one JSON document, and preserve failures."""
    try:
        for name, value in kwargs.items():
            if name != "output_path" and name.endswith("_path") and value:
                validate_read_path(value, tool="cosmos3 fastwam-k2", allow_hf=False)
        validate_write_path(kwargs["output_path"], tool="cosmos3 fastwam-k2", required=True)
        result = function(**kwargs)
    except Exception as exc:
        typer.echo(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result))


@json_stdout_contract
def prepare_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Prepare pinned, matched RoboLab task inputs for the two policy arms."""
    _invoke(prepare_evaluation_inputs, input_path=input_path, output_path=output_path)


@json_stdout_contract
def full_wam_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Run the exact full-WAM baseline through native closed-loop RoboLab."""
    _invoke(run_variant, input_path=input_path, output_path=output_path, variant="full-wam")


@json_stdout_contract
def fastwam_k2_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    baseline_path: str = typer.Option(..., "--baseline-path"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Run the K=2 runtime overlay against the matched full-WAM preparation."""
    _invoke(
        run_variant,
        input_path=input_path,
        baseline_path=baseline_path,
        output_path=output_path,
        variant="fastwam-k2",
    )


@json_stdout_contract
def compare_cmd(
    full_wam_path: str = typer.Option(..., "--full-wam-path"),
    k2_path: str = typer.Option(..., "--k2-path"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Compare matched closed-loop task success and native latency only."""
    _invoke(compare_variants, full_wam_path=full_wam_path, k2_path=k2_path, output_path=output_path)


@json_stdout_contract
def visualize_cmd(
    full_wam_path: str = typer.Option(..., "--full-wam-path"),
    k2_path: str = typer.Option(..., "--k2-path"),
    comparison_path: str = typer.Option(..., "--comparison-path"),
    output_path: str = typer.Option(..., "--output-path"),
) -> None:
    """Write a factual RRD and copied rollout MP4s from completed evidence."""
    _invoke(
        emit_visualization,
        full_wam_path=full_wam_path,
        k2_path=k2_path,
        comparison_path=comparison_path,
        output_path=output_path,
    )
