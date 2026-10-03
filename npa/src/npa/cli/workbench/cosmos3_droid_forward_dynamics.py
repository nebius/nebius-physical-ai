"""CLI bindings for the Cosmos3 DROID forward-dynamics workflow stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.cosmos.droid_forward_dynamics import (
    CHECKPOINT_REVISION,
    controls_droid_forward_dynamics,
    evaluate_droid_forward_dynamics,
    predict_droid_forward_dynamics,
    prepare_droid_forward_dynamics,
    visualize_droid_forward_dynamics,
)


def _invoke(function: Callable[..., dict[str, Any]], **kwargs: Any) -> None:
    try:
        result = function(**kwargs)
    except Exception as exc:
        typer.echo(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, sort_keys=True))


@json_stdout_contract
def droid_fd_prepare_cmd(
    input_path: str = typer.Option(..., "--input-path", help="Held-out DROID selection manifest S3 URI."),
    output_path: str = typer.Option(..., "--output-path", help="Run-scoped preparation S3 prefix."),
) -> None:
    """Build the exact 17-frame composite and 16-step DROID action contract."""

    validate_read_path(input_path, tool="cosmos3 droid-fd-prepare", allow_hf=False)
    validate_write_path(output_path, tool="cosmos3 droid-fd-prepare", required=True)
    _invoke(prepare_droid_forward_dynamics, input_path=input_path, output_path=output_path)


@json_stdout_contract
def droid_fd_predict_cmd(
    input_path: str = typer.Option(..., "--input-path", help="Prepared DROID handoff manifest S3 URI."),
    output_path: str = typer.Option(..., "--output-path", help="Run-scoped true-prediction S3 prefix."),
    seed: int = typer.Option(0, "--seed", min=0),
    checkpoint_revision: str = typer.Option(CHECKPOINT_REVISION, "--checkpoint-revision"),
) -> None:
    """Run native Cosmos forward dynamics with the held-out sample's true actions."""

    validate_read_path(input_path, tool="cosmos3 droid-fd-predict", allow_hf=False)
    validate_write_path(output_path, tool="cosmos3 droid-fd-predict", required=True)
    _invoke(
        predict_droid_forward_dynamics,
        input_path=input_path,
        output_path=output_path,
        seed=seed,
        checkpoint_revision=checkpoint_revision,
    )


@json_stdout_contract
def droid_fd_controls_cmd(
    input_path: str = typer.Option(..., "--input-path", help="Prepared DROID handoff manifest S3 URI."),
    output_path: str = typer.Option(..., "--output-path", help="Run-scoped controls S3 prefix."),
    seed: int = typer.Option(0, "--seed", min=0),
    checkpoint_revision: str = typer.Option(CHECKPOINT_REVISION, "--checkpoint-revision"),
) -> None:
    """Run matched temporal-permutation and zero-action native controls."""

    validate_read_path(input_path, tool="cosmos3 droid-fd-controls", allow_hf=False)
    validate_write_path(output_path, tool="cosmos3 droid-fd-controls", required=True)
    _invoke(
        controls_droid_forward_dynamics,
        input_path=input_path,
        output_path=output_path,
        seed=seed,
        checkpoint_revision=checkpoint_revision,
    )


@json_stdout_contract
def droid_fd_evaluate_cmd(
    prepared_path: str = typer.Option(..., "--prepared-path", help="Prepared handoff manifest S3 URI."),
    prediction_path: str = typer.Option(..., "--prediction-path", help="True-prediction manifest S3 URI."),
    controls_path: str = typer.Option(..., "--controls-path", help="Control-prediction manifest S3 URI."),
    output_path: str = typer.Option(..., "--output-path", help="Run-scoped metrics S3 prefix."),
) -> None:
    """Measure actual held-out RGB error and true-vs-control sensitivity."""

    for option, value in (
        ("--prepared-path", prepared_path),
        ("--prediction-path", prediction_path),
        ("--controls-path", controls_path),
    ):
        validate_read_path(value, tool="cosmos3 droid-fd-evaluate", option=option, allow_hf=False)
    validate_write_path(output_path, tool="cosmos3 droid-fd-evaluate", required=True)
    _invoke(
        evaluate_droid_forward_dynamics,
        prepared_path=prepared_path,
        prediction_path=prediction_path,
        controls_path=controls_path,
        output_path=output_path,
    )


@json_stdout_contract
def droid_fd_visualize_cmd(
    prepared_path: str = typer.Option(..., "--prepared-path"),
    prediction_path: str = typer.Option(..., "--prediction-path"),
    controls_path: str = typer.Option(..., "--controls-path"),
    evaluation_path: str = typer.Option(..., "--evaluation-path"),
    output_path: str = typer.Option(..., "--output-path", help="Run-scoped RRD S3 prefix."),
) -> None:
    """Write and independently verify a synchronized observation/prediction RRD."""

    for option, value in (
        ("--prepared-path", prepared_path),
        ("--prediction-path", prediction_path),
        ("--controls-path", controls_path),
        ("--evaluation-path", evaluation_path),
    ):
        validate_read_path(value, tool="cosmos3 droid-fd-visualize", option=option, allow_hf=False)
    validate_write_path(output_path, tool="cosmos3 droid-fd-visualize", required=True)
    _invoke(
        visualize_droid_forward_dynamics,
        prepared_path=prepared_path,
        prediction_path=prediction_path,
        controls_path=controls_path,
        evaluation_path=evaluation_path,
        output_path=output_path,
    )


__all__ = [
    "droid_fd_controls_cmd",
    "droid_fd_evaluate_cmd",
    "droid_fd_predict_cmd",
    "droid_fd_prepare_cmd",
    "droid_fd_visualize_cmd",
]
