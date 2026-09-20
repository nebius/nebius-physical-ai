"""Thin CLI for the shared SeedVR2 video-restoration implementation."""

from __future__ import annotations

from enum import Enum
import json
from typing import Callable

import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.cli.path_contract import (
    PathContractError,
    validate_read_path,
    validate_write_path,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.schemas import (
    MODEL_FILES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    RestoreRequest,
    VideoArtifactRequest,
)


app = typer.Typer(
    name="seedvr2",
    help="Restore low-resolution video with official ByteDance SeedVR2-3B.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    """Supported machine-readable output formats."""

    json = "json"
    text = "text"


def _paths(input_path: str, output_path: str) -> tuple[str, str]:
    try:
        return (
            validate_read_path(
                input_path, tool="seedvr2", allow_hf=False, required=True
            ),
            validate_write_path(output_path, tool="seedvr2", required=True),
        )
    except PathContractError as exc:
        typer.echo(f"SeedVR2 failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _optional_read_path(path: str) -> str:
    if not path:
        return ""
    try:
        return validate_read_path(path, tool="seedvr2", allow_hf=False, required=True)
    except PathContractError as exc:
        typer.echo(f"SeedVR2 failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _run(operation: Callable, request: object, output_format: OutputFormat) -> None:
    try:
        document = operation(request)
    except (runtime.SeedVR2Error, ValueError) as exc:
        typer.echo(f"SeedVR2 failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    else:
        typer.echo(
            f"status={document.get('status', 'unknown')} "
            f"run_id={document.get('run_id', 'n/a')}"
        )


@app.command("probe")
@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def probe_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Decode the exact input video and publish a verified media manifest."""

    input_path, output_path = _paths(input_path, output_path)
    _run(
        artifacts.probe,
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        ),
        output_format,
    )


@app.command("restore")
@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def restore_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    probe_path: str = typer.Option(
        "", "--probe-path", help="Required for non-dry restoration."
    ),
    output_height: int = typer.Option(480, "--output-height", min=16),
    output_width: int = typer.Option(640, "--output-width", min=16),
    seed: int = typer.Option(666, "--seed"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Run pinned one-step SeedVR2-3B and publish immutable artifacts."""

    input_path, output_path = _paths(input_path, output_path)
    probe_path = _optional_read_path(probe_path)
    _run(
        runtime.restore,
        RestoreRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            probe_path=probe_path,
            output_height=output_height,
            output_width=output_width,
            seed=seed,
            dry_run=dry_run,
        ),
        output_format,
    )


@app.command("verify")
@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def verify_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Re-download and independently verify a delivered SeedVR2 result."""

    input_path, output_path = _paths(input_path, output_path)
    _run(
        artifacts.verify,
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        ),
        output_format,
    )


@app.command("review")
@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def review_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Build a non-blended bicubic-baseline versus SeedVR2 review package."""

    input_path, output_path = _paths(input_path, output_path)
    _run(
        artifacts.review,
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        ),
        output_format,
    )


@app.command("status")
@resolve_typer_defaults
@intent_boundary(OperationIntent.OBSERVE)
def status_cmd() -> None:
    """Report local batch-stage readiness without claiming model-cache state."""

    typer.echo(json.dumps({"status": "ready", "weights_baked": False}, sort_keys=True))


@app.command("system-info")
@resolve_typer_defaults
@intent_boundary(OperationIntent.OBSERVE)
def system_info_cmd() -> None:
    """Report immutable source, model, and payload identities."""

    typer.echo(
        json.dumps(
            {
                "source": {
                    "repository": SOURCE_REPOSITORY,
                    "revision": SOURCE_REVISION,
                    "license": "Apache-2.0",
                },
                "model": {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "license": "Apache-2.0",
                    "runtime_fetch": True,
                    "files": MODEL_FILES,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("list")
@resolve_typer_defaults
@intent_boundary(OperationIntent.OBSERVE)
def list_cmd() -> None:
    """List the real video-restoration and evidence operations."""

    typer.echo(
        json.dumps(
            {"capabilities": ["probe", "restore", "verify", "review"]},
            sort_keys=True,
        )
    )
