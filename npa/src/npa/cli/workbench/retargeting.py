"""npa workbench retargeting - motion retargeting commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.workbench.retargeting import (
    SUPPORTED_SOURCE_FORMATS,
    RetargetingError,
    build_retargeting_manifest,
    write_result,
)

app = typer.Typer(
    name="retargeting",
    help="Motion retargeting for SONIC locomotion workflows.",
    no_args_is_help=True,
)
console = Console(stderr=True)

WORKFLOW_PATH = Path("npa/workflows/workbench/skypilot/retargeting.yaml")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class SourceFormat(str, Enum):
    amass = "amass"
    bvh = "bvh"
    isaac_lab = "isaac-lab"
    mocap_json = "mocap-json"
    usd = "usd"


@app.command("run")
def run_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="S3 or local source motion path.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "-o",
        help="S3 or local path for retargeted motions and manifest.",
    ),
    source_format: SourceFormat = typer.Option(
        SourceFormat.mocap_json,
        "--source-format",
        help="Source motion format.",
    ),
    embodiment: str = typer.Option("unitree-g1", "--embodiment", help="Target robot embodiment."),
    retarget_map: str = typer.Option("", "--retarget-map", help="Optional retarget-map path or URI."),
    frame_rate: int = typer.Option(50, "--frame-rate", help="Output frame rate in Hz."),
    max_frames: int = typer.Option(0, "--max-frames", help="Maximum frames to process; 0 means all."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the manifest artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Retarget source motion artifacts into the SONIC embodiment schema."""

    try:
        result = build_retargeting_manifest(
            input_path=input_path,
            output_path=output_path,
            source_format=source_format.value,
            embodiment=embodiment,
            retarget_map=retarget_map,
            frame_rate=frame_rate,
            max_frames=max_frames,
        )
        payload = asdict(result)
        effective_dry_run = dry_run or _env_dry_run()
        payload["dry_run"] = effective_dry_run
        if not effective_dry_run:
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except RetargetingError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("workflow")
def workflow_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the SkyPilot YAML template for retargeting."""

    _emit({"workflow": str(WORKFLOW_PATH)}, output)


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show retargeting tool status."""

    _emit(
        {
            "backend": "retargeting",
            "status": "available",
            "workflow": str(WORKFLOW_PATH),
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List supported retargeting source formats."""

    _emit({"source_formats": list(SUPPORTED_SOURCE_FORMATS)}, output)


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
