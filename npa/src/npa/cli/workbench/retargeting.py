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

from npa.deploy.images import container_image_for_tool
from npa.workbench.retargeting import (
    SUPPORTED_SOURCE_FORMATS,
    RetargetingError,
    run_retargeting,
)

app = typer.Typer(
    name="retargeting",
    help="Motion retargeting for SONIC locomotion workflows.",
    no_args_is_help=True,
)
console = Console(stderr=True)

WORKFLOW_PATH = Path("npa/workflows/workbench/skypilot/retargeting.yaml")
DEFAULT_RETARGETING_IMAGE_ENV = "NPA_RETARGETING_IMAGE"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class SourceFormat(str, Enum):
    auto = "auto"
    soma_csv = "soma-csv"
    bones_seed_csv = "bones-seed-csv"
    deploy_pkl = "deploy-pkl"
    teleop_pkl = "teleop-pkl"
    motion_lib = "motion-lib"
    bvh = "bvh"


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
        SourceFormat.auto,
        "--source-format",
        help="Source format accepted by upstream SONIC preprocessors.",
    ),
    embodiment: str = typer.Option("unitree-g1", "--embodiment", help="Target robot embodiment."),
    retarget_map: str = typer.Option(
        "",
        "--retarget-map",
        help="Optional external SOMA/GMR retarget-map path or URI for provenance.",
    ),
    frame_rate: int = typer.Option(30, "--frame-rate", help="Output frame rate in Hz."),
    source_frame_rate: int = typer.Option(
        0,
        "--source-frame-rate",
        help="Source data frame rate in Hz; 0 lets the upstream converter use target FPS.",
    ),
    max_frames: int = typer.Option(0, "--max-frames", help="Maximum frames to process; 0 means all."),
    individual: bool = typer.Option(
        True,
        "--individual/--combined",
        help="Write one PKL per motion when the upstream converter supports it.",
    ),
    num_workers: int = typer.Option(4, "--num-workers", help="Parallel worker count."),
    sonic_home: str = typer.Option(
        "",
        "--sonic-home",
        envvar="SONIC_HOME",
        help="Path to a GR00T-WholeBodyControl checkout; defaults to SONIC_HOME or auto-fetch.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan the preprocess without writing outputs."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Retarget source motion artifacts into the SONIC embodiment schema."""

    try:
        effective_dry_run = dry_run or _env_dry_run()
        result = run_retargeting(
            input_path=input_path,
            output_path=output_path,
            source_format=source_format.value,
            embodiment=embodiment,
            retarget_map=retarget_map,
            frame_rate=frame_rate,
            source_frame_rate=source_frame_rate,
            max_frames=max_frames,
            individual=individual,
            num_workers=num_workers,
            sonic_home=sonic_home,
            dry_run=effective_dry_run,
        )
        payload = asdict(result)
        payload["dry_run"] = effective_dry_run
    except RetargetingError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("workflow")
def workflow_cmd(
    image: str = typer.Option(
        "",
        "--image",
        envvar=DEFAULT_RETARGETING_IMAGE_ENV,
        help="Retargeting workflow image. Also settable with NPA_RETARGETING_IMAGE.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the SkyPilot YAML template for retargeting."""

    _emit(
        {
            "workflow": str(WORKFLOW_PATH),
            "image_env": DEFAULT_RETARGETING_IMAGE_ENV,
            "image": image.strip() or container_image_for_tool("retargeting"),
        },
        output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show retargeting tool status."""

    _emit(
        {
            "backend": "sonic-motion-lib-converter",
            "status": "available",
            "workflow": str(WORKFLOW_PATH),
            "source_formats": list(SUPPORTED_SOURCE_FORMATS),
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
