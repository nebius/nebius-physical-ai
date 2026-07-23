"""npa workbench foxglove - Lichtblick (OSS Foxglove) MCAP/log web viewer."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer
from rich.console import Console

from npa.workbench.foxglove import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    FoxgloveError,
    build_launch_plan,
)

app = typer.Typer(
    name="foxglove",
    help="Foxglove (Lichtblick MPL-2.0 OSS) web viewer for MCAP / ROS-bag logs.",
    no_args_is_help=True,
)
console = Console(stderr=True)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.command("serve")
def serve_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="S3 or local MCAP/ROS-bag artifact to open in the viewer.",
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="Optional S3 or local path for viewer session/layout output.",
    ),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host/interface to bind."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="HTTP port to serve on."),
    image: str = typer.Option(
        "",
        "--image",
        help="Override the npa-foxglove image ref (defaults to the pinned tag).",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Plan a Foxglove viewer serving an MCAP artifact staged from S3.

    Prints the resolved image, the container run command, and a deep-linked
    viewer URL. Actual container launch is performed by the workflow/deploy path;
    this keeps the command infra-free and testable.
    """

    try:
        plan = build_launch_plan(
            input_path=input_path,
            host=host,
            port=port,
            image=image,
            output_path=output_path,
        )
    except FoxgloveError as exc:
        _fail(str(exc))
        return
    _emit(plan.to_dict(), output)


# Alias: `launch` reads the same as `serve` for parity with other viewers.
@app.command("launch")
def launch_cmd(
    input_path: str = typer.Option(..., "--input-path", "-i", help="S3 or local MCAP/ROS-bag artifact."),
    output_path: str = typer.Option("", "--output-path", "-o", help="Optional S3/local output path."),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host/interface to bind."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="HTTP port to serve on."),
    image: str = typer.Option("", "--image", help="Override the npa-foxglove image ref."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Alias for ``serve``: plan a Foxglove viewer for an MCAP artifact."""

    serve_cmd(
        input_path=input_path,
        output_path=output_path,
        host=host,
        port=port,
        image=image,
        output=output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show Foxglove tool status."""

    _emit(
        {
            "backend": "foxglove",
            "implementation": "lichtblick",
            "license": "MPL-2.0",
            "tier": "service",
            "port": DEFAULT_PORT,
            "status": "available",
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List artifact formats the Foxglove viewer can open."""

    _emit({"formats": ["mcap", "bag", "db3"]}, output)


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
