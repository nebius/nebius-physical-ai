"""npa workbench lichtblick - open-source, Foxglove-compatible MCAP/log web viewer."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer
from rich.console import Console

from npa.workbench.lichtblick import (
    DEFAULT_CAMERA_TOPIC,
    DEFAULT_FPS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LichtblickError,
    serve_viewer,
)

app = typer.Typer(
    name="lichtblick",
    help="Lichtblick (MPL-2.0) — an open-source, Foxglove-compatible MCAP / ROS-bag log viewer.",
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
        help="S3/local MCAP artifact, or (with --from-frames) an S3 prefix / dir of camera frames.",
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="Optional S3 or local path for viewer session/layout output.",
    ),
    from_frames: bool = typer.Option(
        False,
        "--from-frames/--no-from-frames",
        help="Treat --input-path as a camera-frame sequence (e.g. sim2real rollout/augment "
        "frames) and pack it into an MCAP of foxglove.CompressedImage messages.",
    ),
    topic: str = typer.Option(DEFAULT_CAMERA_TOPIC, "--topic", help="MCAP topic for exported frames."),
    fps: float = typer.Option(DEFAULT_FPS, "--fps", help="Playback rate for exported frames."),
    execute: bool = typer.Option(
        False,
        "--execute/--plan",
        help="Actually stage from S3 and run the viewer container (default: print the plan only).",
    ),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host/interface to bind."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="HTTP port to serve on."),
    image: str = typer.Option(
        "",
        "--image",
        help="Override the npa-lichtblick image ref (defaults to the pinned tag).",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Serve a robotics log in Lichtblick, staged from S3.

    With ``--from-frames`` the sim2real rollout/augment camera frames at
    ``--input-path`` are packed into a real MCAP of foxglove.CompressedImage
    messages. With ``--execute`` the artifact is staged and the viewer container
    is launched so the log is live at the printed deep-linked URL; otherwise the
    resolved plan is printed (infra-free).
    """

    try:
        plan = serve_viewer(
            input_path=input_path,
            host=host,
            port=port,
            image=image,
            output_path=output_path,
            from_frames=from_frames,
            topic=topic,
            fps=fps,
            execute=execute,
        )
    except LichtblickError as exc:
        _fail(str(exc))
        return
    _emit(plan.to_dict(), output)


# Alias: `launch` reads the same as `serve` for parity with other viewers.
@app.command("launch")
def launch_cmd(
    input_path: str = typer.Option(..., "--input-path", "-i", help="S3/local MCAP, or (with --from-frames) camera frames."),
    output_path: str = typer.Option("", "--output-path", "-o", help="Optional S3/local output path."),
    from_frames: bool = typer.Option(False, "--from-frames/--no-from-frames", help="Pack a camera-frame sequence into MCAP."),
    topic: str = typer.Option(DEFAULT_CAMERA_TOPIC, "--topic", help="MCAP topic for exported frames."),
    fps: float = typer.Option(DEFAULT_FPS, "--fps", help="Playback rate for exported frames."),
    execute: bool = typer.Option(False, "--execute/--plan", help="Stage and launch the viewer container."),
    host: str = typer.Option(DEFAULT_HOST, "--host", help="Host/interface to bind."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="HTTP port to serve on."),
    image: str = typer.Option("", "--image", help="Override the npa-lichtblick image ref."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Alias for ``serve``: stage and view a robotics log in Lichtblick."""

    serve_cmd(
        input_path=input_path,
        output_path=output_path,
        from_frames=from_frames,
        topic=topic,
        fps=fps,
        execute=execute,
        host=host,
        port=port,
        image=image,
        output=output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show Lichtblick tool status."""

    _emit(
        {
            "backend": "lichtblick",
            "compatibility": "foxglove-compatible",
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
    """List artifact formats the Lichtblick viewer can open."""

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
