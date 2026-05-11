"""Standalone LeRobotDataset to MP4 conversion command."""

from __future__ import annotations

from enum import Enum

import typer
from rich.console import Console

from npa.adapter.lerobot.render import (
    LeRobotMP4RenderError,
    render_lerobot_to_mp4_result,
)


console = Console(stderr=True)


class RendererName(str, Enum):
    matplotlib = "matplotlib"
    rerun = "rerun"


class LayoutName(str, Enum):
    single = "single"
    side_by_side = "side-by-side"
    overlay = "overlay"


def lerobot_to_mp4_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "--input",
        "-i",
        help="Local or s3:// LeRobotDataset directory.",
    ),
    renderer: RendererName = typer.Option(
        RendererName.matplotlib,
        "--renderer",
        help="Rendering engine.",
        case_sensitive=False,
    ),
    predictions_path: str = typer.Option(
        "", "--predictions-path", help="Local or s3:// GR00T prediction artifact path."
    ),
    layout: LayoutName = typer.Option(
        LayoutName.single,
        "--layout",
        help="Visualization layout.",
        case_sensitive=False,
    ),
    output_path: str = typer.Option(
        "lerobot-viz.mp4",
        "--output-path",
        "--output",
        "-o",
        help="Local or s3:// MP4 output path.",
    ),
    duration: float | None = typer.Option(
        None,
        "--duration",
        help="Rendered duration in seconds. Defaults to source duration capped at 10s.",
    ),
    resolution: str = typer.Option(
        "1280x720", "--resolution", help="Video resolution as WIDTHxHEIGHT."
    ),
    fps: int = typer.Option(30, "--fps", help="Output frame rate."),
    title: str = typer.Option("", "--title", help="Optional render title."),
) -> None:
    """Convert a LeRobotDataset trajectory to an MP4 visualization."""
    try:
        result = render_lerobot_to_mp4_result(
            input_path=input_path,
            output_path=output_path,
            renderer=renderer.value,
            predictions_path=predictions_path or None,
            layout=layout.value,
            duration=duration,
            resolution=resolution,
            fps=fps,
            title=title or None,
        )
    except LeRobotMP4RenderError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]Conversion complete.[/green]")
    console.print(f"  output: {result.saved_to}")
    console.print(f"  local:  {result.local_path}")
    console.print("  format: MP4")
