"""LeRobot visualization CLI command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from npa.adapter.lerobot.render import (
    LeRobotMP4RenderError,
    _finalize_output_path as _finalize_output_path,
    _materialize_lerobot_path as _materialize_lerobot_path,
    _materialize_predictions_path as _materialize_predictions_path,
    _prepare_output_path as _prepare_output_path,
    render_lerobot_to_mp4_result,
)


console = Console(stderr=True)


class BackendName(str, Enum):
    matplotlib = "matplotlib"
    rerun = "rerun"


class LayoutName(str, Enum):
    single = "single"
    side_by_side = "side-by-side"
    overlay = "overlay"


@dataclass(frozen=True)
class VizRenderResult:
    local_path: Path
    saved_to: str
    duration_s: float
    resolution: tuple[int, int]
    fps: int
    frame_count: int


class VizCLIError(Exception):
    """Raised for user-facing viz CLI failures."""


def lerobot_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "--input",
        "-i",
        help="Local or s3:// LeRobotDataset directory.",
    ),
    backend: BackendName = typer.Option(
        BackendName.matplotlib,
        "--backend",
        help="Rendering backend.",
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
    """Render a LeRobotDataset trajectory to an MP4 visualization."""
    try:
        result = render_lerobot(
            input_path=input_path,
            backend=backend.value,
            predictions_path=predictions_path or None,
            layout=layout.value,
            output_path=output_path,
            duration_s=duration,
            resolution=resolution,
            fps=fps,
            title=title,
        )
    except VizCLIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]Render complete.[/green]")
    console.print(f"  output: {result.saved_to}")
    console.print(f"  local:  {result.local_path}")
    console.print(f"  frames: {result.frame_count}")
    console.print(f"  fps:    {result.fps}")
    console.print(f"  size:   {result.resolution[0]}x{result.resolution[1]}")


def render_lerobot(
    *,
    input_path: str,
    backend: str,
    predictions_path: str | None,
    layout: str,
    output_path: str,
    duration_s: float | None,
    resolution: str,
    fps: int,
    title: str = "",
) -> VizRenderResult:
    try:
        result = render_lerobot_to_mp4_result(
            input_path=input_path,
            output_path=output_path,
            renderer=backend,
            duration=duration_s,
            predictions_path=predictions_path,
            layout=layout,
            title=title or None,
            resolution=resolution,
            fps=fps,
        )
    except LeRobotMP4RenderError as exc:
        raise VizCLIError(str(exc)) from exc

    return VizRenderResult(
        local_path=result.local_path,
        saved_to=result.saved_to,
        duration_s=result.duration_s,
        resolution=result.resolution,
        fps=result.fps,
        frame_count=result.frame_count,
    )
