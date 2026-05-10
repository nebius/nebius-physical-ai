"""LeRobot visualization CLI command."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console

from npa.cli.viz.backends import BackendUnavailable, get_backend


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
        ..., "--input-path", "--input", "-i", help="Local or s3:// LeRobotDataset directory."
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
        "lerobot-viz.mp4", "--output-path", "--output", "-o", help="Local or s3:// MP4 output path."
    ),
    duration: float | None = typer.Option(
        None, "--duration", help="Rendered duration in seconds. Defaults to source duration capped at 10s."
    ),
    resolution: str = typer.Option("1280x720", "--resolution", help="Video resolution as WIDTHxHEIGHT."),
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
    from npa.viz.lerobot import (
        G1_JOINT_CONNECTIONS,
        VizDataError,
        load_render_inputs,
        parse_resolution,
    )

    if fps <= 0:
        raise VizCLIError(f"fps must be positive, got {fps}")
    try:
        renderer = get_backend(backend)
        parsed_resolution = parse_resolution(resolution)
    except (BackendUnavailable, VizDataError) as exc:
        raise VizCLIError(str(exc)) from exc

    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    try:
        local_input = _materialize_lerobot_path(input_path, temp_dirs)
        local_predictions = (
            _materialize_predictions_path(predictions_path, temp_dirs)
            if predictions_path
            else None
        )
        render_inputs = load_render_inputs(
            local_input,
            predictions_path=local_predictions,
            layout=layout,
            duration_s=duration_s,
            output_fps=fps,
        )
        local_output = _prepare_output_path(output_path, temp_dirs)
        renderer.render(
            render_inputs.skeleton_data,
            render_inputs.predictions_data,
            layout,
            local_output,
            parsed_resolution,
            fps,
            render_inputs.duration_s,
            title or render_inputs.title,
            G1_JOINT_CONNECTIONS,
        )
        saved_to = _finalize_output_path(local_output, output_path)
        return VizRenderResult(
            local_path=local_output,
            saved_to=saved_to,
            duration_s=render_inputs.duration_s,
            resolution=parsed_resolution,
            fps=fps,
            frame_count=int(render_inputs.skeleton_data.shape[0]),
        )
    except VizDataError as exc:
        raise VizCLIError(str(exc)) from exc
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _materialize_lerobot_path(
    input_path: str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    if not _is_s3_uri(input_path):
        return Path(input_path)

    from npa.clients.storage import StorageClient

    temp_dir = tempfile.TemporaryDirectory(prefix="npa-viz-lerobot-")
    temp_dirs.append(temp_dir)
    return Path(StorageClient.from_environment().download_directory(input_path, temp_dir.name))


def _materialize_predictions_path(
    predictions_path: str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    if not _is_s3_uri(predictions_path):
        return Path(predictions_path)

    from npa.clients.storage import StorageClient

    temp_dir = tempfile.TemporaryDirectory(prefix="npa-viz-predictions-")
    temp_dirs.append(temp_dir)
    return Path(StorageClient.from_environment().download_path(predictions_path, temp_dir.name))


def _prepare_output_path(
    output_path: str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    if _is_s3_uri(output_path):
        temp_dir = tempfile.TemporaryDirectory(prefix="npa-viz-output-")
        temp_dirs.append(temp_dir)
        local_output = Path(temp_dir.name) / _s3_leaf_name(output_path)
    else:
        local_output = Path(output_path)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    return local_output


def _finalize_output_path(local_output: Path, output_path: str) -> str:
    if not local_output.exists():
        raise VizCLIError(f"Renderer did not create output file: {local_output}")
    if not _is_s3_uri(output_path):
        return str(local_output)

    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().upload_file(str(local_output), output_path)


def _s3_leaf_name(uri: str) -> str:
    parsed = urlparse(uri)
    name = Path(parsed.path.rstrip("/")).name
    return name or "lerobot-viz.mp4"
