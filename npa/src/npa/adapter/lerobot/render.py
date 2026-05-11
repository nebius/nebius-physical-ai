"""Core LeRobotDataset-to-MP4 rendering logic."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

RendererName = Literal["matplotlib", "rerun"]
LayoutName = Literal["single", "side-by-side", "overlay"]


@dataclass(frozen=True)
class LeRobotMP4RenderResult:
    local_path: Path
    saved_to: str
    duration_s: float
    resolution: tuple[int, int]
    fps: int
    frame_count: int


class LeRobotMP4RenderError(Exception):
    """Raised for user-facing LeRobot MP4 rendering failures."""


def render_lerobot_to_mp4(
    input_path: Path | str,
    output_path: Path | str,
    renderer: RendererName = "matplotlib",
    duration: float | None = None,
    predictions_path: Path | str | None = None,
    layout: LayoutName = "single",
    title: str | None = None,
    *,
    resolution: str | tuple[int, int] = "1280x720",
    fps: int = 30,
) -> Path:
    """Render a LeRobotDataset trajectory to MP4 and return the local output path."""
    return render_lerobot_to_mp4_result(
        input_path=input_path,
        output_path=output_path,
        renderer=renderer,
        duration=duration,
        predictions_path=predictions_path,
        layout=layout,
        title=title,
        resolution=resolution,
        fps=fps,
    ).local_path


def render_lerobot_to_mp4_result(
    *,
    input_path: Path | str,
    output_path: Path | str,
    renderer: RendererName | str = "matplotlib",
    duration: float | None = None,
    predictions_path: Path | str | None = None,
    layout: LayoutName | str = "single",
    title: str | None = None,
    resolution: str | tuple[int, int] = "1280x720",
    fps: int = 30,
) -> LeRobotMP4RenderResult:
    """Core LeRobot-to-MP4 rendering used by CLI entry points."""
    from npa.cli.viz.backends import BackendUnavailable, get_backend
    from npa.viz.lerobot import (
        G1_JOINT_CONNECTIONS,
        VizDataError,
        load_render_inputs,
        parse_resolution,
    )

    if fps <= 0:
        raise LeRobotMP4RenderError(f"fps must be positive, got {fps}")
    try:
        backend = get_backend(str(renderer))
        parsed_resolution = _parse_resolution(resolution, parse_resolution)
    except (BackendUnavailable, VizDataError) as exc:
        raise LeRobotMP4RenderError(str(exc)) from exc

    output_ref = str(output_path)
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
            layout=str(layout),
            duration_s=duration,
            output_fps=fps,
        )
        local_output = _prepare_output_path(output_ref, temp_dirs)
        backend.render(
            render_inputs.skeleton_data,
            render_inputs.predictions_data,
            str(layout),
            local_output,
            parsed_resolution,
            fps,
            render_inputs.duration_s,
            title or render_inputs.title,
            G1_JOINT_CONNECTIONS,
        )
        saved_to = _finalize_output_path(local_output, output_ref)
        return LeRobotMP4RenderResult(
            local_path=local_output,
            saved_to=saved_to,
            duration_s=render_inputs.duration_s,
            resolution=parsed_resolution,
            fps=fps,
            frame_count=int(render_inputs.skeleton_data.shape[0]),
        )
    except VizDataError as exc:
        raise LeRobotMP4RenderError(str(exc)) from exc
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


def _parse_resolution(
    resolution: str | tuple[int, int],
    parse_resolution,
) -> tuple[int, int]:
    if isinstance(resolution, tuple):
        return resolution
    return parse_resolution(resolution)


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _materialize_lerobot_path(
    input_path: Path | str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    input_ref = str(input_path)
    if not _is_s3_uri(input_ref):
        return Path(input_path)

    from npa.clients.storage import StorageClient

    temp_dir = tempfile.TemporaryDirectory(prefix="npa-viz-lerobot-")
    temp_dirs.append(temp_dir)
    return Path(
        StorageClient.from_environment().download_directory(input_ref, temp_dir.name)
    )


def _materialize_predictions_path(
    predictions_path: Path | str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    predictions_ref = str(predictions_path)
    if not _is_s3_uri(predictions_ref):
        return Path(predictions_path)

    from npa.clients.storage import StorageClient

    temp_dir = tempfile.TemporaryDirectory(prefix="npa-viz-predictions-")
    temp_dirs.append(temp_dir)
    return Path(
        StorageClient.from_environment().download_path(predictions_ref, temp_dir.name)
    )


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
        raise LeRobotMP4RenderError(
            f"Renderer did not create output file: {local_output}"
        )
    if not _is_s3_uri(output_path):
        return str(local_output)

    from npa.clients.storage import StorageClient

    return StorageClient.from_environment().upload_file(str(local_output), output_path)


def _s3_leaf_name(uri: str) -> str:
    parsed = urlparse(uri)
    name = Path(parsed.path.rstrip("/")).name
    return name or "lerobot-viz.mp4"
