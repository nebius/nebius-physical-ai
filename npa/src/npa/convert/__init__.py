"""npa.convert - format conversion primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from npa.adapter.lerobot.render import (
    LeRobotMP4RenderResult,
    render_lerobot_to_mp4_result,
)
from npa.viz.adapters import groot_predictions_to_rerun, lerobot_to_rerun

RendererName = Literal["matplotlib", "rerun"]
LayoutName = Literal["single", "side-by-side", "overlay"]


def lerobot_to_mp4(
    *,
    input_path: str | Path,
    output_path: str | Path = "lerobot-viz.mp4",
    renderer: RendererName | str = "matplotlib",
    predictions_path: str | Path | None = None,
    layout: LayoutName | str = "single",
    duration: float | None = None,
    resolution: str | tuple[int, int] = "1280x720",
    fps: int = 30,
    title: str | None = None,
) -> LeRobotMP4RenderResult:
    """Render a LeRobot dataset trajectory to MP4."""
    return render_lerobot_to_mp4_result(
        input_path=input_path,
        output_path=output_path,
        renderer=renderer,
        predictions_path=predictions_path,
        layout=layout,
        duration=duration,
        resolution=resolution,
        fps=fps,
        title=title,
    )


def lerobot_to_rrd(
    *,
    input_path: str | Path,
    output_path: str | Path,
    duration: float | None = None,
    predictions_path: str | Path | None = None,
) -> Path:
    """Convert a LeRobot dataset, optionally with GR00T predictions, to Rerun RRD."""
    output = Path(output_path)
    if predictions_path:
        groot_predictions_to_rerun(
            predictions_path,
            input_path,
            output,
            duration_s=duration,
        )
    else:
        lerobot_to_rerun(input_path, output, duration_s=duration)
    return output


__all__ = ["lerobot_to_mp4", "lerobot_to_rrd"]
