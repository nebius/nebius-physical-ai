"""SDK surface for the Foxglove embedded viewer tooling.

Mirrors `npa workbench foxglove` so callers (the agent backend, workflows,
notebooks) can convert and inspect MCAP recordings without shelling out.
"""

from __future__ import annotations

from pathlib import Path

from npa.agent_backend.foxglove import foxglove_recording_link

from npa.workbench.foxglove import (
    DEFAULT_FOXGLOVE_EMBED_SRC,
    FOXGLOVE_EMBED_SDK_INTEGRITY,
    FOXGLOVE_EMBED_SDK_VERSION,
    sdk_assets_present,
    sdk_tarball_url,
)
from npa.workbench.foxglove.inspect import (
    McapInfo,
    McapInspectError,
    has_mcap_magic,
    summarize_mcap,
)
from npa.workbench.foxglove.mcap_writer import (
    FrameInput,
    LogInput,
    McapSummary,
    McapWriteError,
    MetricsInput,
    collect_run_inputs,
    convert_run_directory,
    write_run_mcap,
)

__all__ = [
    "DEFAULT_FOXGLOVE_EMBED_SRC",
    "FOXGLOVE_EMBED_SDK_INTEGRITY",
    "FOXGLOVE_EMBED_SDK_VERSION",
    "FrameInput",
    "LogInput",
    "McapInfo",
    "McapInspectError",
    "McapSummary",
    "McapWriteError",
    "MetricsInput",
    "collect_run_inputs",
    "convert_run",
    "export_run",
    "foxglove_recording_link",
    "has_mcap_magic",
    "inspect_mcap",
    "sdk_assets_present",
    "sdk_tarball_url",
    "write_run_mcap",
]


def convert_run(
    *,
    input_path: str | Path,
    output_path: str | Path,
    fps: float = 10.0,
    max_frames: int = 0,
    run_id: str = "",
) -> McapSummary:
    """Convert a run artifact directory into an MCAP recording for Foxglove."""
    return convert_run_directory(
        input_path=input_path,
        output=output_path,
        fps=fps,
        max_frames=max_frames,
        run_id=run_id,
    )


def inspect_mcap(input_path: str | Path) -> McapInfo:
    """Return the channels, schemas, and message counts of an MCAP recording."""
    return summarize_mcap(input_path)


def export_run(
    *,
    input_path: str | Path,
    output_path: str | Path,
    fps: float = 10.0,
    max_frames: int = 0,
    run_id: str = "",
) -> dict:
    """Convert run artifacts to an exportable MCAP."""
    summary = convert_run(
        input_path=input_path,
        output_path=output_path,
        fps=fps,
        max_frames=max_frames,
        run_id=run_id,
    )
    return {
        "summary": summary.to_dict(),
    }
