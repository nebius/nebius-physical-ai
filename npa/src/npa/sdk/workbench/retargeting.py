"""SDK helpers for the Workbench retargeting CLI."""

from __future__ import annotations

from typing import Any

from npa._sdk import make_cli_wrapper
from npa.workbench.retargeting import (
    RetargetingResult,
    metadata_uri_for,
    result_uri_for,
    run_retargeting,
    validate_motion_lib,
    write_metadata,
)


def run(
    *,
    input_path: str,
    output_path: str,
    source_format: str = "auto",
    embodiment: str = "unitree-g1",
    retarget_map: str = "",
    frame_rate: int = 30,
    source_frame_rate: int = 0,
    max_frames: int = 0,
    individual: bool = True,
    num_workers: int = 4,
    dry_run: bool = False,
    sonic_home: str = "",
    **kwargs: Any,
) -> RetargetingResult:
    """Run SONIC retargeting/preprocess and write real motion-lib outputs."""

    return run_retargeting(
        input_path=input_path,
        output_path=output_path,
        source_format=source_format,
        embodiment=embodiment,
        retarget_map=retarget_map,
        frame_rate=frame_rate,
        source_frame_rate=source_frame_rate,
        max_frames=max_frames,
        individual=individual,
        num_workers=num_workers,
        dry_run=dry_run,
        sonic_home=sonic_home,
        **kwargs,
    )


workflow = make_cli_wrapper("npa.cli.workbench.retargeting", "workflow_cmd", "Show retargeting workflow.")
status = make_cli_wrapper("npa.cli.workbench.retargeting", "status_cmd", "Show retargeting status.")
list = make_cli_wrapper("npa.cli.workbench.retargeting", "list_cmd", "List retargeting source formats.")

__all__ = [
    "RetargetingResult",
    "metadata_uri_for",
    "result_uri_for",
    "run",
    "status",
    "validate_motion_lib",
    "workflow",
    "write_metadata",
]
