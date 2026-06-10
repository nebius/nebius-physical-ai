"""Compatibility imports for SONIC ONNX export SDK functions."""

from __future__ import annotations

from typing import Any

from npa._sdk import make_cli_wrapper
from npa.workbench.retargeting import RetargetingResult, run_retargeting
from npa.workbench.sonic import (
    SonicExportError,
    SonicExportResult,
    SonicParityResult,
    export_onnx,
    load_export_metadata,
    validate_onnx_parity,
)
from npa.workbench.sonic.workflow import (
    SonicWorkflowPlan,
    materialize_sonic_workflow,
    submit_sonic_workflow,
)

train = make_cli_wrapper("npa.cli.workbench.sonic.train", "train_cmd", "Run SONIC training.")
eval = make_cli_wrapper(
    "npa.cli.workbench.sonic.eval",
    "eval_cmd",
    "Evaluate a SONIC ONNX policy.",
)
materialize_workflow = materialize_sonic_workflow
submit_workflow = submit_sonic_workflow


def retarget(
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
    """Retarget source motion into the SONIC embodiment schema."""

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


__all__ = [
    "RetargetingResult",
    "SonicExportError",
    "SonicExportResult",
    "SonicParityResult",
    "SonicWorkflowPlan",
    "export_onnx",
    "eval",
    "load_export_metadata",
    "materialize_workflow",
    "materialize_sonic_workflow",
    "retarget",
    "submit_workflow",
    "submit_sonic_workflow",
    "train",
    "validate_onnx_parity",
]
