"""Compatibility imports for SONIC ONNX export SDK functions."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
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

__all__ = [
    "SonicExportError",
    "SonicExportResult",
    "SonicParityResult",
    "SonicWorkflowPlan",
    "export_onnx",
    "eval",
    "load_export_metadata",
    "materialize_workflow",
    "materialize_sonic_workflow",
    "submit_workflow",
    "submit_sonic_workflow",
    "train",
    "validate_onnx_parity",
]
