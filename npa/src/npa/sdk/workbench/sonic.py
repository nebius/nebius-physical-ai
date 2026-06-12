"""Compatibility imports for SONIC ONNX export SDK functions."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.sonic import (
    SonicExportError,
    SonicExportResult,
    SonicParityResult,
    SonicRoutingError,
    classify_gpu_target,
    export_onnx,
    is_datacenter_headless_target,
    is_rt_core_target,
    load_export_metadata,
    validate_gpu_routing,
    validate_onnx_parity,
    validate_render_gpu_target,
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
    "SonicRoutingError",
    "SonicWorkflowPlan",
    "classify_gpu_target",
    "export_onnx",
    "eval",
    "is_datacenter_headless_target",
    "is_rt_core_target",
    "load_export_metadata",
    "materialize_workflow",
    "materialize_sonic_workflow",
    "submit_workflow",
    "submit_sonic_workflow",
    "train",
    "validate_gpu_routing",
    "validate_onnx_parity",
    "validate_render_gpu_target",
]
