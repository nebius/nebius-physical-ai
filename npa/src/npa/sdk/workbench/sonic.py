"""Compatibility imports for SONIC ONNX export SDK functions."""

from __future__ import annotations

from npa.workbench.sonic import (
    SonicExportError,
    SonicExportResult,
    SonicParityResult,
    export_onnx,
    load_export_metadata,
    validate_onnx_parity,
)

__all__ = [
    "SonicExportError",
    "SonicExportResult",
    "SonicParityResult",
    "export_onnx",
    "load_export_metadata",
    "validate_onnx_parity",
]
