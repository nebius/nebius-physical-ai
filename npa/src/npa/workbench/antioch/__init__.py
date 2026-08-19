"""Antioch control-plane integration for NPA Workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .manager import AntiochManager
    from .openpi_bridge import OpenPIWebsocketClient, render_stack
    from .schemas import CollectRequest, OperationRecord, ResumeRequest, SubmitRequest

__all__ = [
    "AntiochManager",
    "CollectRequest",
    "OperationRecord",
    "OpenPIWebsocketClient",
    "ResumeRequest",
    "SubmitRequest",
    "render_stack",
]


def __getattr__(name: str) -> Any:
    """Keep the offline dataset stack optional for health/bridge-only images."""

    exports = {
        "AntiochManager": ("manager", "AntiochManager"),
        "CollectRequest": ("schemas", "CollectRequest"),
        "OperationRecord": ("schemas", "OperationRecord"),
        "OpenPIWebsocketClient": ("openpi_bridge", "OpenPIWebsocketClient"),
        "ResumeRequest": ("schemas", "ResumeRequest"),
        "SubmitRequest": ("schemas", "SubmitRequest"),
        "render_stack": ("openpi_bridge", "render_stack"),
    }
    target = exports.get(name)
    if target is not None:
        from importlib import import_module

        module_name, attribute = target
        return getattr(import_module(f"{__name__}.{module_name}"), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
