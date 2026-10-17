"""BYOF helpers with live infrastructure dependencies loaded on first use."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ByofKubernetesTarget",
    "byof_validation_repo",
    "resolve_byof_project",
    "resolve_byof_kubernetes_target",
    "resolve_byof_resource_yaml",
    "resolve_skypilot_bin",
    "skypilot_config_for_project",
]


def __getattr__(name: str) -> Any:
    """Load the legacy live-helper exports without burdening worker modules."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("npa.workflows.byof.live"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public helpers in interactive module discovery."""

    return sorted({*globals(), *__all__})
