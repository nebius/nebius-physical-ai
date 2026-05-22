"""npa - Nebius Physical AI CLI/SDK.

This package exposes a Python SDK surface that mirrors supported npa CLI
namespaces. The SDK is currently v0: pin the npa version for integrations until
the public API reaches v1 stability.
"""

from __future__ import annotations

import importlib
from typing import Any

from npa import convert, demo, errors, network, rerun, solutions, workflow


def __getattr__(name: str) -> Any:
    if name == "workbench":
        return importlib.import_module(".workbench", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "convert",
    "demo",
    "errors",
    "network",
    "rerun",
    "solutions",
    "workflow",
    "workbench",
]
