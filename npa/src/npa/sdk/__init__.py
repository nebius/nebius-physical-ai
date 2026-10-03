"""npa.sdk - the public Python SDK namespace.

``npa.sdk`` mirrors the supported ``npa`` CLI namespaces. ``npa.sdk.workbench``
holds the per-tool clients; ``fleet``, ``provisioning``, and ``soperator`` hold
the platform-infrastructure surfaces.

Submodules are imported on first access (PEP 562). Importing the namespace
therefore costs nothing, which is what lets the minimal workbench images import
``npa.sdk`` without the full dependency closure.
"""

from __future__ import annotations

import importlib as _importlib
import os as _os
from typing import TYPE_CHECKING

_LIGHT_OPENARM = (
    _os.environ.get("NPA_SKIP_EAGER_IMPORTS", "").strip().lower()
    in {
        "1",
        "true",
        "yes",
    }
    and _os.environ.get("NPA_LIGHT_WORKBENCH_TOOL", "").strip().lower() == "openarm"
)

#: Namespaces reachable as attributes. The OpenArm image narrows this to the one
#: namespace it serves so nothing else can be pulled in behind its back.
_SUBMODULES: tuple[str, ...] = (
    ("workbench",)
    if _LIGHT_OPENARM
    else ("fleet", "provisioning", "soperator", "workbench")
)

__all__ = list(_SUBMODULES)

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.sdk import (  # noqa: F401
        fleet,
        provisioning,
        soperator,
        workbench,
    )


def __getattr__(name: str):
    """Import an SDK namespace on first access (PEP 562)."""
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_SUBMODULES)
