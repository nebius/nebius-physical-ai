"""Compatibility namespace for the npa SDK."""

from __future__ import annotations

import os

_LIGHT_OPENARM = (
    os.environ.get("NPA_SKIP_EAGER_IMPORTS", "").strip().lower()
    in {
        "1",
        "true",
        "yes",
    }
    and os.environ.get("NPA_LIGHT_WORKBENCH_TOOL", "").strip().lower() == "openarm"
)

if _LIGHT_OPENARM:
    from . import workbench

    __all__ = ["workbench"]
else:
    from . import fleet, provisioning, soperator, workbench

    __all__ = ["fleet", "provisioning", "soperator", "workbench"]
