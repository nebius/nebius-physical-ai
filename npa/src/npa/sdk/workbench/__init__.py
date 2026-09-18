"""npa.sdk.workbench - per-tool SDK clients.

Every name in :data:`__all__` is a supported entrypoint. Submodules are imported
on first access (PEP 562) so importing this namespace does not pull every tool's
dependency closure into the process.
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

_ALL_TOOLS: tuple[str, ...] = (
    "alpamayo2_super",
    "byof",
    "cosmos",
    "cosmos2",
    "cosmos3",
    "curobo",
    "data",
    "dataset",
    "detection_training",
    "foxglove",
    "insights",
    "lancedb",
    "lichtblick",
    "mjlab",
    "nurec",
    "openarm",
    "retargeting",
    "robocasa",
    "scenario_gen",
    "sim2real",
    "sim2real_envgen",
    "sim_to_real",
    "sonic",
    "token_factory",
    "trigger",
    "vlm_eval",
    "workflow",
)

#: Tool clients reachable as attributes. The OpenArm image narrows this to the
#: one tool it serves.
_TOOLS: tuple[str, ...] = ("openarm",) if _LIGHT_OPENARM else _ALL_TOOLS

#: Shared helpers that live in the implementation layer and are re-exported here
#: because they have no per-tool SDK module of their own.
_ALIASES: dict[str, str] = (
    {} if _LIGHT_OPENARM else {"training_config": "npa.workbench.training_config"}
)

__all__ = sorted([*_TOOLS, *_ALIASES])

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workbench import training_config  # noqa: F401

    from npa.sdk.workbench import (  # noqa: F401
        alpamayo2_super,
        byof,
        cosmos,
        cosmos2,
        cosmos3,
        curobo,
        data,
        dataset,
        detection_training,
        foxglove,
        insights,
        lancedb,
        lichtblick,
        mjlab,
        nurec,
        openarm,
        retargeting,
        robocasa,
        scenario_gen,
        sim2real,
        sim2real_envgen,
        sim_to_real,
        sonic,
        token_factory,
        trigger,
        vlm_eval,
        workflow,
    )


def __getattr__(name: str):
    """Import a tool client on first access (PEP 562)."""
    if name in _TOOLS:
        module = _importlib.import_module(f"{__name__}.{name}")
    elif name in _ALIASES:
        module = _importlib.import_module(_ALIASES[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return list(__all__)
