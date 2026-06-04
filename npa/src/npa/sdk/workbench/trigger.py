"""Compatibility SDK namespace for Workbench retriggers."""

from __future__ import annotations

from npa.workbench.trigger import run_once, watch
from npa.workflows.sim_to_real_trigger import (
    PipelineLaunch,
    SimToRealTriggerError,
    TriggerConfig,
    TriggerObject,
    TriggerResult,
    TriggerWatermark,
)

__all__ = [
    "PipelineLaunch",
    "SimToRealTriggerError",
    "TriggerConfig",
    "TriggerObject",
    "TriggerResult",
    "TriggerWatermark",
    "run_once",
    "watch",
]
