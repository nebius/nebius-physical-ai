"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb

from . import cosmos, data, detection_training, sim2real, sim2real_envgen, sonic, trigger, vlm_eval

__all__ = [
    "cosmos",
    "data",
    "detection_training",
    "lancedb",
    "sim2real",
    "sim2real_envgen",
    "sonic",
    "trigger",
    "vlm_eval",
]
