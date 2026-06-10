"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb

from . import (
    cosmos,
    cosmos2,
    cosmos3,
    data,
    detection_training,
    mjlab,
    sim2real,
    sim2real_envgen,
    sonic,
    trigger,
    vlm_eval,
)

__all__ = [
    "cosmos",
    "cosmos2",
    "cosmos3",
    "data",
    "detection_training",
    "lancedb",
    "mjlab",
    "sim2real",
    "sim2real_envgen",
    "sonic",
    "trigger",
    "vlm_eval",
]
