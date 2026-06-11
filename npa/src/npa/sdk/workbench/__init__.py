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
    retargeting,
    sim2real,
    sim2real_envgen,
    sonic,
    token_factory,
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
    "retargeting",
    "sim2real",
    "sim2real_envgen",
    "sonic",
    "token_factory",
    "trigger",
    "vlm_eval",
]
