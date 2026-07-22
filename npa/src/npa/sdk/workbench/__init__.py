"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb, training_config

from . import (
    byof,
    cosmos,
    cosmos2,
    cosmos3,
    data,
    dataset,
    detection_training,
    mjlab,
    retargeting,
    sim2real,
    sim2real_envgen,
    sonic,
    token_factory,
    trigger,
    vlm_eval,
    workflow,
)

__all__ = [
    "byof",
    "cosmos",
    "cosmos2",
    "cosmos3",
    "data",
    "dataset",
    "detection_training",
    "lancedb",
    "mjlab",
    "retargeting",
    "sim2real",
    "sim2real_envgen",
    "sonic",
    "token_factory",
    "training_config",
    "trigger",
    "vlm_eval",
    "workflow",
]
