"""Compatibility namespace for workbench SDK functions."""

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
    from . import openarm

    __all__ = ["openarm"]
else:
    from npa.workbench import lancedb, training_config

    from . import (
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
        lichtblick,
        mjlab,
        nurec,
        openarm,
        retargeting,
        robocasa,
        scenario_gen,
        sim2real,
        sim2real_envgen,
        sonic,
        token_factory,
        trigger,
        vlm_eval,
        workflow,
    )

    __all__ = [
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
        "sonic",
        "token_factory",
        "training_config",
        "trigger",
        "vlm_eval",
        "workflow",
    ]
