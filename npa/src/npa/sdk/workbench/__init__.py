"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb

from . import cosmos, data, detection_training, sonic, vlm_eval

__all__ = ["cosmos", "data", "detection_training", "lancedb", "sonic", "vlm_eval"]
