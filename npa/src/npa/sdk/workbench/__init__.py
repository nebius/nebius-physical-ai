"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb

from . import data, detection_training, vlm_eval

__all__ = ["data", "detection_training", "lancedb", "vlm_eval"]
