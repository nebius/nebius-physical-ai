"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.solutions.workbench import lancedb

from . import detection_training

__all__ = ["detection_training", "lancedb"]
