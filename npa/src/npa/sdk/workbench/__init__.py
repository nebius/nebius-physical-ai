"""Compatibility namespace for workbench SDK functions."""

from __future__ import annotations

from npa.workbench import lancedb

from . import detection_training

__all__ = ["detection_training", "lancedb"]
