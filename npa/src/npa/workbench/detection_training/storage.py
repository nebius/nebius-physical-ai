"""Deprecated compatibility shim for npa.solutions.workbench.detection_training.storage."""

from __future__ import annotations

from npa.workbench._compat import install_shim

install_shim(__name__, "npa.solutions.workbench.detection_training.storage", globals())
