"""Deprecated compatibility shim for npa.solutions.workbench.groot."""

from __future__ import annotations

from npa.workbench._compat import install_shim

install_shim(__name__, "npa.solutions.workbench.groot", globals())
