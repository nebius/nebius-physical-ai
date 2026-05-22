"""Deprecated compatibility shim for npa.solutions.workbench.genesis."""

from __future__ import annotations

from npa.workbench._compat import install_shim

install_shim(__name__, "npa.solutions.workbench.genesis", globals())
