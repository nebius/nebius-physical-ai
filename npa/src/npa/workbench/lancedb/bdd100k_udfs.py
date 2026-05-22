"""Deprecated compatibility shim for npa.solutions.workbench.lancedb.bdd100k_udfs."""

from __future__ import annotations

from npa.workbench._compat import install_shim

install_shim(__name__, "npa.solutions.workbench.lancedb.bdd100k_udfs", globals())
