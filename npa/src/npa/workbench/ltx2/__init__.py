"""LTX-2.5 workbench tool: licence facts and generated-video validation.

The container definition lives in ``npa/docker/workbench/ltx2/``; it bakes no
LTX-2.5 code and no LTX-2.5 weights, and fetches both at run time against the
operator's own Hugging Face entitlement for the gated weights repository.

This package holds the parts that must be usable without a GPU: the record of
which licence text governs LTX-2.5 and where to read it, and the decoded-video
check that decides whether a generation run actually produced a clip.
"""

from __future__ import annotations

from npa.workbench.ltx2 import licensing

__all__ = ["licensing"]
