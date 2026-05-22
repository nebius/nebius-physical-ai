"""Deprecated compatibility namespace for npa.solutions.workbench."""

from __future__ import annotations

import warnings

warnings.warn(
    "npa.workbench is deprecated, use npa.solutions.workbench",
    DeprecationWarning,
    stacklevel=2,
)
from npa.solutions.workbench import *  # noqa: E402,F403
from npa.solutions.workbench import __all__ as __all__  # noqa: E402
