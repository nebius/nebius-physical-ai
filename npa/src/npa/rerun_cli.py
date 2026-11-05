"""Resolve the Rerun CLI paired with the active Python runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def resolve_rerun_cli() -> str:
    """Return an executable Rerun CLI for the active runtime.

    Args:
        None.

    Returns:
        The interpreter-sibling executable, or the executable found on PATH.

    Raises:
        RuntimeError: No executable Rerun CLI is available.
    """

    sibling = Path(sys.executable).with_name("rerun")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)

    discovered = shutil.which("rerun")
    if discovered:
        candidate = Path(discovered)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return discovered

    raise RuntimeError("Rerun CLI is unavailable")
