"""Shared setup for NPA CLI modules."""

from __future__ import annotations

import os
import sys


def _plain_output_requested() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return True
    if os.environ.get("CI"):
        return True
    return not sys.stdout.isatty() or not sys.stderr.isatty()


def _configure_rich_plain_output() -> None:
    if not _plain_output_requested():
        return

    os.environ.setdefault("NO_COLOR", "1")
    try:
        import typer.rich_utils as typer_rich_utils
    except ImportError:
        return

    typer_rich_utils.COLOR_SYSTEM = None
    typer_rich_utils.FORCE_TERMINAL = False


_configure_rich_plain_output()
