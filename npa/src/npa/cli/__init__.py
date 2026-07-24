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

    # Rich (>=13) treats FORCE_COLOR=0 as "force terminal on" (it only checks
    # ``force_color != ""``), so on a host that exports FORCE_COLOR=0 with a
    # non-dumb TERM, every ``Console`` still emits bold SGR escapes even though
    # NO_COLOR is set -- NO_COLOR strips color but not bold. That corrupts the
    # plain-substring output assertions across the CLI suite whenever it runs
    # under tmux/CI (TERM != "dumb"). TTY_COMPATIBLE is evaluated before
    # FORCE_COLOR in Rich's terminal detection, so pinning it to "0" forces
    # every Rich console -- not just Typer's help renderer -- to plain output.
    os.environ["TTY_COMPATIBLE"] = "0"

    try:
        import typer.rich_utils as typer_rich_utils
    except ImportError:
        return

    typer_rich_utils.COLOR_SYSTEM = None
    typer_rich_utils.FORCE_TERMINAL = False


_configure_rich_plain_output()
