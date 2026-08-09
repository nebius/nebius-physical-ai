"""Canonical same-interpreter invocation for internal NPA subprocesses."""

from __future__ import annotations

from collections.abc import Sequence
import sys


def internal_cli_argv(args: Sequence[str] = ()) -> list[str]:
    """Return the supported PATH-independent NPA module invocation."""

    return [sys.executable, "-m", "npa", *[str(value) for value in args]]


__all__ = ["internal_cli_argv"]
