"""Lightweight console-script entry for the ``npa`` CLI.

Kept deliberately import-light. ``npa.cli.main`` eagerly imports the entire
command tree (agent / workbench / cluster / convert / …), which transitively
pulls in heavy dependencies such as ``boto3``, ``paramiko``, ``rerun`` and
``numpy`` — hundreds of milliseconds. A bare ``npa --version`` / ``npa -V`` does
not need any of that, so answer it here before importing ``npa.cli.main``.
"""

from __future__ import annotations

import sys

_VERSION_FLAGS = frozenset({"--version", "-V"})


def _is_bare_version_request(argv: list[str]) -> bool:
    """True when the invocation is only a top-level version flag."""
    return len(argv) == 1 and argv[0] in _VERSION_FLAGS


def _print_version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        resolved = version("npa")
    except PackageNotFoundError:  # pragma: no cover - source tree without install
        resolved = "0.0.0.dev0"
    print(f"npa {resolved}")


def main() -> None:
    """Console-script entry point (``npa`` in ``[project.scripts]``)."""
    if _is_bare_version_request(sys.argv[1:]):
        _print_version()
        return
    from npa.cli.main import app_entry

    app_entry()
