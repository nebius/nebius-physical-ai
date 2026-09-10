"""Narrow image entrypoint using the public NCore CLI callback unchanged.

The image aliases this module as npa.cli.entry and npa.__main__ for the existing
SkyPilot shim. The installed public CLI and SDK outside this image are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


def application():
    import typer
    from npa.cli.nurec import convert_colmap_cmd

    root = typer.Typer(no_args_is_help=True)
    workbench = typer.Typer(no_args_is_help=True)
    nurec = typer.Typer(no_args_is_help=True)
    nurec.command("convert-colmap")(convert_colmap_cmd)
    workbench.add_typer(nurec, name="nurec")
    root.add_typer(workbench, name="workbench")
    return root


def main() -> None:
    from npa.workflows.ncore_runtime import (
        NcoreRuntimeError,
        ensure_runtime,
        exec_runtime,
    )

    try:
        runtime = ensure_runtime()
        if Path(sys.prefix) != runtime:
            exec_runtime("npa.workflows.ncore", sys.argv[1:])
    except (NcoreRuntimeError, OSError):
        # Preserve the public JSON failure contract; never expose downloader or
        # cache diagnostics (which can include private paths) in worker output.
        if any(a == "--output-format=json" for a in sys.argv) or any(
            a == "--output-format" and b == "json"
            for a, b in zip(sys.argv, sys.argv[1:])
        ):
            print(
                '{"status":"failed","error":"NCore runtime preparation or integrity check failed"}'
            )
        else:
            print(
                "NCore runtime preparation or integrity check failed", file=sys.stderr
            )
        raise SystemExit(1) from None
    application()()


if __name__ == "__main__":
    main()
