"""Outer pi0.5 evaluator entrypoint that distrusts the Kit process exit code."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from npa.workflows.sim2real.pi05_closed_loop import (
    build_parser,
    parse_args,
    validate_success_report,
)
from npa.workflows.sim2real.workflow_io import read_json

__all__ = ["build_parser", "main"]


def main(argv: list[str] | None = None) -> int:
    """Run Kit as a child and require the declared successful report bytes."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(forwarded)
    environment = dict(os.environ)
    environment["NPA_PI05_CLOSED_LOOP_CHILD"] = "1"
    command = [
        sys.executable,
        "-m",
        "npa.workflows.sim2real.pi05_closed_loop",
        *forwarded,
    ]
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"pi0.5 Kit evaluator failed with exit {completed.returncode}"
        )
    with tempfile.TemporaryDirectory(prefix="npa-pi05-report-readback-") as directory:
        report = read_json(args.output_uri, directory=Path(directory))
    validate_success_report(report, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
