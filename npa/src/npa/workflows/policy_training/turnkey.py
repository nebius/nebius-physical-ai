"""Expose the public policy pipeline as stateless stages of the Workbench runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def build_parser() -> argparse.ArgumentParser:
    """Define the public stage argument contract.

    Args:
        None.
    Returns:
        Parser shared by workers and catalog validation.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    flags = {
        "prepare": ("recipe", "run-id", "workflow-sha256"),
        "curate": ("input-uri",),
        "split": ("input-uri",),
        "train": ("input-uri", "phase", "iteration", "phase-uri", "parent-uri"),
        "evaluate": ("input-uri", "phase"),
        "gate": ("input-uri",),
        "export": ("input-uri",),
        "serve": ("input-uri",),
        "report": ("input-uri",),
    }
    for stage, names in flags.items():
        child = commands.add_parser(stage)
        for name in (*names, "output-uri"):
            child.add_argument("--" + name, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one stage with private diagnostics and a machine-readable result.

    Args:
        argv: Optional process arguments.
    Returns:
        Zero after verified publication.
    Raises:
        Exception: A native component or artifact exchange fails.
    """
    from .diagnostics import _failure
    from .turnkey_store import publish

    args = build_parser().parse_args(argv)
    os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="policy-public-") as temporary:
            workspace = Path(temporary)
            output = workspace / "output"
            output.mkdir()
            _execute_with_diagnostics(args, workspace, output)
            if args.stage != "serve":
                publish(output, args.output_uri)
            if (
                args.stage == "report"
                and not json.loads((output / "proof.json").read_text())["qualified"]
            ):
                raise ValueError(
                    "deployment qualification failed; measured HTML proof was retained"
                )
    except Exception as exc:
        _failure(args.output_uri.rstrip("/") + "-diagnostics/error.json", exc)
        raise SystemExit(f"public policy stage failed ({type(exc).__name__})") from None
    print(
        json.dumps(
            {"stage": args.stage, "status": "completed", "output_uri": args.output_uri}
        )
    )
    return 0


def _execute_with_diagnostics(args, workspace, output):
    from .diagnostics import _cleanup

    try:
        _execute(args, workspace, output)
    except Exception:
        _cleanup(lambda: _retain_logs(args, workspace), args.output_uri)
        raise


def _retain_logs(args, workspace):
    import shutil
    from .turnkey_store import publish

    logs = workspace / "failure-logs"
    logs.mkdir()
    for path in workspace.rglob("*.log"):
        if path.is_relative_to(logs):
            continue
        target = logs / path.relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    publish(logs, args.output_uri.rstrip("/") + "-diagnostics/logs/")


def _execute(args, workspace, output):
    if args.stage in {"prepare", "curate", "split"}:
        from . import turnkey_data

        getattr(turnkey_data, args.stage)(args, workspace, output)
    elif args.stage in {"train", "evaluate", "gate", "export"}:
        from . import turnkey_training

        getattr(turnkey_training, args.stage)(args, workspace, output)
    elif args.stage == "serve":
        from .turnkey_serving import serve

        serve(args, workspace, output)
    else:
        from .turnkey_report import report

        report(args, workspace, output)


if __name__ == "__main__":
    raise SystemExit(main())
