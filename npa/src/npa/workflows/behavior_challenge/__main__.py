"""Expose local protocol planning and the internal BEHAVIOR workflow stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import evaluate
from .protocol import make_plan, verify_upstream


def main() -> None:
    """Plan locally or invoke the licensed evaluator through the workflow stage.

    Args:
        None; reads command-line arguments.
    Returns:
        None.
    Raises:
        SystemExit: Arguments are invalid.
        ValueError: The requested evaluation violates the supported protocol.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Freeze a recipe without simulation or asset access"
    )
    plan.add_argument("--recipe-path", type=Path, required=True)
    plan.add_argument("--upstream-root", type=Path, required=True)
    run = commands.add_parser(
        "evaluate", help="Internal worker stage; use the workbench workflow"
    )
    run.add_argument("--input-path", required=True)
    run.add_argument("--output-path", required=True)
    run.add_argument("--policy-readme-uri", required=True)
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--evaluator-python", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "plan":
        verify_upstream(args.upstream_root)
        result = make_plan(
            json.loads(args.recipe_path.read_bytes()), args.upstream_root
        )
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
