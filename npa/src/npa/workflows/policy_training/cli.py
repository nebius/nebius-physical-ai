"""Expose policy pipeline stages to the workflow tool catalog."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used by stage workers and argv contract checks.

    Args:
        None.
    Returns:
        Stage argument parser.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    options = {
        "curate": ("input-uri", "output-uri", "policy-uri"),
        "split": ("input-uri", "output-uri", "seed"),
        "batch": (
            "settings-uri",
            "stage",
            "partition",
            "split-uri",
            "input-uri",
            "output-uri",
            "run-id",
            "iteration",
        ),
        "gate": (
            "evaluation-uri",
            "candidate-uri",
            "split-uri",
            "policy-uri",
            "phase",
            "output-uri",
        ),
    }
    for command, flags in options.items():
        child = commands.add_parser(command)
        for flag in flags:
            child.add_argument("--" + flag, required=True)
    return parser


def main() -> None:
    """Execute one configured stage; propagate failures without printing inputs.

    Args:
        None.
    Returns:
        None.
    Raises:
        SystemExit: Invalid arguments or stage failure.
    """
    from .batch import batch
    from .data import curate, split
    from .gate import gate

    arguments = vars(build_parser().parse_args())
    command = arguments.pop("command")
    try:
        {"curate": curate, "split": split, "batch": batch, "gate": gate}[command](
            **arguments
        )
    except Exception as exc:
        # Worker diagnostics can contain private dataset paths and job output.
        raise SystemExit(
            f"policy stage failed ({type(exc).__name__}); inspect private worker evidence"
        ) from None
