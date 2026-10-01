"""Expose individual video sweep stages to the standard workflow runtime."""

from __future__ import annotations

import argparse
import math
import sys

from npa.workflows.video_sweep.execution import generate, review
from npa.workflows.video_sweep.planning import prepare
from npa.workflows.video_sweep.publication import publish
from npa.workflows.video_sweep.tracking import lineage


def build_parser() -> argparse.ArgumentParser:
    """Construct the stage parser audited by the toolRef argv guardrail.

    Args:
        None.
    Returns:
        Argument parser.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "generate", "review", "lineage", "publish")
    )
    parser.add_argument("--root-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument(
        "--generator",
        choices=("cosmos-transfer2.5", "cosmos3-nano"),
        default="cosmos-transfer2.5",
    )
    parser.add_argument("--sources-uri", default="")
    parser.add_argument("--variants-uri", default="")
    parser.add_argument("--reasoner-model", default="nvidia/Cosmos3-Super-Reasoner")
    parser.add_argument("--merge-model", default="nvidia/Nemotron-3_5-Lightning")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one stage and keep input data and credentials out of task logs.

    Args:
        argv: Optional command arguments.
    Returns:
        Zero on success, one on a failed stage.
    Raises:
        SystemExit: Argument parsing fails.
    """
    args = build_parser().parse_args(argv)
    try:
        if (
            args.workers < 1
            or args.samples < 2
            or not math.isfinite(args.threshold)
            or not 0 <= args.threshold <= 1
        ):
            raise ValueError("Invalid worker, sample, or threshold configuration")
        {
            "prepare": prepare,
            "generate": generate,
            "review": review,
            "lineage": lineage,
            "publish": publish,
        }[args.stage](args)
    except Exception as error:  # noqa: BLE001 - do not print provider bodies or input paths
        print(
            f"Video sweep {args.stage} failed ({type(error).__name__}); inspect private stage artifacts and prerequisite checks.",
            file=sys.stderr,
        )
        return 1
    print(f"Video sweep {args.stage} completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
