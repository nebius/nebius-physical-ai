"""Provide runtime-local commands for experimental N1.5 offline policy improvement."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from npa.workbench.groot.offline_evaluation import compare_rollouts
from npa.workbench.groot.offline_feedback import prepare_feedback


def main(argv: list[str] | None = None) -> None:
    """Run preparation, native CUDA training, or recorded rollout comparison.

    Args:
        argv: Optional command arguments; defaults to process arguments.
    Returns:
        None. Writes a single result JSON document to stdout.
    Raises:
        SystemExit: Arguments or runtime inputs are invalid.
    """
    parser = _parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        with redirect_stdout(sys.stderr):
            result = _execute(command, args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))


def _execute(command, args):
    if command == "prepare":
        return prepare_feedback(**args).model_dump(mode="json")
    if command == "compare":
        return compare_rollouts(**args)
    from npa.workbench.groot.offline_training import train_offline

    return train_offline(**args)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser(
        "prepare", help="Validate LeRobot v2 feedback and write weights"
    )
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--feedback-path", type=Path, required=True)
    prepare.add_argument("--output-path", type=Path, required=True)
    prepare.add_argument("--action-horizon", type=int, required=True)
    prepare.add_argument(
        "--mode", choices=["outcome", "progress", "uniform"], required=True
    )
    prepare.add_argument("--temperature", type=float, default=1.0)
    train = subcommands.add_parser(
        "train", help="Train in a configured native N1.5 CUDA environment"
    )
    for name in (
        "dataset-root",
        "manifest-path",
        "groot-source",
        "base-model",
        "output-dir",
    ):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--data-config", required=True)
    train.add_argument("--embodiment-tag", required=True)
    train.add_argument("--max-steps", type=int, required=True)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--video-backend",
        choices=["torchcodec", "decord", "torchvision_av"],
        default="torchcodec",
    )
    compare = subcommands.add_parser(
        "compare", help="Compare operator-recorded matched policy trials"
    )
    compare.add_argument("--rollouts-path", type=Path, required=True)
    compare.add_argument("--training-receipt-path", type=Path, required=True)
    compare.add_argument("--output-path", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
