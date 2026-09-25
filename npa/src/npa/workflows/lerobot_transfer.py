"""Expose stateless LeRobot transfer stages to the standard workflow interpreter."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from npa.workflows.lerobot_transfer_data import materialize, prepare_dataset, publish


def build_parser() -> argparse.ArgumentParser:
    """Define the workflow stage argument contract.

    Args:
        None.
    Returns:
        Parser for preparation, training, evaluation, and reporting.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-steps", type=int, default=20_000)
    prepare.add_argument("--batch-size", type=int, default=64)
    prepare.add_argument("--validation-episodes", type=int, default=32)
    prepare.add_argument("--test-episodes", type=int, default=64)
    prepare.add_argument("--eval-batch-size", type=int, default=8)
    prepare.add_argument("--minimum-success", type=float, default=0.7)
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=("baseline", "robust"), required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--baseline-path", required=True)
    evaluate.add_argument("--robust-path", required=True)
    report = commands.add_parser("report")
    report.add_argument("--run-id", required=True)
    for command in (prepare, train, evaluate, report):
        command.add_argument("--output-path", required=True)
    for command in (train, evaluate, report):
        command.add_argument("--input-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one stage and publish its verified artifacts.

    Args:
        argv: Command arguments, or process arguments when omitted.
    Returns:
        Zero after complete verified publication.
    Raises:
        ValueError: A recipe or artifact is invalid.
        RuntimeError: Native training or evaluation fails.
        OSError: Artifact exchange fails.
    """
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="lerobot-transfer-") as temporary:
        workspace = Path(temporary)
        output = workspace / "output"
        _execute_stage(args, workspace, output)
        publish(output, args.output_path)
    return 0


def _execute_stage(args: argparse.Namespace, workspace: Path, output: Path) -> None:
    if args.stage == "prepare":
        prepare_dataset(output, _recipe(args))
        return
    prepared = materialize(args.input_path, workspace / "input")
    if args.stage == "train":
        from npa.workflows.lerobot_transfer_training import train_policy

        train_policy(prepared, output, args.arm)
        return
    if args.stage == "evaluate":
        from npa.workflows.lerobot_transfer_eval import evaluate_pair

        baseline = materialize(args.baseline_path, workspace / "baseline")
        robust = materialize(args.robust_path, workspace / "robust")
        evaluate_pair(prepared, baseline, robust, output)
        return
    from npa.workflows.lerobot_transfer_report import report_results

    report_results(prepared, output, args.run_id)


def _recipe(args: argparse.Namespace) -> dict:
    positive = ("train_steps", "batch_size", "eval_batch_size")
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Training steps and batch sizes must be positive")
    if min(args.validation_episodes, args.test_episodes) < 2:
        raise ValueError("Validation and test each require at least two reset seeds")
    if not 0 < args.minimum_success <= 1 or args.seed < 0:
        raise ValueError(
            "Success threshold must be in (0, 1] and seed must be nonnegative"
        )
    return {
        key: getattr(args, key)
        for key in (
            "seed",
            "train_steps",
            "batch_size",
            "validation_episodes",
            "test_episodes",
            "eval_batch_size",
            "minimum_success",
        )
    } | {"validation_seed": 100_000, "test_seed": 100_000 + args.validation_episodes}


if __name__ == "__main__":
    raise SystemExit(main())
