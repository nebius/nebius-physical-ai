"""Expose preparation, real post-training, and held-out evaluation as workflow stages."""

import argparse
import tempfile
from pathlib import Path

from .artifacts import materialize, publish


def build_parser() -> argparse.ArgumentParser:
    """Define the stateless S3 stage argument contract.

    Args:
        None.
    Returns:
        Parser for dataset preparation, training, and evaluation.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--frame-stride", type=int, default=3)
    train = commands.add_parser("train")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint-path", required=True)
    for command in (prepare, train, evaluate):
        command.add_argument("--input-path", required=True)
        command.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a real stage and publish verified results before enforcing quality.

    Args:
        argv: Explicit arguments or process command line.
    Returns:
        Zero when the stage succeeds, one when held-out quality fails.
    Raises:
        ValueError: Source or stage arguments are invalid.
        RuntimeError: Training or execution fails.
        OSError: Artifact exchange fails.
    """
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="antioch-posttrain-") as temporary:
        root = Path(temporary)
        source = materialize(args.input_path, root / "input")
        output = root / "output"
        passed = _execute(args, source, output, root)
        publish(output, args.output_path)
    return 0 if passed else 1


def _execute(args, source, output, root):
    if args.stage == "prepare":
        from .dataset import prepare_dataset

        prepare_dataset(source, output, args.frame_stride)
        return True
    if args.stage == "train":
        from .training import train_model

        train_model(source, output, args.epochs, args.batch_size, args.seed)
        return True
    from .evaluation import evaluate_model

    checkpoint = materialize(args.checkpoint_path, root / "checkpoint")
    return evaluate_model(source, checkpoint, output)
