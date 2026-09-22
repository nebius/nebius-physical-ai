"""Expose local protocol planning and the internal BEHAVIOR workflow stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import evaluate
from .protocol import make_plan, verify_upstream


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-kind",
        choices=(
            "official",
            "rlc",
            "rlc-selected",
            "rlc-specialist",
            "comet12",
            "comet50",
        ),
        default="official",
    )
    for field in ("root", "python", "checkpoint", "archive"):
        parser.add_argument(f"--policy-{field}", type=Path)
    for field in (
        "selected-export-receipt",
        "correlation-manifest",
        "validation-receipt",
    ):
        parser.add_argument(f"--policy-{field}", type=Path)
    parser.add_argument("--policy-stock-correlation-asset", type=Path)
    parser.add_argument("--policy-stock-correlation-sha256")
    parser.add_argument("--policy-task-name")
    parser.add_argument(
        "--policy-execution-variant",
        choices=(
            "native",
            "transition-refresh",
            "final-stage-backtrack",
            "adaptive-short-chunk",
            "adaptive-short-chunk-transition-refresh",
            "native-stage-transition-refresh",
        ),
        default="native",
    )


def _add_runtime_arguments(run):
    run.add_argument("--upstream-root", type=Path, required=True)
    run.add_argument("--evaluator-python", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--port", type=int, default=8000)
    _add_policy_arguments(run)


def _add_campaign_commands(commands):
    from .campaign_runner import aggregate_campaign_worker, evaluate_partition
    from .campaign_status import observe_campaign

    status = commands.add_parser("campaign-status", help="Read durable case progress")
    status.add_argument("--panel-uri", required=True)
    status.add_argument("--output-path", required=True)
    status.set_defaults(handler=observe_campaign)
    worker = commands.add_parser(
        "campaign-worker", help="Internal resumable case worker"
    )
    worker.add_argument("--partition-uri", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-receipt-uri", required=True)
    _add_runtime_arguments(worker)
    worker.set_defaults(handler=evaluate_partition)
    aggregate = commands.add_parser(
        "campaign-aggregate", help="Verify complete original evidence"
    )
    aggregate.add_argument("--receipt-uri", required=True)
    aggregate.set_defaults(handler=aggregate_campaign_worker)
    for command in (worker, aggregate):
        command.add_argument("--panel-uri", required=True)
        command.add_argument("--output-path", required=True)
        command.add_argument("--workspace", type=Path, required=True)


def _plan(args):
    verify_upstream(args.upstream_root)
    return make_plan(json.loads(args.recipe_path.read_bytes()), args.upstream_root)


def main() -> None:
    """Plan locally or invoke internal stages through the Workbench workflow.

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
    plan = commands.add_parser("plan", help="Freeze a recipe without simulation")
    plan.add_argument("--recipe-path", type=Path, required=True)
    plan.add_argument("--upstream-root", type=Path, required=True)
    plan.set_defaults(handler=_plan)
    run = commands.add_parser("evaluate", help="Internal licensed evaluator stage")
    run.add_argument("--input-path", required=True)
    run.add_argument("--output-path", required=True)
    run.add_argument("--policy-readme-uri", required=True)
    _add_runtime_arguments(run)
    run.set_defaults(handler=evaluate)
    _add_campaign_commands(commands)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
