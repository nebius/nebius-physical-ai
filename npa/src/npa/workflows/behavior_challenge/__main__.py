"""Expose local protocol planning and the internal BEHAVIOR workflow stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser.add_argument("--policy-specialist-equivalence-receipt", type=Path)
    parser.add_argument("--policy-specialist-equivalence-sha256")
    parser.add_argument("--policy-specialist-report-admission", type=Path)
    parser.add_argument("--policy-specialist-report-admission-sha256")
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
    status = commands.add_parser("campaign-status", help="Read durable case progress")
    status.add_argument("--panel-uri", required=True)
    status.add_argument("--output-path", required=True)
    status.set_defaults(handler=_campaign_status)
    worker = commands.add_parser(
        "campaign-worker", help="Internal resumable case worker"
    )
    worker.add_argument("--partition-uri", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-receipt-uri", required=True)
    _add_runtime_arguments(worker)
    startup = worker.add_mutually_exclusive_group()
    startup.add_argument("--simulator-startup-spec", type=Path)
    startup.add_argument("--simulator-startup-receipt", type=Path)
    worker.set_defaults(handler=_campaign_worker)
    aggregate = commands.add_parser(
        "campaign-aggregate", help="Verify complete original evidence"
    )
    aggregate.add_argument("--receipt-uri", required=True)
    aggregate.set_defaults(handler=_campaign_aggregate)
    for command in (worker, aggregate):
        command.add_argument("--panel-uri", required=True)
        command.add_argument("--output-path", required=True)
        command.add_argument("--workspace", type=Path, required=True)


def _simulator_startup(args):
    from .simulator_startup import write_simulator_startup_receipt

    return write_simulator_startup_receipt(args.spec_path, args.receipt_path)


def _simulator_startup_child(args):
    from .simulator_startup import run_simulator_startup_child

    run_simulator_startup_child(
        args.upstream_root, args.marker_path, args.shutdown_request_path
    )
    return {"status": "shutdown_returned"}


def _campaign_worker(args):
    from .campaign_runner import evaluate_partition

    return evaluate_partition(args)


def _campaign_aggregate(args):
    from .campaign_runner import aggregate_campaign_worker

    return aggregate_campaign_worker(args)


def _campaign_status(args):
    from .campaign_status import observe_campaign

    return observe_campaign(args)


def _evaluate(args):
    from .execution import evaluate

    return evaluate(args)


def _plan(args):
    from .protocol import make_plan, verify_upstream

    plan = make_plan(json.loads(args.recipe_path.read_bytes()), args.upstream_root)
    verify_upstream(args.upstream_root, plan["upstream_commit"])
    return plan


def _add_simulator_startup_commands(commands) -> None:
    startup = commands.add_parser(
        "simulator-startup", help="Qualify a writable empty-scene simulator startup"
    )
    startup.add_argument("--spec-path", type=Path, required=True)
    startup.add_argument("--receipt-path", type=Path, required=True)
    startup.set_defaults(handler=_simulator_startup)
    child = commands.add_parser("simulator-startup-child", help=argparse.SUPPRESS)
    child.add_argument("--upstream-root", type=Path, required=True)
    child.add_argument("--marker-path", type=Path, required=True)
    child.add_argument("--shutdown-request-path", type=Path, required=True)
    child.set_defaults(handler=_simulator_startup_child)


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
    run.set_defaults(handler=_evaluate)
    _add_simulator_startup_commands(commands)
    _add_campaign_commands(commands)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
