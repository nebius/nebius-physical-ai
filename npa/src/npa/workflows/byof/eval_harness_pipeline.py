"""eval_harness workbench workflow stages (real implementations).

Mirrors the ``newton_pipeline.py`` module pattern: stage functions backed by
an argparse entry point (``main``), a module-level error type, and JSON +
Markdown stage artifacts written to URIs.

Stages:

* ``run`` — evaluate one policy on a task for N episodes and write a report.
* ``compare`` — A/B compare two policies with paired episode seeds and write
  a comparison report.

Invoked as ``python3 -m npa.workflows.byof.eval_harness_pipeline``.  Heavy
imports (simulation backends, VLM clients) stay inside the stage functions so
the CLI surface imports on machines without them installed.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Mapping, Sequence

JUDGES = ("heuristic", "vlm")
SCHEMA_RUN = "npa.workbench.eval_harness.run.v1"
SCHEMA_COMPARE = "npa.workbench.eval_harness.compare.v1"


class EvalHarnessPipelineError(RuntimeError):
    """Raised when an eval_harness pipeline invariant is not met."""


def _resolve_output_path(*, output_path: str = "", output_uri: str = "") -> str:
    """Resolve the canonical output-path spelling and its transitional alias."""
    if output_path and output_uri and output_path != output_uri:
        raise EvalHarnessPipelineError(
            "output_path and compatibility output_uri must identify the same destination"
        )
    resolved = output_path or output_uri
    if not resolved:
        raise EvalHarnessPipelineError("output_path is required")
    return resolved


def _stage_envelope(
    *,
    stage: str,
    schema: str,
    inputs: Mapping[str, Any],
    artifacts: Mapping[str, str],
    summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": schema,
        "stage": stage,
        "status": "completed",
        "inputs": dict(inputs),
        "artifacts": dict(artifacts),
        "summary": dict(summary),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run(
    *,
    task: str,
    policy: str,
    output_path: str = "",
    output_uri: str = "",
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """Evaluate one policy on a task and write JSON + Markdown reports."""
    from npa.workbench.eval_harness.reports import write_run_report
    from npa.workbench.eval_harness.runner import run_policy

    if not task:
        raise EvalHarnessPipelineError("task is required")
    if not policy:
        raise EvalHarnessPipelineError("policy is required")
    output_path = _resolve_output_path(output_path=output_path, output_uri=output_uri)
    if judge not in JUDGES:
        raise EvalHarnessPipelineError(f"judge must be one of {JUDGES}, got {judge!r}")
    report = run_policy(
        task=task,
        policy=policy,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )
    artifacts = write_run_report(output_path, report.as_dict())
    return _stage_envelope(
        stage="run",
        schema=SCHEMA_RUN,
        inputs={
            "task": task,
            "policy": policy,
            "episodes": episodes,
            "seed": seed,
            "judge": judge,
            "max_steps": max_steps,
        },
        artifacts=artifacts,
        summary=report.summary,
    )


def compare(
    *,
    task: str,
    policy_a: str,
    policy_b: str,
    output_path: str = "",
    output_uri: str = "",
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """A/B compare two policies with paired seeds; write comparison reports."""
    from npa.workbench.eval_harness.compare import compare_policies
    from npa.workbench.eval_harness.reports import write_compare_report

    if not task:
        raise EvalHarnessPipelineError("task is required")
    if not policy_a:
        raise EvalHarnessPipelineError("policy_a is required")
    if not policy_b:
        raise EvalHarnessPipelineError("policy_b is required")
    output_path = _resolve_output_path(output_path=output_path, output_uri=output_uri)
    if judge not in JUDGES:
        raise EvalHarnessPipelineError(f"judge must be one of {JUDGES}, got {judge!r}")
    report = compare_policies(
        task=task,
        policy_a=policy_a,
        policy_b=policy_b,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )
    artifacts = write_compare_report(output_path, report.as_dict())
    return _stage_envelope(
        stage="compare",
        schema=SCHEMA_COMPARE,
        inputs={
            "task": task,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "episodes": episodes,
            "seed": seed,
            "judge": judge,
            "max_steps": max_steps,
        },
        artifacts=artifacts,
        summary={
            "success_rate_a": report.summary_a.get("success_rate"),
            "success_rate_b": report.summary_b.get("success_rate"),
            "success_rate_diff": report.success_rate_diff,
            "bootstrap_95": report.bootstrap_95,
        },
    )


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--task", required=True, help="Registered task name.")
    sub.add_argument("--episodes", type=int, default=10)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--output-path", "--output-uri", dest="output_path", required=True)
    sub.add_argument("--judge", default="heuristic", choices=JUDGES)
    sub.add_argument("--max-steps", type=int, default=500)
    sub.add_argument("--vlm-endpoint-url", default="")


def _check_positive(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise EvalHarnessPipelineError("episodes must be >= 1")
    if args.max_steps < 1:
        raise EvalHarnessPipelineError("max_steps must be >= 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_harness_pipeline",
        description="Standardized policy A/B evaluation harness.",
    )
    commands = parser.add_subparsers(dest="stage", required=True)

    run_cmd = commands.add_parser("run")
    _add_common(run_cmd)
    run_cmd.add_argument("--policy", required=True)

    compare_cmd = commands.add_parser("compare")
    _add_common(compare_cmd)
    compare_cmd.add_argument("--policy-a", required=True)
    compare_cmd.add_argument("--policy-b", required=True)
    return parser


def _run_stage(args: argparse.Namespace) -> int:
    _check_positive(args)
    if args.stage == "run":
        run(
            task=args.task,
            policy=args.policy,
            output_path=args.output_path,
            episodes=args.episodes,
            seed=args.seed,
            judge=args.judge,
            max_steps=args.max_steps,
            vlm_endpoint_url=args.vlm_endpoint_url,
        )
    elif args.stage == "compare":
        compare(
            task=args.task,
            policy_a=args.policy_a,
            policy_b=args.policy_b,
            output_path=args.output_path,
            episodes=args.episodes,
            seed=args.seed,
            judge=args.judge,
            max_steps=args.max_steps,
            vlm_endpoint_url=args.vlm_endpoint_url,
        )
    else:
        raise EvalHarnessPipelineError(f"unknown stage {args.stage!r}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
