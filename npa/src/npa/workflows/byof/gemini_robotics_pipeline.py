"""Gemini Robotics 2 API-backed BYOF pipeline.

Orchestrates the three ``workbench.gemini_robotics`` stages — ER planning,
on-device adaptation, and rubric evaluation — and writes content-addressed
receipts for every artifact.  The Gemini API is closed-weight, so all stages
are zero-GPU: the client is injected (real ``GeminiRoboticsClient`` in
production, a fake in tests) and no heavy imports happen at module load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from npa.cli.gemini_robotics import (
    ADAPT_RECEIPT_SCHEMA,
    EVAL_RECEIPT_SCHEMA,
    PLAN_RECEIPT_SCHEMA,
    AdaptationJob,
    EvalResult,
    GeminiRoboticsClient,
    GeminiRoboticsError,
    PlanResult,
    resolve_config,
)

PIPELINE_RECEIPT_SCHEMA = "npa.gemini_robotics.pipeline-receipt.v1"


class GeminiRoboticsPipelineError(RuntimeError):
    """Raised when a pipeline stage or bookkeeping invariant fails."""


@dataclass
class GeminiRoboticsPipelineConfig:
    """Inputs for one Gemini Robotics pipeline run."""

    task: str
    output_dir: str
    images: list[str] = field(default_factory=list)
    model: str = ""
    run_adaptation: bool = False
    dataset_path: str = ""
    adaptation_display_name: str = ""
    adapt_wait: bool = True
    rubric_path: str = ""

    def resolved_model(self, default: str) -> str:
        return self.model or default


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    return _sha256_bytes(raw.encode("utf-8"))


def _client_or_default(client: GeminiRoboticsClient | None) -> GeminiRoboticsClient:
    if client is not None:
        return client
    try:
        config = resolve_config()
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc
    return GeminiRoboticsClient(config)


def run_er_planning_stage(
    config: GeminiRoboticsPipelineConfig,
    client: GeminiRoboticsClient | None = None,
) -> dict[str, Any]:
    """Run the ER planning stage and persist its receipt."""
    from npa.cli.gemini_robotics import PROVISIONAL_MODEL_ID

    active = _client_or_default(client)
    try:
        result: PlanResult = active.plan(
            task=config.task,
            images=[Path(p) for p in config.images],
            model=config.resolved_model(PROVISIONAL_MODEL_ID),
        )
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(f"ER planning stage failed: {exc}") from exc
    receipt: dict[str, Any] = {
        "schema": PLAN_RECEIPT_SCHEMA,
        "task": config.task,
        "model": result.model,
        "plan_text": result.text,
        "safety_calls": result.safety_calls,
        "finish_reason": result.finish_reason,
        "created_at": _utc_now(),
    }
    plan_path = Path(config.output_dir) / "plan.json"
    digest = _write_json(plan_path, receipt)
    receipt["artifact_path"] = str(plan_path)
    receipt["artifact_sha256"] = digest
    return receipt


def run_adaptation_stage(
    config: GeminiRoboticsPipelineConfig,
    client: GeminiRoboticsClient | None = None,
) -> dict[str, Any]:
    """Run the on-device adaptation stage and persist its receipt."""
    from npa.cli.gemini_robotics import PROVISIONAL_MODEL_ID, _read_examples

    if not config.dataset_path:
        raise GeminiRoboticsPipelineError(
            "Adaptation stage requires dataset_path to be set."
        )
    active = _client_or_default(client)
    try:
        examples = _read_examples(Path(config.dataset_path))
        operation = active.submit_adaptation(
            display_name=config.adaptation_display_name or config.task,
            base_model=config.resolved_model(PROVISIONAL_MODEL_ID),
            examples=examples,
        )
        job: AdaptationJob | None = None
        if config.adapt_wait:
            job = active.wait_for_adaptation(operation)
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(f"Adaptation stage failed: {exc}") from exc
    receipt: dict[str, Any] = {
        "schema": ADAPT_RECEIPT_SCHEMA,
        "display_name": config.adaptation_display_name or config.task,
        "base_model": config.resolved_model(PROVISIONAL_MODEL_ID),
        "operation": operation,
        "num_examples": len(examples),
        "done": bool(job.done) if job else False,
        "tuned_model": job.tuned_model if job else "",
        "error": job.error if job else "",
        "created_at": _utc_now(),
    }
    adapt_path = Path(config.output_dir) / "adaptation.json"
    digest = _write_json(adapt_path, receipt)
    receipt["artifact_path"] = str(adapt_path)
    receipt["artifact_sha256"] = digest
    return receipt


def run_eval_stage(
    config: GeminiRoboticsPipelineConfig,
    plan_receipt: Mapping[str, Any],
    client: GeminiRoboticsClient | None = None,
) -> dict[str, Any]:
    """Run the rubric-eval stage against a plan receipt and persist it."""
    from npa.cli.gemini_robotics import PROVISIONAL_MODEL_ID

    if not config.rubric_path:
        raise GeminiRoboticsPipelineError("Eval stage requires rubric_path to be set.")
    active = _client_or_default(client)
    rubric = Path(config.rubric_path).read_text(encoding="utf-8")
    try:
        result: EvalResult = active.eval_plan(
            plan_text=str(plan_receipt.get("plan_text", "")),
            rubric=rubric,
            model=config.resolved_model(PROVISIONAL_MODEL_ID),
        )
    except (GeminiRoboticsError, OSError) as exc:
        raise GeminiRoboticsPipelineError(f"Eval stage failed: {exc}") from exc
    receipt: dict[str, Any] = {
        "schema": EVAL_RECEIPT_SCHEMA,
        "model": result.model,
        "scores": result.scores,
        "summary": result.summary,
        "created_at": _utc_now(),
    }
    eval_path = Path(config.output_dir) / "eval.json"
    digest = _write_json(eval_path, receipt)
    receipt["artifact_path"] = str(eval_path)
    receipt["artifact_sha256"] = digest
    return receipt


def run_pipeline(
    config: GeminiRoboticsPipelineConfig,
    client: GeminiRoboticsClient | None = None,
) -> dict[str, Any]:
    """Run plan → (adapt) → eval and write the pipeline receipt."""
    stages: dict[str, Any] = {}
    plan_receipt = run_er_planning_stage(config, client)
    stages["plan"] = {
        "artifact_path": plan_receipt["artifact_path"],
        "artifact_sha256": plan_receipt["artifact_sha256"],
    }
    if config.run_adaptation:
        adapt_receipt = run_adaptation_stage(config, client)
        stages["adaptation"] = {
            "artifact_path": adapt_receipt["artifact_path"],
            "artifact_sha256": adapt_receipt["artifact_sha256"],
        }
    if config.rubric_path:
        eval_receipt = run_eval_stage(config, plan_receipt, client)
        stages["eval"] = {
            "artifact_path": eval_receipt["artifact_path"],
            "artifact_sha256": eval_receipt["artifact_sha256"],
        }
    receipt: dict[str, Any] = {
        "schema": PIPELINE_RECEIPT_SCHEMA,
        "task": config.task,
        "stages": stages,
        "created_at": _utc_now(),
    }
    receipt_path = Path(config.output_dir) / "pipeline_receipt.json"
    digest = _write_json(receipt_path, receipt)
    receipt["artifact_path"] = str(receipt_path)
    receipt["artifact_sha256"] = digest
    return receipt


def build_parser() -> argparse.ArgumentParser:
    """Build the pipeline CLI parser.

    The API base URL and model ids are override-required: there are no
    defaults because the provisional guesses have never been validated
    against a live endpoint.
    """
    parser = argparse.ArgumentParser(
        prog="gemini_robotics_pipeline",
        description="Gemini Robotics BYOF pipeline stages (provisional API adapter).",
    )
    parser.add_argument(
        "--api-base-url",
        required=True,
        help="Gemini API base URL (required; the provisional guess is never used).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Gemini API key (falls back to GOOGLE_API_KEY).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan", help="Run ER planning and write the receipt.")
    plan_p.add_argument("--task", required=True)
    plan_p.add_argument("--image", action="append", default=[])
    plan_p.add_argument("--model", required=True, help="Gemini model id (required).")
    plan_p.add_argument("--output-dir", required=True)

    adapt_p = sub.add_parser("adapt", help="Submit an adaptation job.")
    adapt_p.add_argument("--dataset-path", required=True)
    adapt_p.add_argument("--display-name", required=True)
    adapt_p.add_argument(
        "--base-model", required=True, help="Base model id (required)."
    )
    adapt_p.add_argument("--output-dir", required=True)
    adapt_p.add_argument("--no-wait", dest="wait", action="store_false", default=True)

    eval_p = sub.add_parser("eval", help="Evaluate a plan against a rubric.")
    eval_p.add_argument("--plan-path", required=True)
    eval_p.add_argument("--rubric-path", required=True)
    eval_p.add_argument("--model", required=True, help="Gemini model id (required).")
    eval_p.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one pipeline stage against an explicitly configured API endpoint."""
    args = build_parser().parse_args(argv)
    try:
        api_config = resolve_config(
            api_key=args.api_key or None, base_url=args.api_base_url
        )
    except GeminiRoboticsError as exc:
        raise GeminiRoboticsPipelineError(str(exc)) from exc
    client = GeminiRoboticsClient(api_config)
    if args.command == "plan":
        pipeline_config = GeminiRoboticsPipelineConfig(
            task=args.task,
            output_dir=args.output_dir,
            images=list(args.image or []),
            model=args.model,
        )
        receipt = run_er_planning_stage(pipeline_config, client)
    elif args.command == "adapt":
        pipeline_config = GeminiRoboticsPipelineConfig(
            task=args.display_name,
            output_dir=args.output_dir,
            model=args.base_model,
            dataset_path=args.dataset_path,
            adaptation_display_name=args.display_name,
            run_adaptation=True,
            adapt_wait=args.wait,
        )
        receipt = run_adaptation_stage(pipeline_config, client)
    elif args.command == "eval":
        pipeline_config = GeminiRoboticsPipelineConfig(
            task=args.plan_path,
            output_dir=args.output_dir,
            model=args.model,
            rubric_path=args.rubric_path,
        )
        plan_receipt = {
            "plan_text": Path(args.plan_path).read_text(encoding="utf-8"),
            "artifact_path": args.plan_path,
            "artifact_sha256": "",
        }
        receipt = run_eval_stage(pipeline_config, plan_receipt, client)
    else:  # pragma: no cover - argparse restricts commands
        raise GeminiRoboticsPipelineError(f"unknown command {args.command!r}")
    print(
        json.dumps(
            {
                "artifact_path": receipt["artifact_path"],
                "artifact_sha256": receipt["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
