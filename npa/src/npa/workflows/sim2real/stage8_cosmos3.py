"""Hosted rollout evaluation; Cosmos3 artifact names remain compatible."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.workflow_io import (
    image_provenance,
    publish_component_record,
    storage,
    write_loop_output,
)


def _aggregate_usage(results: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    requests = [dict(item.get("request") or {}) for item in results]
    priced = bool(requests) and all(
        item.get("cost_usd") is not None for item in requests
    )
    return {
        "provider": "nebius",
        "backend": "token_factory",
        "model": model,
        "request_count": len(requests),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in requests),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in requests),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in requests),
        "aggregate_latency_seconds": round(
            sum(float(item.get("latency_seconds") or 0.0) for item in requests), 6
        ),
        "per_request_latency_seconds": [
            item.get("latency_seconds") for item in requests
        ],
        "retries": sum(int(item.get("retries") or 0) for item in requests),
        "request_ids": [
            item.get("request_id") for item in requests if item.get("request_id")
        ],
        "cost_usd": (
            round(sum(float(item["cost_usd"]) for item in requests), 8)
            if priced
            else None
        ),
        "cost_source": "response_usage" if priced else "unavailable",
    }


def _publish(args: argparse.Namespace, payload: dict[str, Any], work: Path) -> None:
    root = str(args.root_uri).rstrip("/")
    output_uri = (
        f"{root}/vlm_eval/train/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/cosmos3.json"
    )
    write_loop_output(
        output_uri, payload, work / "out", args.outer_iteration, args.inner_iteration
    )
    publish_component_record(
        root_uri=root,
        stage=8,
        name="stage_08_vlm_eval_train",
        tier="WORKS",
        evidence=f"Hosted {args.reason_model} scored every Stage 7 rollout with event-local labels through Token Factory.",
        artifacts={
            "result": output_uri,
            "model": args.reason_model,
            "reason_family": payload["reason_family"],
            "provider": "nebius",
            "backend": "token_factory",
            "evaluator_usage": payload["evaluator_usage"],
            "rollout_count": len(payload["evaluations"]),
            "outer_iteration": args.outer_iteration,
            "inner_iteration": args.inner_iteration,
            "evaluation_execution": payload["evaluation_execution"],
        },
        require_gpu=False,
        execution_provenance=payload["provenance"],
    )


def _result_payload(args, results, reused, provenance) -> dict[str, Any]:
    from npa.workbench.cosmos.reason import hosted_rollout_model_family

    return {
        "schema": "npa.sim2real.cosmos3_evaluator.v1",
        "evaluator": "cosmos3",
        "model": args.reason_model,
        "reason_family": hosted_rollout_model_family(args.reason_model),
        "provider": "nebius",
        "backend": "token_factory",
        "evaluations": results,
        "source_rollout_ids": [str(item["rollout_id"]) for item in results],
        "evaluator_usage": _aggregate_usage(results, model=args.reason_model),
        "provenance": provenance,
        "evaluation_execution": {
            "concurrency": args.evaluation_concurrency,
            "max_frames": args.evaluation_max_frames,
            "reused_rollouts": reused,
            "new_requests": len(results) - reused,
        },
    }


def run(args: argparse.Namespace) -> None:
    """Score every rollout with durable receipts, then publish the Stage 8 barrier.

    Args:
        args: Stage configuration parsed by the canonical workflow adapter.
    Returns:
        None.
    Raises:
        RuntimeError: No complete rollout set or invalid evaluation evidence.
        ValueError: Invalid evaluation settings or duplicate rollout identities.
    """
    from npa.workbench.cosmos.reason import hosted_rollout_model_family
    from npa.workflows.sim2real.hosted_evaluation import evaluate_rollouts

    root = str(args.root_uri).rstrip("/")
    hosted_rollout_model_family(args.reason_model)
    work = Path(tempfile.mkdtemp(prefix="npa-s2r-stage-08-"))
    iteration = f"outer-{args.outer_iteration:02d}/iter-{args.inner_iteration:02d}"
    actions = work / "actions"
    store = storage()
    store.download_directory(f"{root}/actions/train/{iteration}/", str(actions))
    paths = sorted(actions.glob("rollout-*/manifest.json"))
    if not paths:
        raise RuntimeError("Stage 8 found no real Stage 7 rollouts")
    provenance = image_provenance(require_gpu=False)
    results, reused = evaluate_rollouts(
        paths,
        store=store,
        prefix=f"{root}/vlm_eval/train/{iteration}",
        model=args.reason_model,
        threshold=args.threshold,
        source_sha=provenance["source_sha"],
        concurrency=args.evaluation_concurrency,
        max_frames=args.evaluation_max_frames,
    )
    payload = _result_payload(args, results, reused, provenance)
    _publish(args, payload, work)
