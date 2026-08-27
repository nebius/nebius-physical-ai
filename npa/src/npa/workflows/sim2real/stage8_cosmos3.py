"""Hosted Cosmos3 rollout evaluation for canonical Sim2Real Stage 8."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.workflow_io import (
    image_provenance,
    publish_component_record,
    storage,
    write_loop_output,
)


def _aggregate_usage(
    results: list[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    requests = [dict(item.get("request") or {}) for item in results]
    priced = bool(requests) and all(item.get("cost_usd") is not None for item in requests)
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
        "per_request_latency_seconds": [item.get("latency_seconds") for item in requests],
        "retries": sum(int(item.get("retries") or 0) for item in requests),
        "request_ids": [item.get("request_id") for item in requests if item.get("request_id")],
        "cost_usd": (
            round(sum(float(item["cost_usd"]) for item in requests), 8)
            if priced
            else None
        ),
        "cost_source": "response_usage" if priced else "unavailable",
    }


def run(args: argparse.Namespace) -> None:
    """Score every Stage 7 rollout and publish the Stage 8 barrier record."""

    from npa.workbench.cosmos.reason import (
        run_token_factory_rollout_vlm,
        task_description_from_manifest,
    )

    root = str(args.root_uri).rstrip("/")
    work = Path(tempfile.mkdtemp(prefix="npa-s2r-stage-08-"))
    source = (
        f"{root}/actions/train/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/"
    )
    actions = work / "actions"
    storage().download_directory(source, str(actions))
    results: list[dict[str, Any]] = []
    for manifest_path in sorted(actions.glob("rollout-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        observations = list(manifest.get("camera_observations") or [])
        frames = [manifest_path.parent / str(name) for name in observations]
        frames = [path for path in frames if path.is_file()]
        if not frames:
            frames = sorted(manifest_path.parent.glob("camera-*.png"))
        results.append(
            run_token_factory_rollout_vlm(
                model_id=args.reason_model,
                image_paths=frames,
                actions=list(manifest.get("actions") or []),
                task_description=task_description_from_manifest(manifest),
                rollout_id=str(manifest.get("rollout_id") or manifest_path.parent.name),
                threshold=args.threshold,
            )
        )
    if not results:
        raise RuntimeError("Stage 8 found no real Stage 7 rollouts")
    rollout_ids = [str(item["rollout_id"]) for item in results]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise RuntimeError("Stage 8 found duplicate Stage 7 rollout identities")
    evaluator_usage = _aggregate_usage(results, model=args.reason_model)
    payload = {
        "schema": "npa.sim2real.cosmos3_evaluator.v1",
        "evaluator": "cosmos3",
        "model": args.reason_model,
        "provider": "nebius",
        "backend": "token_factory",
        "evaluations": results,
        "source_rollout_ids": rollout_ids,
        "evaluator_usage": evaluator_usage,
        "provenance": image_provenance(require_gpu=False),
    }
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
        evidence="Hosted Cosmos3-Super-Reasoner scored every Stage 7 rollout with event-local labels through Token Factory.",
        artifacts={
            "result": output_uri,
            "model": args.reason_model,
            "provider": evaluator_usage["provider"],
            "backend": "token_factory",
            "evaluator_usage": evaluator_usage,
            "rollout_count": len(results),
            "outer_iteration": args.outer_iteration,
            "inner_iteration": args.inner_iteration,
        },
        require_gpu=False,
        execution_provenance=payload["provenance"],
    )
