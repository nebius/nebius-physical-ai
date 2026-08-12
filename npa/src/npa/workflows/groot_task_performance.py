"""Resolve immutable GR00T trainer output for the offline reference workflow.

Closed-loop PushT/task-performance commands were intentionally removed: no
shipped workflow configured or consumed them, while this PR's evidence is an
offline held-out action-prediction evaluation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from npa.workflows.groot_learning import (
    GROOT_MODEL_CONFIG_CONTRACT,
    GrootVisualizationError,
    _checkpoint_identity,
    _download_prefix,
    _put_json,
    _read_s3_json,
    _resolve_highest_checkpoint_directory,
    _s3_client,
    checkpoint_model_config_contract,
    require_distinct_trained_weights,
)

CHECKPOINT_REF_SCHEMA = "npa.groot.checkpoint_ref.v1"


def resolve_trained_checkpoint(
    training_manifest_uri: str,
    split_manifest_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    run_id: str,
    *,
    baseline_checkpoint_uri: str,
    expected_gpu_count: int,
    expected_max_steps: int,
    expected_save_steps: int,
    expected_save_total_limit: int,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Resolve the exact final checkpoint-N under a validated save contract."""

    client = _s3_client(s3_client)
    manifest = _read_s3_json(client, training_manifest_uri)
    split = _read_s3_json(client, split_manifest_uri)
    if (
        manifest.get("schema") != "npa.groot.finetune.v1"
        or manifest.get("status") != "completed"
        or manifest.get("run_id") != run_id
        or manifest.get("checkpoint_uri") != checkpoint_uri
        or manifest.get("optimizer_step_ok") is not True
        or manifest.get("collective_ok") is not True
        or manifest.get("loss_steps_real") is not True
        or manifest.get("rank_zero_checkpoint_only") is not True
        or int(manifest.get("checkpoint_upload_invocations") or 0) != 1
        or manifest.get("model_config_contract") != GROOT_MODEL_CONFIG_CONTRACT
    ):
        raise GrootVisualizationError(
            "trainer manifest lacks completed immutable-run evidence"
        )

    gpu_count = int(manifest.get("num_gpus") or 0)
    world_size = int(manifest.get("world_size") or 0)
    distinct_gpu_count = int(manifest.get("distinct_gpu_count") or 0)
    if not (
        gpu_count == world_size == distinct_gpu_count == int(expected_gpu_count)
    ):
        raise GrootVisualizationError("trainer did not use the required distinct GPU world")
    expected_ranks = list(range(gpu_count))
    observed_ranks = sorted(int(value) for value in manifest.get("observed_ranks") or [])
    training_ranks = sorted(
        int(value) for value in manifest.get("training_observed_ranks") or []
    )
    if observed_ranks != expected_ranks or training_ranks != expected_ranks:
        raise GrootVisualizationError("trainer manifest lacks complete rank evidence")

    configured_steps = int(manifest.get("max_steps") or 0)
    completed_steps = int(manifest.get("training_step") or 0)
    save_steps = int(manifest.get("save_steps") or 0)
    save_total_limit = int(manifest.get("save_total_limit") or 0)
    if (
        int(expected_max_steps) < 2
        or configured_steps != int(expected_max_steps)
        or completed_steps != configured_steps
        or int(expected_save_steps) != configured_steps
        or save_steps != int(expected_save_steps)
        or int(expected_save_total_limit) < 1
        or save_total_limit != int(expected_save_total_limit)
    ):
        raise GrootVisualizationError(
            "trainer step/checkpoint schedule differs from the preflight contract"
        )
    checkpoint_steps = sorted(
        {int(value) for value in manifest.get("checkpoint_steps") or []}
    )
    if completed_steps not in checkpoint_steps:
        raise GrootVisualizationError("trainer lacks its final checkpoint")

    training_plan = split.get("training_plan") or {}
    per_device_batch = int(manifest.get("per_device_batch_size") or 0)
    accumulation = int(manifest.get("gradient_accumulation_steps") or 0)
    effective_global = gpu_count * per_device_batch * accumulation
    if (
        split.get("schema") != "npa.groot.episode_split.v1"
        or split.get("run_id") != run_id
        or split.get("status") != "prepared"
        or int(training_plan.get("configured_max_steps") or 0) != configured_steps
        or int(training_plan.get("effective_max_steps") or 0) != configured_steps
        or int(training_plan.get("global_batch_size") or 0) != effective_global
        or int(manifest.get("global_batch_size") or 0) != effective_global
        or int(training_plan.get("per_device_batch_size") or 0) != per_device_batch
        or int(training_plan.get("gradient_accumulation_steps") or 0) != accumulation
        or effective_global < 1
    ):
        raise GrootVisualizationError("trainer and split preflight contracts differ")

    with tempfile.TemporaryDirectory(prefix="npa-groot-checkpoint-ref-") as tmp:
        root = Path(tmp)
        candidate_root = root / "candidate"
        _download_prefix(client, checkpoint_uri, candidate_root)
        checkpoint_path, checkpoint_step = _resolve_highest_checkpoint_directory(
            candidate_root
        )
        if checkpoint_step != completed_steps:
            raise GrootVisualizationError(
                "latest uploaded checkpoint does not equal the completed optimizer step"
            )
        identity = _checkpoint_identity(checkpoint_path)
        checkpoint_model_config_contract(checkpoint_path)
        baseline_path = root / "baseline"
        _download_prefix(client, baseline_checkpoint_uri, baseline_path)
        baseline_identity = _checkpoint_identity(baseline_path)
        checkpoint_model_config_contract(baseline_path)
        require_distinct_trained_weights(baseline_identity, identity)

    resolved_uri = checkpoint_uri.rstrip("/") + f"/checkpoint-{checkpoint_step}/"
    result = {
        "schema": CHECKPOINT_REF_SCHEMA,
        "status": "resolved",
        "run_id": run_id,
        "checkpoint": {
            "uri": resolved_uri,
            "resolved_checkpoint_step": checkpoint_step,
            **identity,
        },
        "baseline_checkpoint": {"uri": baseline_checkpoint_uri, **baseline_identity},
        "weights_differ": True,
        "model_config_contract": GROOT_MODEL_CONFIG_CONTRACT,
        "training_manifest_uri": training_manifest_uri,
        "split_manifest_uri": split_manifest_uri,
        "training": {
            "base_model": manifest.get("base_model"),
            "base_model_revision": manifest.get("base_model_revision"),
            "groot_repo_ref": manifest.get("groot_repo_ref"),
            "gpu_model": manifest.get("gpu_model"),
            "gpu_count": gpu_count,
            "world_size": world_size,
            "distinct_gpu_count": distinct_gpu_count,
            "optimizer_steps": completed_steps,
            "global_batch_size": effective_global,
            "per_device_batch_size": per_device_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_global_batch_size": effective_global,
            "checkpoint_steps": checkpoint_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "observed_ranks": observed_ranks,
            "training_observed_ranks": training_ranks,
            "rank_zero_checkpoint_only": True,
            "checkpoint_upload_invocations": 1,
            "loss_decreased": manifest.get("loss_decreased") is True,
            "training_examples": int(manifest.get("training_examples") or 0),
            "aggregate_train_loss": manifest.get("aggregate_train_loss"),
            "final_step_loss": manifest.get("final_step_loss", manifest.get("final_loss")),
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True).add_parser(
        "resolve-trained-checkpoint"
    )
    command.add_argument("--training-manifest-uri", required=True)
    command.add_argument("--split-manifest-uri", required=True)
    command.add_argument("--checkpoint-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--baseline-checkpoint-uri", required=True)
    command.add_argument("--expected-gpu-count", type=int, required=True)
    command.add_argument("--expected-max-steps", type=int, required=True)
    command.add_argument("--expected-save-steps", type=int, required=True)
    command.add_argument("--expected-save-total-limit", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = vars(args).copy()
    values.pop("command")
    resolve_trained_checkpoint(**values)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
