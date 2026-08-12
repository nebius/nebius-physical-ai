"""Dynamic immutable-checkpoint resolution for the offline GR00T workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows import groot_task_performance as resolver
from npa.workflows.groot_learning import (
    GROOT_MODEL_CONFIG_CONTRACT,
    GrootVisualizationError,
)


@pytest.mark.parametrize("steps", [4, 17, 10000])
def test_resolver_selects_manifest_completed_checkpoint_not_a_yaml_literal(
    monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    run_id = "run"
    training_uri = "s3://bucket/run/checkpoints/candidate/npa_groot_finetune_manifest.json"
    split_uri = "s3://bucket/run/reports/split/manifest.json"
    candidate_uri = "s3://bucket/run/checkpoints/candidate/"
    baseline_uri = "s3://bucket/run/checkpoints/baseline/"
    output_uri = "s3://bucket/run/reports/trained-checkpoint.json"
    manifest = {
        "schema": "npa.groot.finetune.v1",
        "status": "completed",
        "run_id": run_id,
        "checkpoint_uri": candidate_uri,
        "optimizer_step_ok": True,
        "collective_ok": True,
        "loss_steps_real": True,
        "rank_zero_checkpoint_only": True,
        "checkpoint_upload_invocations": 1,
        "model_config_contract": GROOT_MODEL_CONFIG_CONTRACT,
        "num_gpus": 2,
        "world_size": 2,
        "distinct_gpu_count": 2,
        "observed_ranks": [0, 1],
        "training_observed_ranks": [0, 1],
        "max_steps": steps,
        "training_step": steps,
        "save_steps": steps,
        "save_total_limit": 1,
        "checkpoint_steps": [steps],
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 2,
        "training_examples": steps * 2,
        "final_step_loss": 1.0,
    }
    split = {
        "schema": "npa.groot.episode_split.v1",
        "status": "prepared",
        "run_id": run_id,
        "training_plan": {
            "configured_max_steps": steps,
            "effective_max_steps": steps,
            "global_batch_size": 2,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
    }
    monkeypatch.setattr(
        resolver,
        "_read_s3_json",
        lambda _client, uri: manifest if uri == training_uri else split,
    )

    def fake_download(_client: object, uri: str, destination: Path) -> None:
        if uri == candidate_uri:
            checkpoint = destination / f"checkpoint-{steps}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"trained")
        else:
            destination.mkdir(parents=True)
            (destination / "model.safetensors").write_bytes(b"baseline")

    monkeypatch.setattr(resolver, "_download_prefix", fake_download)
    monkeypatch.setattr(
        resolver,
        "checkpoint_model_config_contract",
        lambda _path: GROOT_MODEL_CONFIG_CONTRACT,
    )
    stored: dict[str, dict] = {}
    monkeypatch.setattr(
        resolver,
        "_put_json",
        lambda _client, uri, payload: stored.__setitem__(uri, payload),
    )
    result = resolver.resolve_trained_checkpoint(
        training_uri,
        split_uri,
        candidate_uri,
        output_uri,
        run_id,
        baseline_checkpoint_uri=baseline_uri,
        expected_gpu_count=2,
        expected_max_steps=steps,
        expected_save_steps=steps,
        expected_save_total_limit=1,
        s3_client=object(),
    )

    assert result["checkpoint"]["uri"].endswith(f"/checkpoint-{steps}/")
    assert result["checkpoint"]["resolved_checkpoint_step"] == steps
    assert result["training"]["optimizer_steps"] == steps
    assert stored[output_uri] == result


def test_resolver_rejects_schedule_drift_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema": "npa.groot.finetune.v1",
        "status": "completed",
        "run_id": "run",
        "checkpoint_uri": "s3://bucket/candidate/",
        "optimizer_step_ok": True,
        "collective_ok": True,
        "loss_steps_real": True,
        "rank_zero_checkpoint_only": True,
        "checkpoint_upload_invocations": 1,
        "model_config_contract": GROOT_MODEL_CONFIG_CONTRACT,
        "num_gpus": 2,
        "world_size": 2,
        "distinct_gpu_count": 2,
        "observed_ranks": [0, 1],
        "training_observed_ranks": [0, 1],
        "max_steps": 8,
        "training_step": 8,
        "save_steps": 4,
        "save_total_limit": 1,
    }
    monkeypatch.setattr(resolver, "_read_s3_json", lambda *_args: manifest)
    with pytest.raises(GrootVisualizationError, match="schedule"):
        resolver.resolve_trained_checkpoint(
            "s3://bucket/manifest.json",
            "s3://bucket/split.json",
            "s3://bucket/candidate/",
            "s3://bucket/ref.json",
            "run",
            baseline_checkpoint_uri="s3://bucket/baseline/",
            expected_gpu_count=2,
            expected_max_steps=8,
            expected_save_steps=8,
            expected_save_total_limit=1,
            s3_client=object(),
        )
