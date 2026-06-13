"""Tests for mandatory sim2real preamble stages."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows.sim2real_loop import Sim2RealLoopConfig, run_preamble
from npa.workflows.sim2real_stages import (
    DEFAULT_ENV_COUNT,
    effective_env_count,
    effective_heldout_count,
    effective_train_count,
    k8s_image_ready,
    resolve_augment_frame_count,
    run_augment_stage,
    run_envgen_split_stage,
)


def test_effective_env_counts_default_to_legacy_rollout_plus_heldout() -> None:
    config = Sim2RealLoopConfig(
        run_id="counts",
        rollout_count=2,
        heldout_env_count=4,
        env_count=0,
    )
    assert effective_env_count(config) == 6
    assert effective_train_count(config) == 2
    assert effective_heldout_count(config) == 4


def test_effective_env_counts_use_10k_mandatory_split() -> None:
    config = Sim2RealLoopConfig(
        run_id="counts",
        env_count=DEFAULT_ENV_COUNT,
        train_fraction=0.8,
        rollout_count=3,
        heldout_env_count=8,
    )
    assert effective_env_count(config) == 10_000
    assert effective_train_count(config) == 8_000
    assert effective_heldout_count(config) == 2_000


def test_resolve_augment_frame_count_scales_with_rollouts(monkeypatch) -> None:
    monkeypatch.delenv("NPA_SIM2REAL_AUGMENT_FRAME_COUNT", raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_ROLLOUT_COUNT", raising=False)
    assert resolve_augment_frame_count(rollout_count=2) == 16
    assert resolve_augment_frame_count(rollout_count=300) == 1024
    monkeypatch.setenv("NPA_SIM2REAL_AUGMENT_FRAME_COUNT", "64")
    assert resolve_augment_frame_count(rollout_count=2) == 64


def test_preamble_executes_augment_and_envgen_locally(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="preamble-local",
        output_dir=tmp_path,
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        env_count=0,
        rollout_count=2,
        heldout_env_count=4,
        sim_backend="isaac",
    )
    state = run_preamble(config)
    augment = json.loads((tmp_path / "augment" / "manifest.json").read_text())
    assets = json.loads(
        (tmp_path / "stage_02_assets" / "consumed_scene_spec.json").read_text()
    )
    assert augment["status"] in {"executed_reference", "executed"}
    assert assets["sim_backend"] == "isaac"
    assert state["train_env_count"] == 2
    assert state["heldout_env_count"] == 4
    assert state["env_count"] == 6


def test_k8s_image_ready_rejects_bare_tags_and_placeholders() -> None:
    assert not k8s_image_ready("npa-cosmos2-transfer:2.5.0")
    assert not k8s_image_ready("cr.eu-north1.nebius.cloud/<your-registry-id>/npa:tag")
    assert k8s_image_ready(
        "cr.eu-north1.nebius.cloud/example-registry-id/npa-cosmos2-transfer:2.5.0"
    )


def test_augment_stage_uses_seam_reference_for_placeholder_image(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="seam-augment",
        output_dir=tmp_path,
        s3_bucket="bucket",
        s3_endpoint="",
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        augment_image="npa-cosmos2-transfer:2.5.0",
    )
    result = run_augment_stage(config, tmp_path)
    assert result["component"]["tier"] == "SEAM"
    assert result["manifest"]["status"] == "executed_reference"
    assert (tmp_path / "augment" / "frames" / "index.json").exists()


def test_envgen_split_stage_launches_indexed_shards_when_image_ready(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[int] = []

    def fake_sharded(config, *, envgen):
        calls.append(envgen.shard_count)
        return {"shard_count": envgen.shard_count, "parallelism": 2}

    monkeypatch.setattr(
        "npa.workflows.sim2real.engine.run_envgen_sharded_component",
        fake_sharded,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages.write_split_manifest",
        lambda envgen, output_dir: {
            "uploaded_train": "s3://bucket/run/envs/train/envs.jsonl",
            "uploaded_heldout": "s3://bucket/run/envs/heldout/envs.jsonl",
            "uploaded_manifest": "s3://bucket/run/envs/manifest/split-manifest.json",
            "train_count": 8,
            "heldout_count": 2,
            "raw_count": 10,
            "train_uri": "s3://bucket/run/envs/train/",
            "heldout_uri": "s3://bucket/run/envs/heldout/",
        },
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages._mirror_env_manifests",
        lambda *args, **kwargs: None,
    )

    config = Sim2RealLoopConfig(
        run_id="envgen-sharded",
        output_dir=tmp_path,
        s3_bucket="bucket",
        env_count=10,
        train_fraction=0.8,
        envgen_shard_count=4,
        envgen_image="cr.eu-north1.nebius.cloud/example-registry-id/npa-sim2real-envgen:0.1.1",
    )
    result = run_envgen_split_stage(
        config,
        tmp_path,
        augmented_frames_uri="s3://bucket/run/augment/frames/",
    )
    assert calls == [4]
    assert result["component"]["tier"] == "WORKS"
    assert "indexed GPU shards" in result["component"]["evidence"]
