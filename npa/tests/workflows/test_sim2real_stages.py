"""Tests for mandatory sim2real preamble stages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    Sim2RealStageError,
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
    assert not k8s_image_ready("<your-registry>/npa:tag")
    assert k8s_image_ready(
        "registry.example/operator/npa-cosmos2-transfer:2.5.0"
    )


def test_augment_stage_uses_seam_reference_for_placeholder_image(
    tmp_path: Path,
) -> None:
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


def test_augment_stage_mirrors_k8s_frame_descriptors(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_component(config, *, input_uri, output_uri, local_dir):
        return {
            "manifest": {
                "status": "executed",
                "mode": "cosmos_transfer2.5_gpu",
                "frame_count": 2,
                "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
            },
            "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
        }

    class FakeClient:
        def download_directory(self, uri: str, local_dir: str) -> None:
            frames_dir = Path(local_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "index.json").write_text(
                json.dumps(
                    {
                        "schema": "npa.sim2real.augmented_frames.v1",
                        "frame_count": 1,
                        "frames": [
                            {"frame_id": "frame-00000", "uri": f"{uri}frame-00000.json"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (frames_dir / "frame-00000.json").write_text(
                json.dumps(
                    {
                        "schema": "npa.sim2real.augmented_frame.v1",
                        "frame_id": "frame-00000",
                        "perturbation": "lighting",
                        "status": "cosmos2_transfer_executed",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "npa.workflows.sim2real.engine.run_cosmos2_transfer_component",
        fake_component,
    )
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: FakeClient(),
    )
    config = Sim2RealLoopConfig(
        run_id="k8s-augment",
        output_dir=tmp_path,
        s3_bucket="bucket",
        s3_endpoint="https://storage.example.test",
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        augment_image="registry.example/operator/npa-cosmos2-transfer:2.5.0",
    )

    result = run_augment_stage(config, tmp_path)

    assert result["component"]["tier"] == "WORKS"
    assert (tmp_path / "augment" / "frames" / "index.json").is_file()
    assert (tmp_path / "augment" / "frames" / "frame-00000.json").is_file()


def test_augment_stage_keeps_working_when_frame_mirror_lags(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_component(config, *, input_uri, output_uri, local_dir):
        return {
            "manifest": {
                "status": "executed",
                "mode": "cosmos_transfer2.5_gpu",
                "frame_count": 2,
                "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
            },
            "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
        }

    class FakeClient:
        def download_directory(self, uri: str, local_dir: str) -> None:
            raise OSError(f"not visible yet: {uri}")

    monkeypatch.setattr(
        "npa.workflows.sim2real.engine.run_cosmos2_transfer_component",
        fake_component,
    )
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: FakeClient(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages.time.sleep", lambda _seconds: None
    )
    config = Sim2RealLoopConfig(
        run_id="k8s-augment-lag",
        output_dir=tmp_path,
        s3_bucket="bucket",
        s3_endpoint="https://storage.example.test",
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        augment_image="registry.example/operator/npa-cosmos2-transfer:2.5.0",
    )

    result = run_augment_stage(config, tmp_path)

    assert result["component"]["tier"] == "WORKS"
    assert "falls back to manifest descriptors" in result["component"]["evidence"]
    warning = json.loads(
        (tmp_path / "augment" / "frames" / "mirror-warning.json").read_text()
    )
    assert warning["status"] == "mirror_unavailable"


def test_augment_stage_rejects_descriptor_stub_from_qualified_image(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_component(config, *, input_uri, output_uri, local_dir):
        return {
            "manifest": {
                "status": "executed",
                "mode": "descriptor_stub",
                "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
            },
            "augmented_frames_uri": f"{output_uri.rstrip('/')}/frames/",
        }

    monkeypatch.setattr(
        "npa.workflows.sim2real.engine.run_cosmos2_transfer_component",
        fake_component,
    )
    config = Sim2RealLoopConfig(
        run_id="reject-stub",
        output_dir=tmp_path,
        s3_bucket="bucket",
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        augment_image=(
            "registry.example/operator/npa-cosmos2-transfer:2.5.0"
        ),
    )

    with pytest.raises(Sim2RealStageError, match="did not emit real GPU provenance"):
        run_augment_stage(config, tmp_path)


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
        "npa.workflows.sim2real_envgen.frame_uris_from_augmented_index",
        lambda _uri: ("s3://bucket/run/augment/frames/frame-00000.png",),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages.load_raw_shards",
        lambda envgen, output_dir: (
            [],
            {
                "mode": "downloaded_stage_04_raw_shards",
                "row_count": envgen.env_count,
            },
        ),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages.write_split_manifest",
        lambda envgen, output_dir, **kwargs: {
            "uploaded_train": "s3://bucket/run/envs/train/envs.jsonl",
            "uploaded_heldout": "s3://bucket/run/envs/heldout/envs.jsonl",
            "uploaded_validation": "s3://bucket/run/envs/validation/envs.jsonl",
            "uploaded_gold_heldout": "s3://bucket/run/envs/gold-heldout/envs.jsonl",
            "uploaded_curation": "s3://bucket/run/envs/manifest/curation-manifest.json",
            "uploaded_manifest": "s3://bucket/run/envs/manifest/split-manifest.json",
            "train_count": 8,
            "heldout_count": 2,
            "validation_count": 1,
            "gold_heldout_count": 1,
            "raw_count": 10,
            "train_uri": "s3://bucket/run/envs/train/",
            "heldout_uri": "s3://bucket/run/envs/heldout/",
            "validation_uri": "s3://bucket/run/envs/validation/",
            "gold_heldout_uri": "s3://bucket/run/envs/gold-heldout/",
            "config_digest_leakage": {},
            "coverage": {},
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
        envgen_image="registry.example/operator/npa-envgen:0.1.1",
    )
    result = run_envgen_split_stage(
        config,
        tmp_path,
        augmented_frames_uri="s3://bucket/run/augment/frames/",
    )
    assert calls == [4]
    assert result["component"]["tier"] == "WORKS"
    assert "indexed GPU shards" in result["component"]["evidence"]
