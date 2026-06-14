from __future__ import annotations

import pytest

from npa.workbench.detection_training.schemas import TrainRequest
from npa.workbench.training_config import (
    TrainingConfigError,
    build_training_config,
    overrides_to_mapping,
)


def test_training_config_builds_canonical_env_and_redacts_secrets() -> None:
    config = build_training_config(
        data_path="s3://bucket/data/",
        overrides=["agent.algorithm.learning_rate=3e-4", "terminations.timeout=false"],
        wandb_enabled=True,
        wandb_project="npa-public",
        wandb_run_name="short-run",
        checkpoint_s3_uri="s3://bucket/checkpoints/",
        checkpoint_s3_endpoint_url="https://storage.example",
        checkpoint_s3_access_key_id="access",
        checkpoint_s3_secret_access_key="secret",
    )

    env = config.env()
    assert env["NPA_TRAINING_DATA_PATH"] == "s3://bucket/data/"
    assert env["NPA_TRAINING_WANDB_ENABLED"] == "1"
    assert env["WANDB_PROJECT"] == "npa-public"
    assert env["NPA_CHECKPOINT_S3_URI"] == "s3://bucket/checkpoints/"
    assert env["NPA_CHECKPOINT_S3_ENDPOINT_URL"] == "https://storage.example"
    assert config.public_dict()["checkpoint_s3"]["aws_secret_access_key"] == "set"


def test_training_config_rejects_invalid_override() -> None:
    with pytest.raises(TrainingConfigError, match="KEY=VALUE"):
        build_training_config(overrides=["learning_rate"])


def test_overrides_to_mapping_parses_typed_values() -> None:
    parsed = overrides_to_mapping(["learning_rate=0.001", "domain_randomize=true", "+env.max_steps=12"])

    assert parsed == {
        "learning_rate": 0.001,
        "domain_randomize": True,
        "env.max_steps": 12,
    }


def test_detection_request_applies_canonical_fields_and_supported_overrides() -> None:
    request = TrainRequest(
        view="bdd100k_rider_train",
        lance_uri="s3://bucket/default-lancedb/",
        output_uri="s3://bucket/output/",
        data_path="s3://bucket/custom-lancedb/",
        overrides=["train.epochs=1", "optimizer.learning_rate=0.0003"],
        checkpoint_s3={"uri": "s3://bucket/checkpoints/", "endpoint_url": "https://storage.example"},
        wandb={"enabled": True, "project": "npa-public", "run_name": "short-run", "mode": "offline"},
    )

    assert request.lance_uri == "s3://bucket/custom-lancedb/"
    assert request.epochs == 1
    assert request.learning_rate == 0.0003
    assert request.checkpoint_s3.uri == "s3://bucket/checkpoints/"
    assert request.wandb.enabled is True
