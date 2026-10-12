"""Unit tests for the Gemini -> GPU distillation workflow."""

from __future__ import annotations

from npa.workflows.byof.gemini_robotics_distill import (
    DISTILL_RECEIPT_SCHEMA,
    DistillConfig,
)


def test_distill_config_defaults_to_validated_model() -> None:
    from npa.clients.gemini_robotics import DEFAULT_MODEL_ID

    config = DistillConfig(task="t", output_dir="o")
    assert config.resolved_model() == DEFAULT_MODEL_ID
    config = DistillConfig(task="t", output_dir="o", model="custom")
    assert config.resolved_model() == "custom"


def test_distill_receipt_schema() -> None:
    assert DISTILL_RECEIPT_SCHEMA == "npa.gemini_robotics.distill.v1"
