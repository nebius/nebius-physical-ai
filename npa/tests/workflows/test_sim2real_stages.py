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


def test_preamble_executes_augment_and_envgen_locally(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="preamble-local",
        output_dir=tmp_path,
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        env_count=0,
        rollout_count=2,
        heldout_env_count=4,
    )
    state = run_preamble(config)
    augment = json.loads((tmp_path / "augment" / "manifest.json").read_text())
    assert augment["status"] in {"executed_reference", "executed"}
    assert state["train_env_count"] == 2
    assert state["heldout_env_count"] == 4
    assert state["env_count"] == 6
