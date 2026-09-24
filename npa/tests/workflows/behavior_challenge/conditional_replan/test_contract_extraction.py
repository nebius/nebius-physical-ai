"""Tests for fixed split, noise, anchors, and feature serialization."""

from __future__ import annotations

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.contract import ReplanContract
from npa.workflows.behavior_challenge.conditional_replan.extraction import (
    build_features,
    proposal_errors,
)


def _contract() -> dict:
    return {
        "schema": "npa.behavior.parent-diagnostic-conditional-gate-fit-contract.v1",
        "status": "preregistered_before_parent_proposal_extraction",
        "features": {"dimension": 1233, "dtype": "float32", "minimum_std": 1e-6},
        "gate": {
            "check_offset": 16,
            "threshold_decoded_delta": 0.0,
            "numpy_gelu": "0.5*x*(1+erf(x/sqrt(2)))",
        },
        "membership": {
            "episodes": 180,
            "fit_episodes": 160,
            "validation_episodes": 20,
            "rows_per_episode": 12,
        },
        "models": {
            "initialization_seed": 1033,
            "parameter_dtype": "float32",
            "linear_control": {"widths": [1233, 1]},
            "mlp": {
                "widths": [1233, 256, 64, 1],
                "activation": "GELU_exact_erf",
                "dropout": 0.0,
            },
        },
        "optimization": _optimization(),
    }


def _optimization() -> dict:
    return {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.001,
        "batch_size": 64,
        "epoch_shuffle_seed": 1729,
        "maximum_epochs": 200,
        "early_stop_patience_epochs": 20,
        "gradient_clip_global_l2": 1.0,
        "huber_delta": 1.0,
    }


def test_split_noise_and_anchor_are_deterministic() -> None:
    contract = ReplanContract.from_mapping(_contract())
    first = contract.nested_partition(tuple(range(180)))
    second = contract.nested_partition(tuple(reversed(range(180))))
    assert first == second and len(first["validation"]) == 20
    assert contract.selected_starts(96) == (0, 32, 64)
    assert np.array_equal(
        contract.noise(7, 32, 3, "fresh"), contract.noise(7, 32, 3, "fresh")
    )
    assert not np.array_equal(
        contract.noise(7, 32, 3, "fresh"), contract.noise(7, 32, 3, "chunk_start")
    )


def test_feature_layout_and_errors_match_qualified_formulas() -> None:
    cont, fresh = np.zeros((16, 23), np.float32), np.ones((16, 23), np.float32)
    feature = build_features(
        cont, fresh, np.zeros(32, np.float32), np.ones(32, np.float32)
    )
    assert feature.shape == (1233,) and feature.dtype == np.float32
    expert = np.zeros((2, 16, 23), np.float32)
    errors = proposal_errors(expert, np.ones_like(expert), np.full_like(expert, 2))
    assert tuple(value.tolist() for value in errors) == (
        [1.0, 1.0],
        [4.0, 4.0],
        [1.0, 1.0],
    )


def test_extraction_rejects_tampered_dtype_and_mask() -> None:
    values = np.zeros((16, 23), np.float32)
    with pytest.raises(ValueError, match="representation"):
        build_features(
            values.astype(np.float64),
            values,
            np.zeros(32, np.float32),
            np.zeros(32, np.float32),
        )
    with pytest.raises(ValueError, match="binary"):
        build_features(
            values,
            values,
            np.zeros(32, np.float32),
            np.zeros(32, np.float32),
            np.full(16, 0.5, np.float32),
        )
