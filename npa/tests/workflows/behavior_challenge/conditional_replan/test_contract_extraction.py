"""Tests for fixed split, noise, anchors, and feature serialization."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.contract import (
    ReplanContract,
    canonical_sha256,
    fixed_semantic_config,
)
from npa.workflows.behavior_challenge.conditional_replan.extraction import (
    build_features,
    proposal_errors,
)


def _contract() -> dict:
    nested = _nested_identity(tuple(range(180)))
    value = {
        "schema": "npa.behavior.parent-diagnostic-conditional-gate-fit-contract.v1",
        "status": "preregistered_before_parent_proposal_extraction",
        "membership": {
            "episodes": 180,
            "fit_episodes": 160,
            "validation_episodes": 20,
            "rows_per_episode": 12,
            "development_report_or_campaign_holdout_fit": False,
            "row_weighting": "uniform_equal_to_episode_weighting",
            "split_sha256": "a" * 64,
            "nested_canonical_sha256": nested,
        },
        "optimization": _optimization(),
    }
    value.update(fixed_semantic_config())
    return value


def _optimization() -> dict:
    return {
        "amsgrad": False,
        "backend": "locked217_torch2.7.1_cpu",
        "batch_size": 64,
        "betas": [0.9, 0.999],
        "cpu_threads": 1,
        "deterministic_algorithms": True,
        "drop_last": False,
        "early_stop_patience_epochs": 20,
        "epoch_shuffle_seed": 1729,
        "epsilon": 1e-8,
        "foreach": False,
        "fused": False,
        "gradient_clip_global_l2": 1.0,
        "huber_delta": 1.0,
        "learning_rate": 0.001,
        "learning_rate_schedule": "constant",
        "loss": "Huber_scaled_target",
        "maximum_epochs": 200,
        "minimum_epochs": 1,
        "optimizer": "AdamW",
        "reduction": "mean",
        "refit_after_selection": False,
        "selection": "earliest_strictly_lowest_validation_mean_huber",
        "weight_decay": 0.001,
    }


def _nested_identity(episodes: tuple[int, ...]) -> str:
    ranked = sorted(
        episodes,
        key=lambda episode: (
            hashlib.sha256(f"task1-gate-v1:{episode}".encode()).hexdigest(),
            episode,
        ),
    )
    nested = {"validation": ranked[:20], "fit": ranked[20:]}
    return canonical_sha256(nested)


def _validated(candidate: dict | None = None) -> ReplanContract:
    value = candidate or _contract()
    return ReplanContract.from_mapping(
        value,
        expected_split_sha256="a" * 64,
        expected_nested_canonical_sha256=_nested_identity(tuple(range(180))),
    )


def test_split_noise_and_anchor_are_deterministic() -> None:
    contract = _validated()
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


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("target", "definition", "fresh_error_minus_continue_error"),
        ("features", "serialization_order", ["reversed"]),
        ("features", "input_scaling", "none"),
        ("gate", "refresh_when", "always"),
        ("gate", "threshold_decoded_delta", 999.0),
        ("models", "initialization_seed", 1033.0),
        ("models", "initialization", "custom"),
    ),
)
def test_contract_rejects_semantic_configuration_mutations(section, key, value) -> None:
    candidate = _contract()
    candidate[section][key] = value
    with pytest.raises(ValueError, match="semantic configuration"):
        _validated(candidate)


@pytest.mark.parametrize("key", tuple(_optimization()))
def test_contract_rejects_every_optimization_mutation(key) -> None:
    candidate = _contract()
    candidate["optimization"][key] = _different(candidate["optimization"][key])
    with pytest.raises(ValueError, match="optimization differs"):
        _validated(candidate)


@pytest.mark.parametrize("key", tuple(_contract()["membership"]))
def test_contract_rejects_every_membership_mutation(key) -> None:
    candidate = _contract()
    candidate["membership"][key] = _different(candidate["membership"][key])
    with pytest.raises(ValueError, match="contract differs|membership differs"):
        _validated(candidate)


def test_contract_rejects_digest_binding_and_partition_mutations() -> None:
    candidate = _contract()
    with pytest.raises(ValueError, match="split SHA-256"):
        ReplanContract.from_mapping(
            candidate,
            expected_split_sha256="UPPER",
            expected_nested_canonical_sha256=candidate["membership"][
                "nested_canonical_sha256"
            ],
        )
    contract = _validated(candidate)
    with pytest.raises(ValueError, match="nested TRAIN membership"):
        contract.nested_partition(tuple(range(1, 181)))


def _different(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.25
    if isinstance(value, list):
        return [*value, "changed"]
    return f"{value}-changed"
