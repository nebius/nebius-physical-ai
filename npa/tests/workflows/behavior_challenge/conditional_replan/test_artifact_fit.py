"""Tests for deterministic artifact and fixed Torch fitting seams."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.artifact import (
    load_artifact,
    predict_decoded,
    write_artifact,
)
from npa.workflows.behavior_challenge.conditional_replan.contract import (
    fixed_semantic_config,
)
from npa.workflows.behavior_challenge.conditional_replan.fit import (
    FitResult,
    _require_membership,
    _train_model,
    export_result,
    fit_models,
    make_models,
)
from npa.workflows.behavior_challenge.conditional_replan.fit_data import FitData


def test_artifact_roundtrip_is_deterministic_and_numpy_only(
    tmp_path, artifact, monkeypatch
) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    write_artifact(first, artifact)
    write_artifact(second, artifact)
    assert first.read_bytes() == second.read_bytes()
    monkeypatch.setitem(sys.modules, "torch", None)
    loaded = load_artifact(first)
    assert predict_decoded(loaded, np.zeros((2, 1233), np.float32), "mlp").shape == (2,)


def test_fit_and_export_preserve_model_schema_and_parity(fit_data) -> None:
    torch = pytest.importorskip("torch")
    models = make_models(torch)
    scalers = {
        "input_mean": np.zeros(1233, np.float32),
        "input_std": np.ones(1233, np.float32),
        "target_mean": np.zeros(1, np.float32),
        "target_std": np.ones(1, np.float32),
    }
    selection = {name: {"selected_epoch": 1} for name in models}
    result = FitResult(models, scalers, {}, selection)
    semantics = fixed_semantic_config()
    metadata = {
        "schema": "npa.behavior.parent-diagnostic-conditional-gate-artifact.v1",
        "selection": {
            name: row["selected_epoch"] for name, row in result.selection.items()
        },
        "development_or_report_used": False,
        "feature_config": semantics["features"],
        "target_config": semantics["target"],
        "model_config": semantics["models"],
        "gate_config": semantics["gate"],
    }
    artifact = export_result(result, metadata)
    assert artifact.arrays["mlp.0.weight"].shape == (256, 1233)
    assert set(result.selection) == {"linear_control", "mlp"}


def test_export_rejects_selection_different_from_fit_result(fit_data) -> None:
    torch = pytest.importorskip("torch")
    result = FitResult(
        make_models(torch),
        _unit_scalers(),
        {},
        {"linear_control": {"selected_epoch": 1}, "mlp": {"selected_epoch": 2}},
    )
    metadata = _artifact_metadata({"linear_control": 1, "mlp": 1})
    with pytest.raises(ValueError, match="selection differs"):
        export_result(result, metadata)


def test_artifact_rejects_shape_or_nonfinite(tmp_path, artifact) -> None:
    artifact.arrays["target_std"][0] = np.nan
    with pytest.raises(ValueError, match="target_std"):
        write_artifact(tmp_path / "bad.npz", artifact)


def test_artifact_exclusive_write_preserves_existing_file(tmp_path, artifact) -> None:
    target = tmp_path / "gate.npz"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        write_artifact(target, artifact)
    assert target.read_bytes() == b"existing"


def test_artifact_exclusive_write_rejects_dangling_symlink(tmp_path, artifact) -> None:
    outside = tmp_path / "outside.npz"
    target = tmp_path / "gate.npz"
    target.symlink_to(outside)
    with pytest.raises(FileExistsError):
        write_artifact(target, artifact)
    assert target.is_symlink() and not outside.exists()


def test_fit_rejects_incomplete_fixed_membership(fit_data) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="membership"):
        fit_models(torch, fit_data)


def test_fixed_training_selects_earliest_strict_minimum() -> None:
    torch = pytest.importorskip("torch")
    model = make_models(torch)["linear_control"]
    x, y = np.zeros((8, 1233), np.float32), np.zeros((8, 1), np.float32)
    partitions = np.asarray(["fit"] * 4 + ["validation"] * 4)
    logs, selection = _train_model(torch, model, x, y, partitions)
    values = [row["validation_mean_huber"] for row in logs]
    assert selection["validation_mean_huber"] == min(values)
    assert selection["selected_epoch"] == values.index(min(values)) + 1
    assert selection["epochs_executed"] <= 200


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("target_config", "definition", "fresh_error_minus_continue_error"),
        ("feature_config", "serialization_order", ["reversed"]),
        ("gate_config", "threshold_decoded_delta", 999.0),
        ("gate_config", "refresh_when", "always"),
        ("model_config", "initialization_seed", 1033.0),
        ("model_config", "initialization", "custom"),
    ),
)
def test_artifact_rejects_semantic_configuration_mutations(
    artifact, section, key, value
) -> None:
    artifact.metadata[section][key] = value
    with pytest.raises(ValueError, match="semantic configuration"):
        artifact.validated()


@pytest.mark.parametrize("epoch", (None, "bogus", True, 0, -1, 201))
def test_artifact_rejects_invalid_selected_epoch(artifact, epoch) -> None:
    artifact.metadata["selection"]["mlp"] = epoch
    with pytest.raises(ValueError, match="selected epoch"):
        artifact.validated()


def test_fit_membership_rejects_episode_in_both_partitions() -> None:
    fit_episodes = np.repeat(np.arange(160), 12)
    validation_episodes = np.repeat(np.arange(159, 179), 12)
    episodes = np.concatenate((fit_episodes, validation_episodes))
    fit_rows = np.tile(np.arange(12), 160)
    validation_rows = np.tile(np.arange(12), 20)
    validation_rows[:12] += 12
    rows = np.concatenate((fit_rows, validation_rows))
    partitions = np.asarray(["fit"] * 1920 + ["validation"] * 240)
    order = np.lexsort((rows, episodes))
    data = _empty_fit_data(episodes[order], rows[order], partitions[order])
    with pytest.raises(ValueError, match="globally unique"):
        _require_membership(data)


def _empty_fit_data(episodes, rows, partitions) -> FitData:
    count = len(episodes)
    return FitData(
        np.zeros((count, 1233), np.float32),
        np.zeros(count, np.float64),
        np.zeros(count, np.float64),
        np.zeros(count, np.float64),
        episodes,
        rows,
        partitions,
    )


def _unit_scalers() -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros(1233, np.float32),
        "input_std": np.ones(1233, np.float32),
        "target_mean": np.zeros(1, np.float32),
        "target_std": np.ones(1, np.float32),
    }


def _artifact_metadata(selection: dict[str, int]) -> dict:
    semantics = fixed_semantic_config()
    return {
        "schema": "npa.behavior.parent-diagnostic-conditional-gate-artifact.v1",
        "selection": selection,
        "development_or_report_used": False,
        "feature_config": semantics["features"],
        "target_config": semantics["target"],
        "model_config": semantics["models"],
        "gate_config": semantics["gate"],
    }
