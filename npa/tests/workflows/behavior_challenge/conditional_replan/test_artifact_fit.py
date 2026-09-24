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
from npa.workflows.behavior_challenge.conditional_replan.fit import (
    FitResult,
    _train_model,
    export_result,
    fit_models,
    make_models,
)


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
    metadata = {
        "schema": "npa.behavior.parent-diagnostic-conditional-gate-artifact.v1",
        "selection": {
            name: row["selected_epoch"] for name, row in result.selection.items()
        },
        "development_or_report_used": False,
        "feature_config": {},
        "target_config": {},
        "model_config": {},
        "gate_config": {},
    }
    artifact = export_result(result, metadata)
    assert artifact.arrays["mlp.0.weight"].shape == (256, 1233)
    assert set(result.selection) == {"linear_control", "mlp"}


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
