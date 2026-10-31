"""Verify complete compiled OpenPI evaluation and stderr progress reporting."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from npa.workflows.byof import openpi_pipeline as pipeline


class _LossModel:
    def __init__(self):
        self.calls = []

    def compute_loss(self, rng, observation, targets, *, train):
        self.calls.append((rng, observation, train))
        return targets


def _install_loss_runtime(monkeypatch, compiled):
    def compile_model(method, **kwargs):
        compiled.append((method, kwargs))
        return method

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            random=SimpleNamespace(
                key=lambda seed: seed,
                fold_in=lambda seed, index: (seed, index),
            ),
            device_get=lambda values: values,
        ),
    )
    monkeypatch.setitem(sys.modules, "openpi", ModuleType("openpi"))
    monkeypatch.setitem(
        sys.modules,
        "openpi.shared",
        SimpleNamespace(nnx_utils=SimpleNamespace(module_jit=compile_model)),
    )
    monkeypatch.setitem(
        sys.modules,
        "openpi.training",
        SimpleNamespace(sharding=SimpleNamespace(set_mesh=lambda mesh: nullcontext())),
    )


def test_compiled_loss_visits_every_sample_with_the_declared_noise(monkeypatch, capsys):
    compiled = []
    _install_loss_runtime(monkeypatch, compiled)
    model = _LossModel()
    loader = [
        ("first-observation", np.array([[1, 3], [5, 7]])),
        ("second-observation", np.array([[9, 11], [13, 15]])),
    ]
    args = argparse.Namespace(seed=17, batch_size=2)
    losses = pipeline._heldout_model_losses(args, model, loader, object(), 4)
    assert losses == [2.0, 6.0, 10.0, 14.0]
    assert compiled == [(model.compute_loss, {"static_argnames": ("train",)})]
    assert model.calls == [
        ((17, 0), "first-observation", False),
        ((17, 1), "second-observation", False),
    ]
    output = capsys.readouterr()
    assert output.out == ""
    progress = [json.loads(line) for line in output.err.splitlines()]
    assert [row["completed"] for row in progress] == [0, 4]
    assert all(row["total"] == 4 for row in progress)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_compiled_loss_rejects_non_finite_samples(monkeypatch, invalid):
    _install_loss_runtime(monkeypatch, [])
    loader = [("observation", np.array([[1.0], [invalid]]))]
    args = argparse.Namespace(seed=17, batch_size=2)
    with pytest.raises(pipeline.OpenPIPipelineError, match="loss is non-finite"):
        pipeline._heldout_model_losses(args, _LossModel(), loader, object(), 2)


def test_compiled_loss_rejects_an_incomplete_heldout_set(monkeypatch):
    _install_loss_runtime(monkeypatch, [])
    loader = [("observation", np.array([[1.0], [2.0]]))]
    args = argparse.Namespace(seed=17, batch_size=2)
    with pytest.raises(pipeline.OpenPIPipelineError, match="sample count is incorrect"):
        pipeline._heldout_model_losses(args, _LossModel(), loader, object(), 4)


def test_progress_preserves_stdout_and_reports_step_milestones(monkeypatch, capsys):
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: 5.0)
    for completed in [0, 1, 2, 99, 100, 101, 200]:
        pipeline._log_progress("training", completed, 200, 1.0, 0.25)
    output = capsys.readouterr()
    assert output.out == ""
    progress = [json.loads(line) for line in output.err.splitlines()]
    assert [row["completed"] for row in progress] == [0, 1, 100, 200]
    assert all(
        row
        == {
            "schema": "npa.workbench.openpi.progress.v1",
            "phase": "training",
            "completed": completed,
            "total": 200,
            "elapsed_seconds": 4.0,
            "loss": 0.25,
        }
        for row, completed in zip(progress, [0, 1, 100, 200])
    )
