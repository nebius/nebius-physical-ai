"""Negative controls for public benchmark inputs and legacy metric consumers."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from npa.cli.workbench import vlm_eval as cli_vlm_eval
from npa.workbench import vlm_eval


@pytest.mark.parametrize("dataset", ["", " ", "\t\n"])
def test_direct_dataset_loading_requires_an_explicit_path(dataset: str) -> None:
    with pytest.raises(vlm_eval.VlmEvalError, match="--dataset is required"):
        vlm_eval.load_benchmark_dataset(dataset)


def test_benchmark_default_remains_the_packaged_sample() -> None:
    report = vlm_eval.benchmark_vlm_eval(dataset="", backend="stub")
    assert report.dataset_path == str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH)
    assert report.item_count == 4


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_frames": 0}, "--max-frames must be positive"),
        ({"max_frames": -1}, "--max-frames must be positive"),
        ({"timeout_s": 0}, "--timeout-s must be positive"),
        ({"timeout_s": -1}, "--timeout-s must be positive"),
        ({"thresholds": []}, "--thresholds must include"),
        ({"thresholds": [-0.1]}, "--thresholds values must be between"),
        ({"thresholds": [1.1]}, "--thresholds values must be between"),
        ({"models": [""]}, "--models must include"),
        ({"rubrics": [""]}, "--rubrics must include"),
        ({"frame_selection": "invalid"}, "--frame-selection must be one of"),
    ],
)
def test_scalar_errors_precede_rollout_materialization(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any], message: str
) -> None:
    for name in (
        "_materialized_input",
        "_normalize_backend",
        "_resolve_api_key",
        "_call_openai_compatible",
    ):
        monkeypatch.setattr(
            vlm_eval,
            name,
            lambda *_args, **_kwargs: pytest.fail("rollout/backend/provider activity"),
        )
    with pytest.raises(vlm_eval.VlmEvalError, match=message):
        vlm_eval.benchmark_vlm_eval(dataset="isaac-agency", backend="api", **arguments)


def test_legacy_metrics_sort_below_measured_balanced_accuracy() -> None:
    measured = vlm_eval.benchmark_vlm_eval(backend="stub").best_config
    legacy = replace(
        measured,
        metrics=replace(measured.metrics, specificity=None, balanced_accuracy=None),
    )
    assert sorted([legacy, measured], key=vlm_eval._benchmark_rank_key) == [
        measured,
        legacy,
    ]


@pytest.mark.parametrize("omit", [False, True])
def test_legacy_metric_text_uses_na(
    omit: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = asdict(vlm_eval.benchmark_vlm_eval(backend="stub"))
    payload["written_uri"] = "benchmark.json"
    metrics = payload["best_config"]["metrics"]
    for name in ("specificity", "balanced_accuracy"):
        if omit:
            metrics.pop(name)
        else:
            metrics[name] = None
    cli_vlm_eval._emit_benchmark(payload, cli_vlm_eval.OutputFormat.text)
    output = capsys.readouterr().out
    assert "specificity: n/a" in output
    assert "balanced_accuracy: n/a" in output


def test_task_precedence_is_shared_with_preselected_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vlm_eval, "_explicit_task_text", lambda _task: "shared task")
    assert vlm_eval._resolve_task_text(tmp_path, "sim-to-real") == "shared task"
    item = vlm_eval.load_benchmark_dataset("isaac-agency").items[0]
    frames, task = vlm_eval._select_structural_frames_and_task(
        item, frame_selection="sequence", max_frames=6, selection_cache={}
    )
    assert len(frames) == 6
    assert task == "shared task"
