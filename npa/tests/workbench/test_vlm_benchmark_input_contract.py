"""Negative controls for public benchmark inputs and legacy metric consumers."""

from __future__ import annotations

from dataclasses import asdict, replace
import errno
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
    sample = vlm_eval.load_benchmark_dataset(
        str(vlm_eval.DEFAULT_SAMPLE_BENCHMARK_PATH)
    )
    assert report.item_count == len(sample.items) == 5
    assert any(
        item.id == "progress-without-terminal-fail" and item.expected_label is False
        for item in sample.items
    )


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


@pytest.mark.parametrize(
    "rubric",
    [vlm_eval.DEFAULT_RUBRIC, ("Require visible terminal evidence. " * 40).strip()],
)
def test_public_benchmark_accepts_long_inline_rubrics(rubric: str) -> None:
    assert len(rubric.encode()) > 255
    report = vlm_eval.benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="stub",
        rubrics=[rubric],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert report.item_count == 2
    assert report.best_config.config.rubric == rubric
    assert {
        case.item_id: case.expected_label for case in report.best_config.results
    } == {
        "cube-elevated-positive": True,
        "robot-grasp-lift-negative": False,
    }


@pytest.mark.parametrize("prefix", ["", "@"])
@pytest.mark.parametrize("relative", [False, True])
def test_benchmark_rubric_file_resolution_is_preserved(
    prefix: str, relative: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "explicit-rubric.txt"
    path.write_text("Require a visible terminal state.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rubric = prefix + (path.name if relative else str(path))
    report = vlm_eval.benchmark_vlm_eval(
        dataset="isaac-agency",
        backend="stub",
        rubrics=[rubric],
        thresholds=[0.5],
        frame_selection="sequence",
        max_frames=6,
    )
    assert report.best_config.config.rubric == "Require a visible terminal state."


@pytest.mark.parametrize("rubric", ["default", "dataset-named"])
def test_dataset_named_rubrics_keep_precedence(rubric: str) -> None:
    assert vlm_eval._resolve_benchmark_rubrics(
        [rubric], dataset_rubrics={rubric: "Named fixture rubric"}, dataset_path=""
    ) == [(rubric, "Named fixture rubric")]


def test_explicit_missing_rubric_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(vlm_eval.VlmEvalError, match="rubric file does not exist"):
        vlm_eval.benchmark_vlm_eval(
            dataset="isaac-agency",
            backend="stub",
            rubrics=["@" + str(tmp_path / "missing.txt")],
        )


def test_explicit_overlong_rubric_file_is_not_inline_text() -> None:
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval.benchmark_vlm_eval(
            dataset="isaac-agency",
            backend="stub",
            rubrics=["@" + vlm_eval.DEFAULT_RUBRIC],
        )


@pytest.mark.parametrize("prefix", ["", "@"])
def test_rubric_stat_permission_error_is_not_inline_text(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def inaccessible(_path: Path) -> bool:
        raise PermissionError(errno.EACCES, "Synthetic permission denial")

    monkeypatch.setattr(Path, "is_file", inaccessible)
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval._resolve_benchmark_rubrics(
            [prefix + "protected-rubric.txt"], dataset_rubrics={}, dataset_path=""
        )


@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_explicit_unreadable_rubric_file_fails_closed(
    invalid_utf8: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.txt"
    path.write_bytes(b"\xff" if invalid_utf8 else b"Never substitute inline text.")
    if not invalid_utf8:
        original_read = Path.read_text

        def denied_read(candidate: Path, *args: Any, **kwargs: Any) -> str:
            if candidate == path:
                raise PermissionError(errno.EACCES, "Synthetic read permission denial")
            return original_read(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied_read)
    with pytest.raises(vlm_eval.VlmEvalError, match="Unable to read rubric file"):
        vlm_eval.benchmark_vlm_eval(
            dataset="isaac-agency", backend="stub", rubrics=["@" + str(path)]
        )


@pytest.mark.parametrize(
    "construction", ["legacy11", "landed16", "agency18", "keyword"]
)
def test_metric_constructor_preserves_landed_fields(construction: str) -> None:
    core = {
        "total": 9,
        "correct": 7,
        "agreement": 0.7778,
        "accuracy": 0.7778,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "true_positives": 4,
        "true_negatives": 3,
        "false_positives": 1,
        "false_negatives": 1,
    }
    landed = {
        "confusion_matrix": {
            "actual_positive": {"predicted_positive": 4, "predicted_negative": 1},
            "actual_negative": {"predicted_positive": 1, "predicted_negative": 3},
        },
        "false_positive_rate": 0.25,
        "false_negative_rate": 0.2,
        "false_positive_item_ids": ("negative-1",),
        "false_negative_item_ids": ("positive-1",),
    }
    agency = {"specificity": 0.75, "balanced_accuracy": 0.775}
    if construction == "legacy11":
        metrics = vlm_eval.VlmBenchmarkMetrics(*core.values())
        expected = {
            **core,
            "confusion_matrix": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "false_positive_item_ids": (),
            "false_negative_item_ids": (),
            "specificity": None,
            "balanced_accuracy": None,
        }
    elif construction == "landed16":
        metrics = vlm_eval.VlmBenchmarkMetrics(*core.values(), *landed.values())
        expected = {**core, **landed, "specificity": None, "balanced_accuracy": None}
    elif construction == "agency18":
        metrics = vlm_eval.VlmBenchmarkMetrics(
            *core.values(), *landed.values(), *agency.values()
        )
        expected = {**core, **landed, **agency}
    else:
        metrics = vlm_eval.VlmBenchmarkMetrics(**agency, **landed, **core)
        expected = {**core, **landed, **agency}
    assert asdict(metrics) == expected
    if construction in {"landed16", "agency18"}:
        assert list(asdict(metrics)) == [*core, *landed, *agency]
