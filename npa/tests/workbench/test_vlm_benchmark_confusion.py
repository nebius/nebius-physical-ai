"""Verify benchmark failure membership through the public sweep and report APIs."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from npa.sdk.workbench.vlm_eval import benchmark
from npa.workbench import vlm_eval


def _write_dataset(tmp_path: Path, items: list[dict]) -> Path:
    dataset = tmp_path / "benchmark.json"
    dataset.write_text(json.dumps({"items": items}), encoding="utf-8")
    return dataset


def _fixture_item(item_id: str, label: bool, score: float) -> dict:
    return {
        "id": item_id,
        "rollout": "unused-fixture-frames",
        "expected_label": label,
        "fixture_score": score,
    }


def _saved_fixture_sweep(tmp_path: Path) -> dict:
    dataset = _write_dataset(
        tmp_path,
        [
            _fixture_item("positive-at-boundary", True, 0.8),
            _fixture_item("negative-at-boundary", False, 0.8),
            _fixture_item("positive-below-boundary", True, 0.79),
            _fixture_item("negative-below-boundary", False, 0.79),
        ],
    )
    report = benchmark(dataset=str(dataset), backend="stub", thresholds=(0.79, 0.8))
    output = tmp_path / "report.json"
    vlm_eval.write_benchmark_report(asdict(report), output_path=str(output))
    return json.loads(output.read_text())


def test_sweep_keeps_failure_ids_at_the_inclusive_score_boundary(tmp_path) -> None:
    saved = _saved_fixture_sweep(tmp_path)
    assert saved["schema_version"] == "npa_vlm_eval_benchmark_report_v2"
    configs = {
        row["config"]["success_threshold"]: row for row in saved["ranked_configs"]
    }
    assert saved["best_config"] == saved["ranked_configs"][0]
    assert configs[0.8]["metrics"]["false_positive_item_ids"] == [
        "negative-at-boundary"
    ]
    assert configs[0.8]["metrics"]["false_negative_item_ids"] == [
        "positive-below-boundary"
    ]
    assert configs[0.8]["metrics"]["confusion_matrix"] == {
        "actual_positive": {"predicted_positive": 1, "predicted_negative": 1},
        "actual_negative": {"predicted_positive": 1, "predicted_negative": 1},
    }
    assert configs[0.79]["metrics"]["false_positive_item_ids"] == [
        "negative-at-boundary",
        "negative-below-boundary",
    ]
    assert configs[0.79]["metrics"]["false_negative_item_ids"] == []
    for row in saved["ranked_configs"]:
        for case in row["results"]:
            assert case["predicted_label"] is (
                case["score"] >= row["config"]["success_threshold"]
            )
            assert case["score_source"] == "fixture" and case["evidence"] is None


@pytest.mark.parametrize(
    "identities",
    [
        [{"id": "control"}, {"id": " control "}],
        [{"id": "control"}, {"name": "control"}],
        [{"id": "item-002"}, {}],
        [{"id": "1"}, {"id": 1}],
    ],
)
def test_duplicate_normalized_ids_fail_before_evaluation(
    tmp_path, monkeypatch, identities
) -> None:
    items = [
        {"rollout": "unused-frames", "expected_label": True, **identity}
        for identity in identities
    ]
    dataset = _write_dataset(tmp_path, items)

    def unexpected_evaluation(**kwargs):
        pytest.fail("duplicate IDs must fail before any provider evaluation")

    monkeypatch.setattr(vlm_eval, "evaluate_vlm", unexpected_evaluation)
    with pytest.raises(vlm_eval.VlmEvalError, match="item IDs must be unique"):
        benchmark(dataset=str(dataset), backend="api")


@pytest.mark.parametrize(
    ("labels_and_scores", "expected_f1"),
    [
        ([(True, 1.0)] + [(True, 0.0)] * 5, 0.2857),
        ([(False, 1.0), (True, 0.0)], 0.0),
        ([(False, 0.0)], None),
    ],
)
def test_f1_uses_exact_counts_until_final_rounding(
    tmp_path, labels_and_scores, expected_f1
) -> None:
    items = [
        _fixture_item(f"case-{index}", label, score)
        for index, (label, score) in enumerate(labels_and_scores)
    ]
    dataset = _write_dataset(tmp_path, items)

    report = benchmark(dataset=str(dataset), backend="stub", thresholds=(0.8,))

    assert report.best_config.metrics.f1 == expected_f1
