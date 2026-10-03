"""Verify benchmark failure membership through the public sweep and report APIs."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from npa.sdk.workbench.vlm_eval import benchmark
from npa.workbench import vlm_eval


@pytest.fixture(scope="module")
def live_rate_assertion():
    path = Path(__file__).resolve().parents[1] / "e2e/test_token_factory_e2e.py"
    spec = importlib.util.spec_from_file_location("benchmark_live_assertions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._assert_benchmark_error_rates


@pytest.mark.parametrize(
    ("counts", "rates"),
    [
        ((2, 3, 1, 1), (0.25, 0.3333)),
        ((0, 1, 2, 0), (0.6667, None)),
        ((1, 0, 0, 2), (None, 0.6667)),
        ((0, 0, 0, 0), (None, None)),
    ],
)
def test_live_rate_assertion_uses_class_denominators(
    live_rate_assertion, counts, rates
) -> None:
    expected_counts = dict(
        zip(
            ("true_positive", "true_negative", "false_positive", "false_negative"),
            counts,
            strict=True,
        )
    )
    metrics = dict(
        zip(("false_positive_rate", "false_negative_rate"), rates, strict=True)
    )
    live_rate_assertion(metrics, expected_counts)
    for field, expected_rate in metrics.items():
        corrupted = {**metrics, field: 0.0 if expected_rate is None else 1.0}
        with pytest.raises(AssertionError):
            live_rate_assertion(corrupted, expected_counts)


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

    if expected_f1 is None:
        # A zero denominator remains undefined, but a generated benchmark
        # cannot measure balanced accuracy from one class. Keep both contracts.
        assert vlm_eval._safe_ratio(0, 0) is None
        with pytest.raises(
            vlm_eval.VlmEvalError,
            match="at least one pass and one fail expected_label",
        ):
            benchmark(dataset=str(dataset), backend="stub", thresholds=(0.8,))
        return
    if all(label for label, _score in labels_and_scores):
        # A true negative supplies the required negative class without changing
        # the exact TP/FP/FN counts or the F1 rounding oracle under test.
        items.append(_fixture_item("negative-control", False, 0.0))
        dataset = _write_dataset(tmp_path, items)

    report = benchmark(dataset=str(dataset), backend="stub", thresholds=(0.8,))

    assert report.best_config.metrics.f1 == expected_f1


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("provider_success", [True, False])
def test_confusion_uses_serialized_score_not_provider_claim(
    tmp_path, monkeypatch, backend, provider_success
) -> None:
    """Exercise the combined transport, provider-claim and benchmark contracts."""
    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    dataset = _write_dataset(
        tmp_path,
        [
            {"id": "positive", "rollout": frame.name, "expected_label": True},
            {"id": "negative", "rollout": frame.name, "expected_label": False},
        ],
    )
    response = {
        "model": "MiniMaxAI/MiniMax-M3",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "success": provider_success,
                            "score": 0.79996,
                            "rationale": "fixture",
                        }
                    )
                },
            }
        ],
    }
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **kwargs: response
    )
    report = benchmark(
        dataset=str(dataset),
        backend=backend,
        models=("MiniMaxAI/MiniMax-M3",),
        thresholds=(0.8, 0.80001),
        use_fixture_scores=False,
    )
    output = tmp_path / "report.json"
    vlm_eval.write_benchmark_report(asdict(report), output_path=str(output))
    saved = json.loads(output.read_text())
    configs = {
        row["config"]["success_threshold"]: row for row in saved["ranked_configs"]
    }
    assert configs[0.8]["metrics"]["false_positive_item_ids"] == ["negative"]
    assert configs[0.8]["metrics"]["false_negative_item_ids"] == []
    assert configs[0.80001]["metrics"]["false_positive_item_ids"] == []
    assert configs[0.80001]["metrics"]["false_negative_item_ids"] == ["positive"]
    for threshold, config in configs.items():
        for case in config["results"]:
            assert case["score"] == 0.8
            assert case["predicted_label"] is (0.8 >= threshold)
            assert case["provider_success"] is provider_success
            assert case["provider_success_matches_score_gate"] is (
                provider_success == case["predicted_label"]
            )
            assert json.loads(case["evidence"]["provider"]["raw_response"]) == response
