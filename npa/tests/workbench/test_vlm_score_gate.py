"""Check serialized score gates and provider claims at evaluation boundaries."""

from dataclasses import asdict
import json

import pytest
from PIL import Image

from npa.workbench import vlm_eval


@pytest.mark.parametrize("backend", ["stub", "api", "self-hosted"])
@pytest.mark.parametrize(
    ("score", "threshold", "expected_score", "passed"),
    [(0.79996, 0.8, 0.8, True), (0.80004, 0.80001, 0.8, False)],
)
def test_override_gate_uses_serialized_score(
    tmp_path, backend, score, threshold, expected_score, passed
):
    result = vlm_eval.evaluate_vlm(
        input_path="unused-rollout",
        output_path=str(tmp_path / "result.json"),
        backend=backend,
        score=score,
        success_threshold=threshold,
    )

    assert result.score == expected_score
    assert result.passed is passed
    assert result.status == ("passed" if passed else "needs_iteration")
    assert result.evidence is None
    assert result.provider_success is None
    assert result.provider_success_matches_score_gate is None


def _record_response(monkeypatch, tmp_path, provider_success, score):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    verdict = {"success": provider_success, "score": score, "rationale": "test verdict"}
    response = {
        "model": "MiniMaxAI/MiniMax-M3",
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}
        ],
    }
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_kwargs: "test-key")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **_kwargs: response
    )
    return frame, response


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("provider_success", [True, False])
@pytest.mark.parametrize(
    ("score", "threshold", "passed"),
    [
        (0.0, 0.8, False),
        (0.9, 0.8, True),
        (0.8, 0.8, True),
        (0.79996, 0.8, True),
        (0.80004, 0.80001, False),
    ],
)
def test_saved_provider_claim_does_not_replace_score_gate(
    monkeypatch, tmp_path, backend, provider_success, score, threshold, passed
):
    frame, response = _record_response(monkeypatch, tmp_path, provider_success, score)
    target = tmp_path / "evaluation.json"
    result = vlm_eval.evaluate_vlm(
        input_path=str(frame),
        output_path=str(target),
        backend=backend,
        model="MiniMaxAI/MiniMax-M3",
        success_threshold=threshold,
    )
    vlm_eval.write_result(asdict(result), result_uri=result.result_uri)
    saved = json.loads(target.read_text())

    assert saved["passed"] is passed
    assert saved["passed"] is (saved["score"] >= saved["success_threshold"])
    assert saved["status"] == ("passed" if passed else "needs_iteration")
    assert saved["provider_success"] is provider_success
    assert saved["provider_success_matches_score_gate"] is (provider_success == passed)
    assert json.loads(saved["evidence"]["provider"]["raw_response"]) == response


def _benchmark_dataset(tmp_path, rollout):
    dataset = tmp_path / "benchmark.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "boundary",
                    "rollout": str(rollout),
                    "expected_label": True,
                    "fixture_score": 0.79996,
                },
                {
                    "id": "negative-control",
                    "rollout": str(rollout),
                    "expected_label": False,
                    "fixture_score": 0.0,
                },
            ]
        )
    )
    return dataset


@pytest.mark.parametrize("backend", ["stub", "api", "self-hosted"])
def test_fixture_benchmark_keeps_gate_without_provider_claims(
    monkeypatch, tmp_path, backend
):
    dataset = _benchmark_dataset(tmp_path, "unused-rollout")
    monkeypatch.setattr(
        vlm_eval,
        "_call_openai_compatible",
        lambda **_kwargs: pytest.fail("fixture scoring must not call a provider"),
    )
    report = vlm_eval.benchmark_vlm_eval(
        dataset=str(dataset),
        backend=backend,
        thresholds=[0.8],
        use_fixture_scores=True,
    )
    case = report.best_config.results[0]

    assert case.score == 0.8
    assert case.passed is True
    assert case.predicted_label is True
    assert case.status == "passed"
    assert case.score_source == "fixture"
    assert case.evidence is None
    assert case.provider_success is None
    assert case.provider_success_matches_score_gate is None
    control = report.best_config.results[1]
    assert control.expected_label is False
    assert control.score == 0.0
    assert control.passed is False
    assert control.predicted_label is False
    assert control.evidence is None
    assert control.provider_success is None


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("provider_success", [True, False])
@pytest.mark.parametrize(("score", "passed"), [(0.0, False), (0.9, True)])
def test_benchmark_preserves_provider_claim_and_uses_score_for_prediction(
    monkeypatch, tmp_path, backend, provider_success, score, passed
):
    frame, response = _record_response(monkeypatch, tmp_path, provider_success, score)
    dataset = _benchmark_dataset(tmp_path, frame)
    report = vlm_eval.benchmark_vlm_eval(
        dataset=str(dataset),
        backend=backend,
        thresholds=[0.8],
        models=["MiniMaxAI/MiniMax-M3"],
    )
    case = report.best_config.results[0]

    assert case.passed is passed
    assert case.predicted_label is passed
    assert case.status == ("passed" if passed else "needs_iteration")
    assert case.provider_success is provider_success
    assert case.provider_success_matches_score_gate is (provider_success == passed)
    assert case.evidence is not None
    assert json.loads(case.evidence.provider.raw_response) == response
    control = report.best_config.results[1]
    assert control.expected_label is False
    assert control.passed is passed
    assert control.predicted_label is passed
    assert control.provider_success is provider_success
    assert control.evidence is not None
    assert json.loads(control.evidence.provider.raw_response) == response
