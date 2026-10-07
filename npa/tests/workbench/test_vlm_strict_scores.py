"""Prove self-hosted verdict fields fail before result construction or repair."""

import json
import math

import pytest
import httpx
from PIL import Image
from typer.testing import CliRunner

from npa.workbench import vlm_eval
from npa.cli.main import app


@pytest.mark.parametrize(
    "score", [99, "99", True, -0.1, 1.1, float("nan"), float("inf"), None, [], {}]
)
def test_self_hosted_invalid_scores_fail_without_repair(score):
    payload = json.dumps({"success": False, "score": score, "rationale": "failure"})
    with pytest.raises(
        vlm_eval.VlmEvalError, match=r"score must be a finite number in \[0, 1\]"
    ):
        vlm_eval.parse_structured_response(payload)


@pytest.mark.parametrize("score", [0, 0.0, 0.74, 1, 1.0])
def test_self_hosted_valid_scores_are_preserved(score):
    result = vlm_eval.parse_structured_response(
        json.dumps({"success": False, "score": score, "rationale": "visible"})
    )
    assert result.score == score
    assert result.success is False
    assert result.provider_success is False


@pytest.mark.parametrize("rationale", [None, "", " ", 7, [], {}])
def test_self_hosted_rationale_requires_nonempty_string(rationale):
    payload = json.dumps({"success": False, "score": 0.5, "rationale": rationale})
    with pytest.raises(
        vlm_eval.VlmEvalError, match="rationale must be a nonempty string"
    ):
        vlm_eval.parse_structured_response(payload)


def test_validated_result_does_not_reach_legacy_clamp(monkeypatch):
    def fail_on_repair(_value):
        pytest.fail("validated model score reached legacy clamp")

    monkeypatch.setattr(vlm_eval, "_clamp_score", fail_on_repair)
    result = vlm_eval._result_from_structured(
        backend="self-hosted",
        input_path="rollout",
        output_path="result",
        task="task",
        model="model",
        success_threshold=0.8,
        frame_selection="keyframes",
        frame_count=1,
        rubric=vlm_eval.DEFAULT_RUBRIC,
        structured=vlm_eval.VlmStructuredResponse(False, 0.74268, "visible"),
        provider_call_made=False,
    )
    assert result.score == 0.7427
    assert result.passed is False


@pytest.mark.parametrize("score", [99, "99", True])
def test_invalid_score_never_constructs_result(monkeypatch, tmp_path, score):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "green").save(frame)
    completion = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"success": False, "score": score, "rationale": "failure"}
                    )
                },
            }
        ],
    }
    calls = []

    def transport(**_kwargs):
        calls.append(True)
        return completion

    def fail_on_result(**_kwargs):
        pytest.fail("invalid model score reached result construction")

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", transport)
    monkeypatch.setattr(vlm_eval, "_result_from_structured", fail_on_result)
    with pytest.raises(vlm_eval.VlmEvalError, match="score must be a finite number"):
        vlm_eval.evaluate_vlm(input_path=str(frame), output_path=str(tmp_path / "out"))
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize(
    "score", [99, "99", True, -0.1, 1.1, float("nan"), float("inf"), None, [], {}]
)
def test_invalid_score_cli_rejects_before_artifact_publication(
    monkeypatch, tmp_path, backend, score
):
    source = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "green").save(source)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs["request"])
        return {
            "model": kwargs["request"]["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "success": False,
                                "score": score,
                                "rationale": "synthetic invalid-score control",
                            }
                        )
                    },
                }
            ],
        }

    def forbidden_network(*_, **__):
        pytest.fail("Hermetic invalid-score control attempted networking")

    monkeypatch.setattr(httpx.Client, "send", forbidden_network)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", completion)
    output = tmp_path / "out"
    invocation = CliRunner().invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            str(source),
            "--output-path",
            str(output),
            "--backend",
            backend,
            "--model",
            "MiniMaxAI/MiniMax-M3",
            "--endpoint-url",
            "https://example.test/v1",
            "--output",
            "json",
        ],
    )
    assert invocation.exit_code == 1
    expected = (
        "JSON contains a non-finite number"
        if backend == "api" and isinstance(score, float) and not math.isfinite(score)
        else "score must be a finite number"
    )
    assert expected in invocation.output
    assert len(calls) == 1
    assert not output.exists()
