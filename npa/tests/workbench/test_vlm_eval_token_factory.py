from __future__ import annotations

from dataclasses import asdict
import json

import pytest
from PIL import Image

from npa.workbench.vlm_eval import (
    VlmEvalError,
    _resolve_api_key,
    _resolve_endpoint_url,
    evaluate_vlm,
)


def test_api_backend_defaults_to_token_factory_served_vision_model(tmp_path) -> None:
    """The Token Factory API serves MiniMax-M3, not vlm_eval's self-hosted
    default (Qwen2-VL-7B, which 404s), so the api backend must pick the served
    model unless --model is overridden."""
    from npa.clients.token_factory import DEFAULT_VISION_MODEL

    result = evaluate_vlm(
        input_path="s3://ignored",
        output_path=str(tmp_path / "out.json"),
        backend="api",
        score=0.9,  # skips the real VLM call
    )
    assert result.model == DEFAULT_VISION_MODEL


def test_api_backend_defaults_to_token_factory_base_url(monkeypatch) -> None:
    for key in ("VLM_EVAL_API_BASE_URL", "OPENAI_BASE_URL", "NEBIUS_TOKEN_FACTORY_BASE_URL", "NEBIUS_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    url = _resolve_endpoint_url(backend="api", endpoint_url="")
    assert url == "https://api.tokenfactory.nebius.com/v1/"


def test_api_backend_honors_explicit_endpoint(monkeypatch) -> None:
    url = _resolve_endpoint_url(backend="api", endpoint_url="http://localhost:9000/v1")
    assert url == "http://localhost:9000/v1"


def test_api_backend_accepts_token_factory_key(monkeypatch) -> None:
    monkeypatch.delenv("VLM_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "tf-key")
    assert _resolve_api_key(backend="api", api_key_env="VLM_EVAL_API_KEY") == "tf-key"


def test_api_backend_requires_a_key(monkeypatch) -> None:
    for key in ("VLM_EVAL_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(VlmEvalError):
        _resolve_api_key(backend="api", api_key_env="VLM_EVAL_API_KEY")


@pytest.mark.parametrize(("model", "constrained"), [
    ("MiniMaxAI/MiniMax-M3", False), ("vendor/explicit-vision", True),
])
def test_api_judge_uses_model_specific_json_mode(monkeypatch, model, constrained) -> None:
    from npa.workbench import vlm_eval
    requests = []

    def post(**kwargs):
        requests.append(kwargs["request"])
        return _completion(model=model)

    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    result = vlm_eval._call_openai_compatible(
        backend="api", model=model, endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY", prompt="Return JSON", frames=[], timeout_s=120,
    )
    assert result.score == 0.9
    assert ("response_format" in requests[0]) is constrained
    if not constrained:
        assert requests[0]["chat_template_kwargs"] == {"thinking_mode": "disabled"}


def test_malformed_minimax_json_remains_an_error(monkeypatch) -> None:
    from npa.workbench import vlm_eval
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", lambda **kwargs: _completion(
        content='{"{"}success":true,"score":1}'
    ))
    with pytest.raises(VlmEvalError, match="JSON could not be parsed"):
        vlm_eval._call_openai_compatible(
            backend="api", model="MiniMaxAI/MiniMax-M3",
            endpoint_url="https://example.test/v1", api_key_env="TEST_KEY",
            prompt="Return JSON", frames=[], timeout_s=120,
        )


_VALID_CONTENT = '{"success":true,"score":0.9,"rationale":"target reached"}'


def _completion(*, content=_VALID_CONTENT, model="MiniMaxAI/MiniMax-M3", finish="stop"):
    return {
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def _call_completion(monkeypatch, completion, *, backend="api", model="MiniMaxAI/MiniMax-M3"):
    from npa.workbench import vlm_eval

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", lambda **kwargs: completion)
    return vlm_eval._call_openai_compatible(
        backend=backend, model=model, endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY", prompt="Return JSON", frames=[], timeout_s=120,
    )


@pytest.mark.parametrize("score", [7, -0.1, float("nan"), float("inf"), True, "0.9", None])
def test_api_judge_rejects_invalid_scores_before_result(monkeypatch, score) -> None:
    content = json.dumps({"success": True, "score": score, "rationale": "target reached"})
    with pytest.raises(VlmEvalError, match="finite number|non-finite number"):
        _call_completion(monkeypatch, _completion(content=content))


@pytest.mark.parametrize("content", [
    '{"success":"yes","score":0.9,"rationale":"target reached"}',
    '{"score":0.9,"rationale":"target reached"}',
    '{"success":true,"score":0.1,"score":0.9,"rationale":"target reached"}',
    '```json\n' + _VALID_CONTENT + '\n```',
    'Evaluation: ' + _VALID_CONTENT,
    _VALID_CONTENT + ' trailing explanation',
    '{"success":true,"score":0.9}',
    '{"success":true,"score":0.9,"rationale":null}',
    '{"success":true,"score":0.9,"rationale":"  "}',
    '{"success":true,"score":1e999,"rationale":"target reached"}',
    '[true, 0.9, "target reached"]',
    {"success": True, "score": 0.9, "rationale": "target reached"},
])
def test_api_judge_rejects_invalid_complete_contract(monkeypatch, content) -> None:
    with pytest.raises(VlmEvalError, match="Hosted VLM response"):
        _call_completion(monkeypatch, _completion(content=content))


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_api_judge_rejects_incomplete_output_even_when_json_valid(monkeypatch, finish) -> None:
    with pytest.raises(VlmEvalError, match="finish_reason=stop"):
        _call_completion(monkeypatch, _completion(finish=finish))


def test_api_judge_requires_completion_metadata(monkeypatch) -> None:
    completion = _completion()
    del completion["choices"][0]["finish_reason"]
    with pytest.raises(VlmEvalError, match="finish_reason=stop"):
        _call_completion(monkeypatch, completion)


@pytest.mark.parametrize("model", [None, "", "  ", 7])
def test_api_judge_requires_actual_model_identity(monkeypatch, model) -> None:
    with pytest.raises(VlmEvalError, match="identify the served model"):
        _call_completion(monkeypatch, _completion(model=model))


@pytest.mark.parametrize("model", ["nvidia/Nemotron-3_5-Lightning", "MiniMaxAI/MiniMax-M3"])
def test_api_judge_rejects_canonical_model_mismatch(monkeypatch, model) -> None:
    with pytest.raises(VlmEvalError, match="does not match"):
        _call_completion(monkeypatch, _completion(model="vendor/other-model"), model=model)


@pytest.mark.parametrize("score", [0, 0.74, 1])
def test_api_judge_preserves_valid_scores(monkeypatch, score) -> None:
    content = json.dumps({"success": False, "score": score, "rationale": "visible evidence"})
    result = _call_completion(monkeypatch, _completion(content=content))
    assert result.success is False
    assert result.score == score
    assert result.rationale == "visible evidence"


def test_self_hosted_judge_keeps_legacy_parsing_without_completion_metadata(monkeypatch) -> None:
    completion = {"choices": [{"message": {"content":
        '```json\n{"success":"yes","score":7,"rationale":"legacy"}\n```'}}]}
    result = _call_completion(monkeypatch, completion, backend="self-hosted")
    assert result.success is True
    assert result.score == 1.0
    assert result.served_model is None


def test_api_custom_alias_preserves_request_and_reports_actual_judged_model(
    monkeypatch, tmp_path,
) -> None:
    from npa.workbench import vlm_eval

    frame = tmp_path / "synthetic.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    requests = []

    def post(**kwargs):
        requests.append(kwargs)
        return _completion(model="vendor/served-vision")

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **kwargs: "test-key")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    result = evaluate_vlm(
        input_path=str(frame), output_path=str(tmp_path / "evaluation.json"),
        backend="api", model="vendor/explicit-alias",
        endpoint_url="https://example.test/v1", task="Judge the green diagram",
    )
    assert requests[0]["request"]["model"] == "vendor/explicit-alias"
    assert requests[0]["url"] == "https://example.test/v1/chat/completions"
    saved = json.loads(json.dumps(asdict(result)))
    assert saved["model"] == "vendor/explicit-alias"
    assert saved["served_model"] == "vendor/served-vision"
    assert saved["score"] == 0.9
    assert saved["passed"] is True
    assert saved["frame_count"] == 1
