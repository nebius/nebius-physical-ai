from __future__ import annotations

import pytest

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
        return {"choices": [{"message": {"content":
            '{"success":true,"score":0.9,"rationale":"target reached"}'}}]}

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
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", lambda **kwargs: {
        "choices": [{"message": {"content": '{"{"}success":true,"score":1}'}}]
    })
    with pytest.raises(VlmEvalError, match="JSON could not be parsed"):
        vlm_eval._call_openai_compatible(
            backend="api", model="MiniMaxAI/MiniMax-M3",
            endpoint_url="https://example.test/v1", api_key_env="TEST_KEY",
            prompt="Return JSON", frames=[], timeout_s=120,
        )
