from __future__ import annotations

import pytest

from npa.workbench.vlm_eval import (
    VlmEvalError,
    _resolve_api_key,
    _resolve_endpoint_url,
    evaluate_vlm,
)


def test_api_backend_defaults_to_token_factory_served_vision_model(tmp_path) -> None:
    """The Token Factory API does not serve vlm_eval's self-hosted default
    (Qwen2-VL-7B 404s there), so the api backend must pick the hosted vision
    default unless --model is overridden."""
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


def test_api_backend_model_env_override_repoints_the_default(tmp_path, monkeypatch) -> None:
    """NPA_VLM_API_MODEL repoints the api-backend default without changing a
    workflow's argv, so a running workflow stays resumable when a model leaves
    the public endpoint. An explicit --model still wins."""
    from npa.clients.token_factory import DEFAULT_VISION_MODEL, VISION_MODEL_ENV

    monkeypatch.setenv(VISION_MODEL_ENV, "openbmb/MiniCPM-V-4_5")
    result = evaluate_vlm(
        input_path="s3://ignored",
        output_path=str(tmp_path / "out.json"),
        backend="api",
        score=0.9,
    )
    assert result.model == "openbmb/MiniCPM-V-4_5"
    pinned_default = evaluate_vlm(
        input_path="s3://ignored",
        output_path=str(tmp_path / "out3.json"),
        backend="api",
        model=DEFAULT_VISION_MODEL,
        score=0.9,
    )
    assert pinned_default.model == "openbmb/MiniCPM-V-4_5"  # same rule as caption
    explicit = evaluate_vlm(
        input_path="s3://ignored",
        output_path=str(tmp_path / "out2.json"),
        backend="api",
        model="some/explicit-model",
        score=0.9,
    )
    assert explicit.model == "some/explicit-model"
