from __future__ import annotations

import pytest

from npa.workbench.vlm_eval import VlmEvalError, _resolve_api_key, _resolve_endpoint_url


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
