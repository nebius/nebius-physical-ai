from __future__ import annotations

import json

import httpx
import pytest

from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    TokenFactoryClient,
    TokenFactoryError,
    resolve_config,
)


def _client(handler) -> TokenFactoryClient:
    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_resolve_config_defaults_to_token_factory_base_url() -> None:
    config = resolve_config(api_key="abc", environ={})
    assert config.base_url == DEFAULT_BASE_URL
    assert config.chat_completions_url == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert config.models_url == "https://api.tokenfactory.nebius.com/v1/models"


def test_resolve_config_reads_nebius_api_key_from_env() -> None:
    config = resolve_config(environ={"NEBIUS_API_KEY": "env-key"})
    assert config.api_key == "env-key"


def test_resolve_config_base_url_override() -> None:
    config = resolve_config(
        api_key="abc",
        environ={"NEBIUS_TOKEN_FACTORY_BASE_URL": "https://example.com/v1"},
    )
    assert config.base_url == "https://example.com/v1"
    assert config.chat_completions_url == "https://example.com/v1/chat/completions"


def test_resolve_config_requires_api_key() -> None:
    with pytest.raises(TokenFactoryError):
        resolve_config(environ={})


def test_resolve_config_allows_missing_key_when_not_required() -> None:
    config = resolve_config(environ={}, require_api_key=False)
    assert config.api_key == ""


def test_chat_completion_text_sends_bearer_and_parses_content() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "a caption"}}]},
        )

    client = _client(handler)
    text = client.chat_completion_text(
        model="some-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
    )

    assert text == "a caption"
    assert seen["auth"] == "Bearer test-key"
    assert seen["url"] == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["max_tokens"] == 128


def test_chat_completion_missing_content_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _client(handler)
    with pytest.raises(TokenFactoryError):
        client.chat_completion_text(model="m", messages=[{"role": "user", "content": "x"}])


def test_chat_completion_http_error_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = _client(handler)
    with pytest.raises(TokenFactoryError) as exc:
        client.chat_completion(model="m", messages=[{"role": "user", "content": "x"}])
    assert "429" in str(exc.value)


def test_list_models_returns_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.tokenfactory.nebius.com/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "model-a"}, {"id": "model-b"}, {}]},
        )

    client = _client(handler)
    assert client.list_models() == ["model-a", "model-b"]


def test_chat_completion_requires_model_and_messages() -> None:
    client = _client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(TokenFactoryError):
        client.chat_completion(model="", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(TokenFactoryError):
        client.chat_completion(model="m", messages=[])
