from __future__ import annotations

import json

import httpx
import pytest

from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    TokenFactoryClient,
    TokenFactoryError,
    resolve_config,
    split_reasoning,
)


def _message_client(message: dict) -> TokenFactoryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": message}]})

    config = resolve_config(api_key="test-key", environ={})
    return TokenFactoryClient(config, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


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


# --- reasoning-model response shapes -----------------------------------------


def test_split_reasoning_strips_inline_think_block() -> None:
    # Cosmos 3 shape: reasoning trace inline as a leading <think>...</think>.
    visible, reasoning = split_reasoning(
        {"content": "<think>\nweigh the options\n</think>\n1. approach 2. grasp"}
    )
    assert visible == "1. approach 2. grasp"
    assert reasoning == "weigh the options"


def test_split_reasoning_uses_reasoning_field_when_content_null() -> None:
    # Kimi/GLM shape: content null, trace in a separate reasoning field.
    visible, reasoning = split_reasoning({"content": None, "reasoning": "thinking hard"})
    assert visible == ""
    assert reasoning == "thinking hard"


def test_split_reasoning_truncated_think_has_no_visible_text() -> None:
    # finish_reason=length mid-think: opening tag, no close, no answer.
    visible, reasoning = split_reasoning({"content": "<think>still reasoning when it ran"})
    assert visible == ""
    assert reasoning == "still reasoning when it ran"


def test_split_reasoning_plain_content_unchanged() -> None:
    visible, reasoning = split_reasoning({"content": "just the answer"})
    assert visible == "just the answer"
    assert reasoning is None


def test_chat_completion_message_returns_visible_and_reasoning() -> None:
    client = _message_client({"content": "<think>plan</think>do it", "reasoning": None})
    visible, reasoning = client.chat_completion_message(
        model="nvidia/Cosmos3-Super-Reasoner", messages=[{"role": "user", "content": "x"}]
    )
    assert visible == "do it"
    assert reasoning == "plan"


def test_chat_completion_text_strips_inline_think() -> None:
    client = _message_client({"content": "<think>plan</think>do it"})
    text = client.chat_completion_text(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert "<think>" not in text
    assert text == "do it"


def test_chat_completion_text_raises_on_reasoning_only_response() -> None:
    # Regression: str(None) used to return the literal string "None".
    client = _message_client({"content": None, "reasoning": "all thinking, no answer"})
    with pytest.raises(TokenFactoryError) as exc:
        client.chat_completion_text(model="m", messages=[{"role": "user", "content": "x"}])
    assert "reasoning-only" in str(exc.value)
