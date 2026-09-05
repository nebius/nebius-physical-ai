"""Model defaults must come from observed chat availability, without paid calls."""

from __future__ import annotations

import json
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from npa.cli import agent_routing as routing

from .test_agent_backend_render import (
    _clear_rendered_agent_backend_modules,
    _import_rendered_backend,
)


def catalog(*models):
    return {"data": list(models)}


def rich(model, modality="text->text", **extra):
    return {"id": model, "architecture": {"modality": modality}, **extra}


def test_default_prefers_available_configured_model_then_ordered_fallback():
    observed = routing.parse_model_catalog(catalog(
        rich(routing.CHEAP_MODEL), rich(routing.REASONING_MODEL), rich(routing.STANDARD_MODEL)
    ))
    selected = routing.model_availability(
        routing.REASONING_MODEL, [routing.CHEAP_MODEL, routing.STANDARD_MODEL], observed
    )
    assert selected["default_model"] == routing.REASONING_MODEL
    observed = routing.parse_model_catalog(catalog(rich(routing.STANDARD_MODEL), rich(routing.CHEAP_MODEL)))
    selected = routing.model_availability(
        routing.REASONING_MODEL, [routing.CHEAP_MODEL, routing.STANDARD_MODEL], observed
    )
    assert selected["default_model"] == routing.CHEAP_MODEL
    assert routing.REASONING_MODEL not in selected["models"]
    assert selected["availability_status"] == "available"


def test_allowlist_limits_default_but_preserves_full_provider_catalog():
    observed = routing.parse_model_catalog(catalog(
        rich(routing.REASONING_MODEL), rich(routing.CHEAP_MODEL), rich("BAAI/bge-en-icl", "text->embedding")
    ))
    selected = routing.model_availability(
        routing.REASONING_MODEL, [routing.CHEAP_MODEL], observed,
        allowed_models=[routing.CHEAP_MODEL],
    )
    assert selected["default_model"] == routing.CHEAP_MODEL
    assert set(selected["models"]) == {routing.REASONING_MODEL, routing.CHEAP_MODEL, "BAAI/bge-en-icl"}
    selected = routing.model_availability(
        routing.REASONING_MODEL, [routing.STANDARD_MODEL], observed,
        allowed_models=[routing.STANDARD_MODEL],
    )
    assert selected["default_model"] is None
    assert selected["availability_status"] == "unavailable"


@pytest.mark.parametrize("modality", ["text->embedding", "text->image", "text->audio"])
def test_explicit_nonchat_metadata_cannot_be_overridden_by_config_or_known_name(modality):
    observed = routing.parse_model_catalog(catalog(rich(routing.CHEAP_MODEL, modality)))
    result = routing.model_availability(routing.CHEAP_MODEL, [routing.CHEAP_MODEL], observed)
    assert result["default"] is None
    assert result["availability_status"] == "unavailable"
    assert result["models"] == [routing.CHEAP_MODEL]


@pytest.mark.parametrize("modality", ["text->text", "text+image->text", "text + image -> text"])
def test_verbose_chat_modality_supports_new_provider_models(modality):
    observed = routing.parse_model_catalog(catalog(rich("example/new-chat", modality)))
    assert routing.model_availability("example/new-chat", [], observed)["default"] == "example/new-chat"


def test_basic_catalog_supports_known_chat_but_never_guesses_embedding_or_custom_capability():
    observed = routing.parse_model_catalog(catalog(
        {"id": routing.STANDARD_MODEL}, {"id": "BAAI/bge-en-icl"}, {"id": "example/unknown"}
    ))
    assert observed["chat_models"] == [routing.STANDARD_MODEL]
    for model in ("BAAI/bge-en-icl", "example/unknown"):
        result = routing.model_availability(model, [], observed)
        assert result["default"] is None
        assert result["availability_status"] == "unknown"
    assert routing.model_availability(routing.STANDARD_MODEL, [], observed)["default"] == routing.STANDARD_MODEL


@pytest.mark.parametrize("payload", [None, [], {}, {"data": None}, {"error": "private provider error", "data": []},
                                   {"data": [None]}, {"data": [{"id": 1}]}, {"data": [{"id": " padded "}]},
                                   {"data": [{"id": "example/chat", "architecture": None}]},
                                   {"data": [{"id": "example/chat", "architecture": {}}]},
                                   {"data": [rich("example/chat"), rich("example/chat", "text->embedding")]}])
def test_failed_or_malformed_catalog_is_unknown_and_does_not_invent_configured_models(payload):
    observed = routing.parse_model_catalog(payload)
    assert observed is None
    result = routing.model_availability(routing.CHEAP_MODEL, [routing.CHEAP_MODEL], observed)
    assert result["default"] is None
    assert result["models"] == []
    assert result["catalog_status"] == result["availability_status"] == "unknown"


def test_empty_successful_catalog_is_unavailable_and_inactive_model_is_not_selected():
    for payload in (catalog(), catalog(rich(routing.CHEAP_MODEL, status="deleted"))):
        result = routing.model_availability(routing.CHEAP_MODEL, [], routing.parse_model_catalog(payload))
        assert result["default"] is None
        assert result["catalog_status"] == "available"
        assert result["availability_status"] == "unavailable"


@pytest.fixture
def backend(monkeypatch, tmp_path):
    _clear_rendered_agent_backend_modules()
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name="npa_models_test_backend")
    monkeypatch.setattr(module, "LLM_MODEL", routing.REASONING_MODEL)
    monkeypatch.setattr(module, "LLM_MODELS_ENV", "")
    monkeypatch.setattr(module, "DEFAULT_LLM_MODELS", [routing.CHEAP_MODEL, routing.STANDARD_MODEL, routing.REASONING_MODEL])
    monkeypatch.setattr(module, "_provider_api_key", lambda _provider: "synthetic-test-value")
    module.STATE_PATH = tmp_path / "state.json"
    module._STATE_STORE = None
    module.PRELOAD_STOCK_DEMO = False
    monkeypatch.setattr(module, "_agent_s3_client_optional", lambda: (None, {"bucket": ""}))
    monkeypatch.setattr(module, "_agent_k8s_backends", lambda: {})
    monkeypatch.setattr(module, "_rerun_ready_state", lambda **_kwargs: {"ready": False})
    yield module
    sys.modules.pop("npa_models_test_backend", None)
    _clear_rendered_agent_backend_modules()


def transport(monkeypatch, module, payload):
    calls = []

    def get(url, **kwargs):
        assert kwargs["params"] == {"verbose": "true"}
        calls.append(url)
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(module.httpx, "get", get)
    return calls


def test_rendered_models_and_session_agree_on_fallback_and_share_observed_cache(backend, monkeypatch):
    calls = transport(monkeypatch, backend, catalog(rich("BAAI/bge-en-icl", "text->embedding"), rich(routing.CHEAP_MODEL)))
    client = TestClient(backend.app)
    models = client.get("/models").json()
    session = client.get("/session").json()
    assert models["ok"] is True
    assert models["default_model"] == models["default"] == routing.CHEAP_MODEL
    assert session["llm"]["model"] == routing.CHEAP_MODEL
    assert {k: session["llm"][k] for k in models if k != "ok"} == {k: v for k, v in models.items() if k != "ok"}
    assert len(calls) == 1
    assert models["models"] == [routing.CHEAP_MODEL, "BAAI/bge-en-icl"]
    assert routing.REASONING_MODEL not in json.dumps(models)
    models["models"].clear()
    assert backend.models()["models"]  # Callers cannot mutate cached provider facts.
    assert client.get("/models?refresh=true").json()["default"] == routing.CHEAP_MODEL
    assert len(calls) == 2
    assert "BAAI/bge-en-icl" in backend._fetch_token_factory_models()


@pytest.mark.parametrize("failure", ["http", "json", "missing-key", "empty"])
def test_rendered_discovery_failure_or_empty_catalog_has_honest_state_without_error_leak(backend, monkeypatch, failure):
    if failure == "missing-key":
        monkeypatch.setattr(backend, "_provider_api_key", lambda _provider: "")
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if failure == "empty":
            return httpx.Response(200, json=catalog(), request=httpx.Request("GET", url))
        if failure == "json":
            return httpx.Response(200, content=b"private non-json body", request=httpx.Request("GET", url))
        raise httpx.ConnectError("private transport diagnostic", request=httpx.Request("GET", url))

    monkeypatch.setattr(backend.httpx, "get", get)
    client = TestClient(backend.app)
    result = client.get("/models").json()
    session = client.get("/session").json()
    assert result["ok"] is True  # Grounded/session operations remain available.
    assert result["models"] == []
    assert result["default_model"] is None
    assert session["llm"]["model"] is None
    assert result["availability_status"] == ("unavailable" if failure == "empty" else "unknown")
    assert "private" not in json.dumps(result)
    assert len(calls) == (0 if failure == "missing-key" else 1)


def test_rendered_allowlist_default_and_explicit_model_override_preserve_routing(backend, monkeypatch):
    monkeypatch.setattr(backend, "LLM_MODELS_ENV", routing.CHEAP_MODEL)
    transport(monkeypatch, backend, catalog(rich(routing.REASONING_MODEL), rich(routing.CHEAP_MODEL), rich("example/explicit")))
    assert backend.models()["default"] == routing.CHEAP_MODEL
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(backend, "_provider_chat", chat)
    result, _, selected = backend._chat_with_resilience(messages=[], requested_model="example/explicit")
    assert selected == "example/explicit"
    assert len(calls) == 1 and calls[0]["model"] == "example/explicit"
    assert result["choices"][0]["message"]["content"] == "ok"


def test_rendered_catalog_cache_is_bound_to_provider_key_and_refresh_recovers_unknown(backend, monkeypatch):
    calls = transport(monkeypatch, backend, {})
    assert backend.models()["availability_status"] == "unknown"
    assert backend.models()["availability_status"] == "unknown"
    assert len(calls) == 1
    calls = transport(monkeypatch, backend, catalog(rich(routing.CHEAP_MODEL)))
    assert backend.models()["availability_status"] == "unknown"
    assert not calls
    assert backend.models(refresh=True)["default"] == routing.CHEAP_MODEL
    assert len(calls) == 1
    monkeypatch.setattr(backend, "_provider_api_key", lambda _provider: "rotated-synthetic-value")
    calls = transport(monkeypatch, backend, catalog(rich(routing.STANDARD_MODEL)))
    result = backend.models()
    assert result["default"] == routing.STANDARD_MODEL
    assert len(calls) == 1
    assert "rotated-synthetic-value" not in repr(backend._MODELS_CACHE)
    assert routing.CHEAP_MODEL not in result["models"]
