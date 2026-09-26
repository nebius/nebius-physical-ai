"""Prove the shipped Jev router is reached by the rendered agent chat path."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from .test_agent_backend_render import (
    _clear_rendered_agent_backend_modules,
    _import_rendered_backend,
)
from .test_jev_model_router import FAST, REASONING, _response


@pytest.fixture
def backend(monkeypatch, tmp_path):
    _clear_rendered_agent_backend_modules()
    monkeypatch.setattr(
        "npa.cli.agent._stage_agent_npa_source", lambda *args, **kwargs: None
    )
    module = _import_rendered_backend(
        monkeypatch, tmp_path, module_name="jev_test_backend"
    )
    module.STATE_PATH = tmp_path / "state.json"
    module._STATE_STORE = None
    monkeypatch.setenv("NPA_AGENT_MODEL_ROUTER", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    monkeypatch.setattr(module, "_available_llm_models", lambda: [FAST, REASONING])
    monkeypatch.setattr(module, "_configured_llm_providers", lambda: ["token_factory"])
    monkeypatch.setattr(module, "LLM_MODELS_ENV", "")
    yield module
    _clear_rendered_agent_backend_modules()


def test_chat_exposes_actual_route_and_generation_cache_usage(backend, monkeypatch):
    router = importlib.import_module("agent_backend.model_router")
    routed = []
    generated = []

    def classify(_post, body, _key, _timeout):
        routed.append(body)
        return _response(REASONING)

    def generate(**kwargs):
        generated.append(kwargs)
        return {
            "choices": [{"message": {"content": "A synthetic answer."}}],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 5,
                "total_tokens": 205,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        }

    monkeypatch.setattr(router, "_classify", classify)
    monkeypatch.setattr(backend, "_provider_chat", generate)
    monkeypatch.setattr(backend, "_semantic_route", lambda text: {"mode": "none"})
    monkeypatch.setattr(backend, "_maybe_retrieval_grounded", lambda text: None)
    result = (
        TestClient(backend.app)
        .post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Write a limerick about violet clouds."}
                ]
            },
        )
        .json()
    )
    assert result["model"] == REASONING
    assert result["model_routing"]["status"] == "accepted"
    assert result["model_routing"]["served_model"] == REASONING
    assert result["model_routing"]["usage"]["input_tokens"] == 200
    assert result["usage"]["cached_tokens"] == 100
    assert len(routed) == len(generated) == 1
    assert routed[0]["state"] == {"request": "Write a limerick about violet clouds."}
    assert generated[0]["messages"][0]["role"] == "system"
    assert generated[0]["extra"] == {}


def test_grounded_and_explicit_and_vision_requests_never_classify(backend, monkeypatch):
    router = importlib.import_module("agent_backend.model_router")

    def forbidden(*args, **kwargs):
        pytest.fail("Jev must not run on this path")

    monkeypatch.setattr(router, "_classify", forbidden)
    monkeypatch.setattr(backend, "_provider_chat", lambda **kwargs: {"choices": []})
    result = (
        TestClient(backend.app)
        .post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "show me the workbench tools catalog"}
                ]
            },
        )
        .json()
    )
    assert result["grounded"] is True
    for settings in ({"requested_model": FAST}, {"tier": "vision"}):
        data, _, _ = backend._chat_with_resilience(
            messages=[{"role": "user", "content": "synthetic"}],
            use_model_router=True,
            **settings,
        )
        assert "model_routing" not in data


def test_selected_model_failure_retains_existing_generation_fallback(
    backend, monkeypatch
):
    router = importlib.import_module("agent_backend.model_router")
    monkeypatch.setattr(router, "_classify", lambda *args: _response(REASONING))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == REASONING:
            raise RuntimeError("synthetic outage")
        return {"choices": []}

    monkeypatch.setattr(backend, "_provider_chat", generate)
    data, _, actual = backend._chat_with_resilience(
        messages=[{"role": "user", "content": "synthetic"}],
        use_model_router=True,
    )
    assert actual == FAST
    assert calls == [REASONING, FAST]
    assert data["model_routing"]["selected_model"] == REASONING
    assert data["model_routing"]["served_model"] == FAST


@pytest.mark.parametrize("available", [[], ["other/model"], [FAST]])
def test_unobserved_or_single_eligible_model_never_classifies(
    backend, monkeypatch, available
):
    router = importlib.import_module("agent_backend.model_router")

    def forbidden(*args, **kwargs):
        pytest.fail("Jev must not choose models outside observed availability")

    monkeypatch.setattr(router, "_classify", forbidden)
    monkeypatch.setattr(backend, "_available_llm_models", lambda: available)
    monkeypatch.setattr(backend, "_provider_chat", lambda **kwargs: {"choices": []})
    data, _, _ = backend._chat_with_resilience(
        messages=[{"role": "user", "content": "synthetic"}],
        use_model_router=True,
    )
    assert data.get("model_routing", {}).get("status") != "accepted"


def test_explicit_operator_allowlist_limits_router_candidates(backend, monkeypatch):
    router = importlib.import_module("agent_backend.model_router")

    def forbidden(*args, **kwargs):
        pytest.fail("Jev must respect the operator's single-model allowlist")

    monkeypatch.setattr(router, "_classify", forbidden)
    monkeypatch.setattr(backend, "LLM_MODELS_ENV", FAST)
    monkeypatch.setattr(backend, "_provider_chat", lambda **kwargs: {"choices": []})
    data, _, model = backend._chat_with_resilience(
        messages=[{"role": "user", "content": "synthetic"}],
        use_model_router=True,
    )
    assert model == FAST
    assert data["model_routing"]["reason"] == "fewer_than_two_candidates"
