"""Exercise Jev transport, abstention, model policy, and private configuration."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from npa.agent_backend import model_router as router
from npa.cli import agent_llm_config, agent_routing

FAST, REASONING = tuple(router.MODEL_CRITERIA)


def _response(choice=REASONING, confidence=0.9):
    probabilities = {FAST: 0.02, REASONING: 0.97, "none": 0.01}
    if choice != REASONING:
        probabilities[choice], probabilities[REASONING] = 0.97, probabilities[choice]
    return {
        "model": router.JEV_MODEL,
        "answers": {
            "model": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 200, "output_tokens": 8},
    }


def test_real_http_serialization_and_response_validation():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        ladder, decision = router.route_generation_ladder(
            "Prove this invariant.",
            [FAST, REASONING],
            enabled=True,
            api_key="synthetic-key",
            post=client.post,
        )
    assert ladder == [REASONING, FAST]
    assert decision["selected_model"] == REASONING
    assert decision["status"] == "accepted"
    assert decision["usage"] == {"input_tokens": 200, "output_tokens": 8}
    assert decision["latency_seconds"] >= 0
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == router.JEV_ENDPOINT
    assert requests[0].headers["authorization"] == "Bearer synthetic-key"
    assert body["state"] == {"request": "Prove this invariant."}
    assert set(body["questions"]["model"]["criteria"]) == {FAST, REASONING, "none"}
    assert "synthetic-key" not in json.dumps(decision)


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": False},
        {"requested_model": FAST},
        {"tier": "vision"},
        {"ladder": [FAST]},
        {"api_key": ""},
    ],
)
def test_bypass_never_calls_classifier(settings):
    arguments = {
        "enabled": True,
        "api_key": "synthetic-key",
        "ladder": [FAST, REASONING],
    }
    arguments.update(settings)

    def forbidden(*args, **kwargs):
        pytest.fail("Classifier called for a bypassed request")

    ladder, _ = router.route_generation_ladder("text", post=forbidden, **arguments)
    assert ladder == arguments["ladder"]


@pytest.mark.parametrize("choice,confidence", [("none", 0.99), (REASONING, 0.79)])
def test_abstention_preserves_baseline(choice, confidence):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_response(choice, confidence))
        )
    ) as client:
        ladder, decision = router.route_generation_ladder(
            "ambiguous",
            [FAST, REASONING],
            enabled=True,
            api_key="synthetic-key",
            post=client.post,
        )
    assert ladder == [FAST, REASONING]
    assert decision["status"] == "abstained"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(model="unexpected/model"),
        lambda value: value.update(answers={}),
        lambda value: value["answers"]["model"].update(type="noul"),
        lambda value: value["answers"]["model"].update(choice="unapproved/model"),
        lambda value: value["answers"]["model"].update(choice=[]),
        lambda value: value["answers"]["model"].update(confidence=True),
        lambda value: value["answers"]["model"].update(confidence=-1),
        lambda value: value["answers"]["model"].update(confidence=float("nan")),
        lambda value: value["answers"]["model"]["probabilities"].pop("none"),
        lambda value: value["answers"]["model"]["probabilities"].update(none=2),
        lambda value: value["answers"]["model"].update(choice=FAST),
    ],
)
def test_invalid_answers_do_not_select_a_model(mutation):
    payload = copy.deepcopy(_response())
    mutation(payload)
    # Inject decoded data so malformed non-finite JSON is validated too.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(router, "_classify", lambda *args: payload)
        ladder, decision = router.route_generation_ladder(
            "text", [FAST, REASONING], enabled=True, api_key="synthetic-key"
        )
    assert ladder == [FAST, REASONING]
    assert decision["reason"] == "invalid_response"
    assert decision["selected_model"] is None


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_provider_failure_is_sanitized_and_does_not_change_baseline(status):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, text="private-provider-diagnostic")
        )
    ) as client:
        ladder, decision = router.route_generation_ladder(
            "text",
            [FAST, REASONING],
            enabled=True,
            api_key="synthetic-key",
            post=client.post,
        )
    assert ladder == [FAST, REASONING]
    assert decision["http_status"] == status
    assert "private-provider-diagnostic" not in json.dumps(decision)


def test_transport_error_and_malformed_json_preserve_baseline():
    for response in (
        httpx.ConnectError("private detail"),
        httpx.Response(200, text="no"),
    ):

        def handle(request):
            if isinstance(response, Exception):
                raise response
            return response

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            ladder, decision = router.route_generation_ladder(
                "text",
                [FAST, REASONING],
                enabled=True,
                api_key="synthetic-key",
                post=client.post,
            )
        assert ladder == [FAST, REASONING]
        assert decision["status"] == "unavailable"
        assert "private detail" not in json.dumps(decision)


def test_router_configuration_is_opt_in_and_rejects_env_injection(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    assert agent_llm_config._model_router_env("token_factory") == []
    monkeypatch.setenv("NPA_AGENT_MODEL_ROUTER", "jev")
    assert agent_llm_config._model_router_env("token_factory") == [
        "NPA_AGENT_MODEL_ROUTER=jev",
        "TYPESAFE_API_KEY=synthetic-key",
    ]
    with pytest.raises(ValueError, match="Token Factory only"):
        agent_llm_config._model_router_env("other")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key\nOTHER=value")
    with pytest.raises(ValueError, match="valid TYPESAFE_API_KEY"):
        agent_llm_config._model_router_env("token_factory")


def test_identical_routes_still_issue_two_provider_requests():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=_response())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        for _ in range(2):
            router.route_generation_ladder(
                "same text",
                [FAST, REASONING],
                enabled=True,
                api_key="synthetic-key",
                post=client.post,
            )
    assert len(calls) == 2


def test_router_redacts_credentials_before_the_http_request():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_response())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        router.route_generation_ladder(
            "Explain Bearer synthetic-private-value without copying the token.",
            [FAST, REASONING],
            enabled=True,
            api_key="synthetic-router-key",
            post=client.post,
        )
    assert "synthetic-private-value" not in json.dumps(requests)
    assert "synthetic-router-key" not in json.dumps(requests)
    assert "Explain" in requests[0]["state"]["request"]


def test_saved_router_key_uses_private_staging_not_shell_argv(monkeypatch):
    monkeypatch.setenv("NPA_AGENT_MODEL_ROUTER", "jev")
    monkeypatch.setattr(
        agent_llm_config,
        "load_credentials",
        lambda: SimpleNamespace(tokens={"TYPESAFE_API_KEY": "synthetic-router-key"}),
    )
    ssh = MagicMock()
    agent_llm_config.write_agent_llm_env(
        ssh,
        api_key="synthetic-generation-key",
        provider="token_factory",
        model=FAST,
        providers=["token_factory"],
        models=[FAST, REASONING],
    )
    private_calls = ssh.upload_private_text.call_args_list
    assert len(private_calls) == 1
    assert "TYPESAFE_API_KEY=synthetic-router-key" in str(private_calls[0])
    assert "NPA_AGENT_MODEL_ROUTER=jev" in str(private_calls[0])
    assert "synthetic-router-key" not in str(ssh.run_or_raise.call_args_list)
    assert "synthetic-generation-key" not in str(ssh.run_or_raise.call_args_list)


def test_missing_router_credential_rejected_during_runtime_resolution(monkeypatch):
    monkeypatch.setenv("NPA_AGENT_MODEL_ROUTER", "jev")
    monkeypatch.setattr(
        agent_llm_config, "load_credentials", lambda: SimpleNamespace(tokens={})
    )
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        agent_llm_config.resolve_agent_llm_runtime(
            {},
            llm_config_file="",
            requested_model="",
            requested_models=[],
            defaults=("token_factory", "synthetic-key", FAST, (FAST, REASONING)),
            normalize_models=list,
        )


@pytest.mark.parametrize(
    "details,extra,expected",
    [
        ({"cached_tokens": 20}, {}, 20),
        (None, {"prompt_cache_hit_tokens": 0}, 0),
        (None, {"prompt_cache_hit_tokens": 20}, 20),
        (None, {}, None),
        ({"cached_tokens": True}, {}, None),
        ({"cached_tokens": -1}, {}, None),
        ({"cached_tokens": 101}, {}, None),
    ],
)
def test_usage_keeps_provider_cache_evidence_without_inventing_missing_counts(
    details, extra, expected
):
    summary = agent_routing.usage_summary(
        {
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": details,
                **extra,
            }
        }
    )
    assert summary.get("cached_tokens") == expected
