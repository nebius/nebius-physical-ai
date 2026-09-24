"""Prove Jev selection cannot change tool authority or silently repeat paid requests."""

import json
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists import routing
from npa.agent_backend.specialists.config import ModelEndpoint, Profile, TeamConfig
from npa.agent_backend.specialists.team import SpecialistTeam


class _Client:
    def __init__(self):
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        return {
            "model": arguments["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Complete"},
                }
            ],
        }


@pytest.fixture
def routed(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = Profile(
        name="repair",
        description="Repair this exact workspace",
        workspace=workspace,
        model="synthetic/small",
        fallback_models=[ModelEndpoint(model="synthetic/reasoning")],
        model_router="jev",
        model_criteria={
            "synthetic/small": "Simple changes",
            "synthetic/reasoning": "Complex changes",
        },
        read_paths=["source"],
        write_paths=["source"],
    )
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="repair"
    )
    client = _Client()
    team = SpecialistTeam(config, clients={"repair": client})
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-credential")
    monkeypatch.setattr(routing, "load_credentials", lambda: SimpleNamespace(tokens={}))
    team.submit("Repair difficult source", specialist="repair", task_id="task")
    return SimpleNamespace(team=team, profile=profile, client=client)


def _decision(status="accepted", selected="synthetic/reasoning"):
    return {
        "provider": "typesafe",
        "model": "jev-test",
        "status": status,
        "selected_model": selected,
        "usage": {"input_tokens": 15, "output_tokens": 3},
    }


def test_explicit_workspace_still_routes_model_once_with_durable_intent(
    routed, monkeypatch
):
    attempts = []
    original_policy = routed.profile.model_dump()

    def classify(text, candidates, **options):
        intent = routed.team.status("task")["route"]["model_selection"]
        assert intent["status"] == "started"
        assert intent["api_call_attempted"] == "unknown"
        assert text == "Repair difficult source"
        assert candidates == routed.profile.model_criteria
        attempts.append(options)
        return _decision()

    monkeypatch.setattr(routing, "classify_generation_model", classify)
    assert routed.team.work_once("repair")["status"] == "completed"
    assert routed.client.calls[0]["model"] == "synthetic/reasoning"
    assert routed.profile.model_dump() == original_policy
    receipt = routed.team.status("task")["route"]["model_selection"]
    assert receipt["api_call_attempted"] is True
    assert receipt["endpoint_order"] == ["synthetic/reasoning", "synthetic/small"]
    assert receipt["usage"] == {"input_tokens": 15, "output_tokens": 3}
    assert "synthetic-credential" not in json.dumps(routed.team.status("task"))
    assert routed.team.work_once("repair") is None
    assert len(attempts) == 1


def test_restart_after_routing_never_spends_again(routed, monkeypatch):
    monkeypatch.setattr(
        routing, "classify_generation_model", lambda *_a, **_k: _decision()
    )
    routed_task = routing._route_model(
        routed.profile, routed.team.store._get("task"), routed.team.store
    )
    assert (
        routed_task["route"]["model_selection"]["effective_model"]
        == "synthetic/reasoning"
    )
    monkeypatch.setattr(
        routing,
        "classify_generation_model",
        lambda *_a, **_k: pytest.fail("router replay"),
    )
    restarted = SpecialistTeam(routed.team.config, clients={"repair": routed.client})
    assert restarted.work_once("repair")["status"] == "completed"
    assert routed.client.calls[0]["model"] == "synthetic/reasoning"


def test_crash_before_router_receipt_is_unknown_not_zero_cost_or_replayed(
    routed, monkeypatch
):
    def lost(*args, **kwargs):
        raise RuntimeError("lost response")

    monkeypatch.setattr(routing, "classify_generation_model", lost)
    assert routed.team.work_once("repair")["status"] == "needs_attention"
    receipt = routed.team.status("task")["route"]["model_selection"]
    assert receipt["api_call_attempted"] == "unknown"
    assert receipt["usage"] == {}
    routed.team.reconcile("task", retry=True)
    monkeypatch.setattr(
        routing,
        "classify_generation_model",
        lambda *_a, **_k: pytest.fail("paid retry"),
    )
    result = routed.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert "routing result was lost" in result["error"]
    assert routed.client.calls == []


def test_missing_key_is_explicit_fallback_without_router_http(routed, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    from npa.agent_backend import model_router

    monkeypatch.setattr(
        model_router,
        "_request_decision",
        lambda *_a, **_k: pytest.fail("unexpected HTTP"),
    )
    routed.team.work_once("repair")
    receipt = routed.team.status("task")["route"]["model_selection"]
    assert receipt["api_call_attempted"] is False
    assert receipt["reason"] == "missing_credential"
    assert receipt["fallback"] is True
    assert receipt["effective_model"] == "synthetic/small"


@pytest.mark.parametrize(
    "decision",
    [
        _decision("abstained", None),
        {
            **_decision("unavailable", None),
            "reason": "provider_transport_error",
            "usage": {},
        },
    ],
)
def test_provider_abstention_and_failure_retain_attempt_and_fallback(
    routed, monkeypatch, decision
):
    monkeypatch.setattr(
        routing, "classify_generation_model", lambda *_a, **_k: decision
    )
    routed.team.work_once("repair")
    receipt = routed.team.status("task")["route"]["model_selection"]
    assert receipt["api_call_attempted"] is True
    assert receipt["fallback"] is True
    assert receipt["usage"] == decision["usage"]
    assert routed.client.calls[0]["model"] == "synthetic/small"


def test_router_cannot_select_different_authority_or_endpoint(routed, monkeypatch):
    monkeypatch.setattr(
        routing,
        "classify_generation_model",
        lambda *_a, **_k: _decision(selected="other/endpoint"),
    )
    result = routed.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert routed.client.calls == []


@pytest.mark.parametrize("control", ["cancel", "pause"])
def test_control_during_router_call_prevents_generation(routed, monkeypatch, control):
    def classify(*args, **kwargs):
        if control == "cancel":
            routed.team.cancel("task")
        else:
            routed.team.pause(task_id="task")
        return _decision()

    monkeypatch.setattr(routing, "classify_generation_model", classify)
    result = routed.team.work_once("repair")
    assert result["status"] == ("cancelled" if control == "cancel" else "running")
    assert result["route"]["model_selection"]["api_call_attempted"] is True
    assert routed.client.calls == []


def test_rejected_routed_endpoint_falls_back_within_original_grants(
    routed, monkeypatch
):
    monkeypatch.setattr(
        routing, "classify_generation_model", lambda *_a, **_k: _decision()
    )
    original = routed.client.chat_completion

    def completion(**arguments):
        response = original(**arguments)
        if len(routed.client.calls) == 1:
            response["choices"][0]["finish_reason"] = "length"
        return response

    monkeypatch.setattr(routed.client, "chat_completion", completion)
    assert routed.team.work_once("repair")["status"] == "running"
    assert routed.team.work_once("repair")["status"] == "completed"
    assert [call["model"] for call in routed.client.calls] == [
        "synthetic/reasoning",
        "synthetic/small",
    ]
    assert (
        routed.client.calls[0]["extra"]["tools"]
        == routed.client.calls[1]["extra"]["tools"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_criteria": {"other/model": "Outside policy"}},
        {"model_criteria": {"synthetic/small": "", "synthetic/reasoning": "Reason"}},
        {"fallback_models": [ModelEndpoint(model="synthetic/small")]},
        {"model_router": "explicit"},
    ],
)
def test_router_configuration_cannot_ambiguously_bind_model_ids(routed, change):
    with pytest.raises(ValueError):
        Profile.model_validate({**routed.profile.model_dump(), **change})
