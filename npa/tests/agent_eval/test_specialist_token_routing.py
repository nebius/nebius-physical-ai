"""Verify model-driven routing preserves grants, paid attempts and exact endpoint choices."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists import routing, token_router
from npa.agent_backend.specialists.config import (
    ModelEndpoint,
    Profile,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.clients.token_factory import TokenFactoryError


def _response(selected="synthetic/reasoning"):
    return {
        "model": "synthetic/classifier",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"selected_model": selected}),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.last_request_metrics = {"attempts": 1, "retries": 0}

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        return deepcopy(self.response)


def _profile(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Profile(
        name="repair",
        description="Repair this workspace",
        workspace=workspace,
        model="synthetic/fast",
        fallback_models=[ModelEndpoint(model="synthetic/reasoning")],
        model_router="token_factory",
        require_model_route=True,
        routing_model=ModelEndpoint(
            model="synthetic/classifier", key_env="ROUTER_TEST_KEY"
        ),
        model_criteria={
            "synthetic/fast": "Routine repairs",
            "synthetic/reasoning": "Deep diagnosis",
        },
        read_paths=["source"],
        write_paths=["source"],
    )


@pytest.fixture
def routed(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    classifier = _Client(_response())
    worker = _Client(
        {
            "model": "synthetic/reasoning",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Done"},
                }
            ],
        }
    )
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="repair"
    )
    team = SpecialistTeam(config, clients={"repair": worker})
    monkeypatch.setenv("ROUTER_TEST_KEY", "synthetic-router-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(routing, "load_credentials", lambda: SimpleNamespace(tokens={}))
    monkeypatch.setattr(token_router, "TokenFactoryClient", lambda **_k: classifier)
    team.submit("Diagnose the physics timestep", specialist="repair", task_id="task")
    return SimpleNamespace(
        team=team, profile=profile, worker=worker, classifier=classifier
    )


def test_actual_choice_drives_langgraph_without_jev_or_changed_grants(routed):
    before = routed.profile.model_dump()
    result = routed.team.work_once("repair")
    assert result["status"] == "completed"
    route = result["route"]["model_selection"]
    assert route["provider"] == "token_factory"
    assert route["model"] == "synthetic/classifier"
    assert (
        route["selected_model"]
        == routed.worker.calls[0]["model"]
        == "synthetic/reasoning"
    )
    assert route["status"] == "accepted" and route["api_call_attempted"] is True
    assert route["model_verified"] is True
    assert route["usage"] == {
        "input_tokens": 100,
        "output_tokens": 8,
        "cached_input_tokens": 40,
    }
    assert routed.profile.model_dump() == before
    request = routed.classifier.calls[0]
    assert "tools" not in request.get("extra", {})
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["selected_model"]["enum"] == [
        "synthetic/fast",
        "synthetic/reasoning",
        "none",
    ]
    assert "synthetic-router-key" not in json.dumps(result)


def test_route_is_journaled_before_request_and_reused_after_restart(
    routed, monkeypatch
):
    original = routed.classifier.chat_completion

    def request(**arguments):
        intent = routed.team.status("task")["route"]["model_selection"]
        assert intent["status"] == "started"
        assert intent["api_call_attempted"] == "unknown"
        return original(**arguments)

    monkeypatch.setattr(routed.classifier, "chat_completion", request)
    routing._route_model(
        routed.profile, routed.team.store._get("task"), routed.team.store
    )
    restarted = SpecialistTeam(routed.team.config, clients={"repair": routed.worker})
    assert restarted.work_once("repair")["status"] == "completed"
    assert len(routed.classifier.calls) == 1


@pytest.mark.parametrize(
    "defect", ["outside", "abstain", "model", "truncated", "json", "fields", "tools"]
)
def test_invalid_or_abstained_route_preserves_usage_and_blocks_generation(
    routed, defect
):
    response = routed.classifier.response
    choice = response["choices"][0]
    if defect in {"outside", "abstain"}:
        choice["message"]["content"] = json.dumps(
            {"selected_model": "other/model" if defect == "outside" else "none"}
        )
    elif defect == "model":
        response["model"] = "wrong/classifier"
    elif defect == "truncated":
        choice["finish_reason"] = "length"
    elif defect == "json":
        choice["message"]["content"] = "not JSON"
    elif defect == "fields":
        choice["message"]["content"] = (
            '{"selected_model":"synthetic/fast","write_paths":["."]}'
        )
    else:
        choice["message"]["tool_calls"] = [{"function": {"name": "edit_file"}}]
    result = routed.team.work_once("repair")
    assert result["status"] == "needs_attention"
    assert result["route"]["model_selection"]["usage"]["input_tokens"] == 100
    assert result["route"]["model_selection"]["model_verified"] is (defect != "model")
    assert routed.worker.calls == []
    routed.team.reconcile("task", retry=True)
    assert routed.team.work_once("repair")["status"] == "needs_attention"
    assert len(routed.classifier.calls) == 1


def test_missing_classifier_credential_makes_no_provider_calls(routed, monkeypatch):
    monkeypatch.delenv("ROUTER_TEST_KEY")
    result = routed.team.work_once("repair")
    assert result["status"] == "needs_attention"
    route = result["route"]["model_selection"]
    assert route["api_call_attempted"] is False
    assert route["reason"] == "missing_credential"
    assert routed.classifier.calls == routed.worker.calls == []


@pytest.mark.parametrize(
    "error", [TokenFactoryError("private-provider-body"), RuntimeError("lost response")]
)
def test_failed_or_lost_router_attempt_is_never_replayed(routed, monkeypatch, error):
    calls = []

    def fail(**arguments):
        calls.append(arguments)
        raise error

    monkeypatch.setattr(routed.classifier, "chat_completion", fail)
    first = routed.team.work_once("repair")
    assert first["status"] == "needs_attention"
    assert "private-provider-body" not in json.dumps(first)
    assert first["route"]["model_selection"]["usage"] == {}
    routed.team.reconcile("task", retry=True)
    assert routed.team.work_once("repair")["status"] == "needs_attention"
    assert len(calls) == 1 and routed.worker.calls == []


def test_router_configuration_binds_policy_and_requires_classifier(routed):
    data = routed.profile.model_dump()
    changed = Profile.model_validate(
        {**data, "routing_model": {"model": "other/classifier"}}
    )
    assert fingerprint(changed) != fingerprint(routed.profile)
    for change in (
        {"routing_model": None},
        {"model_router": "jev"},
        {"model_criteria": {}},
    ):
        with pytest.raises(ValueError):
            Profile.model_validate({**data, **change})


def test_transport_has_one_attempt_and_honors_endpoint_options(routed, monkeypatch):
    from npa.clients.token_factory import TokenFactoryClient

    instances = []

    def construct(**arguments):
        instances.append(TokenFactoryClient(**arguments))
        return routed.classifier

    monkeypatch.setattr(token_router, "TokenFactoryClient", construct)
    routed.profile.routing_model.model_options = {"temperature": 0.1}
    token_router._classify(
        routed.profile.routing_model,
        "Repair",
        routed.profile.model_criteria,
        api_key="synthetic-router-key",
    )
    assert instances[0]._retry_attempts == 1
    assert routed.classifier.calls[0]["extra"] == {"temperature": 0.1}


@pytest.mark.parametrize("bad", [True, -1, 1.5, "10", None])
def test_invalid_usage_is_unknown_instead_of_coerced_to_billable_counts(bad):
    response = _response()
    response["usage"]["prompt_tokens"] = bad
    assert token_router._usage(response) == {"output_tokens": 8}
