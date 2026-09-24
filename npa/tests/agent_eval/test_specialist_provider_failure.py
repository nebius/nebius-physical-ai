"""Prove typed provider failures preserve evidence and never broaden fallback authority."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from npa.agent_backend.specialists import graph, provider
from npa.agent_backend.specialists.config import ModelEndpoint, Profile, TeamConfig
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.clients.token_factory import (
    TokenFactoryClient,
    TokenFactoryConfig,
    TokenFactoryError,
)

_SECRET = "synthetic-private-credential"


def _response(model):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Complete"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = Profile(
        name="repair",
        description="Repair the granted source",
        model="test/primary",
        fallback_models=[ModelEndpoint(model="test/backup")],
        workspace=workspace,
        read_paths=["source"],
        write_paths=["source"],
    )
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="repair"
    )
    return SimpleNamespace(profile=profile, config=config)


def _team(setup, handler):
    client = TokenFactoryClient(
        config=TokenFactoryConfig(
            base_url="https://models.example/v1", api_key=_SECRET
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_attempts=1,
    )
    team = SpecialistTeam(setup.config, clients={"repair": client})
    team.submit("Finish within existing grants", specialist="repair", task_id="task")
    return team


def test_404_fallback_preserves_failure_after_restart_and_success(setup):
    calls = []
    original_policy = setup.profile.model_dump()

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(404, json={"detail": _SECRET})
        return httpx.Response(200, json=_response(body["model"]))

    team = _team(setup, handler)
    assert team.work_once("repair")["status"] == "running"
    restarted = SpecialistTeam(setup.config, clients=team.clients)
    assert restarted.work_once("repair")["status"] == "completed"
    events = restarted.status("task")["events"]
    failures = [event for event in events if event["type"] == "provider_failure"]
    assert len(failures) == 1
    assert failures[0]["model"] == "test/primary"
    assert failures[0]["status_code"] == 404
    assert failures[0]["request_metrics"]["attempts"] == 1
    assert failures[0]["usage_missing"] is True and failures[0]["usage"] == {}
    assert [call["model"] for call in calls] == ["test/primary", "test/backup"]
    assert calls[1]["messages"][:2] == calls[0]["messages"][:2]
    assert calls[1]["tools"] == calls[0]["tools"]
    assert setup.profile.model_dump() == original_policy
    assert _SECRET not in json.dumps(events)


@pytest.mark.parametrize("status", [404, 401, 403, 429, 500, 503])
def test_no_unauthorized_or_unavailable_alternative_is_invented(setup, status):
    setup.profile.fallback_models = []
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"detail": _SECRET})

    team = _team(setup, handler)
    assert team.work_once("repair")["status"] == "needs_attention"
    assert len(calls) == 1
    events = team.status("task")["events"]
    assert len([event for event in events if event["type"] == "provider_failure"]) == 1
    assert not any(event["type"] == "model_fallback" for event in events)
    assert _SECRET not in json.dumps(events)


def test_transport_failure_escalates_once_without_fallback(setup):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout(_SECRET, request=request)

    team = _team(setup, handler)
    assert team.work_once("repair")["status"] == "needs_attention"
    assert len(calls) == 1
    event = next(
        item
        for item in team.status("task")["events"]
        if item["type"] == "provider_failure"
    )
    assert event["failure_kind"] == "provider_transport_error"
    assert event["status_code"] is None
    assert event["request_metrics"]["retries"] == 0
    assert _SECRET not in json.dumps(team.status("task"))


def test_error_text_alone_cannot_authorize_fallback(setup):
    class Client:
        def chat_completion(self, **kwargs):
            raise TokenFactoryError("Token Factory request failed (404): " + _SECRET)

    team = SpecialistTeam(setup.config, clients={"repair": Client()})
    team.submit("Inspect", specialist="repair", task_id="task")
    assert team.work_once("repair")["status"] == "needs_attention"
    event = next(
        item
        for item in team.status("task")["events"]
        if item["type"] == "provider_failure"
    )
    assert event["status_code"] is None
    assert event["api_call_attempted"] == "unknown"
    assert _SECRET not in json.dumps(team.status("task"))


def test_constructed_specialist_client_disables_hidden_retries(setup, monkeypatch):
    monkeypatch.setenv(setup.profile.key_env, _SECRET)
    client = graph._client(setup.profile)
    assert client._retry_attempts == 1


def test_metrics_allowlist_drops_strings_secrets_and_nonfinite_values():
    client = SimpleNamespace(
        last_request_metrics={
            "attempts": "credential",
            "retries": True,
            "latency_seconds": float("nan"),
            "status_code": 404,
            "headers": {"Authorization": _SECRET},
        }
    )
    assert provider._metrics(client, None) == {"status_code": 404}


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_other_http_errors_do_not_use_available_backup(setup, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"detail": _SECRET})

    team = _team(setup, handler)
    assert team.work_once("repair")["status"] == "needs_attention"
    assert len(calls) == 1
    assert not any(
        event["type"] == "model_fallback" for event in team.status("task")["events"]
    )


def test_exhausted_endpoints_preserve_each_missing_usage_failure(setup):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(404, json={"detail": _SECRET})

    team = _team(setup, handler)
    assert team.work_once("repair")["status"] == "running"
    assert team.work_once("repair")["status"] == "needs_attention"
    assert calls == ["test/primary", "test/backup"]
    failures = [
        event
        for event in team.status("task")["events"]
        if event["type"] == "provider_failure"
    ]
    assert [event["model"] for event in failures] == calls
    assert all(event["usage_missing"] and event["usage"] == {} for event in failures)


def test_checkpoint_error_never_retains_response_body(setup):
    import sqlite3

    team = _team(setup, lambda request: httpx.Response(403, json={"detail": _SECRET}))
    assert team.work_once("repair")["status"] == "needs_attention"
    path = setup.config.state_directory / "task.checkpoints.sqlite"
    with sqlite3.connect(path) as database:
        rows = database.execute(
            "SELECT value FROM writes WHERE channel = '__error__'"
        ).fetchall()
    assert len(rows) == 1
    assert _SECRET.encode() not in rows[0][0]
