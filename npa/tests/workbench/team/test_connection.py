"""Prove saved CLI and agent connections share authentication without crossing endpoints."""

import httpx
import pytest

from npa.workbench.team import connection
from npa.workbench.team.client import TeamClient
from npa.workbench.team.errors import AuthenticationError, TeamError
from npa.workbench.team.sessions import SavedSession, SessionStore


def _save(tmp_path, monkeypatch, **settings):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("NPA_TEAM_TOKEN", raising=False)
    session = SavedSession(
        endpoint="https://team.example.test", verified_subject="researcher", **settings
    )
    SessionStore().save(
        "research",
        session,
        token="test-personal-key" if session.auth_mode == "local" else None,
    )
    return session


def test_saved_credentials_are_not_forwarded_by_endpoint_override(
    tmp_path, monkeypatch
):
    _save(tmp_path, monkeypatch)
    with pytest.raises(TeamError, match="endpoint differs"):
        connection.connection_settings(endpoint="https://elsewhere.example.test")


def test_explicit_credentials_work_without_any_saved_login(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path))
    key = tmp_path / "key"
    key.write_text("test-explicit-key")
    key.chmod(0o600)
    session, token = connection.connection_settings(
        endpoint="https://team.example.test", token_file=key
    )
    assert token == "test-explicit-key"
    assert session.verified_subject is None
    assert not (tmp_path / "team").exists()


def test_sdk_connect_uses_same_saved_authenticated_api(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch)
    observed = []

    def handle(request):
        observed.append((request.url.path, request.headers["Authorization"]))
        return httpx.Response(200, json={"subject": "researcher"})

    def factory(endpoint, token):
        return TeamClient(endpoint, token, transport=httpx.MockTransport(handle))

    client, _ = connection.open_connection(client_factory=factory)
    try:
        assert client.whoami()["subject"] == "researcher"
    finally:
        client.close()
    assert observed == [("/v1/me", "Bearer test-personal-key")] * 2


@pytest.mark.parametrize("auth_mode", ["local", "nebius"])
def test_changed_identity_is_rejected_before_any_run_operation(
    tmp_path, monkeypatch, auth_mode
):
    settings = {"auth_mode": auth_mode}
    if auth_mode == "nebius":
        settings["nebius_profile"] = "human"
    _save(tmp_path, monkeypatch, **settings)
    if auth_mode == "local":
        monkeypatch.setenv("NPA_TEAM_TOKEN", "test-unrelated-ambient-key")
    monkeypatch.setattr(connection, "fresh_token", lambda profile: "test-current-token")
    observed, clients = [], []

    def handle(request):
        observed.append(request.url.path)
        return httpx.Response(200, json={"subject": "another-person"})

    def factory(endpoint, token):
        client = TeamClient(endpoint, token, transport=httpx.MockTransport(handle))
        clients.append(client)
        return client

    with pytest.raises(AuthenticationError, match="account changed"):
        connection.open_connection(client_factory=factory)
    assert observed == ["/v1/me"]
    assert clients[0].http.is_closed


@pytest.mark.parametrize("content", [None, "not a certificate"])
def test_invalid_private_ca_returns_actionable_error(tmp_path, content):
    certificate = tmp_path / "ca.pem"
    if content is not None:
        certificate.write_text(content)
    with pytest.raises(TeamError, match="check the supplied file"):
        TeamClient("https://team.example.test", "test-key", ca_file=certificate)


def test_sdk_submission_uses_saved_placement_and_replays_receipt(tmp_path, monkeypatch):
    import json
    from npa.sdk.workbench.team import open_connection, submit_workflow

    _save(tmp_path, monkeypatch, workspace="robotics", cluster="east")
    submitted = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"subject": "researcher"})
        submitted.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "retained-run"})

    def factory(endpoint, token):
        return TeamClient(endpoint, token, transport=httpx.MockTransport(handle))

    client, session = open_connection(client_factory=factory)
    try:
        first = submit_workflow(client, session, {"stages": []})
        replay = submit_workflow(client, session, {"stages": []})
        another = submit_workflow(client, session, {"stages": []}, new_run=True)
    finally:
        client.close()
    assert first["id"] == replay["id"]
    assert replay["submission"]["reused"]
    assert submitted[0] == submitted[1]
    assert submitted[0]["workspace"] == "robotics"
    assert submitted[0]["cluster"] == "east"
    assert (
        another["submission"]["idempotency_key"]
        != first["submission"]["idempotency_key"]
    )


def test_explicit_credential_and_endpoint_override_do_not_inherit_saved_ca(
    tmp_path, monkeypatch
):
    _save(tmp_path, monkeypatch, ca_file=str(tmp_path / "old-ca.pem"))
    monkeypatch.setenv("NPA_TEAM_TOKEN", "test-explicit-new-key")
    session, token = connection.connection_settings(endpoint="https://new.example.test")
    assert session.ca_file is None
    assert token == "test-explicit-new-key"
