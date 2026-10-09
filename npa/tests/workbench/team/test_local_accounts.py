"""Verify local keys, browser revocation, isolation, and stable ownership after SSO linking."""

import json
import sqlite3
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.workbench.team import app as team_cli
from npa.workbench.team.account_administration import issue_key_file, link_identity
from npa.workbench.team.account_authentication import AccountAuthentication
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.api import create_app
from npa.workbench.team.authorization import authorize, bind_execution
from npa.workbench.team.browser_sessions import SESSION_COOKIE
from npa.workbench.team.errors import AuthenticationError, ConflictError, TeamError
from npa.workbench.team.models import BrowserLogin, SubmitRequest, TeamConfig
from npa.workbench.team.service import TeamService, binding_snapshot


@pytest.fixture
def local(config):
    config = config.model_copy(
        update={
            "identity": None,
            "account_namespace": uuid.uuid4(),
            "browser_login": BrowserLogin(public_url="https://testserver"),
        }
    )
    accounts = Accounts(config)
    people = [accounts.create(name, ["researchers"]) for name in ("alice", "bob")]
    data = config.model_dump(mode="json")
    for allocation, person in zip(
        data["workspaces"]["robotics"]["allocations"], people, strict=True
    ):
        allocation["subject"] = person["id"]
    config = TeamConfig.model_validate(data)
    keys = [accounts.issue_key(person["id"]) for person in people]
    service = TeamService(lambda: config)
    app = create_app(None, service=service)
    with TestClient(app, base_url="https://testserver") as client:
        yield SimpleNamespace(
            config=config,
            accounts=accounts,
            people=people,
            keys=keys,
            service=service,
            app=app,
            client=client,
        )


def _headers(local, index=0):
    return {"Authorization": "Bearer " + local.keys[index][1]}


def _sign_in(local, index=0, **headers):
    return local.client.post(
        "/auth/key",
        json={"access_key": local.keys[index][1]},
        headers={"Origin": "https://testserver", "X-Workbench-Login": "1", **headers},
    )


def test_real_local_keys_persist_as_hashes_and_never_need_external_identity(local):
    assert local.config.identity is None
    profile = local.client.get("/v1/me", headers=_headers(local)).json()
    assert profile["subject"] == local.people[0]["id"]
    assert profile["display_name"] == "alice"
    assert profile["workspaces"][0]["allocated"]
    assert (
        Accounts(local.config).authenticate(local.keys[0][1])[0].subject
        == profile["subject"]
    )
    with sqlite3.connect(local.accounts.path) as db:
        dump = "\n".join(db.iterdump())
    assert all(secret not in dump for _, secret in local.keys)
    assert local.accounts.path.stat().st_mode & 0o777 == 0o600


def test_key_login_cookie_csrf_and_immediate_revocation(local):
    response = _sign_in(local)
    assert response.status_code == 204
    assert all(
        flag in response.headers["set-cookie"]
        for flag in ("Secure", "HttpOnly", "SameSite=lax")
    )
    assert local.keys[0][1] not in response.text
    profile = local.client.get("/v1/me").json()
    assert profile["display_name"] == "alice"
    assert local.client.post("/auth/logout").status_code == 403
    local.accounts.revoke(local.keys[0][0])
    assert local.client.get("/v1/me").status_code == 401
    assert local.client.get("/v1/me", headers=_headers(local)).status_code == 401
    assert local.client.get("/v1/me", headers=_headers(local, 1)).status_code == 200


@pytest.mark.parametrize("origin", ["https://attacker.example.test", "null", ""])
def test_key_login_cannot_be_forced_by_another_origin(local, origin):
    assert _sign_in(local, Origin=origin).status_code == 403
    assert SESSION_COOKIE not in local.client.cookies


def test_invalid_keys_and_local_session_ids_are_not_credentials(local):
    for invalid in ("", "alice", local.keys[0][0], local.keys[0][1] + "x"):
        response = local.client.get(
            "/v1/me", headers={"Authorization": "Bearer " + invalid}
        )
        assert response.status_code == 401
        assert local.keys[0][1] not in response.text
    assert _sign_in(local).status_code == 204
    cookie = local.client.cookies.get(SESSION_COOKIE)
    response = local.client.get("/v1/me", headers={"Authorization": "Bearer " + cookie})
    assert response.status_code == 401


def test_disable_and_group_changes_apply_to_sessions_and_admitted_work(local):
    actor = local.accounts.authenticate(local.keys[0][1])[0]
    assert _sign_in(local).status_code == 204
    local.accounts.update(actor.subject, groups=[])
    assert local.client.get("/v1/me").json()["workspaces"] == []
    with pytest.raises(TeamError):
        authorize(local.config, actor, "robotics", "runner")
    local.accounts.update(actor.subject, disabled=True)
    assert local.client.get("/v1/me").status_code == 401
    with pytest.raises(AuthenticationError):
        authorize(local.config, actor, "robotics", "runner")


def test_user_isolation_and_sso_link_preserve_job_and_namespace(
    local, config, tokens, workflow
):
    first = local.accounts.authenticate(local.keys[0][1])[0]
    binding = bind_execution(local.config, first, "robotics", "east")
    request = SubmitRequest(
        workspace="robotics",
        cluster="east",
        idempotency_key="before-sso",
        workflow=workflow,
    )
    record, _ = local.service.ledger.create(first, request, binding_snapshot(binding))
    assert (
        local.client.get(
            f"/v1/runs/{record['id']}", headers=_headers(local, 1)
        ).status_code
        == 404
    )
    linked = local.config.model_copy(update={"identity": config.identity})
    auth = AccountAuthentication(linked, external=tokens.verifier)
    with pytest.raises(AuthenticationError):
        auth.verify("Bearer " + tokens.sign(sub=first.subject))
    link_identity(linked, first.subject, config.identity.issuer, "provider-person")
    actor = auth.verify(
        "Bearer " + tokens.sign(sub="provider-person", groups=["administrator"])
    )
    assert actor == first
    assert (
        bind_execution(linked, actor, "robotics", "east").namespace == binding.namespace
    )
    assert local.service.get(actor, record["id"])["id"] == record["id"]
    reused, created = local.service.ledger.create(
        actor, request, binding_snapshot(binding)
    )
    assert not created and reused["id"] == record["id"]
    with pytest.raises(ConflictError):
        link_identity(
            linked, local.people[1]["id"], config.identity.issuer, "provider-person"
        )


def test_account_namespace_is_immutable_and_emails_never_link_automatically(local):
    changed = local.config.model_copy(update={"account_namespace": uuid.uuid4()})
    with pytest.raises(ConflictError, match="namespace changed"):
        Accounts(changed)
    with pytest.raises(TeamError, match="configured external provider"):
        link_identity(
            local.config, local.people[0]["id"], "https://other.test", "alice"
        )


def test_key_file_delivery_is_private_and_does_not_overwrite(local, tmp_path):
    output = tmp_path / "access-key"
    receipt = issue_key_file(local.config, local.people[0]["id"], output)
    assert output.stat().st_mode & 0o777 == 0o600
    secret = output.read_text().strip()
    assert secret not in json.dumps(receipt)
    assert local.accounts.authenticate(secret)[0].subject == local.people[0]["id"]
    before = len(local.accounts.list()[0]["keys"])
    with pytest.raises(FileExistsError):
        issue_key_file(local.config, local.people[0]["id"], output)
    assert len(local.accounts.list()[0]["keys"]) == before


def test_local_cli_creates_revokes_and_disables_real_accounts(local, tmp_path):
    config = tmp_path / "team.json"
    config.write_text(local.config.model_dump_json())
    runner = CliRunner()
    created = runner.invoke(
        team_cli, ["account", "create", "--config", str(config), "--name", "casey"]
    )
    assert created.exit_code == 0, created.output
    user_id = json.loads(created.stdout)["id"]
    destination = tmp_path / "casey-key"
    issued = runner.invoke(
        team_cli,
        [
            "account",
            "issue-key",
            "--config",
            str(config),
            "--user",
            user_id,
            "--output-file",
            str(destination),
        ],
    )
    assert issued.exit_code == 0, issued.output
    key_id = json.loads(issued.stdout)["key_id"]
    assert destination.read_text().strip() not in issued.stdout
    revoked = runner.invoke(
        team_cli, ["account", "revoke-key", "--config", str(config), "--key-id", key_id]
    )
    assert revoked.exit_code == 0, revoked.output
    disabled = runner.invoke(
        team_cli,
        ["account", "update", "--config", str(config), "--user", user_id, "--disabled"],
    )
    assert disabled.exit_code == 0, disabled.output
