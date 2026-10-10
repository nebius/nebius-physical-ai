"""Verify local keys, API revocation, isolation, and stable ownership after SSO linking."""

import json
import sqlite3
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.workbench.team import app as team_cli
from npa.workbench.team.account_administration import (
    issue_key_file,
    link_identity,
    unlink_identity,
)
from npa.workbench.team.account_authentication import AccountAuthentication
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.api import create_app
from npa.workbench.team.authorization import authorize, bind_execution
from npa.workbench.team.errors import AuthenticationError, ConflictError, TeamError
from npa.workbench.team.models import Actor, SubmitRequest, TeamConfig
from npa.workbench.team.service import TeamService, binding_snapshot


@pytest.fixture
def local(config):
    config = config.model_copy(
        update={
            "identity": None,
            "account_namespace": uuid.uuid4(),
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


def test_configuration_rejects_retired_browser_login_setting(config):
    policy = config.model_dump(mode="json")
    policy["browser_login"] = {"public_url": "https://workbench.example.test"}
    with pytest.raises(ValueError, match="browser_login"):
        TeamConfig.model_validate(policy)


def test_key_revocation_immediately_blocks_bearer_authentication(local):
    profile = local.client.get("/v1/me", headers=_headers(local)).json()
    assert profile["display_name"] == "alice"
    local.accounts.revoke(local.keys[0][0])
    assert local.client.get("/v1/me", headers=_headers(local)).status_code == 401
    assert local.client.get("/v1/me", headers=_headers(local, 1)).status_code == 200


def test_invalid_keys_and_cookie_values_are_not_credentials(local):
    for invalid in ("", "alice", local.keys[0][0], local.keys[0][1] + "x"):
        response = local.client.get(
            "/v1/me", headers={"Authorization": "Bearer " + invalid}
        )
        assert response.status_code == 401
        assert local.keys[0][1] not in response.text
    response = local.client.get(
        "/v1/me", headers={"Cookie": "__Host-workbench-session=forged"}
    )
    assert response.status_code == 401


def test_disable_and_group_changes_apply_to_sessions_and_admitted_work(local):
    actor = local.accounts.authenticate(local.keys[0][1])[0]
    local.accounts.update(actor.subject, groups=[])
    assert (
        local.client.get("/v1/me", headers=_headers(local)).json()["workspaces"] == []
    )
    with pytest.raises(TeamError):
        authorize(local.config, actor, "robotics", "runner")
    local.accounts.update(actor.subject, disabled=True)
    assert local.client.get("/v1/me", headers=_headers(local)).status_code == 401
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


def test_retired_identity_removal_is_exact_repeatable_and_allows_provider_change(
    local, config
):
    owner, other = [person["id"] for person in local.people]
    old = local.config.model_copy(update={"identity": config.identity})
    link_identity(old, owner, config.identity.issuer, "old-person")
    replacement = config.identity.model_copy(update={"issuer": "https://new.test"})
    new = local.config.model_copy(update={"identity": replacement})
    for user, issuer, subject in (
        (other, config.identity.issuer, "old-person"),
        (owner, replacement.issuer, "old-person"),
        (owner, config.identity.issuer, "different-person"),
    ):
        assert not unlink_identity(new, user, issuer, subject)["unlinked"]
    local.accounts.update(owner, disabled=True)
    assert unlink_identity(new, owner, config.identity.issuer, "old-person")["unlinked"]
    assert not unlink_identity(new, owner, config.identity.issuer, "old-person")[
        "unlinked"
    ]
    local.accounts.update(owner, disabled=False)
    link_identity(new, owner, replacement.issuer, "new-person")
    actor = local.accounts.resolve(
        Actor(issuer=replacement.issuer, subject="new-person")
    )
    assert actor == local.accounts.authenticate(local.keys[0][1])[0]
    with sqlite3.connect(local.accounts.path) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM account_audit WHERE action='identity-unlink'"
            ).fetchone()[0]
            == 1
        )


def test_unlink_cli_uses_authoritative_database_and_reports_repeated_removal(
    local, config, tmp_path
):
    linked = local.config.model_copy(update={"identity": config.identity})
    owner = local.people[0]["id"]
    link_identity(linked, owner, config.identity.issuer, "old-person")
    path = tmp_path / "local-only.json"
    path.write_text(local.config.model_dump_json())
    args = [
        "account",
        "unlink",
        "--config",
        str(path),
        "--user",
        owner,
        "--issuer",
        config.identity.issuer,
        "--subject",
        "old-person",
    ]
    for removed in (True, False):
        result = CliRunner().invoke(team_cli, args)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {"user_id": owner, "unlinked": removed}
    assert local.accounts.authenticate(local.keys[0][1])[0].subject == owner
