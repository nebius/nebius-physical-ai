"""Exercise native Nebius human identity verification without live IAM access."""

from __future__ import annotations

import copy
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from npa.workbench.team.account_administration import link_identity, unlink_identity
from npa.workbench.team.account_authentication import AccountAuthentication
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.api import create_app
from npa.workbench.team.authorization import bind_execution
from npa.workbench.team.errors import AuthenticationError, ConflictError, TeamError
from npa.workbench.team.models import SubmitRequest, TeamConfig
from npa.workbench.team.nebius_authentication import (
    NEBIUS_ISSUER,
    PROFILE_ENDPOINT,
    NebiusProfileVerifier,
)
from npa.workbench.team.service import TeamService, binding_snapshot

TENANT_ID = "tenant-example"
FEDERATION_ID = "federation-example"
HUMAN_SUBJECT = "tenantuseraccount-human"


def _profile(subject=HUMAN_SUBJECT, tenants=None, **overrides):
    """Build a public-REST-cased successful ProfileService response fixture."""
    profile = {
        "id": "useraccount-human",
        "federationInfo": {
            "federationId": FEDERATION_ID,
            "federationUserAccountId": "federated-human",
        },
        "tenants": tenants
        if tenants is not None
        else [
            {
                "tenantId": TENANT_ID,
                "tenantUserAccountId": subject,
                "tenantUserAccountState": "ACTIVE",
            }
        ],
        "userAccountState": "ACTIVE",
    }
    profile.update(overrides)
    return {"userProfile": profile}


def _profile_verifier(identity, handler):
    """Create a profile verifier backed by one hermetic HTTP transport."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return NebiusProfileVerifier(identity, client=client)


@pytest.fixture
def native(config):
    """Create two local accounts under a native Nebius identity configuration."""
    policy = config.model_dump(mode="json")
    policy["identity"] = None
    policy["account_namespace"] = str(uuid.uuid4())
    policy["nebius_identity"] = {
        "tenant_id": TENANT_ID,
        "federation_id": FEDERATION_ID,
    }
    configured = TeamConfig.model_validate(policy)
    accounts = Accounts(configured)
    people = [accounts.create(name, ["researchers"]) for name in ("alice", "bob")]
    policy = configured.model_dump(mode="json")
    for allocation, person in zip(
        policy["workspaces"]["robotics"]["allocations"], people, strict=True
    ):
        allocation["subject"] = person["id"]
    configured = TeamConfig.model_validate(policy)
    keys = [accounts.issue_key(person["id"]) for person in people]
    return SimpleNamespace(
        config=configured, accounts=accounts, people=people, keys=keys
    )


def test_nebius_configuration_requires_local_accounts_and_excludes_generic_jwt(config):
    policy = config.model_dump(mode="json")
    policy["nebius_identity"] = {"tenant_id": TENANT_ID}
    with pytest.raises(ValueError, match="mutually exclusive"):
        TeamConfig.model_validate(policy)
    policy["identity"] = None
    with pytest.raises(ValueError, match="requires account_namespace"):
        TeamConfig.model_validate(policy)
    policy["account_namespace"] = str(uuid.uuid4())
    assert TeamConfig.model_validate(policy).nebius_identity.tenant_id == TENANT_ID


def test_profile_service_uses_fixed_endpoint_and_verified_tenant_subject(native):
    observed = []

    def handler(request):
        observed.append(request)
        return httpx.Response(200, json=_profile(roles=["cloud-admin"]))

    verifier = _profile_verifier(native.config.nebius_identity, handler)
    actor = verifier.verify("Bearer human-token")
    assert actor.issuer == NEBIUS_ISSUER
    assert actor.subject == HUMAN_SUBJECT
    assert actor.groups == frozenset()
    assert len(observed) == 1
    assert str(observed[0].url) == PROFILE_ENDPOINT
    assert observed[0].method == "GET"
    assert observed[0].headers["Authorization"] == "Bearer human-token"


def test_profile_service_allows_an_unconstrained_federation(native):
    identity = native.config.nebius_identity.model_copy(update={"federation_id": None})
    payload = _profile()
    del payload["userProfile"]["federationInfo"]

    def handler(request):
        return httpx.Response(200, json=payload)

    verifier = _profile_verifier(identity, handler)
    assert verifier.verify("Bearer caller-token").subject == HUMAN_SUBJECT


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"serviceAccountProfile": {"info": {}}},
        {"anonymousProfile": {}},
        _profile(id=""),
        _profile(userAccountState="INACTIVE"),
        _profile(tenants=[{"tenantId": "tenant-other"}]),
        _profile(
            tenants=[
                {
                    "tenantId": TENANT_ID,
                    "tenantUserAccountId": HUMAN_SUBJECT,
                    "tenantUserAccountState": "ACTIVE",
                },
                {
                    "tenantId": TENANT_ID,
                    "tenantUserAccountId": "tenantuseraccount-duplicate",
                    "tenantUserAccountState": "ACTIVE",
                },
            ]
        ),
        _profile(
            tenants=[
                {
                    "tenantId": TENANT_ID,
                    "tenantUserAccountId": HUMAN_SUBJECT,
                    "tenantUserAccountState": "BLOCKED",
                }
            ]
        ),
        _profile(federationInfo={"federationId": "federation-other"}),
        {"userProfile": []},
    ],
)
def test_profile_service_rejects_untrusted_or_malformed_profiles(native, payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    verifier = _profile_verifier(native.config.nebius_identity, handler)
    with pytest.raises(AuthenticationError) as raised:
        verifier.verify("Bearer caller-token")
    assert str(raised.value) == "Nebius identity could not be verified"
    assert "caller-token" not in str(raised.value)


@pytest.mark.parametrize("status", [302, 401, 502])
def test_profile_service_failures_and_redirects_do_not_expose_details(native, status):
    def handler(request):
        return httpx.Response(
            status,
            headers={"Location": "https://untrusted.example.test"},
            text="provider detail that must not escape",
        )

    verifier = _profile_verifier(native.config.nebius_identity, handler)
    with pytest.raises(AuthenticationError) as raised:
        verifier.verify("Bearer caller-token")
    assert str(raised.value) == "Nebius identity could not be verified"
    assert "provider detail" not in str(raised.value)


def test_profile_service_rejects_invalid_json_and_provider_errors(native):
    def malformed(request):
        return httpx.Response(200, content=b"not-json")

    verifier = _profile_verifier(native.config.nebius_identity, malformed)
    with pytest.raises(AuthenticationError, match="could not be verified"):
        verifier.verify("Bearer caller-token")

    def unavailable(request):
        raise httpx.ConnectError("provider unavailable", request=request)

    verifier = _profile_verifier(native.config.nebius_identity, unavailable)
    with pytest.raises(AuthenticationError, match="could not be verified"):
        verifier.verify("Bearer caller-token")


def test_linked_human_and_local_key_resolve_to_same_actor_and_namespace(native):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_profile(groups=["cloud-admin"]))

    local_actor = native.accounts.authenticate(native.keys[0][1])[0]
    link_identity(native.config, local_actor.subject, NEBIUS_ISSUER, HUMAN_SUBJECT)
    service = TeamService(lambda: native.config)
    app = create_app(
        None,
        service=service,
        verifier=_profile_verifier(native.config.nebius_identity, handler),
    )
    with TestClient(app, base_url="https://testserver") as client:
        local = client.get(
            "/v1/me", headers={"Authorization": f"Bearer {native.keys[0][1]}"}
        )
        human = client.get("/v1/me", headers={"Authorization": "Bearer human-token"})
    assert local.status_code == human.status_code == 200
    assert local.json() == human.json()
    assert len(calls) == 1
    human_actor = AccountAuthentication(
        native.config,
        external=_profile_verifier(native.config.nebius_identity, handler),
    ).verify("Bearer human-token")
    assert human_actor == local_actor
    assert bind_execution(
        native.config, human_actor, "robotics", "east"
    ) == bind_execution(native.config, local_actor, "robotics", "east")
    assert human_actor.groups == frozenset({"researchers"})


def test_unlinked_humans_and_foreign_owners_remain_denied(native, workflow):
    def handler(request):
        return httpx.Response(200, json=_profile())

    alice = native.accounts.authenticate(native.keys[0][1])[0]
    binding = bind_execution(native.config, alice, "robotics", "east")
    request = SubmitRequest(
        workspace="robotics",
        cluster="east",
        idempotency_key="owned-by-alice",
        workflow=workflow,
    )
    service = TeamService(lambda: native.config)
    record, _ = service.ledger.create(alice, request, binding_snapshot(binding))
    app = create_app(
        None,
        service=service,
        verifier=_profile_verifier(native.config.nebius_identity, handler),
    )
    with TestClient(app) as client:
        unlinked = client.get("/v1/me", headers={"Authorization": "Bearer human-token"})
        foreign = client.get(
            f"/v1/runs/{record['id']}",
            headers={"Authorization": f"Bearer {native.keys[1][1]}"},
        )
    assert unlinked.status_code == 401
    assert foreign.status_code == 404


def test_local_disablement_denies_linked_human_and_local_key(native):
    def handler(request):
        return httpx.Response(200, json=_profile())

    user_id = native.people[0]["id"]
    link_identity(native.config, user_id, NEBIUS_ISSUER, HUMAN_SUBJECT)
    service = TeamService(lambda: native.config)
    app = create_app(
        None,
        service=service,
        verifier=_profile_verifier(native.config.nebius_identity, handler),
    )
    native.accounts.update(user_id, disabled=True)
    with TestClient(app) as client:
        local = client.get(
            "/v1/me", headers={"Authorization": f"Bearer {native.keys[0][1]}"}
        )
        human = client.get("/v1/me", headers={"Authorization": "Bearer human-token"})
    assert local.status_code == human.status_code == 401


def test_existing_local_account_database_remains_compatible_with_nebius(config):
    previous_policy = config.model_dump(mode="json")
    previous_policy["identity"] = None
    previous_policy["account_namespace"] = str(uuid.uuid4())
    prior = TeamConfig.model_validate(previous_policy)
    accounts = Accounts(prior)
    person = accounts.create("casey", ["researchers"])
    _, key = accounts.issue_key(person["id"])
    existing = accounts.authenticate(key)[0]
    native_policy = prior.model_dump(mode="json")
    native_policy["nebius_identity"] = {"tenant_id": TENANT_ID}
    native = TeamConfig.model_validate(native_policy)
    resumed = Accounts(native).authenticate(key)[0]
    assert resumed == existing


def test_nebius_identity_changes_require_server_restart(native):
    policies = [native.config]
    service = TeamService(lambda: policies[0])
    changed = native.config.model_dump(mode="json")
    changed["nebius_identity"]["federation_id"] = "federation-reconfigured"
    policies[0] = TeamConfig.model_validate(changed)
    with pytest.raises(ConflictError, match="restart the service"):
        service.config()


def test_nebius_verifier_never_falls_back_to_ambient_administrator_tokens(
    native, monkeypatch
):
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "administrator-token")
    observed = []

    def handler(request):
        observed.append(request.headers["Authorization"])
        return httpx.Response(200, json=_profile())

    verifier = _profile_verifier(native.config.nebius_identity, handler)
    with pytest.raises(AuthenticationError, match="valid bearer token"):
        verifier.verify("")
    assert observed == []
    assert verifier.verify("Bearer caller-token").subject == HUMAN_SUBJECT
    assert observed == ["Bearer caller-token"]


def test_link_only_accepts_the_native_provider_and_never_changes_ownership(native):
    owner = native.accounts.authenticate(native.keys[0][1])[0]
    before = copy.deepcopy(native.accounts.list())
    with pytest.raises(TeamError, match="configured external provider"):
        link_identity(
            native.config, owner.subject, "https://other.example.test", HUMAN_SUBJECT
        )
    link_identity(native.config, owner.subject, NEBIUS_ISSUER, HUMAN_SUBJECT)
    with pytest.raises(ConflictError, match="already has an external identity"):
        link_identity(
            native.config,
            owner.subject,
            NEBIUS_ISSUER,
            "tenantuseraccount-second",
        )
    assert native.accounts.authenticate(native.keys[0][1])[0] == owner
    assert native.accounts.list()[0]["id"] == before[0]["id"]


def test_unlink_denies_nebius_login_but_preserves_local_access_and_namespace(native):
    owner = native.accounts.authenticate(native.keys[0][1])[0]
    link_identity(native.config, owner.subject, NEBIUS_ISSUER, HUMAN_SUBJECT)
    verifier = _profile_verifier(
        native.config.nebius_identity, lambda _: httpx.Response(200, json=_profile())
    )
    app = create_app(
        None, service=TeamService(lambda: native.config), verifier=verifier
    )
    before = bind_execution(native.config, owner, "robotics", "east").namespace
    with TestClient(app) as client:
        human = {"Authorization": "Bearer human-token"}
        local = {"Authorization": "Bearer " + native.keys[0][1]}
        assert client.get("/v1/me", headers=human).status_code == 200
        assert unlink_identity(
            native.config, owner.subject, NEBIUS_ISSUER, HUMAN_SUBJECT
        )["unlinked"]
        assert client.get("/v1/me", headers=human).status_code == 401
        assert client.get("/v1/me", headers=local).status_code == 200
    assert native.accounts.authenticate(native.keys[0][1])[0] == owner
    assert bind_execution(native.config, owner, "robotics", "east").namespace == before
