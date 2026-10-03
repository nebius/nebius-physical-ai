"""A concurrent agent deploy must not lose shared credentials during teardown."""

from __future__ import annotations

import pytest

from npa.cli import agent_iam
from npa.clients import agent_iam_binding, credentials, nebius


@pytest.fixture
def owned_account(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", tmp_path / "credentials.yaml")
    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    agent_iam.record_agent_iam_resource(
        "project-test",
        "access_key",
        {"id": "key-test", "service_account_id": "account-test"},
    )
    monkeypatch.setattr(
        nebius, "get_service_account_id_by_name", lambda *a, **kw: "account-test"
    )
    monkeypatch.setattr(
        nebius,
        "list_access_keys_for_service_account",
        lambda *a, **kw: [{"id": "key-test", "service_account_id": "account-test"}],
    )

    # Every provider boundary is isolated. Returned state below models a peer
    # finishing deployment between initial teardown inventory and its recheck.
    def forbid_provider(*args, **kwargs):
        raise AssertionError("unexpected real provider access")

    monkeypatch.setattr(nebius, "_run_json", forbid_provider)
    mutations = []
    monkeypatch.setattr(
        nebius, "delete_access_key", lambda key: mutations.append(("key", key))
    )
    monkeypatch.setattr(agent_iam, "_verify_access_key_absent", lambda key: None)
    monkeypatch.setattr(
        agent_iam_binding,
        "remove_created_agent_account",
        lambda *args: mutations.append(("account", args[-1])),
    )
    return mutations


@pytest.mark.parametrize("recheck", ["peer", "unreadable"])
def test_late_dependency_or_unreadable_inventory_retains_storage_keys(
    owned_account, monkeypatch, recheck
):
    calls = []

    def dependents(project, account):
        calls.append((project, account))
        if len(calls) == 1:
            return []
        if recheck == "unreadable":
            raise nebius.NebiusError("PermissionDenied: compute inventory unavailable")
        return ["concurrent-peer (instance-test)"]

    monkeypatch.setattr(agent_iam, "_provider_agent_dependents", dependents)
    monkeypatch.setattr(
        agent_iam,
        "_receipt_proves_agent_graphs_absent",
        lambda *args: (False, "no exact graph receipt"),
    )
    with pytest.raises(agent_iam.AgentIAMCleanupError):
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            strict=True,
            on_status=lambda message: None,
        )
    assert len(calls) >= 2, "test must reach the final provider dependency recheck"
    assert owned_account == [], (
        "a late peer or unverified inventory must preserve its keys"
    )
    assert "key-test" in agent_iam._recorded_access_keys("project-test")


def test_verified_last_account_cleanup_deletes_key_then_account(
    owned_account, monkeypatch
):
    monkeypatch.setattr(agent_iam, "_provider_agent_dependents", lambda *args: [])
    agent_iam.report_agent_iam(
        project_id="project-test",
        remaining_agents=0,
        purge=True,
        strict=True,
        on_status=lambda message: None,
    )
    assert owned_account == [("key", "key-test"), ("account", "account-test")]
