"""Durable project binding rollback and final-agent teardown behavior."""

import json
import os
import subprocess

import pytest

from npa.cli import agent_iam
from npa.clients import agent_iam_binding, credentials, nebius


@pytest.fixture(autouse=True)
def isolate_provider_boundaries(mocker):
    mocker.patch.object(agent_iam_binding, "verify_agent_project_scope")
    mocker.patch.object(
        agent_iam_binding,
        "remove_created_agent_account",
        side_effect=lambda project, tenant, account: nebius.delete_service_account(
            account
        ),
    )
    mocker.patch.object(
        nebius,
        "_run_json",
        side_effect=AssertionError("unexpected provider call in wrapper test"),
    )


def binding_record(kind="agent_group"):
    value = {
        "id": kind + "-test",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "service_account_id": "account-test",
        "group_name": "npa-agent-project-editors",
        "role": "editor",
        "created_by": "npa",
        "ownership_source": "provider-create-response",
    }
    if kind != "agent_group":
        value["group_id"] = "agent_group-test"
    return value


def recorded_binding(kind="agent_group"):
    record = binding_record(kind)
    agent_iam.record_agent_iam_resource("project-test", kind, record)
    return {kind: {record["id"]: record}}


def stub_account(mocker, account="account-test", owned=True):
    mocker.patch.object(nebius, "get_service_account_id_by_name", return_value=account)
    mocker.patch.object(nebius, "list_access_keys_for_service_account", return_value=[])
    mocker.patch.object(agent_iam, "agent_iam_owned", return_value=owned)
    return mocker.patch.object(agent_iam, "_provider_agent_dependents", return_value=[])


def test_no_tenant_permission_fallback_on_reused_storage_failure(mocker):
    mocker.patch.object(
        nebius, "get_service_account_id_by_name", return_value="account-test"
    )
    mocker.patch.object(nebius, "ensure_service_account", return_value="account-test")
    tenant = mocker.patch.object(nebius, "ensure_editors_membership")
    fallback = mocker.patch.object(nebius, "_saved_storage_credentials")
    mocker.patch.object(
        agent_iam_binding,
        "ensure_agent_project_binding",
        side_effect=nebius.NebiusError("PermissionDenied"),
    )
    delete_account = mocker.patch.object(nebius, "delete_service_account")
    with pytest.raises(nebius.NebiusError, match="PermissionDenied"):
        nebius.bootstrap_agent_environment(
            "project-test",
            "tenant-test",
            "region-test",
            reuse_storage_credentials={"s3_bucket": "test-bucket"},
        )
    fallback.assert_not_called()
    tenant.assert_not_called()
    delete_account.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [nebius.NebiusError("PermissionDenied"), OSError("local persistence failure")],
)
def test_created_bindings_roll_back_before_account(mocker, failure):
    mocker.patch.object(nebius, "get_service_account_id_by_name", return_value=None)

    def account(*args, on_created, **kwargs):
        on_created("account-test")
        return "account-test"

    mocker.patch.object(nebius, "ensure_service_account", side_effect=account)

    def grant(**kwargs):
        kwargs["on_resource_created"]("agent_group", binding_record())
        raise failure

    mocker.patch.object(
        agent_iam_binding, "ensure_agent_project_binding", side_effect=grant
    )
    calls = []

    def cleanup(project, records, *, on_removed):
        assert "agent_group-test" in records["agent_group"]
        assert agent_iam.agent_iam_binding_resources(project)["agent_group"]
        calls.append("binding")
        on_removed("agent_group", "agent_group-test")
        return ["agent_group"]

    mocker.patch.object(
        agent_iam_binding, "cleanup_agent_project_binding", side_effect=cleanup
    )
    mocker.patch.object(
        nebius,
        "delete_service_account",
        side_effect=lambda identity: calls.append("account"),
    )
    with pytest.raises(type(failure)):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    assert calls == ["binding", "account"]
    assert not agent_iam.agent_iam_binding_resources("project-test")
    assert not agent_iam.agent_iam_owned("project-test", "account-test")


def test_failed_binding_rollback_retains_account_and_exact_journal(mocker):
    mocker.patch.object(nebius, "get_service_account_id_by_name", return_value=None)

    def account(*args, on_created, **kwargs):
        on_created("account-test")
        return "account-test"

    mocker.patch.object(nebius, "ensure_service_account", side_effect=account)

    def grant(**kwargs):
        kwargs["on_resource_created"]("agent_group", binding_record())
        raise nebius.NebiusError("grant verification failed")

    mocker.patch.object(
        agent_iam_binding, "ensure_agent_project_binding", side_effect=grant
    )
    mocker.patch.object(
        agent_iam_binding,
        "cleanup_agent_project_binding",
        side_effect=nebius.NebiusError("rollback denied"),
    )
    delete = mocker.patch.object(nebius, "delete_service_account")
    with pytest.raises(nebius.NebiusError, match="grant verification"):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    delete.assert_not_called()
    assert agent_iam.agent_iam_owned("project-test", "account-test")
    data, path = agent_iam._agent_iam_records()
    assert data["agent_iam"]["projects"]["project-test"]["status"] == "partial"
    assert agent_iam.agent_iam_binding_resources("project-test")["agent_group"]
    assert path.stat().st_mode & 0o777 == 0o600


def test_external_operation_receives_id_even_when_credential_journal_write_fails(
    mocker,
):
    mocker.patch.object(nebius, "get_service_account_id_by_name", return_value=None)
    mocker.patch.object(
        agent_iam, "record_agent_iam_resource", side_effect=OSError("write failed")
    )
    delete = mocker.patch.object(nebius, "delete_service_account")
    external = mocker.Mock()

    def account(*args, on_created, **kwargs):
        on_created("account-test")
        return "account-test"

    mocker.patch.object(nebius, "ensure_service_account", side_effect=account)
    with pytest.raises(OSError, match="write failed"):
        nebius.bootstrap_agent_environment(
            "project-test",
            "tenant-test",
            "region-test",
            reuse_storage_credentials={},
            on_resource_created=external,
        )
    external.assert_called_once_with(
        "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    delete.assert_called_once_with("account-test")


def test_unwritable_creation_journal_blocks_provider_creation(mocker):
    mocker.patch.object(nebius, "get_service_account_id_by_name", return_value=None)
    mocker.patch.object(
        agent_iam,
        "preflight_agent_iam_journal",
        side_effect=OSError("journal unavailable"),
    )
    account = mocker.patch.object(nebius, "ensure_service_account")
    with pytest.raises(OSError, match="journal unavailable"):
        nebius.bootstrap_agent_environment(
            "project-test", "tenant-test", "region-test", reuse_storage_credentials={}
        )
    account.assert_not_called()


@pytest.mark.parametrize("kind", ["symlink", "fifo", "invalid-yaml", "invalid-journal"])
def test_invalid_journal_fails_closed_without_hanging(tmp_path, monkeypatch, kind):
    path = tmp_path / "private" / "credentials.yaml"
    path.parent.mkdir(mode=0o700)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("{}")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_text("a: [" if kind == "invalid-yaml" else "agent_iam: malformed\n")
    with pytest.raises(
        (
            OSError,
            ValueError,
            agent_iam.AgentIAMCleanupError,
            credentials.CredentialStoreError,
        )
    ):
        recorded_binding()


def test_concurrent_process_journals_preserve_both_creations(tmp_path, monkeypatch):
    path = tmp_path / "private" / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    agent_iam.preflight_agent_iam_journal()
    base = binding_record()
    script = """
import json, sys
from pathlib import Path
from npa.clients import credentials
from npa.cli.agent_iam import record_agent_iam_resource
credentials.CREDENTIALS_PATH = Path(sys.argv[1])
record_agent_iam_resource('project-test', 'agent_group', json.loads(sys.argv[2]))
"""
    import sys

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                json.dumps({**base, "id": f"group-{index}"}),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(4)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
    assert set(
        agent_iam.agent_iam_binding_resources("project-test")["agent_group"]
    ) == {f"group-{index}" for index in range(4)}


def test_account_removal_preserves_unreconciled_binding_receipts():
    recorded_binding()
    agent_iam.record_agent_iam_resource(
        "project-test", "service_account", {"id": "account-test", "name": "npa-agent"}
    )
    assert agent_iam.clear_agent_iam_record("project-test", "account-test")
    assert agent_iam.agent_iam_binding_resources("project-test")["agent_group"]
    assert not agent_iam.agent_iam_owned("project-test", "account-test")


def test_teardown_deletes_bindings_before_owned_account(mocker):
    recorded_binding()
    stub_account(mocker)
    calls = []
    mocker.patch.object(
        agent_iam_binding,
        "cleanup_agent_project_binding",
        side_effect=lambda *args, **kwargs: calls.append("binding") or ["agent_group"],
    )
    mocker.patch.object(
        nebius,
        "delete_service_account",
        side_effect=lambda identity: calls.append("account"),
    )
    agent_iam.report_agent_iam(
        project_id="project-test",
        remaining_agents=0,
        purge=True,
        on_status=lambda message: None,
        strict=True,
    )
    assert calls == ["binding", "account"]


@pytest.mark.parametrize("peer", ["local", "provider", "late-provider"])
def test_teardown_retains_bindings_for_any_dependent_agent(mocker, peer):
    recorded_binding()
    dependents = stub_account(mocker)
    if peer == "provider":
        dependents.return_value = ["peer"]
    elif peer == "late-provider":
        dependents.side_effect = [[], ["peer"]]
    cleanup = mocker.patch.object(agent_iam_binding, "cleanup_agent_project_binding")
    delete = mocker.patch.object(nebius, "delete_service_account")
    try:
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=1 if peer == "local" else 0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
    except agent_iam.AgentIAMCleanupError:
        assert peer != "local"
    cleanup.assert_not_called()
    delete.assert_not_called()


@pytest.mark.parametrize("absent", [True, False])
def test_owned_bindings_can_clean_up_with_absent_or_reused_account(mocker, absent):
    recorded_binding()
    dependents = stub_account(
        mocker, account="" if absent else "account-test", owned=False
    )
    cleanup = mocker.patch.object(
        agent_iam_binding, "cleanup_agent_project_binding", return_value=["agent_group"]
    )
    delete = mocker.patch.object(nebius, "delete_service_account")
    deleted = agent_iam.report_agent_iam(
        project_id="project-test",
        remaining_agents=0,
        purge=True,
        on_status=lambda message: None,
    )
    assert deleted == ["agent_group"]
    assert all(
        call.args == ("project-test", "account-test")
        for call in dependents.call_args_list
    )
    cleanup.assert_called_once()
    delete.assert_not_called()


def test_failed_binding_cleanup_preserves_account(mocker):
    recorded_binding()
    stub_account(mocker)
    mocker.patch.object(
        agent_iam_binding,
        "cleanup_agent_project_binding",
        side_effect=nebius.NebiusError("shared group"),
    )
    delete = mocker.patch.object(nebius, "delete_service_account")
    with pytest.raises(agent_iam.AgentIAMCleanupError, match="partial"):
        agent_iam.report_agent_iam(
            project_id="project-test",
            remaining_agents=0,
            purge=True,
            on_status=lambda message: None,
            strict=True,
        )
    delete.assert_not_called()
    assert agent_iam.agent_iam_binding_resources("project-test")["agent_group"]


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [], "next_page_token": "another-page"},
        {"items": [], "next_page_token": None},
        {
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "other-project",
                    },
                    "spec": {},
                }
            ]
        },
        {
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "project-test",
                    },
                    "spec": {"account": "malformed"},
                }
            ]
        },
        {
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "project-test",
                    },
                    "spec": {"account": {"service_account": "malformed"}},
                }
            ]
        },
        {
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "project-test",
                    },
                    "spec": {"service_account_id": 123},
                }
            ]
        },
        {
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "project-test",
                    },
                    "spec": {
                        "account": {"service_account": {"id": "account-test"}},
                        "service_account_id": "another-account",
                    },
                }
            ]
        },
    ],
)
def test_incomplete_or_ambiguous_compute_inventory_never_proves_absence(
    mocker, payload
):
    mocker.patch.object(nebius, "_run_json", return_value=payload)
    with pytest.raises(nebius.NebiusError):
        agent_iam._provider_agent_dependents("project-test", "account-test")


def test_complete_compute_inventory_distinguishes_unattached_and_attached_vms(mocker):
    mocker.patch.object(
        nebius,
        "_run_json",
        return_value={
            "items": [
                {
                    "metadata": {
                        "id": "vm-empty",
                        "name": "unattached",
                        "parent_id": "project-test",
                    },
                    "spec": {},
                },
                {
                    "metadata": {
                        "id": "vm-attached",
                        "name": "attached",
                        "parent_id": "project-test",
                    },
                    "spec": {"account": {"service_account": {"id": "account-test"}}},
                },
            ]
        },
    )
    assert agent_iam._provider_agent_dependents("project-test", "account-test") == [
        "attached (vm-attached)"
    ]


@pytest.mark.parametrize(
    "identity", [" account-test ", "account test", "-account-test"]
)
def test_nonexact_attachment_ids_fail_closed(mocker, identity):
    mocker.patch.object(
        nebius,
        "_run_json",
        return_value={
            "items": [
                {
                    "metadata": {
                        "id": "vm-test",
                        "name": "vm",
                        "parent_id": "project-test",
                    },
                    "spec": {"service_account_id": identity},
                },
            ]
        },
    )
    with pytest.raises(nebius.NebiusError, match="non-exact"):
        agent_iam._provider_agent_dependents("project-test", "account-test")


@pytest.mark.parametrize(
    "message",
    ["metadata.id is not found", "PermissionDenied: NotFound", "response missing id"],
)
def test_key_projection_failure_is_not_resource_absence(mocker, message):
    mocker.patch.object(
        nebius, "_access_key_metadata_scalar", side_effect=nebius.NebiusError(message)
    )
    with pytest.raises(nebius.NebiusError):
        agent_iam._verify_access_key_absent("key-test")


def test_provider_omitted_empty_compute_items_proves_terminal_empty_inventory(mocker):
    mocker.patch.object(nebius, "_run_json", return_value={})
    assert agent_iam._provider_agent_dependents("project-test", "account-test") == []


@pytest.mark.parametrize(
    "payload",
    [
        {"items": None},
        {"error": "denied"},
        {"unexpected": []},
        {"next_page_token": "next"},
        {"next_page_token": None},
    ],
)
def test_omitted_compute_default_does_not_accept_malformed_or_partial_pages(
    mocker, payload
):
    mocker.patch.object(nebius, "_run_json", return_value=payload)
    with pytest.raises(nebius.NebiusError):
        agent_iam._provider_agent_dependents("project-test", "account-test")
