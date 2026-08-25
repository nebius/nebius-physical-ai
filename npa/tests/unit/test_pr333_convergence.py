"""Adversarial convergence regressions for PR #333 lifecycle evidence."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml
from typer.testing import CliRunner

from npa import teardown_receipts
from npa.cli.main import app


runner = CliRunner()


def _write_project_config(
    path: Path,
    *,
    alias: str = "demo",
    project_id: str = "project-a",
    project: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": alias,
                "projects": {
                    alias: {
                        "project_id": project_id,
                        "tenant_id": "tenant-a",
                        **dict(project or {}),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _agent_terminal_verification(*, iam_complete: bool = True) -> dict[str, object]:
    return {
        "exact_instance_absent": True,
        "terraform_destroy_completed": True,
        "terraform_dependency_graph": sorted(
            {
                "compute_instance",
                "boot_disk",
                "network",
                "subnet",
                "security_group",
                "public_ip",
            }
        ),
        "iam_cleanup_complete": iam_complete,
        "iam_disposition": "deleted" if iam_complete else "verification_unresolved",
    }


def _record_terminal_cloud_receipts() -> None:
    common = {"project_alias": "demo", "project_id": "project-a"}
    teardown_receipts.record_teardown_event(
        phase="workflow_audit",
        resource="all-managed-jobs",
        terminal_state="verified_absent",
        action={"kind": "read_only_managed_job_audit"},
        verification={"nonterminal_job_ids": [], "detail": ""},
        **common,
    )
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_deleted",
        identity={
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-a",
        },
        action={"kind": "terraform_agent_destroy", "purge_iam": True},
        verification=_agent_terminal_verification(),
        **common,
    )
    teardown_receipts.record_teardown_event(
        phase="bucket",
        resource="bucket-a",
        terminal_state="verified_absent",
        action={"kind": "none"},
        verification={"bucket_absent": True},
        **common,
    )
    teardown_receipts.record_teardown_event(
        phase="storage_iam",
        resource="storage-account-a",
        terminal_state="verified_absent",
        identity={
            "project_id": "project-a",
            "service_account_id": "storage-account-a",
        },
        action={"kind": "exact_provider_check", "mutation": False},
        verification={
            "provider_outcome": "verified_absent",
            "exact_service_account_absent": True,
        },
        **common,
    )


def test_unresolved_agent_iam_cannot_authorize_dependent_retirement() -> None:
    from npa.cli import cleanup as cleanup_cli

    event = {
        "phase": "agent",
        "resource": "agent",
        "terminal_state": "verified_deleted",
        "action": {"kind": "terraform_agent_destroy", "purge_iam": True},
        "verification": _agent_terminal_verification(iam_complete=False),
        "errors": [],
    }

    assert cleanup_cli._event_authorizes_cloud_absence(event) is False


def test_direct_forget_preserves_project_with_unresolved_agent_iam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    _write_project_config(config_module.CONFIG_PATH)
    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "agent_iam": {
                    "version": 1,
                    "projects": {
                        "project-a": {
                            "status": "partial",
                            "resources": {
                                "service_account": {
                                    "id": "agent-account-a",
                                    "name": "npa-agent",
                                    "project_id": "project-a",
                                    "created_by": "npa",
                                },
                                "access_keys": {},
                            },
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["configure", "--forget-project", "demo"])

    assert result.exit_code != 0
    assert "demo" in yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]
    assert credentials_module.CREDENTIALS_PATH.exists()


def test_unscoped_full_cleanup_fails_closed_on_agent_scope_and_auth_residue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.clients import config as config_module

    _write_project_config(
        config_module.CONFIG_PATH,
        project={
            "agents": {
                "agent": {
                    "schema_version": 1,
                    "project_id": "project-a",
                    "instance_id": "instance-a",
                }
            }
        },
    )
    auth = Path.home() / ".npa" / "agents" / "demo" / "agent" / "auth.env"
    auth.parent.mkdir(parents=True)
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    monkeypatch.setattr(
        cleanup_cli,
        "_storage_iam_full_check",
        lambda *_a, **_k: ("verified absent", False, "fully_clean", "verified_terminal"),
    )
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda _sky: ([], "", "verified_empty"),
    )

    result = runner.invoke(
        app, ["cleanup", "--full", "--yes", "--keep-sky", "--json"]
    )

    assert result.exit_code != 0, result.output
    payload = json.loads(result.output)
    assert payload["result"] != "fully_cleaned"
    assert payload["verification_unresolved"] is True
    assert auth.exists()


def test_failed_terraform_retirement_preserves_credentials_until_final_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.cli.cluster.terraform_runtime import TerraformResidue
    from npa.clients import config as config_module

    _write_project_config(config_module.CONFIG_PATH)
    _record_terminal_cloud_receipts()
    residue_path = Path.home() / ".npa" / "terraform-data" / "cluster" / "owned"
    residue_path.mkdir(parents=True)
    residue = TerraformResidue("Terraform state", residue_path, True)
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_runtime.collect_terraform_residue",
        lambda: [residue],
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_runtime.remove_terraform_residue",
        lambda _item: "injected deletion failure",
    )
    monkeypatch.setattr(
        cleanup_cli,
        "_storage_iam_full_check",
        lambda *_a, **_k: ("verified absent", False, "fully_clean", "verified_terminal"),
    )
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda _sky: ([], "", "verified_empty"),
    )
    cleared: list[str] = []
    monkeypatch.setattr(
        cleanup_cli,
        "_full_credential_labels",
        lambda: ["Hugging Face token"],
    )
    monkeypatch.setattr(
        cleanup_cli,
        "_clear_full_credentials",
        lambda: cleared.append("credentials") or ["Hugging Face token"],
    )

    result = runner.invoke(
        app,
        ["cleanup", "--project", "demo", "--full", "--yes", "--keep-sky", "--json"],
    )

    assert result.exit_code != 0
    assert cleared == []


def test_agent_record_project_must_match_parent_project_stanza() -> None:
    from npa.cli.agent_records import AgentRecordState, decode_agent_record
    from npa.clients import config as config_module

    _write_project_config(
        config_module.CONFIG_PATH,
        project={
            "agents": {
                "agent": {
                    "schema_version": 1,
                    "project_id": "project-b",
                    "instance_id": "instance-a",
                }
            }
        },
    )

    decoded = decode_agent_record("demo", "agent")

    assert decoded.state is AgentRecordState.CONFLICTING
    assert "parent project" in decoded.detail


def test_local_cleanup_receipt_is_nonterminal_when_final_residue_reappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.cli.cluster.terraform_runtime import TerraformResidue

    first_path = Path.home() / ".npa" / "terraform-data" / "cluster" / "first"
    later_path = Path.home() / ".npa" / "terraform-data" / "cluster" / "later"
    first_path.mkdir(parents=True)
    first = TerraformResidue("Terraform state", first_path, True)
    later = TerraformResidue("Terraform state", later_path, True)
    calls = 0

    def inventory() -> list[TerraformResidue]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [first]
        later_path.mkdir(parents=True, exist_ok=True)
        return [later]

    def remove(_item: TerraformResidue) -> str:
        first_path.rmdir()
        return ""

    monkeypatch.setattr(
        "npa.cli.cluster.terraform_runtime.collect_terraform_residue", inventory
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_runtime.remove_terraform_residue", remove
    )
    monkeypatch.setattr(
        cleanup_cli, "_nonterminal_jobs", lambda _sky: ([], "", "verified_empty")
    )

    result = runner.invoke(app, ["cleanup", "--yes", "--keep-sky", "--json"])

    assert result.exit_code != 0, result.output
    event = teardown_receipts.latest_phase_states()["local_cleanup"]
    assert event["terminal_state"] != "completed"
    assert event["verification"]["remaining_terraform_count"] == 1


def test_direct_agent_iam_primitive_rejects_unverified_caller_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_iam import AgentIAMCleanupError, purge_agent_iam

    deleted: list[str] = []
    monkeypatch.setattr(
        "npa.clients.nebius.delete_access_key", lambda key_id: deleted.append(key_id)
    )
    monkeypatch.setattr(
        "npa.clients.nebius.delete_service_account",
        lambda account_id: deleted.append(account_id),
    )

    with pytest.raises(AgentIAMCleanupError, match="authoritative|verified"):
        purge_agent_iam(
            {
                "project_id": "project-a",
                "service_account_id": "agent-account-a",
                "service_account_name": "npa-agent",
                "access_keys": [{"id": "agent-key-a"}],
            },
            on_status=lambda _message: None,
        )

    assert deleted == []


def test_receipt_aggregation_preserves_unresolved_siblings_and_cross_file_conflicts() -> None:
    from npa.cli import cleanup as cleanup_cli

    common = {"project_id": "project-a"}
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="agent-a",
        terminal_state="verification_failed",
        project_alias="alpha",
        identity={"agent_name": "agent-a", "instance_id": "instance-a", **common},
        errors=["IAM unresolved"],
        **common,
    )
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="agent-b",
        terminal_state="verified_deleted",
        project_alias="alpha",
        identity={"agent_name": "agent-b", "instance_id": "instance-b", **common},
        action={"kind": "terraform_agent_destroy", "purge_iam": True},
        verification=_agent_terminal_verification(),
        **common,
    )
    for index in range(3):
        teardown_receipts.record_teardown_event(
            phase=f"padding-{index}",
            resource=f"padding-{index}",
            terminal_state="completed",
            project_alias="alpha",
            **common,
        )
    teardown_receipts.record_teardown_event(
        phase="storage_iam",
        resource="storage-account-a",
        terminal_state="verified_absent",
        project_alias="alpha",
        identity={"service_account_id": "storage-account-a", **common},
        action={"kind": "exact_provider_check", "mutation": False},
        verification={
            "provider_outcome": "verified_absent",
            "exact_service_account_absent": True,
        },
        **common,
    )
    teardown_receipts.record_teardown_event(
        phase="storage_iam",
        resource="storage-account-b",
        terminal_state="verified_deleted",
        project_alias="beta",
        identity={"service_account_id": "storage-account-b", **common},
        action={"kind": "delete_npa_owned_service_account"},
        verification={"provider_outcome": "present"},
        **common,
    )

    states = teardown_receipts.latest_phase_states(project_id="project-a")

    assert states["agent"]["terminal_state"] == "verification_failed"
    assert cleanup_cli._event_authorizes_cloud_absence(states["storage_iam"]) is False


def test_storage_delete_requires_exact_provider_absence_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    _write_project_config(config_module.CONFIG_PATH)
    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "storage-account-a",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                }
            }
        ),
        encoding="utf-8",
    )
    identity = nebius_module.ServiceAccountIdentity(
        account_id="storage-account-a",
        name="lerobot-training",
        project_id="project-a",
        tenant_id="tenant-a",
        profile="",
    )
    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", lambda *_a, **_k: identity
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_id_by_name",
        lambda *_a, **_k: "storage-account-a",
    )
    monkeypatch.setattr(
        nebius_module, "list_access_keys_for_service_account", lambda *_a, **_k: []
    )
    monkeypatch.setattr(nebius_module, "delete_service_account", lambda *_a, **_k: None)

    result = runner.invoke(
        app,
        ["storage", "service-account", "delete", "--project", "demo", "--yes"],
    )

    assert result.exit_code != 0, result.output
    assert credentials_module.CREDENTIALS_PATH.exists()
    events = teardown_receipts.latest_phase_states(project_id="project-a")
    assert not cleanup_cli_event_authorizes_storage(events.get("storage_iam", {}))


def cleanup_cli_event_authorizes_storage(event: dict[str, object]) -> bool:
    from npa.cli import cleanup as cleanup_cli

    return cleanup_cli._event_authorizes_cloud_absence(event)


def test_conflicting_provider_vm_service_account_paths_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_iam import _provider_agent_dependents
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "_run_json",
        lambda *_a, **_k: {
            "items": [
                {
                    "metadata": {"id": "instance-a", "name": "agent-a"},
                    "spec": {
                        "account": {
                            "service_account": {"id": "agent-account-a"},
                            "service_account_id": "agent-account-b",
                        },
                        "service_account_id": "agent-account-a",
                    },
                }
            ]
        },
    )

    with pytest.raises(nebius_module.NebiusError, match="conflict|disagree"):
        _provider_agent_dependents("project-a", "agent-account-a")


def test_replacement_agent_key_blocks_service_account_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_iam import AgentIAMCleanupError, purge_agent_iam

    authoritative = {
        "project_id": "project-a",
        "service_account_id": "agent-account-a",
        "service_account_name": "npa-agent",
        "access_keys": [{"id": "agent-key-a"}],
        "owned_by_npa": True,
        "inventory_verified": True,
        "inventory_error": "",
        "dependents": [],
    }
    monkeypatch.setattr(
        "npa.cli.agent_iam.agent_iam_leftovers", lambda _project: authoritative
    )
    calls: list[str] = []
    monkeypatch.setattr(
        "npa.clients.nebius.delete_access_key",
        lambda key_id: calls.append(f"key:{key_id}"),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.list_access_keys_for_service_account",
        lambda *_a, **_k: [{"id": "replacement-key"}],
    )
    monkeypatch.setattr(
        "npa.clients.nebius.delete_service_account",
        lambda account_id: calls.append(f"account:{account_id}"),
    )

    with pytest.raises(AgentIAMCleanupError, match="replacement|remaining|empty"):
        purge_agent_iam(authoritative, on_status=lambda _message: None)

    assert calls == ["key:agent-key-a"]


def test_agent_status_later_failure_wins_over_older_complete_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import agent_status
    from npa.clients import config as config_module

    _write_project_config(config_module.CONFIG_PATH)
    common = {
        "phase": "agent",
        "resource": "agent",
        "project_alias": "demo",
        "project_id": "project-a",
        "identity": {
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-a",
        },
    }
    teardown_receipts.record_teardown_event(
        terminal_state="verified_deleted",
        action={"kind": "terraform_agent_destroy", "purge_iam": True},
        verification=_agent_terminal_verification(),
        **common,
    )
    teardown_receipts.record_teardown_event(
        terminal_state="verification_failed",
        action={"kind": "terraform_agent_destroy", "purge_iam": True},
        verification={
            **_agent_terminal_verification(),
            "iam_cleanup_complete": False,
            "iam_disposition": "verification_unresolved",
        },
        errors=["later provider verification failed"],
        **common,
    )

    result = agent_status.partial_agent_status("demo", "agent")

    assert result["classification"] != "VERIFIED_ABSENT"
    assert result["current_verification"] != "terminal_exact_agent_receipt"
