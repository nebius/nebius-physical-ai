"""Adversarial convergence regressions for PR #333 lifecycle evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest
import yaml
from typer.testing import CliRunner

from npa import teardown_receipts
from npa.cli.main import app
from npa.project_destroy import DestroyPhase


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
        "local_state_retired": True,
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


def test_agent_receipt_without_local_retirement_cannot_authorize_convergence() -> None:
    from npa.teardown_receipts import teardown_event_authorizes_convergence

    verification = _agent_terminal_verification()
    verification.pop("local_state_retired")
    event = {
        "phase": "agent",
        "resource": "agent",
        "terminal_state": "verified_deleted",
        "project_id": "project-a",
        "identity": {
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-a",
        },
        "action": {"kind": "terraform_agent_destroy", "purge_iam": True},
        "verification": verification,
        "errors": [],
    }

    assert teardown_event_authorizes_convergence(event) is False


def test_unresolved_agent_iam_blocks_project_destroy_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import project_destroy
    from npa.clients import config as config_module

    _write_project_config(config_module.CONFIG_PATH)
    order: list[str] = []
    phases = [
        DestroyPhase("workflows", (("npa", "workflow-list"),), "workflows"),
        DestroyPhase(
            "agents", (("npa", "agent-destroy"),), "agents", ("workflows",)
        ),
        DestroyPhase(
            "controller",
            (("npa", "controller-delete"),),
            "controller",
            ("workflows", "agents"),
        ),
        DestroyPhase(
            "clusters",
            (("npa", "cluster-delete"),),
            "clusters",
            ("workflows", "controller"),
        ),
        DestroyPhase(
            "bucket",
            (("npa", "bucket-delete"),),
            "bucket",
            ("workflows", "agents", "controller", "clusters"),
        ),
        DestroyPhase(
            "storage_iam",
            (("npa", "storage-iam-delete"),),
            "storage IAM",
            ("workflows", "agents", "controller", "clusters", "bucket"),
        ),
        DestroyPhase(
            "delete_project",
            (),
            "project",
            (
                "workflows",
                "agents",
                "controller",
                "clusters",
                "bucket",
                "storage_iam",
            ),
        ),
        DestroyPhase(
            "forget_alias",
            (("npa", "forget-alias"),),
            "config",
            (
                "workflows",
                "agents",
                "controller",
                "clusters",
                "bucket",
                "storage_iam",
            ),
        ),
    ]

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        order.append(command[1])
        if command[1] == "workflow-list":
            return subprocess.CompletedProcess(
                command, 0, stdout='{"runs": []}', stderr=""
            )
        if command[1] == "agent-destroy":
            return subprocess.CompletedProcess(
                command,
                2,
                stdout=json.dumps(
                    {
                        "infrastructure_absent": True,
                        "iam_cleanup_complete": False,
                    }
                ),
                stderr="agent IAM remains",
            )
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(
        project_destroy,
        "_delete_owned_empty_project",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("project deletion must remain blocked")
        ),
    )

    result = project_destroy.execute_project_destroy("demo", phases, runner=run)

    assert order == ["workflow-list", "agent-destroy"]
    statuses = {item["phase"]: item["status"] for item in result["phases"]}
    assert statuses == {
        "workflows": "completed",
        "agents": "partial",
        "controller": "skipped_dependency",
        "clusters": "skipped_dependency",
        "bucket": "skipped_dependency",
        "storage_iam": "skipped_dependency",
        "delete_project": "skipped_dependency",
        "forget_alias": "skipped_dependency",
    }
    assert result["status"] == "partial"


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


def test_final_cleanup_receipt_failure_restores_credentials_and_project_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients.project_credential_store import write_project_credentials

    _write_project_config(config_module.CONFIG_PATH)
    write_project_credentials(
        "project-a",
        {
            "storage": {
                "aws_access_key_id": "fixture-access-key",
                "aws_secret_access_key": "fixture-secret-key",
            }
        },
        alias="demo",
    )
    saved = yaml.safe_load(
        credentials_module.CREDENTIALS_PATH.read_text(encoding="utf-8")
    )
    saved["tokens"] = {"HF_TOKEN": "fixture-hf-token"}
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(saved), encoding="utf-8"
    )
    _record_terminal_cloud_receipts()
    monkeypatch.setattr(
        cleanup_cli,
        "_storage_iam_full_check",
        lambda *_a, **_k: ("verified absent", False, "fully_clean", "verified_terminal"),
    )
    monkeypatch.setattr(
        cleanup_cli, "_nonterminal_jobs", lambda _sky: ([], "", "verified_empty")
    )
    real_record = teardown_receipts.record_teardown_event
    local_receipt_calls = 0

    def fail_final_local_receipt(**kwargs):  # noqa: ANN003, ANN202
        nonlocal local_receipt_calls
        if kwargs.get("phase") == "local_cleanup":
            local_receipt_calls += 1
            if local_receipt_calls == 2:
                raise OSError("injected final receipt failure")
        return real_record(**kwargs)

    monkeypatch.setattr(
        teardown_receipts, "record_teardown_event", fail_final_local_receipt
    )

    result = runner.invoke(
        app,
        ["cleanup", "--project", "demo", "--full", "--yes", "--keep-sky", "--json"],
    )

    assert result.exit_code == 1, result.output
    assert local_receipt_calls == 2
    retained = yaml.safe_load(
        credentials_module.CREDENTIALS_PATH.read_text(encoding="utf-8")
    )
    assert retained["tokens"]["HF_TOKEN"] == "fixture-hf-token"
    assert "project-a" in retained["project_credentials"]["projects"]
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert "demo" in configured["projects"]
    event = teardown_receipts.latest_phase_states(
        project_alias="demo", project_id="project-a"
    )["local_cleanup"]
    assert event["terminal_state"] == "in_progress"


def test_project_config_retirement_failure_restores_project_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients.project_credential_store import write_project_credentials

    _write_project_config(config_module.CONFIG_PATH)
    write_project_credentials(
        "project-a",
        {
            "storage": {
                "aws_access_key_id": "fixture-access-key",
                "aws_secret_access_key": "fixture-secret-key",
            }
        },
        alias="demo",
    )
    _record_terminal_cloud_receipts()
    monkeypatch.setattr(
        cleanup_cli,
        "_storage_iam_full_check",
        lambda *_a, **_k: ("verified absent", False, "fully_clean", "verified_terminal"),
    )
    monkeypatch.setattr(
        cleanup_cli, "_nonterminal_jobs", lambda _sky: ([], "", "verified_empty")
    )
    monkeypatch.setattr(
        config_module,
        "forget_project",
        lambda _alias: (_ for _ in ()).throw(OSError("injected config write failure")),
    )

    result = runner.invoke(
        app,
        ["cleanup", "--project", "demo", "--full", "--yes", "--keep-sky", "--json"],
    )

    assert result.exit_code == 1, result.output
    retained = yaml.safe_load(
        credentials_module.CREDENTIALS_PATH.read_text(encoding="utf-8")
    )
    assert "project-a" in retained["project_credentials"]["projects"]
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert "demo" in configured["projects"]


def test_agent_terraform_delete_failure_preserves_auth_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_local_state import (
        AgentLocalRetirementError,
        cleanup_agent_local_files,
    )
    from npa.deploy import provisioner

    agent_dir = Path.home() / ".npa" / "agents" / "demo" / "agent"
    agent_dir.mkdir(parents=True)
    auth = agent_dir / "auth.env"
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    tf_dir = provisioner.working_dir_path("demo", "agent")
    tf_dir.mkdir(parents=True)
    state = tf_dir / "terraform.tfstate"
    state.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "npa.cli.agent_local_state.shutil.rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("injected deletion failure")),
    )

    with pytest.raises(AgentLocalRetirementError, match="injected deletion failure"):
        cleanup_agent_local_files("demo", "agent")

    assert auth.exists()
    assert state.exists()


def test_agent_credential_retirement_restores_auth_after_partial_directory_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_local_state import (
        AgentLocalRetirementError,
        cleanup_agent_local_files,
    )
    from npa.deploy import provisioner

    agent_dir = Path.home() / ".npa" / "agents" / "demo" / "agent"
    agent_dir.mkdir(parents=True)
    auth = agent_dir / "auth.env"
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    recovery = agent_dir / "recovery.json"
    recovery.write_text('{"instance_id":"instance-a"}\n', encoding="utf-8")
    tf_dir = provisioner.working_dir_path("demo", "agent")
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}\n", encoding="utf-8")
    real_rmtree = __import__("shutil").rmtree

    def partially_delete_then_fail(path: Path) -> None:
        if Path(path) == agent_dir:
            auth.unlink()
            raise OSError("injected partial credential-directory deletion")
        real_rmtree(path)

    monkeypatch.setattr(
        "npa.cli.agent_local_state.shutil.rmtree", partially_delete_then_fail
    )

    with pytest.raises(
        AgentLocalRetirementError,
        match="injected partial credential-directory deletion",
    ):
        cleanup_agent_local_files("demo", "agent")

    assert auth.read_text(encoding="utf-8") == "AGENT_PASSWORD=fixture-only\n"
    assert recovery.exists()


def test_agent_credential_retirement_restores_recovery_after_parent_prune_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_local_state import (
        AgentLocalRetirementError,
        cleanup_agent_local_files,
    )
    from npa.deploy import provisioner

    agent_dir = Path.home() / ".npa" / "agents" / "demo" / "agent"
    agent_dir.mkdir(parents=True)
    auth = agent_dir / "auth.env"
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    recovery = agent_dir / "recovery.json"
    recovery.write_text('{"instance_id":"instance-a"}\n', encoding="utf-8")
    tf_dir = provisioner.working_dir_path("demo", "agent")
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}\n", encoding="utf-8")
    real_rmdir = Path.rmdir

    def fail_agent_parent_prune(path: Path) -> None:
        if path == agent_dir.parent:
            raise OSError("injected empty-parent prune failure")
        real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_agent_parent_prune)

    with pytest.raises(
        AgentLocalRetirementError,
        match="injected empty-parent prune failure",
    ):
        cleanup_agent_local_files("demo", "agent")

    assert auth.read_text(encoding="utf-8") == "AGENT_PASSWORD=fixture-only\n"
    assert recovery.read_text(encoding="utf-8") == (
        '{"instance_id":"instance-a"}\n'
    )
    assert agent_dir.is_dir()


def test_agent_destroy_iam_failure_preserves_auth_and_agent_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.cli.agent_iam import AgentIAMCleanupError
    from npa.clients import config as config_module
    from npa.provisioning_journal import ProvisioningOperation

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
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo --name agent",
    )
    auth = Path.home() / ".npa" / "agents" / "demo" / "agent" / "auth.env"
    auth.parent.mkdir(parents=True)
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.cli.agent_iam.report_destroyed_agent_iam",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AgentIAMCleanupError("injected IAM convergence failure")
        ),
    )

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "demo", "--yes", "--json"]
    )

    assert result.exit_code == 2, result.output
    assert auth.exists()
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["agents"]["agent"]["instance_id"] == (
        "instance-a"
    )
    event = teardown_receipts.latest_phase_states(project_id="project-a")["agent"]
    assert event["terminal_state"] == "partial"


def test_agent_destroy_terminal_receipt_failure_preserves_auth_and_agent_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module
    from npa.provisioning_journal import ProvisioningOperation

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
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo --name agent",
    )
    auth = Path.home() / ".npa" / "agents" / "demo" / "agent" / "auth.env"
    auth.parent.mkdir(parents=True)
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    iam_calls: list[str] = []
    monkeypatch.setattr(
        "npa.cli.agent_iam.report_destroyed_agent_iam",
        lambda *_a, **_k: iam_calls.append("verified") or "deleted",
    )
    real_record = teardown_receipts.record_teardown_event
    states: list[str] = []

    def fail_terminal_agent_receipt(**kwargs):  # noqa: ANN003, ANN202
        if kwargs.get("phase") == "agent":
            state = str(kwargs.get("terminal_state") or "")
            states.append(state)
            if state == "verified_deleted":
                raise OSError("injected terminal agent receipt failure")
        return real_record(**kwargs)

    monkeypatch.setattr(
        teardown_receipts, "record_teardown_event", fail_terminal_agent_receipt
    )

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "demo", "--yes", "--json"]
    )

    assert result.exit_code != 0, result.output
    assert iam_calls == ["verified"]
    assert states == ["in_progress", "verified_deleted", "partial"]
    assert auth.exists()
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["agents"]["agent"]["instance_id"] == (
        "instance-a"
    )
    latest = teardown_receipts.latest_phase_states(project_id="project-a")["agent"]
    assert latest["terminal_state"] == "partial"


def test_agent_destroy_interruption_cannot_leave_authoritative_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module
    from npa.provisioning_journal import ProvisioningOperation
    from npa.teardown_receipts import teardown_event_authorizes_convergence

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
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo --name agent",
    )
    auth = Path.home() / ".npa" / "agents" / "demo" / "agent" / "auth.env"
    auth.parent.mkdir(parents=True)
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.cli.agent_iam.report_destroyed_agent_iam", lambda *_a, **_k: "deleted"
    )
    monkeypatch.setattr(
        agent_module,
        "_cleanup_agent_local_files",
        lambda *_a, **_k: (_ for _ in ()).throw(
            KeyboardInterrupt("injected local retirement interruption")
        ),
    )

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "demo", "--yes", "--json"]
    )

    assert result.exit_code != 0, result.output
    assert auth.exists()
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["agents"]["agent"]["instance_id"] == (
        "instance-a"
    )
    event = teardown_receipts.latest_phase_states(project_id="project-a")["agent"]
    assert event["terminal_state"] != "verified_deleted"
    assert teardown_event_authorizes_convergence(event) is False


def test_cleanup_agent_receipt_failure_restores_credentials_and_saved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.clients import config as config_module
    from npa.teardown_receipts import teardown_event_authorizes_convergence

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
    recovery = auth.parent / "recovery.json"
    recovery.write_text('{"instance_id":"instance-a"}\n', encoding="utf-8")
    verification = _agent_terminal_verification()
    verification.pop("local_state_retired")
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_deleted",
        project_alias="demo",
        project_id="project-a",
        identity={
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-a",
        },
        action={"kind": "terraform_agent_destroy", "purge_iam": True},
        verification=verification,
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    real_record = teardown_receipts.record_teardown_event

    def fail_local_terminal_receipt(**kwargs):  # noqa: ANN003, ANN202
        event_verification = kwargs.get("verification")
        if (
            kwargs.get("phase") == "agent"
            and kwargs.get("terminal_state") == "verified_deleted"
            and isinstance(event_verification, dict)
            and event_verification.get("local_state_retired") is True
        ):
            raise OSError("injected cleanup terminal receipt failure")
        return real_record(**kwargs)

    monkeypatch.setattr(
        teardown_receipts, "record_teardown_event", fail_local_terminal_receipt
    )

    safe, detail = cleanup_cli._agent_lifecycle_allows_project_retirement(
        "demo", "project-a", retire=True
    )

    assert safe is False
    assert "receipt failure" in detail
    assert auth.read_text(encoding="utf-8") == "AGENT_PASSWORD=fixture-only\n"
    assert recovery.exists()
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["agents"]["agent"]["instance_id"] == (
        "instance-a"
    )
    event = teardown_receipts.latest_phase_states(project_id="project-a")["agent"]
    assert event["terminal_state"] == "partial"
    assert teardown_event_authorizes_convergence(event) is False


def test_agent_record_retirement_failure_cannot_delete_auth_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module
    from npa.provisioning_journal import ProvisioningOperation

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
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo --name agent",
    )
    auth = Path.home() / ".npa" / "agents" / "demo" / "agent" / "auth.env"
    auth.parent.mkdir(parents=True)
    auth.write_text("AGENT_PASSWORD=fixture-only\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.cli.agent_iam.report_destroyed_agent_iam",
        lambda *_a, **_k: "deleted",
    )
    monkeypatch.setattr(
        agent_module,
        "_remove_agent_record",
        lambda *_a, **_k: (_ for _ in ()).throw(
            OSError("injected agent-record write failure")
        ),
    )

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "demo", "--yes", "--json"]
    )

    assert result.exit_code != 0, result.output
    assert auth.exists()
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["agents"]["agent"]["instance_id"] == (
        "instance-a"
    )


def test_forget_project_requires_durable_recovery_receipt_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients import config as config_module

    _write_project_config(
        config_module.CONFIG_PATH,
        project={
            "terraform_state": {
                "bucket": "fixture-state-bucket",
                "access_key": "fixture-access-key",
                "secret_key": "fixture-secret-key",
            }
        },
    )
    monkeypatch.setattr(
        teardown_receipts,
        "record_teardown_event",
        lambda **_kwargs: (_ for _ in ()).throw(
            OSError("injected receipt store failure")
        ),
    )

    result = runner.invoke(app, ["configure", "--forget-project", "demo"])

    assert result.exit_code != 0, result.output
    configured = yaml.safe_load(config_module.CONFIG_PATH.read_text(encoding="utf-8"))
    assert configured["projects"]["demo"]["terraform_state"]["access_key"] == (
        "fixture-access-key"
    )


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
