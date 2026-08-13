from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.agent import app as agent_app
from npa.cli.cluster import app as cluster_app
from npa.cli.cluster import terraform_lifecycle as cluster_tf
from npa.cli.main import app as npa_app
from npa.cli.workbench.workflow import app as workflow_app
from npa.orchestration.npa_workflow.submission_state import (
    submission_state_path,
    update_submission_state,
)
from npa.orchestration.skypilot.workflow import ManagedJobEvidence
from npa.teardown_receipts import list_teardown_receipts, record_teardown_event


runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))


def test_agent_exact_id_absence_needs_no_project_stanza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *args, **kwargs: None
    )

    result = runner.invoke(
        agent_app,
        [
            "destroy",
            "--project-id",
            "project-1",
            "--instance-id",
            "instance-1",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "already_absent"
    assert payload["verified"] is True
    assert payload["identity_source"] == "explicit_exact_arguments"


def test_agent_receipt_absence_needs_no_project_stanza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="in_progress",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "agents": [
                {
                    "agent_name": "agent",
                    "instance_id": "instance-1",
                    "project_id": "project-1",
                }
            ],
        },
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *args, **kwargs: None
    )

    result = runner.invoke(
        agent_app,
        ["destroy", "--receipt", path.stem, "--yes", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "already_absent"
    assert payload["identity"]["instance_id"] == "instance-1"


def test_recovery_selectors_are_exposed_consistently_in_help() -> None:
    commands = (
        (
            agent_app,
            ["destroy", "--help"],
            ("--receipt", "--project-id", "--instance-id"),
        ),
        (
            cluster_app,
            ["down", "--help"],
            ("--receipt", "--project-id", "--cluster-id"),
        ),
        (workflow_app, ["cancel", "--help"], ("--receipt", "--project-id", "--job-id")),
        (
            npa_app,
            ["skypilot", "cleanup-controller", "--help"],
            ("--receipt", "--project-id", "--cluster-id"),
        ),
        (
            npa_app,
            ["storage", "service-account", "delete", "--help"],
            ("--receipt", "--project-id", "--id"),
        ),
    )
    for app, args, expected in commands:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        for option in expected:
            assert option in result.output


def test_agent_exact_provider_failure_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.clients.nebius import NebiusError

    def unavailable(*_args, **_kwargs):
        raise NebiusError("RBAC denied")

    monkeypatch.setattr("npa.clients.nebius.get_compute_instance_identity", unavailable)
    result = runner.invoke(
        agent_app,
        [
            "destroy",
            "--project-id",
            "project-1",
            "--instance-id",
            "instance-1",
            "--yes",
        ],
    )
    assert result.exit_code == 2
    assert "unresolved" in result.output
    assert "Nothing was deleted" in result.output


def test_cluster_no_state_exact_absence_never_initializes_terraform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cluster.exceptions import ClusterNotFoundError

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('cluster_name = "cluster-1"\n')

    class MissingCluster:
        def get_cluster(self, *_args, **_kwargs):
            raise ClusterNotFoundError("not found")

    monkeypatch.setattr("npa.cluster.api.MK8sClient", MissingCluster)
    monkeypatch.setattr(
        cluster_tf,
        "_require_bin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Terraform/provider initialization must not run")
        ),
    )
    result = runner.invoke(
        cluster_app,
        [
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--project-id",
            "project-1",
            "--cluster-id",
            "cluster-id-1",
            "--force",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["outcome"] == "already_absent"
    assert not (tf_dir / ".terraform").exists()


def test_cluster_no_state_insufficient_identity_fails_before_terraform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    monkeypatch.setattr(
        cluster_tf,
        "_require_bin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Terraform must not run")
        ),
    )
    result = runner.invoke(
        cluster_app,
        [
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--project-id",
            "project-1",
            "--force",
        ],
    )
    assert result.exit_code == 2
    assert "--cluster-id" in result.output
    assert not (tf_dir / ".terraform").exists()


def test_workflow_reserved_ledger_is_not_submitted_without_s3_or_sky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.orchestration.npa_workflow.run_resolution import RunResolution

    update_submission_state("demo", "reserved-run", {"launch_state": "reserved"})
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.resolve_run",
        lambda *_args, **_kwargs: RunResolution(
            run_id="reserved-run",
            project="demo",
            not_submitted=True,
            source="durable_submission_receipt",
        ),
    )
    result = runner.invoke(
        workflow_app,
        ["cancel", "reserved-run", "--project", "demo", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "not_submitted"
    assert payload["detected_state"] == "NOT_SUBMITTED"
    assert payload["cloud_calls"] is False


def test_workflow_plan_only_receipt_is_not_submitted_without_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.orchestration.npa_workflow.run_resolution import RunResolution

    update_submission_state(
        "demo",
        "planned-run",
        {
            "workflow": {
                "name": "physical-ai-data-factory",
                "steps": [{"state": "augment", "status": "planned"}],
            }
        },
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.resolve_run",
        lambda *_args, **_kwargs: RunResolution(
            run_id="planned-run",
            project="demo",
            not_submitted=True,
            source="durable_submission_receipt",
        ),
    )
    result = runner.invoke(
        workflow_app,
        ["cancel", "planned-run", "--project", "demo", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["outcome"] == "not_submitted"


def test_workflow_partial_submission_stays_unresolved_without_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update_submission_state("demo", "partial-run", {"launch": {"status": "launching"}})
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *_args, **_kwargs: ManagedJobEvidence(
            "unavailable", error="SkyPilot removed"
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution._parent_state",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("S3 config removed")),
    )
    result = runner.invoke(
        workflow_app,
        ["cancel", "partial-run", "--project", "demo", "--json"],
    )
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["detected_state"] == "VERIFICATION_UNAVAILABLE"


def test_workflow_ambiguous_legacy_state_is_not_claimed_not_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = submission_state_path("demo", "legacy-run")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "project": "demo",
                "run_id": "legacy-run",
                "workflow": {"name": "legacy-workflow"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *_args, **_kwargs: ManagedJobEvidence(
            "unavailable", error="SkyPilot removed"
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution._parent_state",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("S3 config removed")),
    )

    result = runner.invoke(
        workflow_app,
        ["cancel", "legacy-run", "--project", "demo", "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["outcome"] == "verification_failed"
    assert payload["detected_state"] == "VERIFICATION_UNAVAILABLE"


@pytest.mark.parametrize("terminal_phase", ["controller", "cluster"])
def test_controller_terminal_receipt_noops_without_provider_or_skypilot(
    monkeypatch: pytest.MonkeyPatch, terminal_phase: str
) -> None:
    path = record_teardown_event(
        phase=terminal_phase,
        resource="ctx-1",
        terminal_state="verified_absent",
        project_id="project-1",
        context="ctx-1",
        identity={
            "project_id": "project-1",
            "cluster_id": "cluster-1",
            "context": "ctx-1",
        },
    )
    monkeypatch.setattr(
        "npa.cluster.identity.resolve_verified_cluster_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider must not be inspected")
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup._jobs_controller_clusters",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SkyPilot must not be bootstrapped")
        ),
    )
    from npa.orchestration.skypilot.cleanup import cleanup_jobs_controller

    result = cleanup_jobs_controller(receipt=path.stem, context="ctx-1")
    assert result.ok
    assert result.outcome == "already_absent"
    assert result.no_op is True
    assert len(list_teardown_receipts()) == 1


def test_controller_exact_absent_cluster_noops_without_alias_or_skypilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cluster.exceptions import ClusterNotFoundError

    class MissingCluster:
        def get_cluster(self, *_args, **_kwargs):
            raise ClusterNotFoundError("gone")

    monkeypatch.setattr("npa.cluster.identity.MK8sClient", MissingCluster)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup._jobs_controller_clusters",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("SkyPilot must not be bootstrapped")
        ),
    )
    from npa.orchestration.skypilot.cleanup import cleanup_jobs_controller

    result = cleanup_jobs_controller(
        project_id="project-1",
        cluster_id="cluster-1",
        context="ctx-1",
    )

    assert result.ok
    assert result.outcome == "already_absent"
    assert result.no_op is True
    assert result.resources_removed == []
