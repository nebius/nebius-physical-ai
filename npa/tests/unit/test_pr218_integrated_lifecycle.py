from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from npa import teardown_receipts
from npa.clients.storage_validation import probe_terraform_backend
from npa.cluster.drain import DisruptionBlocker, _delete_exact_pdb
from npa.controller_ownership import (
    ClusterOwnerIdentityMismatchError,
    ControllerOwner,
    bind_controller_owner,
    clear_controller_owner,
    controller_owner,
)
from npa.deploy import provisioner
from npa.orchestration.skypilot.k8s_gpu_catalog import (
    discover_kubernetes_gpu_inventory,
)
from npa.progress import WaitProgress
from npa.project_destroy import DestroyPhase, execute_project_destroy
from npa.provisioning_journal import ProvisioningOperation, operation_context


class S3Error(Exception):
    def __init__(self, code: str, status: int) -> None:
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }
        super().__init__("provider detail intentionally ignored")


class ExactBackendClient:
    def __init__(self, *, state_exists: bool = True, fail_list: bool = False) -> None:
        self.state_exists = state_exists
        self.fail_list = fail_list
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803, ANN201
        if Key == "npa/terraform-state/demo/agent/terraform.tfstate":
            if not self.state_exists:
                raise S3Error("NoSuchKey", 404)
            return {}
        if Key not in self.objects:
            raise S3Error("NoSuchKey", 404)
        return {}

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803, ANN201
        if Key == "npa/terraform-state/demo/agent/terraform.tfstate":
            return {"Body": io.BytesIO(b'{"version":4,"resources":[]}')}
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, IfNoneMatch: str):  # noqa: N803, ANN201
        assert IfNoneMatch == "*"
        self.objects[Key] = Body

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int):  # noqa: N803, ANN201
        if self.fail_list:
            raise S3Error("AccessDenied", 403)
        return {
            "Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]
        }

    def delete_object(self, *, Bucket: str, Key: str):  # noqa: N803, ANN201
        self.deleted.append(Key)
        self.objects.pop(Key, None)


def test_existing_backend_state_also_proves_sibling_write_list_read_delete() -> None:
    client = ExactBackendClient()

    result = probe_terraform_backend(
        bucket="state-bucket",
        state_key="npa/terraform-state/demo/agent/terraform.tfstate",
        endpoint_url="https://storage.example.invalid",
        access_key_id="saved-access",
        secret_access_key="saved-secret",
        client=client,
    )

    assert result.ok
    assert result.code == "existing_state_valid"
    assert result.cleanup_succeeded
    assert client.deleted == [result.probe_key]


def test_backend_prefix_denial_is_not_hidden_by_readable_existing_state() -> None:
    result = probe_terraform_backend(
        bucket="state-bucket",
        state_key="npa/terraform-state/demo/agent/terraform.tfstate",
        endpoint_url="https://storage.example.invalid",
        access_key_id="saved-access",
        secret_access_key="saved-secret",
        client=ExactBackendClient(fail_list=True),
    )

    assert not result.ok
    assert result.code == "forbidden"
    assert result.cleanup_succeeded
    assert "saved-" not in result.summary


def test_remote_backend_context_reaches_every_state_command_and_beats_parent_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "backend.tf").write_text('terraform { backend "s3" {} }\n')
    lock = tmp_path / ".terraform.lock.hcl"
    lock.write_text("provider lock\n")
    monkeypatch.setattr("npa.terraform_lock.validate_provider_lock", lambda _path: None)
    monkeypatch.setattr(provisioner, "_require_terraform", lambda: "terraform")
    monkeypatch.setattr(
        provisioner,
        "_tf_env",
        lambda _path: {
            "PATH": "/bin",
            "AWS_ACCESS_KEY_ID": "poisoned-parent",
            "AWS_SECRET_ACCESS_KEY": "poisoned-parent-secret",
        },
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN202
        calls.append((list(cmd), dict(kwargs["env"])))
        operation = cmd[1:3]
        stdout = ""
        if operation == ["output", "-json"]:
            stdout = '{"instance_id":{"value":"instance-1"}}'
        elif operation == ["state", "list"]:
            stdout = "nebius_compute_v1_instance.main\n"
        elif operation == ["state", "pull"]:
            stdout = '{"version":4}'
        elif operation == ["state", "show"]:
            stdout = 'id = "network-1"\n'
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(provisioner.subprocess, "run", fake_run)
    backend = {
        "access_key": "correct-access",
        "secret_key": "correct-secret",
        "session_token": "correct-session",
        "endpoint": "https://storage.example.invalid",
        "region": "eu-test1",
        "addressing_style": "path",
    }
    provisioner.init(tmp_path, backend_config=backend)
    provisioner.plan(tmp_path)
    provisioner.apply(tmp_path, stream=False)
    provisioner.state_list(tmp_path)
    provisioner.state_pull(tmp_path)
    provisioner.state_resource_id("nebius_vpc_v1_network.main", tmp_path)
    provisioner.outputs(tmp_path)
    provisioner.destroy(tmp_path, stream=False)

    assert {cmd[1] for cmd, _env in calls} >= {
        "init",
        "plan",
        "apply",
        "state",
        "output",
        "destroy",
    }
    for cmd, env in calls:
        assert env["AWS_ACCESS_KEY_ID"] == "correct-access"
        assert env["AWS_SECRET_ACCESS_KEY"] == "correct-secret"
        assert env["AWS_SESSION_TOKEN"] == "correct-session"
        assert "correct-secret" not in " ".join(cmd)
        assert "correct-session" not in " ".join(cmd)


def test_remote_state_command_without_initialized_context_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "backend.tf").write_text('terraform { backend "s3" {} }\n')
    monkeypatch.setattr(provisioner, "_require_terraform", lambda: "terraform")
    monkeypatch.setattr(provisioner, "_tf_env", lambda _path: {"PATH": "/bin"})

    with pytest.raises(
        provisioner.BackendAuthenticationError, match="credentials unavailable"
    ):
        provisioner.state_list(tmp_path)


def _owner(alias: str, project_id: str, cluster_id: str) -> ControllerOwner:
    return ControllerOwner(
        project_alias=alias,
        project_id=project_id,
        cluster_id=cluster_id,
        cluster_name="npa-cluster",
        context="npa-cluster",
        context_fingerprint=f"fingerprint-{project_id}-{cluster_id}",
    )


def test_controller_owner_is_global_and_alias_rename_keeps_immutable_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import controller_ownership as ownership

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "alpha": {"project_id": "project-a"},
                    "renamed": {"project_id": "project-a"},
                    "beta": {"project_id": "project-b"},
                }
            }
        )
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", path)
    bind_controller_owner(_owner("alpha", "project-a", "cluster-a"))

    with pytest.raises(ClusterOwnerIdentityMismatchError):
        bind_controller_owner(_owner("beta", "project-b", "cluster-b"))

    renamed = _owner("renamed", "project-a", "cluster-a")
    bind_controller_owner(renamed)
    assert controller_owner() == renamed
    saved = yaml.safe_load(path.read_text())
    assert saved["skypilot"]["controller_owner"]["project_id"] == "project-a"
    assert all("controller_owner" not in value for value in saved["projects"].values())
    assert clear_controller_owner("", project_id="project-a", cluster_id="cluster-a")
    assert controller_owner() is None


def test_pdb_conflicts_refetch_and_stop_without_deleting_replacement() -> None:
    blocker = DisruptionBlocker(
        namespace="kube-system",
        name="coredns",
        matching_pods=("kube-system/coredns-1",),
        workloads=("kube-system/Deployment/coredns",),
        nodes=("node-a",),
        reason="fixture",
        uid="uid-coredns",
        resource_version="1",
        labels=(("app", "coredns"),),
        annotations=(),
        spec={"minAvailable": 1},
    )
    gets = 0

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        nonlocal gets
        if "get" in cmd:
            gets += 1
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "metadata": {
                            "name": blocker.name,
                            "namespace": blocker.namespace,
                            "uid": blocker.uid,
                            "resourceVersion": str(gets),
                            "labels": dict(blocker.labels),
                            "annotations": {},
                        },
                        "spec": blocker.spec,
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="409 Conflict")

    messages: list[str] = []
    sleeps: list[float] = []
    result = _delete_exact_pdb(
        blocker,
        context="ctx",
        kubeconfig="",
        runner=runner,
        sleeper=sleeps.append,
        on_status=messages.append,
    )

    assert result.returncode == 2
    assert gets == 3
    assert sleeps == [0.25, 0.5]
    assert len(messages) == 2
    assert "preserved" in result.stderr


def test_interruption_is_durable_and_recovery_argv_remains_structured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        resource_type="agent",
        requested_name="agent",
        resume_command="",
        resume_argv=["npa", "agent", "deploy", "--project", "demo"],
        destroy_argv=["npa", "agent", "destroy", "--operation-id", "exact"],
    )

    with pytest.raises(KeyboardInterrupt):
        with operation_context(operation):
            operation.transition("mutating")
            raise KeyboardInterrupt("operator stopped wait")

    summary = operation.recovery_summary()
    assert summary["lifecycle"] == "interrupted"
    assert summary["last_error_type"] == "KeyboardInterrupt"
    assert summary["resume_argv"][-2:] == ["--project", "demo"]
    assert "operator stopped" in summary["last_error"]


def test_status_reconciles_a_dead_operation_owner_to_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import provisioning_journal as journal

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    with monkeypatch.context() as creating:
        creating.setattr(journal.os, "getpid", lambda: 424242)
        operation = ProvisioningOperation.prepare(
            command="npa agent deploy",
            project_alias="demo",
            project_id="project-a",
            resource_type="agent",
            requested_name="agent",
            resume_command="npa agent deploy --project demo",
        )
        operation.transition("mutating")
    monkeypatch.setattr(journal, "_pid_is_alive", lambda _pid: False)

    summary = operation.recovery_summary()

    assert summary["lifecycle"] == "interrupted"
    assert summary["last_error_type"] == "ProcessExited"


def test_receipts_filter_by_immutable_project_and_quarantine_corrupt_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "receipts"
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(root))
    teardown_receipts.record_teardown_event(
        phase="cluster",
        resource="old",
        terminal_state="completed",
        project_alias="reused",
        project_id="project-old",
    )
    teardown_receipts.record_teardown_event(
        phase="cluster",
        resource="new",
        terminal_state="completed",
        project_alias="reused",
        project_id="project-new",
    )
    (root / "corrupt.json").write_text("not json")

    [selected] = teardown_receipts.list_teardown_receipts(
        project_alias="reused", project_id="project-new", legacy="exclude"
    )
    assert selected["project_id"] == "project-new"
    assert selected["operational_status"] == "terminal"
    all_receipts = teardown_receipts.list_teardown_receipts()
    assert any(item["schema_version"] == "unreadable" for item in all_receipts)


def test_partial_agent_status_requires_exact_project_and_agent_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import agent_status
    from npa.clients import config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "reused",
                "projects": {"reused": {"project_id": "project-new"}},
            }
        )
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="target",
        terminal_state="verified_absent",
        project_alias="reused",
        project_id="project-old",
    )
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="other",
        terminal_state="verified_absent",
        project_alias="reused",
        project_id="project-new",
    )

    assert (
        agent_status.partial_agent_status("reused", "target")["classification"]
        == "NOT_FOUND"
    )
    teardown_receipts.record_teardown_event(
        phase="agent",
        resource="target",
        terminal_state="verified_absent",
        project_alias="reused",
        project_id="project-new",
    )
    result = agent_status.partial_agent_status("reused", "target")
    assert result["classification"] == "VERIFIED_ABSENT"
    assert result["project_id"] == "project-new"


def test_project_destroy_continues_independent_work_but_preserves_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config

    class Environment:
        project_id = "project-a"

    monkeypatch.setattr(config, "resolve_environment", lambda _project: Environment())
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    phases = [
        DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory"),
        DestroyPhase("agents", (("npa", "agent-destroy"),), "agents"),
        DestroyPhase(
            "controller", (("npa", "controller-clean"),), "controller", ("workflows",)
        ),
        DestroyPhase(
            "clusters", (("npa", "cluster-down"),), "clusters", ("controller",)
        ),
        DestroyPhase(
            "bucket", (("npa", "bucket-delete"),), "bucket", ("agents", "clusters")
        ),
        DestroyPhase("local_cleanup", (("npa", "cleanup"),), "local"),
        DestroyPhase(
            "forget_alias",
            (("npa", "forget"),),
            "forget",
            (
                "workflows",
                "agents",
                "controller",
                "clusters",
                "bucket",
                "local_cleanup",
            ),
        ),
    ]
    executed: list[str] = []

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        executed.append(cmd[1])
        return subprocess.CompletedProcess(
            cmd,
            1 if cmd[1] == "workflow-list" else 0,
            stdout="",
            stderr="",
        )

    result = execute_project_destroy("demo", phases, runner=runner)

    assert result["status"] == "partial"
    assert executed == ["workflow-list", "agent-destroy", "cleanup"]
    statuses = {item["phase"]: item["status"] for item in result["phases"]}
    assert statuses["controller"] == "skipped_dependency"
    assert statuses["bucket"] == "skipped_dependency"
    assert statuses["forget_alias"] == "skipped_dependency"


def test_project_destroy_skips_not_submitted_workflow_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "demo",
                "projects": {"demo": {"project_id": "project-a"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    commands: list[list[str]] = []

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "runs": [
                        {
                            "run_id": "run-not-submitted",
                            "status": "NOT_SUBMITTED",
                            "submission_state": "NOT_SUBMITTED",
                        }
                    ]
                }
            ),
            stderr="",
        )

    result = execute_project_destroy(
        "demo",
        [DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory")],
        runner=runner,
    )

    assert result["status"] == "success"
    assert commands == [["npa", "workflow-list"]]


def test_full_mocked_project_lifecycle_is_exact_isolated_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import controller_ownership, project_destroy
    from npa.cli import main as main_module
    from npa.cli.main import app
    from npa.clients import config, credentials
    from npa.cluster import state as cluster_state

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    config_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    clusters_dir = tmp_path / ".npa" / "clusters"
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setattr(controller_ownership, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cluster_state, "CLUSTERS_DIR", clusters_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "target",
                "projects": {
                    "target": {
                        "project_id": "project-target",
                        "tenant_id": "tenant-target",
                        "region": "eu-test1",
                        "terraform_state": {
                            "bucket": "bucket-target",
                            "endpoint": "https://storage.example.invalid",
                        },
                        "agents": {
                            "agent": {
                                "instance_id": "instance-target",
                                "project_id": "project-target",
                                "region": "eu-test1",
                            }
                        },
                    },
                    "unrelated": {
                        "project_id": "project-unrelated",
                        "tenant_id": "tenant-unrelated",
                        "region": "eu-test2",
                        "agents": {
                            "agent": {
                                "instance_id": "instance-unrelated",
                                "project_id": "project-unrelated",
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "serviceaccount-target",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-target",
                    "service_account_managed_by": "npa",
                },
                "storage_setup": {
                    "version": 1,
                    "projects": {
                        "project-unrelated": {
                            "status": "partial",
                            "phase": "rollback_incomplete",
                            "resources": {
                                "service_account": {
                                    "id": "serviceaccount-unrelated",
                                    "name": "lerobot-training",
                                    "created_by": "npa",
                                    "project_id": "project-unrelated",
                                    "attempt_id": "attempt-unrelated",
                                }
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    target_cluster = cluster_state.ClusterState(
        name="cluster-target",
        cluster_id="cluster-target-id",
        project_id="project-target",
        region="eu-test1",
        node_count=1,
        node_platform="cpu-d3",
        node_preset="4vcpu-16gb",
        k8s_version="1.31",
        subnet_id="subnet-target",
        created_at="2026-08-08T00:00:00Z",
    )
    unrelated_cluster = cluster_state.ClusterState(
        name="cluster-unrelated",
        cluster_id="cluster-unrelated-id",
        project_id="project-unrelated",
        region="eu-test2",
        node_count=1,
        node_platform="cpu-d3",
        node_preset="4vcpu-16gb",
        k8s_version="1.31",
        subnet_id="subnet-unrelated",
        created_at="2026-08-08T00:00:00Z",
    )
    cluster_state.save_cluster_state(target_cluster)
    cluster_state.save_cluster_state(unrelated_cluster)
    bind_controller_owner(
        ControllerOwner(
            project_alias="target",
            project_id="project-target",
            cluster_id="cluster-target-id",
            cluster_name="cluster-target",
            context="cluster-target",
            context_fingerprint="fingerprint-target",
        )
    )
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        resume_command="",
        project_alias="target",
        project_id="project-target",
        tenant_id="tenant-target",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        backend={
            "bucket": "bucket-target",
            "endpoint": "https://storage.example.invalid",
            "state_key": "npa/target/agent.tfstate",
            "credential_source": "project_saved",
        },
    )
    teardown_receipts.record_teardown_event(
        phase="workflow",
        resource="run-not-submitted",
        terminal_state="not_submitted",
        project_alias="target",
        project_id="project-target",
        identity={
            "project_alias": "target",
            "project_id": "project-target",
            "run_id": "run-not-submitted",
            "submission_status": "NOT_SUBMITTED",
        },
    )

    provider = {
        "instances": {"instance-target", "instance-unrelated"},
        "controllers": {"cluster-target-id", "cluster-unrelated-id"},
        "clusters": {"cluster-target-id", "cluster-unrelated-id"},
        "buckets": {"bucket-target", "bucket-unrelated"},
        "service_accounts": {
            "serviceaccount-target",
            "serviceaccount-unrelated",
        },
        "access_keys": {"accesskey-target", "accesskey-unrelated"},
    }
    commands: list[list[str]] = []

    def fake_runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        command = list(cmd)
        commands.append(command)
        if command[1:5] == ["workbench", "workflow", "list", "--project"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "runs": [
                            {
                                "run_id": "run-not-submitted",
                                "status": "NOT_SUBMITTED",
                                "submission_state": "NOT_SUBMITTED",
                            }
                        ]
                    }
                ),
                stderr="",
            )
        if command[1:3] == ["agent", "destroy"]:
            assert command[command.index("--project") + 1] == "target"
            provider["instances"].discard("instance-target")
        elif command[1:3] == ["skypilot", "cleanup-controller"]:
            assert command[command.index("--cluster-id") + 1] == "cluster-target-id"
            provider["controllers"].discard("cluster-target-id")
        elif command[1:3] == ["cluster", "down"]:
            assert command[command.index("--cluster-id") + 1] == "cluster-target-id"
            provider["clusters"].discard("cluster-target-id")
            cluster_state.state_file("cluster-target").unlink(missing_ok=True)
        elif command[1:4] == ["storage", "bucket", "delete"]:
            assert command[command.index("--name") + 1] == "bucket-target"
            provider["buckets"].discard("bucket-target")
        elif command[1:4] == ["storage", "service-account", "delete"]:
            provider["service_accounts"].discard("serviceaccount-target")
            provider["access_keys"].discard("accesskey-target")
        elif command[1:3] == ["configure", "--forget-project"]:
            main_module._forget_project("target")
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    real_execute = project_destroy.execute_project_destroy
    monkeypatch.setattr(
        project_destroy,
        "execute_project_destroy",
        lambda project, phases, on_phase=None: real_execute(
            project, phases, runner=fake_runner, on_phase=on_phase
        ),
    )
    real_record = teardown_receipts.record_teardown_event

    def fallback_receipt(**kwargs):  # noqa: ANN003, ANN202
        identity = kwargs.get("identity") or {}
        if kwargs.get("phase") == "project_config" and identity.get("operations"):
            raise ValueError("simulated full receipt persistence failure")
        return real_record(**kwargs)

    monkeypatch.setattr(teardown_receipts, "record_teardown_event", fallback_receipt)

    result = CliRunner().invoke(
        app, ["destroy", "--project", "target", "--all", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "wrote a minimal safe cleanup receipt" in result.output
    assert not any(
        command[1:4] == ["workbench", "workflow", "cancel"]
        for command in commands
    )
    assert provider == {
        "instances": {"instance-unrelated"},
        "controllers": {"cluster-unrelated-id"},
        "clusters": {"cluster-unrelated-id"},
        "buckets": {"bucket-unrelated"},
        "service_accounts": {"serviceaccount-unrelated"},
        "access_keys": {"accesskey-unrelated"},
    }
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["default_project"] == "unrelated"
    assert set(saved["projects"]) == {"unrelated"}
    assert "controller_owner" not in saved.get("skypilot", {})
    assert cluster_state.load_cluster_state("cluster-target") is None
    assert cluster_state.load_cluster_state("cluster-unrelated") is not None

    destructive_commands = [
        command
        for command in commands
        if tuple(command[1:3])
        in {
            ("agent", "destroy"),
            ("skypilot", "cleanup-controller"),
            ("cluster", "down"),
            ("storage", "bucket"),
            ("storage", "service-account"),
        }
    ]
    before_retry = {key: set(value) for key, value in provider.items()}
    for command in destructive_commands:
        assert fake_runner(command).returncode == 0
    assert provider == before_retry


def test_bucket_name_reuse_is_terminal_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.storage import _wait_for_bucket_gone

    monkeypatch.setattr(
        "npa.clients.nebius.get_bucket_by_name",
        lambda _project, name: {"metadata": {"id": "replacement-id", "name": name}},
    )

    assert not _wait_for_bucket_gone(
        "project-a", "state-bucket", "original-id", 1, bucket_id="original-id"
    )


def test_gpu_inventory_keeps_ready_capacity_when_product_label_is_missing() -> None:
    payload = {
        "items": [
            {
                "metadata": {"name": "gpu-node", "labels": {}},
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "1"},
                    "allocatable": {"nvidia.com/gpu": "1"},
                },
            }
        ]
    }

    inventory = discover_kubernetes_gpu_inventory(
        context="ctx",
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert inventory.allocatable == 1
    assert inventory.eligible_gpu_nodes == 1
    assert inventory.products == ()
    assert inventory.to_dict()["accelerator_product"] == "unknown/unlabeled"
    assert inventory.to_dict()["label_readiness"] == "blocked_missing_product_label"


def test_gpu_inventory_ignores_labels_from_ineligible_nodes() -> None:
    payload = {
        "items": [
            {
                "metadata": {
                    "name": "not-ready",
                    "labels": {"nvidia.com/gpu.product": "FOREIGN-GPU"},
                },
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "False"}],
                    "capacity": {"nvidia.com/gpu": "8"},
                    "allocatable": {"nvidia.com/gpu": "8"},
                },
            },
            {
                "metadata": {"name": "ready-unlabelled", "labels": {}},
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "1"},
                    "allocatable": {"nvidia.com/gpu": "1"},
                },
            },
        ]
    }

    inventory = discover_kubernetes_gpu_inventory(
        context="ctx",
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert inventory.allocatable == 1
    assert inventory.products == ()
    assert inventory.to_dict()["label_readiness"] == "blocked_missing_product_label"


def test_progress_fake_clock_reports_first_periodic_and_terminal_events() -> None:
    now = [10.0]
    messages: list[str] = []
    progress = WaitProgress(
        "controller queue",
        interval=5.0,
        monotonic=lambda: now[0],
        emit=messages.append,
    )

    progress.start("attempt=1 token=hunter2")
    now[0] = 14.0
    progress.tick("attempt=2")
    now[0] = 15.0
    progress.tick("attempt=3 last_error=timeout")
    now[0] = 16.0
    progress.finish("ready")

    assert messages == [
        "controller queue: started; attempt=1 token=<redacted>",
        "controller queue: waiting 5s; attempt=3 last_error=timeout",
        "controller queue: ready after 6s",
    ]
