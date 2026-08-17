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

    def put_object(self, *, Bucket: str, Key: str, Body: bytes):  # noqa: N803, ANN201
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
    assert result.error.kind.value == "authorization"
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


def test_controller_binding_rejects_cluster_from_rolled_back_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import controller_ownership as ownership
    from npa.clients import config
    from npa.cluster import state as cluster_state

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "region-a",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(ownership, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cluster_state, "CLUSTERS_DIR", tmp_path / "clusters")
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    cluster_state.save_cluster_state(
        cluster_state.ClusterState(
            name="gpu",
            cluster_id="cluster-dead",
            project_id="project-a",
            region="region-a",
            node_count=1,
            node_platform="gpu-rtx6000",
            node_preset="1gpu-24vcpu-218gb",
            k8s_version="1.31",
            subnet_id="subnet-a",
            created_at="2026-08-08T00:00:00Z",
        )
    )
    operation = ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias="prod",
        project_id="project-a",
        resource_type="cluster",
        requested_name="gpu",
        resume_command="npa provision-if-absent --project prod",
    )
    operation.transition("mutating")
    operation.record_resource(
        resource_type="managed_kubernetes_cluster",
        requested_name="gpu",
        provider_id="cluster-dead",
        project_id="project-a",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )
    operation.record_failure("apply failed")
    operation.transition("rolled-back")

    candidate = ownership.resolve_controller_candidate("prod", "gpu")
    assert candidate.operation_id == operation.operation_id
    with pytest.raises(ClusterOwnerIdentityMismatchError, match="rolled-back"):
        ownership.verify_live_controller_candidate(candidate)
    assert ownership.controller_owner() is None


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


def test_absent_vm_with_present_owned_service_account_requires_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import agent_status

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        resume_command="npa agent deploy --project demo --name agent",
        project_alias="demo",
        project_id="project-demo",
        tenant_id="tenant-demo",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
    )
    operation.record_resource(
        resource_type="compute_instance",
        requested_name="agent-demo-agent",
        provider_id="instance-demo",
        project_id="project-demo",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )
    operation.record_resource(
        resource_type="agent_service_account",
        requested_name="npa-agent",
        provider_id="serviceaccount-demo",
        project_id="project-demo",
        ownership="created_by_this_operation",
        ownership_source="create-response",
    )
    operation.transition("destroyed")
    monkeypatch.setattr(
        "npa.clients.nebius.get_compute_instance_identity", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_service_account_identity",
        lambda account_id, **_k: {"id": account_id},
    )

    result = agent_status.partial_agent_status("demo", "agent")

    assert result["classification"] == "CLEANUP_REQUIRED"
    assert result["lifecycle"] == "partial"
    assert result["recorded_lifecycle"] == "succeeded"
    assert result["current_verification"] == "provider_verified_present"
    assert result["components"]["vm"][0]["state"] == "verified_absent"
    assert result["components"]["service_account"][0]["state"] == "present"
    assert (
        f"--operation-id {operation.operation_id}"
        in result["recovery"]["exact_cleanup_command"]
    )


def test_21_gib_whole_path_quota_blocks_before_any_mutation_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.agent_quota import _agent_check_whole_path_capacity
    from npa.provisioning_preflight import (
        GIB,
        NETWORK_SSD_BYTES_QUOTA,
        ExistingCapacity,
    )

    mutations: list[str] = []
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _project: "eu-test1"
    )
    monkeypatch.setattr(
        "npa.provisioning_preflight.discover_existing_capacity",
        lambda **_kwargs: ExistingCapacity(),
    )

    def quotas(_tenant: str) -> dict[str, object]:
        names = {
            "compute.instance.count": 20,
            "compute.disk.count": 20,
            NETWORK_SSD_BYTES_QUOTA: 21 * GIB,
            "vpc.ipv4-address.public.count": 20,
            "compute.instance.gpu.rtx6000": 20,
        }
        return {
            "items": [
                {
                    "metadata": {"name": name},
                    "spec": {"region": "eu-test1", "limit": str(limit)},
                    "status": {"usage": "0"},
                }
                for name, limit in names.items()
            ]
        }

    monkeypatch.setattr("npa.clients.nebius.list_quota_allowances", quotas)

    def mutate(kind: str) -> None:
        mutations.append(kind)

    with pytest.raises(Exception, match="1251"):
        _agent_check_whole_path_capacity(
            "project-demo", "tenant-demo", "eu-test1", agent_exists=False
        )
        mutate("terraform/network/control-plane/vm/disk")

    assert mutations == []


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


def test_internal_project_destroy_uses_active_interpreter_without_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from npa import project_destroy

    seen: list[list[str]] = []

    def run(cmd, **_kwargs):  # noqa: ANN001, ANN202
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(project_destroy.subprocess, "run", run)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    completed = project_destroy._run(("npa", "--version"), run)

    assert completed.returncode == 0
    expected = [
        sys.executable,
        "-m",
        "npa",
        "--version",
    ]
    assert seen == [expected]
    assert project_destroy._internal_command_argv(("npa", "--version")) == expected


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


def test_project_destroy_exact_empty_inventory_converges_despite_exit_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"projects": {"demo": {"project_id": "project-a"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))

    result = execute_project_destroy(
        "demo",
        [DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory")],
        runner=lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout='{"runs": []}\n',
            stderr="Warning: SkyPilot update check is unavailable.\n",
        ),
    )

    assert result["status"] == "success"
    assert result["phases"][0]["status"] == "completed"


def test_project_destroy_empty_inventory_never_masks_auth_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"projects": {"demo": {"project_id": "project-a"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    result = execute_project_destroy(
        "demo",
        [DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory")],
        runner=lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout='{"runs": []}', stderr="Permission denied"
        ),
    )
    assert result["status"] == "partial"
    assert "permission" in result["phases"][0]["errors"][0]


def test_agent_destroy_no_vm_id_continues_exact_owned_iam_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.main import app
    from npa.clients import config
    from npa.cli import agent as agent_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "demo",
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                        "agents": {"agent": {"project_id": "project-a"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo",
    )
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "npa.cli.agent_iam.agent_iam_leftovers",
        lambda _project: {
            "project_id": "project-a",
            "service_account_id": "serviceaccount-agent",
            "service_account_name": "npa-agent",
            "access_keys": [{"id": "accesskey-agent"}],
            "owned_by_npa": True,
            "inventory_verified": True,
            "inventory_error": "",
            "dependents": [],
        },
    )
    deleted: list[str] = []
    monkeypatch.setattr("npa.clients.nebius.delete_access_key", deleted.append)
    monkeypatch.setattr("npa.clients.nebius.delete_service_account", deleted.append)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_a: True)

    result = CliRunner().invoke(
        app, ["agent", "destroy", "--project", "demo", "--purge-iam", "--yes", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-agent", "serviceaccount-agent"]
    assert json.loads(result.stdout)["infrastructure_absent"] is True


def test_agent_destroy_shared_iam_degradation_terminalizes_teardown_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.main import app
    from npa.clients import config
    from npa.cli import agent as agent_module
    from npa.cli.agent_iam import AgentIAMCleanupError
    from npa.provisioning_journal import list_operations

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "demo",
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                        "agents": {"agent": {"project_id": "project-a"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="npa agent deploy --project demo",
    )
    monkeypatch.setattr(
        agent_module, "_destroy_agent_terraform", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *_a, **_k: None
    )

    def retain_shared_iam(*_args, **_kwargs) -> None:
        raise AgentIAMCleanupError(
            "agent IAM remains because exact provider inventory reports dependent VMs"
        )

    monkeypatch.setattr(
        "npa.cli.agent_iam.report_destroyed_agent_iam", retain_shared_iam
    )

    result = CliRunner().invoke(
        app, ["agent", "destroy", "--project", "demo", "--yes", "--json"]
    )

    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "partial_iam_cleanup"
    assert payload["infrastructure_absent"] is True
    assert payload["iam_cleanup_complete"] is False
    [teardown] = list_operations(
        project_id="project-a",
        resource_type="agent-teardown",
        requested_name="agent",
    )
    assert teardown.read()["phase"] == "destroyed"


def test_destroyed_agent_operation_needs_no_deleted_backend_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa agent deploy",
        project_alias="forgotten",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        resume_command="",
        backend={
            "bucket": "already-deleted",
            "endpoint": "https://storage.invalid",
            "state_key": "agent.tfstate",
        },
    )
    operation.transition("destroyed")
    monkeypatch.setattr(
        agent_module,
        "_resolve_destroy_tf_vars",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("stale backend must not be reopened")
        ),
    )

    agent_module._destroy_agent_terraform(
        "forgotten",
        "agent",
        operation_id=operation.operation_id,
        project_id="project-a",
    )
    teardown_receipts.record_teardown_event(
        phase="project_destroy_workflows",
        resource="demo",
        terminal_state="partial",
        project_alias="demo",
        project_id="project-a",
        errors=["storage was already removed on a later retry"],
    )


def _owned_project_operation() -> ProvisioningOperation:
    operation = ProvisioningOperation.prepare(
        command="npa project create",
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-test1",
        resource_type="project",
        requested_name="demo",
        resume_command="",
    )
    operation.record_resource(
        resource_type="nebius_project",
        requested_name="demo",
        provider_id="project-a",
        project_id="project-a",
        ownership="created_by_this_operation",
        ownership_source="provider-create-response",
        labels={"tenant_id": "tenant-a"},
    )
    return operation


def test_owned_empty_project_delete_is_exact_receipted_and_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config, nebius

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    },
                    "other": {
                        "project_id": "project-b",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    _owned_project_operation()
    observations = iter(
        [
            nebius.ProjectIdentity("project-a", "demo", "tenant-a", "eu-test1"),
            None,
            None,
        ]
    )
    monkeypatch.setattr("npa.project_destroy.time.sleep", lambda _delay: None)
    monkeypatch.setattr(
        nebius, "get_project_identity", lambda *_a, **_k: next(observations)
    )
    monkeypatch.setattr(
        nebius, "list_project_dependencies", lambda *_a, **_k: {"compute_instances": ()}
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius,
        "delete_project",
        lambda project_id, **_kwargs: deleted.append(project_id),
    )

    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )

    assert result["status"] == "success"
    assert deleted == ["project-a"]
    receipt = teardown_receipts.list_teardown_receipts(project_id="project-a")[0]
    assert receipt["events"][-1]["terminal_state"] == "completed"
    assert not teardown_receipts.list_teardown_receipts(project_id="project-b")


def test_owned_project_delete_waits_for_eventual_provider_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import project_destroy
    from npa.clients import config, nebius

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    _owned_project_operation()
    present = nebius.ProjectIdentity("project-a", "demo", "tenant-a", "eu-test1")
    observations = iter([present, present, None, None])
    monkeypatch.setattr(
        nebius, "get_project_identity", lambda *_a, **_k: next(observations)
    )
    monkeypatch.setattr(nebius, "list_project_dependencies", lambda *_a, **_k: {})
    monkeypatch.setattr(nebius, "delete_project", lambda *_a, **_k: None)
    sleeps: list[float] = []
    monkeypatch.setattr(project_destroy.time, "sleep", sleeps.append)

    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )

    assert result["status"] == "success"
    assert sleeps == [project_destroy.PROJECT_DELETE_VERIFY_INTERVAL_SECONDS] * 3


def test_owned_project_delete_timeout_reports_unstable_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import project_destroy
    from npa.clients import config, nebius

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    _owned_project_operation()
    present = nebius.ProjectIdentity("project-a", "demo", "tenant-a", "eu-test1")
    observations = iter([present, None])
    monkeypatch.setattr(
        nebius, "get_project_identity", lambda *_a, **_k: next(observations)
    )
    monkeypatch.setattr(nebius, "list_project_dependencies", lambda *_a, **_k: {})
    monkeypatch.setattr(nebius, "delete_project", lambda *_a, **_k: None)
    ticks = iter([0.0, 181.0])
    monkeypatch.setattr(project_destroy.time, "monotonic", lambda: next(ticks))

    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )

    assert result["status"] == "partial"
    error = result["phases"][0]["errors"][0]
    assert "stable absence could not be established" in error
    assert "last_observation=absent" in error
    assert "stable_absence_observations=1" in error


def test_owned_project_not_found_retry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config, nebius

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    _owned_project_operation()
    monkeypatch.setattr(nebius, "get_project_identity", lambda *_a, **_k: None)
    monkeypatch.setattr(
        nebius,
        "delete_project",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("NotFound retry must not delete")
        ),
    )

    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )

    assert result["status"] == "success"


def test_delete_project_requires_yes_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa import project_destroy
    from npa.cli.main import app

    monkeypatch.setattr(
        project_destroy,
        "build_project_destroy_plan",
        lambda *_a, **_k: [DestroyPhase("delete_project", (), "exact delete")],
    )
    monkeypatch.setattr(
        project_destroy,
        "execute_project_destroy",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("plan-only command must not execute")
        ),
    )

    result = CliRunner().invoke(
        app,
        ["destroy", "--project", "demo", "--all", "--delete-project", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["status"] == "plan_only"


def test_destroy_retry_replays_original_exact_phase_topology() -> None:
    from npa.cli.main import _restore_recorded_destroy_phases

    restored = _restore_recorded_destroy_phases(
        {
            "topology": {
                "phases": [
                    DestroyPhase(
                        "agents",
                        (("npa", "agent", "destroy", "--name", "agent-a"),),
                        "destroy exact agent",
                    ).to_dict(),
                    DestroyPhase(
                        "bucket",
                        (("npa", "storage", "bucket", "delete", "--name", "b"),),
                        "destroy exact bucket",
                        ("agents",),
                    ).to_dict(),
                ]
            }
        }
    )

    assert restored is not None
    assert [phase.name for phase in restored] == ["agents", "bucket"]
    assert restored[0].commands[0][-1] == "agent-a"
    assert restored[1].requires == ("agents",)


def test_destroy_bucket_fallback_requires_exact_project_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy

    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_project_id="project-a", s3_bucket="s3://bucket-a/"
        ),
    )

    assert project_destroy._project_bucket_name("project-a", "") == "bucket-a"
    assert project_destroy._project_bucket_name("project-b", "") == ""
    assert (
        project_destroy._project_bucket_name("project-b", "s3://state-owned/prefix/")
        == "state-owned"
    )


def test_destroy_retry_replays_completed_phase_without_current_state_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy, teardown_receipts

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="project-a", tenant_id="tenant-a", region="us-central1"
        ),
    )
    teardown_receipts.record_teardown_event(
        phase="project_destroy_workflows",
        resource="demo",
        terminal_state="completed",
        project_alias="demo",
        project_id="project-a",
    )

    calls: list[list[str]] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout='{"runs": []}', stderr="")

    result = project_destroy.execute_project_destroy(
        "demo",
        [DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory")],
        runner=run,
    )

    assert result["status"] == "success"
    assert calls == [["npa", "workflow-list"]]


def test_destroy_resume_replays_recreated_resources_before_project_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy, teardown_receipts

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="project-a", tenant_id="tenant-a", region="us-central1"
        ),
    )
    for name in ("workflows", "agents", "controller", "clusters"):
        teardown_receipts.record_teardown_event(
            phase=f"project_destroy_{name}",
            resource="demo",
            terminal_state="completed",
            project_alias="demo",
            project_id="project-a",
        )
    calls: list[str] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(command[1])
        stdout = '{"runs": []}' if command[1] == "workflow-list" else "{}"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        project_destroy,
        "_delete_owned_empty_project",
        lambda **_kwargs: calls.append("delete-project") or {"outcome": "deleted"},
    )
    phases = [
        DestroyPhase("workflows", (("npa", "workflow-list"),), "workflows"),
        DestroyPhase("agents", (("npa", "agent-destroy"),), "agents", ("workflows",)),
        DestroyPhase(
            "controller", (("npa", "controller-destroy"),), "controller", ("workflows",)
        ),
        DestroyPhase(
            "clusters",
            (("npa", "cluster-destroy"),),
            "clusters",
            ("workflows", "controller"),
        ),
        DestroyPhase(
            "delete_project",
            (),
            "project",
            ("workflows", "agents", "controller", "clusters"),
        ),
    ]

    result = project_destroy.execute_project_destroy("demo", phases, runner=run)

    assert result["status"] == "success"
    assert calls == [
        "workflow-list",
        "agent-destroy",
        "controller-destroy",
        "cluster-destroy",
        "delete-project",
    ]
    assert all(
        phase["evidence"].get("resume_contract") == "reverify_or_replay"
        for phase in result["phases"][:4]
    )


def test_storage_iam_without_exact_generation_is_explicit_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy

    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="project-a", tenant_id="tenant-a", region="us-central1"
        ),
    )
    phase = DestroyPhase(
        "storage_iam",
        (),
        "storage",
        metadata={"generation_ids": [], "logical_names": []},
    )

    result = project_destroy.execute_project_destroy(
        "demo",
        [phase],
        runner=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("identity-less destructive command must not run")
        ),
    )

    assert result["status"] == "success"
    assert result["phases"][0]["commands"] == []
    assert result["phases"][0]["evidence"] == {
        "outcome": "verified_nothing_to_do",
        "identity_source": "exact_project_records_and_receipts",
        "generation_ids": [],
        "command_results": [],
    }


def test_destroy_plan_never_emits_identityless_storage_iam_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy

    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="project-a", tenant_id="tenant-a", region="us-central1"
        ),
    )
    monkeypatch.setattr(
        "npa.clients.config.resolve_terraform_state",
        lambda _project: SimpleNamespace(bucket=""),
    )
    monkeypatch.setattr("npa.cli.agent.resolve_project_agents", lambda _project: {})
    monkeypatch.setattr("npa.cluster.state.list_local_clusters", lambda: [])
    monkeypatch.setattr(
        "npa.controller_ownership.controller_owner", lambda *_args: None
    )
    monkeypatch.setattr(
        "npa.provisioning_journal.list_operations", lambda **_kwargs: []
    )
    monkeypatch.setattr(project_destroy, "_project_bucket_name", lambda *_a: "")
    monkeypatch.setattr(
        project_destroy, "_project_storage_iam_generation_ids", lambda *_a: ()
    )
    monkeypatch.setattr(
        project_destroy, "_project_storage_iam_logical_names", lambda *_a: ()
    )

    phases = project_destroy.build_project_destroy_plan("demo")
    storage = next(phase for phase in phases if phase.name == "storage_iam")

    assert storage.commands == ()
    assert storage.metadata["generation_ids"] == []


def test_receipt_project_delete_plan_finishes_alias_free_full_audit() -> None:
    from npa import project_destroy

    phases = project_destroy.build_receipt_project_delete_plan(
        project="forgotten",
        project_id="project-a",
        tenant_id="tenant-a",
        receipt_id="receipt-a",
    )

    assert [phase.name for phase in phases] == [
        "network",
        "delete_project",
        "final_audit",
    ]
    assert phases[-1].commands == (
        (
            "npa",
            "cleanup",
            "--project",
            "project-a",
            "--full",
            "--yes",
            "--include-sky",
            "--skip-jobs",
            "--attest-no-active-jobs",
            "--json",
        ),
    )


def test_destroy_retry_rechecks_bucket_receipt_and_deletes_present_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from npa import project_destroy, teardown_receipts

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setattr(
        "npa.clients.config.resolve_environment",
        lambda _project: SimpleNamespace(
            project_id="project-a", tenant_id="tenant-a", region="us-central1"
        ),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_identity",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "npa.clients.nebius.get_bucket_by_name",
        lambda *_a, **_k: {"metadata": {"id": "bucket-replacement"}},
    )
    teardown_receipts.record_teardown_event(
        phase="project_destroy_bucket",
        resource="demo",
        terminal_state="completed",
        project_alias="demo",
        project_id="project-a",
    )
    calls: list[list[str]] = []

    def run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, stdout='{"verified_absent":true}', stderr=""
        )

    phase = DestroyPhase(
        "bucket",
        (("npa", "bucket-delete"),),
        "bucket",
        metadata={"logical_name": "bucket-a"},
    )
    result = project_destroy.execute_project_destroy("demo", [phase], runner=run)

    assert result["status"] == "success"
    assert calls == [["npa", "bucket-delete"]]


def test_incident_cleanup_order_continues_independent_phases_and_project_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import project_destroy
    from npa.clients import config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    order: list[str] = []
    phases = [
        DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory"),
        DestroyPhase("agents", (("npa", "agent-destroy"),), "agent"),
        DestroyPhase("clusters", (), "none", ("workflows",)),
        DestroyPhase(
            "bucket", (("npa", "bucket-delete"),), "bucket", ("agents", "clusters")
        ),
        DestroyPhase("storage_iam", (("npa", "storage-iam"),), "storage", ("bucket",)),
        DestroyPhase(
            "delete_project",
            (),
            "project",
            ("workflows", "agents", "clusters", "bucket", "storage_iam"),
        ),
    ]

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        order.append(cmd[1])
        if cmd[1] == "workflow-list":
            return subprocess.CompletedProcess(cmd, 1, stdout='{"runs": []}', stderr="")
        if cmd[1] == "agent-destroy":
            return subprocess.CompletedProcess(
                cmd,
                2,
                stdout=json.dumps(
                    {
                        "infrastructure_absent": True,
                        "iam_cleanup_complete": False,
                    }
                ),
                stderr="agent IAM remains",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(
        project_destroy,
        "_delete_owned_empty_project",
        lambda **_kwargs: order.append("project-delete") or {"outcome": "deleted"},
    )
    result = execute_project_destroy("demo", phases, runner=runner)

    assert order == [
        "workflow-list",
        "agent-destroy",
        "bucket-delete",
        "storage-iam",
        "project-delete",
    ]
    statuses = {item["phase"]: item["status"] for item in result["phases"]}
    assert statuses["agents"] == "degraded"
    assert statuses["bucket"] == "completed"
    assert statuses["storage_iam"] == "completed"
    assert statuses["delete_project"] == "completed"


def test_project_creation_proof_survives_local_alias_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import project_destroy
    from npa.provisioning_journal import ProvisioningOperation

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa fleet deploy",
        project_alias="fleet-created-name",
        project_id="project-a",
        tenant_id="tenant-a",
        region="us-central1",
        resource_type="project",
        requested_name="fleet-created-name",
        ownership_source="fleet-project-create",
        resume_command="npa fleet status",
        destroy_command="npa destroy --all --delete-project",
    )
    operation.record_resource(
        resource_type="nebius_project",
        requested_name="fleet-created-name",
        provider_id="project-a",
        ownership="created_by_this_operation",
        ownership_source="provider-create-response",
        project_id="project-a",
        labels={"tenant_id": "tenant-a"},
    )

    assert (
        project_destroy._project_ownership_operation(
            "later-configured-alias", "project-a", "tenant-a"
        )
        == operation
    )


def test_incident_end_to_end_recovers_iam_then_deletes_owned_project_from_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa import project_destroy
    from npa.cli.agent_iam import report_agent_iam
    from npa.cli.main import app
    from npa.clients import config, nebius

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    },
                    "unrelated": {
                        "project_id": "project-b",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    _owned_project_operation()
    receipt_path = teardown_receipts.record_teardown_event(
        phase="project_destroy",
        resource="demo",
        terminal_state="partial",
        project_alias="demo",
        project_id="project-a",
        identity={
            "project_alias": "demo",
            "project_id": "project-a",
            "tenant_id": "tenant-a",
            "region": "eu-test1",
            "profile": "test-profile",
        },
    )
    order: list[str] = []
    phases = [
        DestroyPhase("workflows", (("npa", "workflow-list"),), "inventory"),
        DestroyPhase("agents", (("npa", "agent-destroy"),), "agent"),
        DestroyPhase("clusters", (), "none", ("workflows",)),
        DestroyPhase(
            "bucket", (("npa", "bucket-delete"),), "bucket", ("agents", "clusters")
        ),
        DestroyPhase("storage_iam", (("npa", "storage-iam"),), "storage", ("bucket",)),
        DestroyPhase("local_cleanup", (("npa", "local-cleanup"),), "local"),
        DestroyPhase(
            "forget_alias",
            (("npa", "forget-alias"),),
            "forget",
            ("workflows", "agents", "clusters", "bucket", "storage_iam"),
        ),
    ]

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        order.append(cmd[1])
        if cmd[1] == "workflow-list":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout='{"runs": []}',
                stderr="Warning: SkyPilot update check skipped",
            )
        if cmd[1] == "agent-destroy":
            return subprocess.CompletedProcess(
                cmd,
                2,
                stdout=json.dumps(
                    {
                        "infrastructure_absent": True,
                        "iam_cleanup_complete": False,
                    }
                ),
                stderr="agent IAM remains",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    initial = execute_project_destroy("demo", phases, runner=runner)
    assert initial["status"] == "partial"
    assert order == [
        "workflow-list",
        "agent-destroy",
        "bucket-delete",
        "storage-iam",
        "local-cleanup",
        "forget-alias",
    ]

    # Model the guarded alias removal while preserving the unrelated project and
    # the durable receipt/operation journal used by the recovery commands.
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "unrelated": {
                        "project_id": "project-b",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "npa.cli.agent_iam.agent_iam_leftovers",
        lambda _project: {
            "project_id": "project-a",
            "service_account_id": "serviceaccount-agent",
            "service_account_name": "npa-agent",
            "access_keys": [{"id": "accesskey-agent"}],
            "owned_by_npa": True,
            "inventory_verified": True,
            "inventory_error": "",
            "dependents": [],
        },
    )
    monkeypatch.setattr(
        nebius,
        "delete_access_key",
        lambda key_id: order.append(f"purge-key:{key_id}"),
    )
    monkeypatch.setattr(
        nebius,
        "delete_service_account",
        lambda account_id: order.append(f"purge-agent-iam:{account_id}"),
    )
    monkeypatch.setattr("npa.cli.agent_iam.remove_agent_iam_resource", lambda *_a: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_a: True)
    report_agent_iam(
        project_id="project-a",
        remaining_agents=0,
        purge=True,
        strict=True,
        on_status=lambda _message: None,
    )
    teardown_receipts.record_teardown_event(
        phase="project_destroy_network",
        resource="demo",
        terminal_state="completed",
        project_alias="demo",
        project_id="project-a",
    )

    observations = iter(
        [
            nebius.ProjectIdentity("project-a", "demo", "tenant-a", "eu-test1"),
            None,
            None,
        ]
    )
    monkeypatch.setattr("npa.project_destroy.time.sleep", lambda _delay: None)
    monkeypatch.setattr(
        nebius, "get_project_identity", lambda *_a, **_k: next(observations)
    )
    monkeypatch.setattr(
        nebius, "list_project_dependencies", lambda *_a, **_k: {"all": ()}
    )
    monkeypatch.setattr(
        nebius,
        "delete_project",
        lambda project_id, **_kwargs: order.append(f"delete-project:{project_id}"),
    )
    monkeypatch.setattr(
        project_destroy,
        "_run",
        lambda command, _runner: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "result": "fully_cleaned",
                    "operational_residue_present": False,
                    "verification_unresolved": False,
                }
            ),
            stderr="",
        ),
    )
    deleted = CliRunner().invoke(
        app,
        [
            "destroy",
            "--receipt",
            receipt_path.stem,
            "--all",
            "--delete-project",
            "--yes",
            "--json",
        ],
    )
    assert deleted.exit_code == 0, deleted.output
    assert order[-3:] == [
        "purge-key:accesskey-agent",
        "purge-agent-iam:serviceaccount-agent",
        "delete-project:project-a",
    ]
    assert yaml.safe_load(config_path.read_text())["projects"] == {
        "unrelated": {
            "project_id": "project-b",
            "tenant_id": "tenant-a",
            "region": "eu-test1",
        }
    }

    # The same receipt remains a safe idempotent recovery selector after the
    # provider reports NotFound; no second deletion is attempted.
    monkeypatch.setattr(nebius, "get_project_identity", lambda *_a, **_k: None)
    repeated = CliRunner().invoke(
        app,
        [
            "destroy",
            "--receipt",
            receipt_path.stem,
            "--all",
            "--delete-project",
            "--yes",
            "--json",
        ],
    )
    assert repeated.exit_code == 0, repeated.output
    assert order.count("delete-project:project-a") == 1


@pytest.mark.parametrize("mode", ["unowned", "children", "ambiguous"])
def test_project_delete_refuses_unproven_nonempty_or_ambiguous_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    from npa.clients import config, nebius

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    if mode != "unowned":
        _owned_project_operation()
    monkeypatch.setattr(
        nebius,
        "get_project_identity",
        lambda *_a, **_k: nebius.ProjectIdentity(
            "project-a", "demo", "tenant-a", "eu-test1"
        ),
    )
    if mode == "children":
        monkeypatch.setattr(
            nebius,
            "list_project_dependencies",
            lambda *_a, **_k: {"service_accounts": ("sa-a",)},
        )
    elif mode == "ambiguous":
        monkeypatch.setattr(
            nebius,
            "list_project_dependencies",
            lambda *_a, **_k: (_ for _ in ()).throw(
                nebius.NebiusError("schema-invalid inventory")
            ),
        )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius,
        "delete_project",
        lambda project_id, **_kwargs: deleted.append(project_id),
    )

    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )

    assert result["status"] == "partial"
    assert deleted == []


def test_project_delete_receipt_failure_prevents_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients import config, nebius

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "demo": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-test1",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    _owned_project_operation()
    monkeypatch.setattr(
        nebius,
        "get_project_identity",
        lambda *_a, **_k: nebius.ProjectIdentity(
            "project-a", "demo", "tenant-a", "eu-test1"
        ),
    )
    monkeypatch.setattr(nebius, "list_project_dependencies", lambda *_a, **_k: {})
    monkeypatch.setattr(
        teardown_receipts,
        "record_teardown_event",
        lambda **_k: (_ for _ in ()).throw(OSError("receipt unavailable")),
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius,
        "delete_project",
        lambda project_id, **_kwargs: deleted.append(project_id),
    )
    result = execute_project_destroy(
        "demo", [DestroyPhase("delete_project", (), "delete")]
    )
    assert result["status"] == "partial"
    assert deleted == []


def test_remote_controller_absence_allows_safe_downstream_cleanup(
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
    phases = [
        DestroyPhase("workflows", (), "none"),
        DestroyPhase(
            "controller", (("npa", "controller"),), "controller", ("workflows",)
        ),
        DestroyPhase("clusters", (("npa", "cluster"),), "cluster", ("controller",)),
        DestroyPhase("bucket", (("npa", "bucket"),), "bucket", ("clusters",)),
    ]
    commands: list[str] = []

    def runner(cmd, **_kwargs):  # noqa: ANN001, ANN202
        commands.append(cmd[1])
        if cmd[1] == "controller":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "outcome": "degraded_local_metadata",
                        "remote_absence_verified": True,
                    }
                ),
                stderr="stale local row",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    result = execute_project_destroy("demo", phases, runner=runner)

    assert commands == ["controller", "cluster", "bucket"]
    statuses = {item["phase"]: item["status"] for item in result["phases"]}
    assert statuses == {
        "workflows": "completed",
        "controller": "degraded",
        "clusters": "completed",
        "bucket": "completed",
    }
    assert result["status"] == "partial"


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
    failed_cluster = ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias="target",
        project_id="project-target",
        tenant_id="tenant-target",
        region="eu-test1",
        resource_type="cluster",
        requested_name="cluster-target",
        resume_command="npa provision-if-absent --project target",
    )
    failed_cluster.transition("mutating")
    failed_cluster.record_resource(
        resource_type="managed_kubernetes_cluster",
        requested_name="cluster-target",
        provider_id="cluster-first-failed-id",
        project_id="project-target",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )
    failed_cluster.record_failure("simulated first apply failure")
    failed_cluster.record_rollback(
        attempted=True,
        completed=True,
        removed=failed_cluster.read()["resources"],
        preserved=[],
    )
    failed_cluster.transition("rolled-back")
    retry_cluster = ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias="target",
        project_id="project-target",
        tenant_id="tenant-target",
        region="eu-test1",
        resource_type="cluster",
        requested_name="cluster-target",
        resume_command="npa provision-if-absent --project target",
    )
    retry_cluster.transition("mutating")
    retry_cluster.record_resource(
        resource_type="managed_kubernetes_cluster",
        requested_name="cluster-target",
        provider_id="cluster-target-id",
        project_id="project-target",
        ownership="created_by_this_operation",
        ownership_source="terraform-output",
    )
    retry_cluster.transition("state-durable")
    retry_cluster.commit()
    assert retry_cluster.operation_id.endswith("-r1")
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
            target_id = command[command.index("--cluster-id") + 1]
            assert target_id in {"cluster-first-failed-id", "cluster-target-id"}
            provider["clusters"].discard(target_id)
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
        lambda project, phases, on_phase=None, exact_identity=None: real_execute(
            project,
            phases,
            runner=fake_runner,
            on_phase=on_phase,
            exact_identity=exact_identity,
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
        command[1:4] == ["workbench", "workflow", "cancel"] for command in commands
    )
    assert provider == {
        "instances": {"instance-unrelated"},
        "controllers": {"cluster-unrelated-id"},
        "clusters": {"cluster-unrelated-id"},
        "buckets": {"bucket-unrelated"},
        # No exact owned storage-IAM generation was recorded, so teardown
        # refuses to infer ownership from a familiar local/provider name.
        "service_accounts": {"serviceaccount-target", "serviceaccount-unrelated"},
        "access_keys": {"accesskey-target", "accesskey-unrelated"},
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
