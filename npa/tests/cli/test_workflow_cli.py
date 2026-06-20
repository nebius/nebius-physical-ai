from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.orchestration.skypilot.workflow import WorkflowResult
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config
from npa.workflows.distill import DistillationError
from npa.workflows.distill_two_vm import TwoVMDistillError


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "command",
    ["submit", "run", "list", "status", "logs", "artifacts", "cancel", "teardown", "distill"],
)
def test_workflow_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "workflow", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


class FakeWorkflowS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = "") -> None:
        del ContentType
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else str(Body).encode("utf-8")

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket: str, Prefix: str):
                contents = [
                    {"Key": key, "Size": len(body)}
                    for (bucket, key), body in sorted(client.objects.items())
                    if bucket == Bucket and key.startswith(Prefix)
                ]
                yield {"Contents": contents}

        return Paginator()


def _patch_workflow_s3(monkeypatch: pytest.MonkeyPatch, fake_s3: FakeWorkflowS3) -> None:
    monkeypatch.setattr("npa.orchestration.skypilot.workflow_state.boto3.client", lambda *args, **kwargs: fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(checkpoint_bucket="", endpoint_url=""),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.load_credentials",
        lambda: SimpleNamespace(
            s3_access_key_id="test-access",
            s3_secret_access_key="test-secret",
            s3_endpoint="https://storage.example",
            s3_bucket="",
        ),
    )


def test_workflow_s3_config_uses_nebius_mount_for_nebius_endpoint() -> None:
    state = WorkflowS3Config(
        bucket="bucket",
        prefix="run-1",
        endpoint_url="https://storage.eu-north1.nebius.cloud",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )

    assert state.uri == "s3://bucket/run-1"
    assert state.sky_mount_source == "nebius://bucket"
    assert state.sky_mount_store == "NEBIUS"


def test_workflow_run_dispatches(mocker) -> None:
    run_mock = mocker.patch(
        "npa.workflows.distill.run_distillation",
        return_value={"run_id": "run-1", "stages": {"train_teacher": {"status": "success"}}},
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "run",
            "distill",
            "--n-envs",
            "2",
            "--action-space",
            "joint",
        ],
    )

    assert result.exit_code == 0
    assert "Workflow complete" in result.output
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["n_envs"] == 2
    assert run_mock.call_args.kwargs["action_space"] == "joint"


def test_workbench_workflow_submit_dispatches_skypilot(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    submit_mock = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=WorkflowResult(status="SUBMITTED", job_id="42", returncode=0),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "run-1",
            "--submit-timeout",
            "30",
            "--secret-env",
            "AWS_ACCESS_KEY_ID",
        ],
    )

    assert result.exit_code == 0
    assert "SUBMITTED" in result.output
    assert "42" in result.output
    submit_mock.assert_called_once()
    assert submit_mock.call_args.args == (yaml_path, "run-1")
    assert submit_mock.call_args.kwargs["timeout"] == 30
    assert submit_mock.call_args.kwargs["secret_envs"] == ["AWS_ACCESS_KEY_ID"]


def test_workbench_workflow_submit_instruments_durable_s3(monkeypatch, mocker, tmp_path) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nexecution: serial\n---\nname: train\nresources:\n  cloud: kubernetes\nrun: |\n  echo HF_TOKEN=hf_testsecret123456\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["content"] = path.read_text(encoding="utf-8")
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow", side_effect=fake_submit_workflow)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "run-1",
            "--durable-s3",
            "--s3-bucket",
            "bucket",
            "--s3-endpoint",
            "https://storage.example",
        ],
    )

    assert result.exit_code == 0
    assert "run_prefix_uri: s3://bucket/run-1" in result.output
    assert captured["run_id"] == "run-1"
    kwargs = captured["kwargs"]
    assert kwargs["extra_env"]["AWS_ACCESS_KEY_ID"] == "test-access"
    assert kwargs["extra_env"]["AWS_SECRET_ACCESS_KEY"] == "test-secret"
    assert "AWS_ACCESS_KEY_ID" in kwargs["secret_envs"]
    assert "AWS_SECRET_ACCESS_KEY" in kwargs["secret_envs"]
    docs = [doc for doc in yaml.safe_load_all(str(captured["content"])) if doc]
    task = docs[1]
    assert task["file_mounts"]["/mnt/npa-workflow-state"]["source"] == "s3://bucket"
    assert task["file_mounts"]["/mnt/npa-workflow-state"]["mode"] == "MOUNT"
    assert task["envs"]["NPA_WORKFLOW_RUN_PREFIX_URI"] == "s3://bucket/run-1"
    assert "npa_workflow_redact_stream" in task["run"]
    manifest = json.loads(fake_s3.objects[("bucket", "run-1/manifest.json")].decode("utf-8"))
    assert manifest["sky_job_id"] == "42"
    assert manifest["stages"]["train"]["log_uri"] == "s3://bucket/run-1/logs/train/run.log"


def test_workbench_workflow_submit_substitutes_vars(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: ${RUN_NAME}\nresources:\n  cloud: ${CLOUD}\nrun: echo ${RUN_NAME}\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["path"] = path
        captured["run_id"] = run_id
        captured["content"] = path.read_text(encoding="utf-8")
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow", side_effect=fake_submit_workflow)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "run-1",
            "--var",
            "RUN_NAME=demo-run",
            "--var",
            "CLOUD=nebius",
        ],
    )

    assert result.exit_code == 0
    assert captured["path"] != yaml_path
    assert captured["run_id"] == "run-1"
    assert captured["content"] == "name: demo-run\nresources:\n  cloud: nebius\nrun: echo demo-run\n"


def test_workbench_workflow_submit_rejects_invalid_var(tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")

    result = runner.invoke(app, ["workbench", "workflow", "submit", str(yaml_path), "--var", "missing-equals"])

    assert result.exit_code == 1
    assert "Invalid --var format. Use KEY=VALUE." in result.output


def test_workbench_workflow_submit_warns_on_unresolved_placeholders(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: ${RUN_NAME}\nrun: echo ${MISSING}\n", encoding="utf-8")

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=WorkflowResult(status="SUBMITTED", job_id="42", returncode=0),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "run-1",
            "--var",
            "RUN_NAME=demo-run",
        ],
    )

    assert result.exit_code == 0
    assert "Warning: unresolved placeholders remain: ${MISSING}" in result.output + result.stderr


def test_workbench_workflow_submit_materializes_sonic_yaml(mocker) -> None:
    yaml_path = REPO_ROOT / "workflows/workbench/skypilot/sonic-train-standalone.yaml"
    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["path"] = path
        captured["run_id"] = run_id
        captured["content"] = path.read_text(encoding="utf-8")
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=fake_submit_workflow,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "sonic-run",
            "--registry",
            "registry.example/workbench",
            "--gpu-target",
            "gpu-rtx6000",
            "--s3-endpoint",
            "https://storage.example",
            "--s3-bucket",
            "proof-bucket",
            "--s3-prefix",
            "sonic-proof/sonic-run",
            "--accelerators",
            "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
            "--var",
            "SONIC_MAX_ITERATIONS=2",
        ],
    )

    assert result.exit_code == 0
    assert captured["run_id"] == "sonic-run"
    docs = [doc for doc in yaml.safe_load_all(str(captured["content"])) if doc]
    task = docs[1]
    envs = task["envs"]
    assert "image_id" not in task["resources"]
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert envs["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    assert envs["SONIC_GPU_TYPE"] == "gpu-rtx6000"
    assert envs["SONIC_IMAGE_VARIANT"] == "sonic-k8s-host-mounted"
    assert envs["S3_ENDPOINT_URL"] == "https://storage.example"
    assert envs["S3_BUCKET"] == "proof-bucket"
    assert envs["SONIC_OUTPUT_PREFIX"] == "sonic-proof/sonic-run/"
    assert envs["SONIC_MAX_ITERATIONS"] == "2"
    assert "image_id" not in task["resources"]
    assert "${" not in "\n".join(str(value) for value in envs.values())


def test_workbench_workflow_submit_materializes_registry_auth(mocker) -> None:
    yaml_path = REPO_ROOT / "workflows/workbench/skypilot/sonic-train-standalone.yaml"
    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["content"] = path.read_text(encoding="utf-8")
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=fake_submit_workflow,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "sonic-run",
            "--registry",
            "registry.example/workbench",
            "--registry-server",
            "registry.example",
            "--registry-username",
            "operator",
            "--registry-password",
            "redacted-test-token",
            "--gpu-target",
            "h100",
            "--use-spot",
            "--s3-endpoint",
            "https://storage.example",
            "--s3-bucket",
            "proof-bucket",
        ],
    )

    assert result.exit_code == 0
    assert "redacted-test-token" not in result.output
    docs = [doc for doc in yaml.safe_load_all(str(captured["content"])) if doc]
    task = docs[1]
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["resources"]["memory"] == 200
    assert task["resources"]["use_spot"] is True
    assert task["envs"]["SKYPILOT_DOCKER_USERNAME"] == "operator"
    assert task["envs"]["SKYPILOT_DOCKER_PASSWORD"] == "redacted-test-token"
    assert task["envs"]["SKYPILOT_DOCKER_SERVER"] == "registry.example"


def test_workbench_workflow_submit_materializes_sonic_mvp_workflow(mocker) -> None:
    yaml_path = REPO_ROOT / "workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml"
    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["content"] = path.read_text(encoding="utf-8")
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=fake_submit_workflow,
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--run-id",
            "sonic-run",
            "--registry",
            "registry.example/workbench",
            "--registry-server",
            "registry.example",
            "--registry-username",
            "operator",
            "--registry-password",
            "redacted-test-token",
            "--gpu-target",
            "h100",
            "--use-spot",
            "--region",
            "eu-north1",
            "--s3-endpoint",
            "https://storage.example",
            "--s3-bucket",
            "proof-bucket",
            "--s3-prefix",
            "sonic-mvp-proof/sonic-run",
            "--var",
            "SONIC_PAYLOAD_MODE=docker",
        ],
    )

    assert result.exit_code == 0
    assert "redacted-test-token" not in result.output
    docs = [doc for doc in yaml.safe_load_all(str(captured["content"])) if doc]
    assert [doc["name"] for doc in docs[1:]] == [
        "sonic-retarget-motion",
        "sonic-g1-finetune",
        "sonic-mujoco-eval",
    ]
    assert docs[1]["resources"]["cloud"] == "kubernetes"
    assert docs[1]["envs"]["AWS_PROFILE"] == "nebius"
    assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == "https://storage.example"
    for task in docs[2:]:
        assert task["resources"]["accelerators"] == "H100:1"
        assert task["resources"]["region"] == "eu-north1"
        assert task["resources"]["use_spot"] is True
        assert "image_id" not in task["resources"]
        assert task["envs"]["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic-mujoco:0.1.3-mvp"
        assert task["envs"]["SONIC_PAYLOAD_MODE"] == "docker"
        assert task["envs"]["AWS_PROFILE"] == "nebius"
        assert task["envs"]["SKYPILOT_DOCKER_PASSWORD"] == "redacted-test-token"
    assert captured["kwargs"]["require_controller_up"] is False


def test_workflow_run_unknown_workflow_errors() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "run", "unknown"])

    assert result.exit_code == 1
    assert "Unknown workflow" in result.output


def test_workflow_run_remote_requires_s3_bucket() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "run", "distill", "--remote"])

    assert result.exit_code == 1
    assert "--remote requires --s3-bucket" in result.output


def test_workflow_status_prints_status(mocker) -> None:
    mocker.patch("npa.workflows.sim2real.monitor.sim2real_run_exists", return_value=False)
    mocker.patch(
        "npa.workflows.distill.get_run_status",
        return_value={"run_id": "run-1", "status": "success", "stages": {}},
    )

    result = runner.invoke(app, ["workbench", "workflow", "status", "run-1"])

    assert result.exit_code == 0
    assert "run-1" in result.output
    assert "success" in result.output


def test_workflow_status_maps_distillation_error(mocker) -> None:
    mocker.patch("npa.workflows.sim2real.monitor.sim2real_run_exists", return_value=False)
    mocker.patch(
        "npa.workflows.distill.get_run_status",
        side_effect=DistillationError("not found"),
    )

    result = runner.invoke(app, ["workbench", "workflow", "status", "missing"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_workflow_logs_prints_stage_logs(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_stage_logs",
        return_value="stage log text",
    )

    result = runner.invoke(app, ["workbench", "workflow", "logs", "run-1", "convert"])

    assert result.exit_code == 0
    assert "stage log text" in result.output


def test_durable_workflow_status_logs_and_artifacts_read_s3(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    manifest = {
        "schema_version": 1,
        "run_id": "run-1",
        "workflow_name": "demo",
        "run_prefix_uri": "s3://bucket/run-1",
        "sky_job_id": "42",
        "stages": {
            "train": {
                "name": "train",
                "sky_job_id": "42",
                "log_uri": "s3://bucket/run-1/logs/train/run.log",
                "status_uri": "s3://bucket/run-1/logs/train/status.json",
                "artifact_uri": "s3://bucket/run-1/artifacts/train/",
            }
        },
    }
    status = {
        "schema_version": 1,
        "run_id": "run-1",
        "stage": "train",
        "state": "SUCCEEDED",
        "tier": "WORKS",
        "start_time": "2026-06-07T00:00:00Z",
        "end_time": "2026-06-07T00:00:01Z",
        "sky_job_id": "42",
        "sky_task_id": "0",
        "artifact_uri": "s3://bucket/run-1/artifacts/train/",
        "log_uri": "s3://bucket/run-1/logs/train/run.log",
        "error_summary": "",
    }
    fake_s3.put_object(Bucket="bucket", Key="run-1/manifest.json", Body=json.dumps(manifest).encode("utf-8"))
    fake_s3.put_object(Bucket="bucket", Key="run-1/logs/train/status.json", Body=json.dumps(status).encode("utf-8"))
    fake_s3.put_object(Bucket="bucket", Key="run-1/logs/train/run.log", Body=b"training complete\n")
    fake_s3.put_object(Bucket="bucket", Key="run-1/artifacts/train/model.bin", Body=b"model")

    status_result = runner.invoke(
        app,
        ["workbench", "workflow", "status", "s3://bucket/run-1", "--json"],
    )
    logs_result = runner.invoke(
        app,
        ["workbench", "workflow", "logs", "s3://bucket/run-1", "--stage", "train"],
    )
    artifacts_result = runner.invoke(
        app,
        ["workbench", "workflow", "artifacts", "s3://bucket/run-1"],
    )

    assert status_result.exit_code == 0
    payload = json.loads(status_result.output)
    assert payload["status"] == "SUCCEEDED"
    assert payload["stages"]["train"]["tier"] == "WORKS"
    assert logs_result.exit_code == 0
    assert "training complete" in logs_result.output
    assert artifacts_result.exit_code == 0
    assert "s3://bucket/run-1/artifacts/train/model.bin" in artifacts_result.output


def test_workflow_logs_maps_distillation_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_stage_logs",
        side_effect=DistillationError("no logs"),
    )

    result = runner.invoke(app, ["workbench", "workflow", "logs", "run-1", "bad"])

    assert result.exit_code == 1
    assert "no logs" in result.output


def test_workflow_distill_dispatches_two_vm_workflow(mocker) -> None:
    distill_mock = mocker.patch(
        "npa.workflows.distill_two_vm.distill",
        return_value={
            "status": "success",
            "run_id": "run-1",
            "s3_base": "s3://bucket/distill/run-1/",
            "stages": {"convert": {"status": "success"}},
        },
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "distill",
            "--skip-infra",
            "--skip-setup",
            "--n-envs",
            "2",
            "--student-policy",
            "act",
        ],
    )

    assert result.exit_code == 0
    assert "Workflow success" in result.output
    distill_mock.assert_called_once()
    assert distill_mock.call_args.kwargs["skip_infra"] is True


def test_workflow_distill_validates_student_policy() -> None:
    result = runner.invoke(
        app,
        ["workbench", "workflow", "distill", "--student-policy", "bad"],
    )

    assert result.exit_code == 1
    assert "student-policy must be act, diffusion, or smolvla" in result.output


def test_workflow_distill_maps_two_vm_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill_two_vm.distill",
        side_effect=TwoVMDistillError("infra failed"),
    )

    result = runner.invoke(app, ["workbench", "workflow", "distill"])

    assert result.exit_code == 1
    assert "infra failed" in result.output


def test_workflow_teardown_destroys_registered_vms(mocker) -> None:
    cfg = WorkbenchConfig(
        endpoint="",
        ssh=SSHConfig(host="host", user="ubuntu", key_path="key"),
        storage=StorageConfig(checkpoint_bucket="s3://bucket/checkpoints/", endpoint_url="url"),
    )
    destroy_mock = mocker.patch("npa.workflows.distill_two_vm._destroy_vm")
    remove_mock = mocker.patch("npa.clients.config.remove_workbench_config")
    mocker.patch("npa.clients.config.resolve_ssh_config", return_value=cfg)
    mocker.patch(
        "npa.clients.nebius.bootstrap_environment",
        return_value={
            "s3_bucket": "bucket",
            "s3_endpoint": "url",
            "nebius_api_key": "key",
            "nebius_secret_key": "secret",
            "nebius_project_id": "project",
            "nebius_region": "eu-north1",
            "iam_token": "token",
            "service_account_id": "sa",
        },
    )

    result = runner.invoke(app, ["workbench", "workflow", "teardown"])

    assert result.exit_code == 0
    assert destroy_mock.call_count == 2
    assert remove_mock.call_count == 2


def test_workflow_teardown_errors_when_no_vms_registered(mocker) -> None:
    from npa.clients.config import ConfigError

    mocker.patch(
        "npa.clients.config.resolve_ssh_config",
        side_effect=ConfigError("missing"),
    )

    result = runner.invoke(app, ["workbench", "workflow", "teardown"])

    assert result.exit_code == 1
    assert "No distill VMs found" in result.output
