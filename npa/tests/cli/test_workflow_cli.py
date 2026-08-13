from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.orchestration.skypilot.workflow import ManagedJobEvidence, WorkflowResult
from npa.orchestration.skypilot.workflow_state import WorkflowS3Config
from npa.workflows.distill import DistillationError
from npa.workflows.distill_two_vm import TwoVMDistillError


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Frozen raw-task fixtures. The submit wrapper accepts a customer's own SkyPilot YAML,
#: so that contract needs a raw task to exercise -- but not a SHIPPED one, which is what
#: made these tests block the catalog's retirement. See tests/fixtures/skypilot/README.md.
SKYPILOT_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/skypilot"


@pytest.mark.parametrize(
    "command",
    [
        "submit",
        "run",
        "list",
        "status",
        "logs",
        "artifacts",
        "cancel",
        "teardown",
        "distill",
    ],
)
def test_workflow_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "workflow", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


class FakeWorkflowS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.get_requests: list[tuple[str, str]] = []

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = ""
    ) -> None:
        del ContentType
        self.objects[(Bucket, Key)] = (
            Body if isinstance(Body, bytes) else str(Body).encode("utf-8")
        )

    def get_object(self, *, Bucket: str, Key: str):
        self.get_requests.append((Bucket, Key))
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket: str, Prefix: str, PaginationConfig=None):
                contents = [
                    {"Key": key, "Size": len(body)}
                    for (bucket, key), body in sorted(client.objects.items())
                    if bucket == Bucket and key.startswith(Prefix)
                ]
                if PaginationConfig:
                    contents = contents[
                        : int(PaginationConfig.get("MaxItems") or len(contents))
                    ]
                yield {"Contents": contents}

        return Paginator()


def _patch_workflow_s3(
    monkeypatch: pytest.MonkeyPatch, fake_s3: FakeWorkflowS3
) -> None:
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.boto3.client",
        lambda *args, **kwargs: fake_s3,
    )
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


def _patch_managed_job_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence("absent"),
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
        return_value={
            "run_id": "run-1",
            "stages": {"train_teacher": {"status": "success"}},
        },
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


def test_workbench_workflow_submit_dispatches_skypilot(
    mocker, monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    submit_mock = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=WorkflowResult(status="SUBMITTED", job_id="42", returncode=0),
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")

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
    assert "run_id: run-1" in result.output
    assert "42" in result.output
    submit_mock.assert_called_once()
    assert submit_mock.call_args.args == (yaml_path, "run-1")
    assert submit_mock.call_args.kwargs["timeout"] == 30
    assert submit_mock.call_args.kwargs["secret_envs"] == ["AWS_ACCESS_KEY_ID"]


def test_submit_missing_secret_fails_before_remote_setup(
    mocker, monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    submit_mock = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        lambda **kwargs: SimpleNamespace(
            endpoint_url="https://storage.example",
            secret_values={},
            missing=("HF_TOKEN",),
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(yaml_path),
            "--project",
            "test-rtx",
            "--secret-env",
            "HF_TOKEN",
        ],
    )

    assert result.exit_code == 1
    assert "HF_TOKEN" in result.output
    assert "test-rtx" in result.output
    submit_mock.assert_not_called()


def test_submit_injects_configured_secret_without_printing_it(
    mocker, monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    secret = "configured-secret-value"
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        lambda **kwargs: SimpleNamespace(
            endpoint_url="https://storage.us-central1.nebius.cloud",
            secret_values={"HF_TOKEN": secret},
            missing=(),
        ),
    )
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
            "--project",
            "test-rtx",
            "--secret-env",
            "HF_TOKEN",
        ],
    )

    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert submit_mock.call_args.kwargs["extra_env"] == {"HF_TOKEN": secret}


def test_workbench_workflow_submit_json_exposes_run_id(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
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
            "json-run-1",
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == "json-run-1"
    assert payload["status"] == "SUBMITTED"


def test_stage_src_uses_resolved_project_storage_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_stage(**kwargs: object) -> str:
        captured.update(kwargs)
        return "s3://bucket/npa-src/npa/"

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source", fake_stage
    )
    from npa.cli.workbench import workflow as workflow_cli

    uri = workflow_cli._stage_npa_src_for_submit(
        {"bucket": "bucket"},
        s3_endpoint="https://storage.example",
        credential_values={
            "AWS_ACCESS_KEY_ID": "configured-access",
            "AWS_SECRET_ACCESS_KEY": "configured-secret",
        },
    )

    assert uri == "s3://bucket/npa-src/npa/"
    assert captured["endpoint_url"] == "https://storage.example"
    assert captured["aws_access_key_id"] == "configured-access"
    assert captured["aws_secret_access_key"] == "configured-secret"


def test_workbench_workflow_submit_instruments_durable_s3(
    monkeypatch, mocker, tmp_path
) -> None:
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
    manifest = json.loads(
        fake_s3.objects[("bucket", "run-1/manifest.json")].decode("utf-8")
    )
    assert manifest["sky_job_id"] == "42"
    assert (
        manifest["stages"]["train"]["log_uri"] == "s3://bucket/run-1/logs/train/run.log"
    )


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
    assert (
        captured["content"]
        == "name: demo-run\nresources:\n  cloud: nebius\nrun: echo demo-run\n"
    )


def test_workbench_workflow_submit_rejects_invalid_var(tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["workbench", "workflow", "submit", str(yaml_path), "--var", "missing-equals"],
    )

    assert result.exit_code == 1
    assert "Invalid --var format. Use KEY=VALUE." in result.output


def test_workbench_workflow_submit_warns_on_unresolved_placeholders(
    mocker, tmp_path
) -> None:
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
    assert (
        "Warning: unresolved placeholders remain: ${MISSING}"
        in result.output + result.stderr
    )


def test_workbench_workflow_submit_materializes_sonic_yaml(mocker) -> None:
    yaml_path = SKYPILOT_FIXTURES / "sonic-train-standalone.yaml"
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
    assert envs["POLICY_IMAGE"] == (
        "registry.example/workbench/npa-sonic:cuda13-b300-0.1.2-k8s-runtime-"
        "sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )
    assert envs["SONIC_GPU_TYPE"] == "gpu-rtx6000"
    assert envs["SONIC_IMAGE_VARIANT"] == "sonic-k8s-host-mounted"
    assert envs["S3_ENDPOINT_URL"] == "https://storage.example"
    assert envs["S3_BUCKET"] == "proof-bucket"
    assert envs["SONIC_OUTPUT_PREFIX"] == "sonic-proof/sonic-run/"
    assert envs["SONIC_MAX_ITERATIONS"] == "2"
    assert "image_id" not in task["resources"]
    assert "${" not in "\n".join(str(value) for value in envs.values())


def test_workbench_workflow_submit_materializes_registry_auth(mocker) -> None:
    yaml_path = SKYPILOT_FIXTURES / "sonic-train-standalone.yaml"
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
    yaml_path = SKYPILOT_FIXTURES / "sonic-locomotion-finetuning.yaml"
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
        assert (
            task["envs"]["POLICY_IMAGE"]
            == "registry.example/workbench/npa-sonic-mujoco:0.1.3-mvp"
        )
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


def test_workflow_list_discovers_npa_workflow_run_manifest(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    manifest = {
        "schema_version": "npa.workflow.run.v1",
        "run_id": "groot-run-4gpu",
        "workflow": "groot-1-7-finetune",
        "run_prefix_uri": "s3://bucket/groot-1-7-finetune/groot-run-4gpu",
        "updated_at": "2026-08-05T13:47:40Z",
        "steps": [],
    }
    fake_s3.put_object(
        Bucket="bucket",
        Key=("groot-1-7-finetune/groot-run-4gpu/npa-workflow/manifest.json"),
        Body=json.dumps(manifest).encode("utf-8"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "list",
            "--workflow-s3-prefix",
            "groot-1-7-finetune",
            "--s3-bucket",
            "bucket",
            "--s3-endpoint",
            "https://storage.example",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "runs": [
            {
                "run_id": "groot-run-4gpu",
                "workflow_name": "groot-1-7-finetune",
                "run_prefix_uri": (
                    "s3://bucket/groot-1-7-finetune/groot-run-4gpu/npa-workflow"
                ),
                "updated_at": "2026-08-05T13:47:40Z",
                "sky_job_id": "",
            }
        ]
    }


def test_workflow_status_prints_status(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real.monitor.sim2real_run_exists", return_value=False
    )
    mocker.patch(
        "npa.workflows.distill.get_run_status",
        return_value={"run_id": "run-1", "status": "success", "stages": {}},
    )

    result = runner.invoke(app, ["workbench", "workflow", "status", "run-1"])

    assert result.exit_code == 0
    assert "run-1" in result.output
    assert "success" in result.output


def test_workflow_status_maps_distillation_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.sim2real.monitor.sim2real_run_exists", return_value=False
    )
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
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: WorkflowResult(
            status="SUCCEEDED", job_id="42", returncode=0
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        lambda *args, **kwargs: [
            {
                "task_id": 0,
                "status": "SUCCEEDED",
                "start_at": "2026-06-07T00:00:00Z",
                "end_at": "2026-06-07T00:00:01Z",
            }
        ],
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
    fake_s3.put_object(
        Bucket="bucket",
        Key="run-1/manifest.json",
        Body=json.dumps(manifest).encode("utf-8"),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key="run-1/logs/train/status.json",
        Body=json.dumps(status).encode("utf-8"),
    )
    fake_s3.put_object(
        Bucket="bucket", Key="run-1/logs/train/run.log", Body=b"training complete\n"
    )
    fake_s3.put_object(
        Bucket="bucket", Key="run-1/artifacts/train/model.bin", Body=b"model"
    )

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


def test_exact_npa_manifest_uri_reconciles_failed_job_and_accelerator(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)

    def fail_legacy_probe(*args, **kwargs):
        raise AssertionError(
            "an exact durable manifest must not probe legacy sim2real S3"
        )

    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists", fail_legacy_probe
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: WorkflowResult(
            status="FAILED", job_id="4", returncode=0
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        lambda *args, **kwargs: [
            {"task_id": 0, "task_name": "annotate", "status": "SUCCEEDED"},
            {"task_id": 1, "task_name": "augment", "status": "FAILED"},
        ],
    )
    manifest = {
        "schema_version": "npa.workflow.run.v1",
        "workflow": "physical-ai-data-factory",
        "run_id": "paidf-1",
        "api_version": "npa.workflow/v0.0.1",
        "run_prefix_uri": "s3://bucket/paidf-1",
        "status": "submitted",
        "sky_job_id": "4",
        "steps": [
            {"state": "annotate", "status": "submitted", "resources_profile": {}},
            {
                "state": "augment",
                "status": "submitted",
                "resources_profile": {
                    "accelerators": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
                },
            },
        ],
    }
    key = "paidf-1/npa-workflow/manifest.json"
    fake_s3.put_object(Bucket="bucket", Key=key, Body=json.dumps(manifest).encode())
    uri = f"s3://bucket/{key}"

    result = runner.invoke(app, ["workbench", "workflow", "status", uri, "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "FAILED"
    assert payload["manifest_uri"] == uri
    assert payload["stages"]["annotate"]["state"] == "SUCCEEDED"
    assert payload["stages"]["augment"]["state"] == "FAILED"
    assert payload["stages"]["augment"]["requested_accelerators"].startswith("RTXPRO-")
    persisted = json.loads(fake_s3.objects[("bucket", key)])
    assert persisted["status"] == "submitted"  # status lookup is read-only
    assert persisted["steps"][0]["status"] == "submitted"


def test_workflow_dns_failure_is_unavailable_and_eight_ledger_stages_remain_visible(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    synthetic_secret = "synthetic-controller-secret"
    dns_error = (
        "dial tcp: lookup controller.synthetic.invalid: no such host; "
        f"AWS_SECRET_ACCESS_KEY={synthetic_secret}"
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(dns_error)),
    )
    stage_names = [f"stage-{index}" for index in range(8)]
    run_id = "paidf-eight-stage-ledger"
    prefix = f"{run_id}/npa-workflow"
    manifest = {
        "schema_version": "npa.workflow.run.v1",
        "workflow": "physical-ai-data-factory",
        "run_id": run_id,
        "api_version": "npa.workflow/v0.0.1",
        "run_prefix_uri": f"s3://bucket/{run_id}",
        "status": "RUNNING",
        "updated_at": "2026-08-04T00:00:00Z",
        # Runtime waves supersede this historical root identity.
        "sky_job_id": "999",
        "steps": [
            {"state": name, "status": "SUBMITTED", "resources_profile": {}}
            for name in stage_names
        ],
    }
    waves = [
        {
            "key": f"wave-{index}",
            "states": [name],
            "attempt": 1,
            "status": "running",
            "sky_status": "PENDING",
            "job_id": str(100 + index),
            "job_name": f"synthetic-{name}",
            "started_at": "2026-08-04T00:00:00Z",
        }
        for index, name in enumerate(stage_names)
    ]
    stages = [
        {
            "key": name,
            "stage": name,
            "attempt": 1,
            "logical_state": "PENDING",
            "managed_job_id": str(100 + index),
            "last_observed_at": "2026-08-04T00:02:00Z",
            "last_heartbeat_at": ("2026-08-04T00:01:00Z" if index == 0 else ""),
            "heartbeat_source": ("scheduler_task_progress" if index == 0 else ""),
            "provenance": "runtime_wave_projection",
        }
        for index, name in enumerate(stage_names)
    ]
    runtime = {
        "schema_version": "npa.workflow.runtime.v1",
        "workflow": "physical-ai-data-factory",
        "run_id": run_id,
        "api_version": "npa.workflow/v0.0.1",
        "status": "running",
        "run_prefix_uri": f"s3://bucket/{run_id}",
        "updated_at": "2026-08-04T00:02:00Z",
        "waves": waves,
        "stages": stages,
    }
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/manifest.json",
        Body=json.dumps(manifest).encode(),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/runtime.json",
        Body=json.dumps(runtime).encode(),
    )
    uri = f"s3://bucket/{prefix}/manifest.json"

    status_result = runner.invoke(
        app, ["workbench", "workflow", "status", uri, "--json"]
    )

    assert status_result.exit_code == 2, status_result.output
    payload = json.loads(status_result.output)
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["manifest_state"] == "available"
    assert payload["last_known"]["state"] in {"PENDING", "RUNNING"}
    assert payload["live_verification"]["error_code"] == "DNS_RESOLUTION_FAILED"
    assert synthetic_secret not in status_result.output
    assert payload["live_verified"] is False
    assert set(payload["stages"]) == set(stage_names)
    assert payload["stages"]["stage-0"]["managed_job_id"] == "100"
    assert payload["stages"]["stage-7"]["managed_job_id"] == "107"
    assert payload["stages"]["stage-0"]["last_heartbeat_at"] == ("2026-08-04T00:01:00Z")
    assert payload["heartbeat_stale"] is True
    assert (
        payload["live_verification"]["attempted_at"]
        > payload["stages"]["stage-0"]["last_heartbeat_at"]
    )

    human = runner.invoke(app, ["workbench", "workflow", "status", uri])
    assert human.exit_code == 2
    assert human.output.startswith("VERIFICATION_UNAVAILABLE")
    assert "last-known state:" in human.output
    assert "npa workbench workflow status" in human.output
    assert synthetic_secret not in human.output

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.tail_live_job_logs",
        lambda **kwargs: subprocess.CompletedProcess(
            ["sky", "jobs", "logs"], 1, "", dns_error
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._resolve_sky_bin", lambda value: "synthetic-sky"
    )
    logs = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "logs",
            uri,
            "--stage",
            "stage-0",
            "--json",
        ],
    )
    assert logs.exit_code == 2, logs.output
    log_payload = json.loads(logs.output)
    assert log_payload["manifest_state"] == "available"
    assert log_payload["persisted_stages"] == stage_names
    assert log_payload["managed_job_id"] == "100"
    assert log_payload["live_log_state"] == "unavailable"
    assert log_payload["error_code"] == "DNS_RESOLUTION_FAILED"
    assert "<none>" not in logs.output
    assert synthetic_secret not in logs.output

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.tail_live_job_logs",
        lambda **kwargs: subprocess.CompletedProcess(
            ["sky", "jobs", "logs"],
            0,
            f"progress AWS_SECRET_ACCESS_KEY={synthetic_secret}\n",
            "",
        ),
    )
    successful_logs = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "logs",
            uri,
            "--stage",
            "stage-0",
            "--json",
        ],
    )
    assert successful_logs.exit_code == 0, successful_logs.output
    assert synthetic_secret not in successful_logs.output
    assert "<redacted>" in json.loads(successful_logs.output)["log"]


def test_workflow_cached_status_is_opt_in_and_skips_live_controller(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    run_id = "cached-explicit-run"
    key = f"{run_id}/npa-workflow/manifest.json"
    fake_s3.put_object(
        Bucket="bucket",
        Key=key,
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "demo",
                "run_id": run_id,
                "api_version": "npa.workflow/v0.0.1",
                "status": "RUNNING",
                "sky_job_id": "55",
                "updated_at": "2026-08-04T00:00:00Z",
                "steps": [{"state": "run", "status": "RUNNING"}],
            }
        ).encode(),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("--cached must not query live SkyPilot")
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            f"s3://bucket/{key}",
            "--cached",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "CACHED"
    assert payload["verification_status"] == "CACHED"
    assert payload["live_verified"] is False
    assert payload["last_known"]["state"] == "RUNNING"


def test_status_discovers_paidf_manifest_from_project_and_run_id(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: WorkflowResult(
            status="SUCCEEDED", job_id="1", returncode=0
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        lambda *args, **kwargs: [
            {"task_id": 0, "task_name": "finalize", "status": "SUCCEEDED"}
        ],
    )
    run_id = "paidf-readme-1"
    key = f"physical-ai-data-factory/{run_id}/npa-workflow/manifest.json"
    manifest = {
        "schema_version": "npa.workflow.run.v1",
        "workflow": "physical-ai-data-factory",
        "run_id": run_id,
        "api_version": "npa.workflow/v0.0.1",
        "run_prefix_uri": f"s3://bucket/physical-ai-data-factory/{run_id}",
        "status": "submitted",
        "sky_job_id": "1",
        "updated_at": "2026-08-04T00:00:00Z",
        "steps": [
            {"state": "finalize", "status": "submitted", "resources_profile": {}}
        ],
    }
    fake_s3.put_object(Bucket="bucket", Key=key, Body=json.dumps(manifest).encode())

    def forbidden_listing(_name: str):
        raise AssertionError("known PAIDF layout must not require ListBucket")

    monkeypatch.setattr(fake_s3, "get_paginator", forbidden_listing)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            run_id,
            "--project",
            "test-rtx",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == run_id
    assert payload["status"] == "SUCCEEDED"
    assert payload["manifest_uri"] == f"s3://bucket/{key}"


@pytest.mark.parametrize(
    ("staged_key", "expected_status", "expected_exit"),
    [
        ("", "NOT_FOUND", 1),
        (
            "physical-ai-data-factory/paidf-never-launched/configs/input.json",
            "NOT_SUBMITTED",
            4,
        ),
    ],
    ids=["planned", "staged"],
)
def test_paidf_status_distinguishes_absent_from_partial_prefix(
    monkeypatch,
    staged_key: str,
    expected_status: str,
    expected_exit: int,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    _patch_managed_job_absent(monkeypatch)
    if staged_key:
        fake_s3.put_object(Bucket="bucket", Key=staged_key, Body=b"{}")

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "paidf-never-launched",
            "--project",
            "test-rtx",
            "--json",
        ],
    )

    assert result.exit_code == expected_exit, result.output
    payload = json.loads(result.output)
    assert payload["status"] == expected_status
    assert payload["sky_job_id"] == ""
    assert payload["manifest_uri"] == (
        "s3://bucket/physical-ai-data-factory/paidf-never-launched/"
        "npa-workflow/manifest.json"
    )
    checks = {item["source"]: item["outcome"] for item in payload["resolution_checks"]}
    assert checks["canonical_paidf_s3_prefix"] == ("found" if staged_key else "absent")
    assert checks["managed_job"] == "absent"
    assert "s3://bucket/paidf-never-launched/manifest.json" not in result.output


def test_paidf_conclusively_absent_cancel_is_repeat_safe(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    _patch_managed_job_absent(monkeypatch)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("never-launched cancel must not clean cloud resources")
        ),
    )
    command = [
        "workbench",
        "workflow",
        "cancel",
        "paidf-never-launched",
        "--project",
        "test-rtx",
        "--json",
    ]
    first = runner.invoke(app, command)
    second = runner.invoke(app, command)

    assert first.exit_code == second.exit_code == 0
    for result in (first, second):
        payload = json.loads(result.output)
        assert payload["outcome"] == "already_absent"
        assert payload["launch_state"] == "not_launched"
        assert payload["verification"] == "conclusively_absent"
        assert payload["cloud_calls"] is False
    assert (
        "bucket",
        "physical-ai-data-factory/paidf-never-launched/npa-workflow/manifest.json",
    ) in fake_s3.get_requests
    assert ("bucket", "paidf-never-launched/manifest.json") not in fake_s3.get_requests


def test_workflow_cancel_distinguishes_terminal_launched_run(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    manifest = {
        "run_id": "ordinary-terminal",
        "workflow_name": "ordinary",
        "sky_job_id": "44",
        "stages": {"train": {"name": "train"}},
    }
    fake_s3.put_object(
        Bucket="bucket",
        Key="ordinary-terminal/manifest.json",
        Body=json.dumps(manifest).encode(),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key="ordinary-terminal/stages/train/status.json",
        Body=json.dumps({"state": "SUCCEEDED"}).encode(),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: WorkflowResult(
            status="SUCCEEDED", job_id="44", returncode=0
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.cancellation.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "found", job_id="44", status="SUCCEEDED"
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal run must not be cancelled again")
        ),
    )

    result = runner.invoke(
        app,
        ["workbench", "workflow", "cancel", "s3://bucket/ordinary-terminal", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "terminal"
    assert payload["status"] == "SUCCEEDED"
    assert payload["cloud_calls"] is False


def test_paidf_terminal_multistage_cancel_without_root_job_id_exits_zero(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    run_id = "paidf-terminal-multistage"
    prefix = f"physical-ai-data-factory/{run_id}/npa-workflow"
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/manifest.json",
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "physical-ai-data-factory",
                "run_id": run_id,
                "status": "submitted",
                "steps": [],
            }
        ).encode(),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/runtime.json",
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.runtime.v1",
                "run_id": run_id,
                "status": "succeeded",
                "waves": [
                    {"key": "annotate", "job_id": "91", "status": "succeeded"},
                    {"key": "curate", "job_id": "92", "status": "succeeded"},
                ],
            }
        ).encode(),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal multistage run must be a no-op")
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "cancel",
            run_id,
            "--project",
            "paidf",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "terminal"
    assert payload["detected_state"] == "SUCCEEDED"
    assert payload["sky_job_ids"] == ["91", "92"]
    assert payload["cloud_calls"] is False
    assert "No cancellation was needed" in payload["message"]


def test_paidf_active_multistage_cancel_targets_each_active_wave_once(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    run_id = "paidf-active-multistage"
    prefix = f"physical-ai-data-factory/{run_id}/npa-workflow"
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/manifest.json",
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "physical-ai-data-factory",
                "run_id": run_id,
                "status": "running",
                "steps": [],
            }
        ).encode(),
    )
    fake_s3.put_object(
        Bucket="bucket",
        Key=f"{prefix}/runtime.json",
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.runtime.v1",
                "run_id": run_id,
                "status": "running",
                "waves": [
                    {
                        "key": "augment",
                        "job_id": "93",
                        "job_name": "paidf-wave-augment",
                        "status": "running",
                    },
                    {"key": "evaluate", "job_id": "94", "status": "succeeded"},
                    {
                        "key": "curate",
                        "job_id": "95",
                        "job_name": "paidf-wave-curate",
                        "status": "submitted",
                    },
                ],
            }
        ).encode(),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.cancellation.lookup_managed_job",
        lambda job_name, *, job_id, **kwargs: ManagedJobEvidence(
            "found", job_id=job_id, status="RUNNING"
        ),
    )
    calls: list[list[tuple[str, str]]] = []

    def cleanup(jobs, *args, **kwargs):  # noqa: ANN001, ANN202
        calls.append(jobs)
        return SimpleNamespace(
            resources_removed=[f"job:{job_id}" for job_id, _name in jobs],
            commands=[],
            errors=[],
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows", cleanup
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "cancel",
            run_id,
            "--project",
            "paidf",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "cancelled"
    assert payload["cancelled_job_ids"] == ["93", "95"]
    assert calls == [[("93", "paidf-wave-augment"), ("95", "paidf-wave-curate")]]


def test_launched_workflow_cancel_uses_guarded_cleanup_and_reports_cancelled(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    run_id = "paidf-launched-123"
    key = f"{run_id}/npa-workflow/manifest.json"
    fake_s3.put_object(
        Bucket="bucket",
        Key=key,
        Body=json.dumps(
            {
                "schema_version": "npa.workflow.run.v1",
                "workflow": "physical-ai-data-factory",
                "run_id": run_id,
                "api_version": "npa.workflow/v0.0.1",
                "run_prefix_uri": f"s3://bucket/{run_id}",
                "status": "submitted",
                "sky_job_id": "55",
                "steps": [{"state": "augment", "status": "running"}],
            }
        ).encode(),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_status",
        lambda *args, **kwargs: WorkflowResult(
            status="RUNNING", job_id="55", returncode=0
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        lambda *args, **kwargs: [
            {"task_id": 0, "task_name": "augment", "status": "RUNNING"}
        ],
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.cancellation.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "found", job_id="55", status="RUNNING"
        ),
    )
    calls: list[tuple[list[tuple[str, str]], str, str, object]] = []

    def fake_cleanup(jobs, actual_run_id, *, cluster="", sky_bin=None):
        calls.append((jobs, actual_run_id, cluster, sky_bin))
        return SimpleNamespace(
            ok=True,
            resources_removed=[actual_run_id],
            errors=[],
            commands=[["pinned-sky", "jobs", "cancel", "--yes", jobs[0][0]]],
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows", fake_cleanup
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "cancel",
            f"s3://bucket/{key}",
            "--sky-bin",
            "/npa/pinned/sky",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "cancelled"
    assert payload["cloud_calls"] is True
    assert payload["errors"] == []
    assert calls == [([("55", run_id)], run_id, "", "/npa/pinned/sky")]


def test_ordinary_missing_workflow_cancel_is_verification_failure(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)

    result = runner.invoke(
        app,
        ["workbench", "workflow", "cancel", "s3://bucket/ordinary-missing", "--json"],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["outcome"] == "verification_failed"
    assert payload["cloud_calls"] is False


def test_ordinary_workflow_missing_manifest_remains_an_error(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    _patch_managed_job_absent(monkeypatch)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "s3://bucket/ordinary-run",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "NOT_FOUND"
    assert payload["verification"] == "conclusively_absent"


def test_verified_never_submitted_status_after_later_sources_are_absent(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    monkeypatch.setenv("HOME", str(tmp_path))
    update_submission_state("demo", "reserved-run", {"launch_state": "reserved"})
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    _patch_managed_job_absent(monkeypatch)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "reserved-run",
            "--project",
            "demo",
            "--s3-bucket",
            "bucket",
            "--json",
        ],
    )

    assert result.exit_code == 4, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "NOT_SUBMITTED"
    assert payload["verification_status"] == "VERIFIED"


def test_prepare_run_persists_current_schema_plan_only_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = (
        REPO_ROOT
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "physical-ai-data-factory.yaml"
    )

    prepared = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "prepare-run",
            str(spec),
            "--project",
            "demo",
            "--json",
        ],
    )

    assert prepared.exit_code == 0, prepared.output
    payload = json.loads(prepared.output)
    assert payload["lifecycle_state"] == "PLAN_ONLY"
    assert payload["submission_state"] == "NOT_SUBMITTED"
    from npa.orchestration.npa_workflow.submission_state import (
        SCHEMA_VERSION,
        load_submission_state,
    )

    ledger = load_submission_state("demo", payload["run_id"])
    assert ledger["schema_version"] == SCHEMA_VERSION
    assert ledger["launch_state"] == "reserved"


def test_legacy_first_run_state_alone_remains_untrusted(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    from npa.orchestration.npa_workflow.first_run_state import prepare_run
    from npa.orchestration.npa_workflow.run_resolution import resolve_run

    prepared = prepare_run(
        project="demo", workflow_identity="legacy-workflow", new_run_id="legacy-run"
    )
    assert prepared.run_id == "legacy-run"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("external verification unavailable")

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.resolve_workflow_s3_config",
        unavailable,
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *_a, **_k: ManagedJobEvidence(
            outcome="unavailable", error="external verification unavailable"
        ),
    )
    resolution = resolve_run(
        "legacy-run", project="demo", allow_local_not_submitted=True
    )
    assert not resolution.not_submitted
    assert resolution.verification_unavailable


def test_paidf_partial_prefix_preserves_exact_workflow_s3_uri(monkeypatch) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    exact_uri = "s3://bucket/archive/physical-ai-data-factory/paidf-custom/npa-workflow"
    fake_s3.put_object(
        Bucket="bucket",
        Key="archive/physical-ai-data-factory/paidf-custom/configs/input.json",
        Body=b"{}",
    )
    _patch_managed_job_absent(monkeypatch)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "paidf-custom",
            "--workflow-s3-uri",
            exact_uri,
            "--json",
        ],
    )

    assert result.exit_code == 4, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "NOT_SUBMITTED"
    assert payload["manifest_pending"] is False
    assert payload["resolution_source"] == "explicit_workflow_s3_uri"
    assert payload["run_prefix_uri"] == exact_uri.removesuffix("/npa-workflow")
    assert payload["manifest_uri"] == f"{exact_uri}/manifest.json"
    assert payload["resolution_checks"][0]["source"] == "explicit_workflow_s3_uri"


def test_manifest_pending_status_logs_artifacts_and_cancel_share_resolution(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.monitor.sim2real_run_exists",
        lambda *args, **kwargs: False,
    )
    run_id = "legacy-active"
    run_prefix = f"physical-ai-data-factory/{run_id}"
    fake_s3.put_object(
        Bucket="bucket", Key=f"{run_prefix}/configs/input.json", Body=b"{}"
    )
    fake_s3.put_object(
        Bucket="bucket", Key=f"{run_prefix}/curate/output.json", Body=b"{}"
    )
    evidence = ManagedJobEvidence(
        "found",
        job_id="81",
        status="RUNNING",
        task_rows=(
            {
                "task_id": 0,
                "task_name": "annotate",
                "status": "SUCCEEDED",
                "last_progress_at": "2026-08-05T11:00:00Z",
            },
            {
                "task_id": 1,
                "task_name": "curate",
                "status": "PENDING",
                "retry_count": 2,
                "last_progress_at": "2026-08-05T12:00:00Z",
            },
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_controller_logs",
        lambda *args, **kwargs: __import__("subprocess").CompletedProcess(
            [],
            0,
            stdout='container not found ("ray-node")\ncontainer not found ("ray-node")\n',
            stderr="",
        ),
    )
    log_calls: list[tuple[str, str]] = []

    def tail_logs(*, job_id, stage, **kwargs):  # noqa: ANN001, ANN202
        log_calls.append((job_id, stage))
        return __import__("subprocess").CompletedProcess(
            [], 0, stdout="live curate log\n", stderr=""
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.tail_live_job_logs", tail_logs
    )
    monkeypatch.setattr(
        "npa.cli.workbench.workflow._resolve_sky_bin", lambda value="": "pinned-sky"
    )
    cleanup_calls: list[str] = []

    def cleanup(jobs, run_id, **kwargs):  # noqa: ANN001, ANN202
        cleanup_calls.extend(job_id for job_id, _job_name in jobs)
        return SimpleNamespace(
            ok=True,
            resources_removed=[run_id],
            errors=[],
            commands=[["pinned-sky", "jobs", "cancel", "--yes", jobs[0][0]]],
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.cleanup.cleanup_launched_workflows", cleanup
    )
    common = [run_id, "--project", "paidf"]

    status_result = runner.invoke(
        app, ["workbench", "workflow", "status", *common, "--json"]
    )
    artifacts_result = runner.invoke(
        app, ["workbench", "workflow", "artifacts", *common, "--json"]
    )
    logs_result = runner.invoke(
        app, ["workbench", "workflow", "logs", *common, "--stage", "curate"]
    )
    cancel_result = runner.invoke(
        app, ["workbench", "workflow", "cancel", *common, "--json"]
    )

    assert status_result.exit_code == artifacts_result.exit_code == 0
    assert logs_result.exit_code == 0, logs_result.output
    assert cancel_result.exit_code == 0, cancel_result.output
    status = json.loads(status_result.output)
    artifacts = json.loads(artifacts_result.output)
    cancelled = json.loads(cancel_result.output)
    assert status["status"] == "RETRYING"
    assert status["manifest_pending"] is True
    assert status["active_stage_name"] == "curate"
    assert status["active_stage_index"] == 2
    assert status["last_heartbeat_at"] == "2026-08-05T12:00:00Z"
    assert status["stages"]["curate"]["scheduler_state"] == "PENDING"
    assert status["stages"]["curate"]["retry_count"] == 2
    assert status["stages"]["curate"]["startup_failure_evidence"] == 2
    assert status["stages"]["curate"]["last_normalized_startup_failure"]
    assert (
        "workflow logs legacy-active --stage curate"
        in status["stages"]["curate"]["log_command"]
    )
    assert (
        status["resolution_source"]
        == artifacts["resolution_source"]
        == ("canonical_paidf_s3_prefix")
    )
    assert artifacts["manifest_pending"] is True
    assert f"s3://bucket/{run_prefix}/curate/output.json" in artifacts["artifacts"]
    assert "live curate log" in logs_result.output
    assert log_calls == [("81", "curate")]
    assert cancelled["outcome"] == "cancelled"
    assert cancelled["sky_job_id"] == "81"
    assert cleanup_calls == ["81"]


@pytest.mark.parametrize("with_receipt", [True, False], ids=["receipt", "managed-job"])
def test_manifest_pending_status_reports_receipt_or_managed_job_provenance(
    monkeypatch, with_receipt: bool
) -> None:
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    run_id = "proven-pending"
    if with_receipt:
        update_submission_state(
            "paidf",
            run_id,
            {
                "workflow": {
                    "name": "physical-ai-data-factory",
                    "run_prefix_uri": (
                        f"s3://bucket/physical-ai-data-factory/{run_id}"
                    ),
                    "steps": [{"state": "curate", "status": "submitted"}],
                },
                "launch": {"status": "submitted", "sky_job_id": "82"},
            },
        )
    evidence = ManagedJobEvidence(
        "found",
        job_id="82",
        status="RUNNING",
        task_rows=({"task_id": 0, "task_name": "curate", "status": "RUNNING"},),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow.workflow_controller_logs",
        lambda *args, **kwargs: __import__("subprocess").CompletedProcess(
            [], 0, stdout="", stderr=""
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            run_id,
            "--project",
            "paidf",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "RUNNING"
    assert payload["manifest_pending"] is True
    assert payload["manifest_state"] == "pending"
    assert payload["resolution_source"] == (
        "durable_submission_receipt" if with_receipt else "managed_job"
    )
    assert payload["active_stage_name"] == "curate"
    assert payload["stages"]["curate"]["scheduler_state"] == "RUNNING"


def test_status_json_reports_provider_unavailable_without_saying_not_found(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    monkeypatch.setattr(
        fake_s3,
        "get_object",
        lambda **kwargs: (_ for _ in ()).throw(
            PermissionError("fixture access denied")
        ),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.run_resolution.lookup_managed_job",
        lambda *args, **kwargs: ManagedJobEvidence(
            "unavailable", error="fixture SkyPilot provider unavailable"
        ),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "status",
            "unavailable-run",
            "--project",
            "paidf",
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["verification"] == "unavailable"
    assert "Run not found" not in result.output
    checks = {item["source"]: item["outcome"] for item in payload["resolution_checks"]}
    assert checks["canonical_paidf_s3_prefix"] == "unavailable"
    assert checks["managed_job"] == "unavailable"


def test_workflow_list_ignores_component_and_staged_source_manifests(
    monkeypatch,
) -> None:
    fake_s3 = FakeWorkflowS3()
    _patch_workflow_s3(monkeypatch, fake_s3)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.workflow_state.resolve_project_storage",
        lambda project=None: StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/",
            endpoint_url="https://storage.example",
        ),
    )
    run_id = "paidf-1"
    durable = {
        "schema_version": "npa.workflow.run.v1",
        "workflow": "physical-ai-data-factory",
        "run_id": run_id,
        "api_version": "npa.workflow/v0.0.1",
        "steps": [],
        "updated_at": "2026-08-04T00:00:00Z",
    }
    objects = {
        f"physical-ai-data-factory/{run_id}/npa-workflow/manifest.json": durable,
        f"physical-ai-data-factory/{run_id}/configs/manifest.json": {
            "schema": "npa.data_factory.configs.v1",
            "run_id": "configs",
        },
        f"physical-ai-data-factory/{run_id}/cosmos_augmented/manifest.json": {
            "schema": "npa.cosmos2.transfer.v1",
            "run_id": run_id,
        },
        "npa-src/npa/manifest.json": {"name": "npa", "run_id": "npa"},
    }
    for key, payload in objects.items():
        fake_s3.put_object(Bucket="bucket", Key=key, Body=json.dumps(payload).encode())

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "list",
            "--project",
            "test-rtx",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    runs = json.loads(result.output)["runs"]
    assert [row["run_id"] for row in runs] == [run_id]
    assert runs[0]["workflow_name"] == "physical-ai-data-factory"
    assert runs[0]["run_prefix_uri"].endswith(f"/{run_id}/npa-workflow")


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
        storage=StorageConfig(
            checkpoint_bucket="s3://bucket/checkpoints/", endpoint_url="url"
        ),
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
