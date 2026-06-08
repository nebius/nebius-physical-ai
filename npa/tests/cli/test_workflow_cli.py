from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.orchestration.skypilot.workflow import WorkflowResult
from npa.workflows.distill import DistillationError
from npa.workflows.distill_two_vm import TwoVMDistillError


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "command",
    ["submit", "run", "status", "logs", "teardown", "distill"],
)
def test_workflow_command_help(command: str) -> None:
    result = runner.invoke(app, ["workbench", "workflow", command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


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
    assert task["resources"]["image_id"] == "docker:registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert envs["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    assert envs["SONIC_GPU_TYPE"] == "gpu-rtx6000"
    assert envs["SONIC_IMAGE_VARIANT"] == "sonic-k8s-host-mounted"
    assert envs["S3_ENDPOINT_URL"] == "https://storage.example"
    assert envs["S3_BUCKET"] == "proof-bucket"
    assert envs["SONIC_OUTPUT_PREFIX"] == "sonic-proof/sonic-run/"
    assert envs["SONIC_MAX_ITERATIONS"] == "2"
    assert "${" not in task["resources"]["image_id"]
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
    assert [doc["name"] for doc in docs[1:]] == ["sonic-g1-finetune", "sonic-mujoco-eval"]
    for task in docs[1:]:
        assert task["resources"]["accelerators"] == "H100:1"
        assert task["resources"]["region"] == "eu-north1"
        assert task["resources"]["use_spot"] is True
        assert "image_id" not in task["resources"]
        assert task["envs"]["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic-mujoco:0.1.3-mvp"
        assert task["envs"]["SONIC_PAYLOAD_MODE"] == "docker"
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
