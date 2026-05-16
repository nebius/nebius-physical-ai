from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import SSHConfig, StorageConfig, WorkbenchConfig
from npa.workflows.distill import DistillationError
from npa.workflows.distill_two_vm import TwoVMDistillError


runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    ["run", "status", "logs", "teardown", "distill"],
)
def test_workflow_command_help(command: str) -> None:
    result = runner.invoke(app, ["workflow", command, "--help"])

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


def test_workflow_run_unknown_workflow_errors() -> None:
    result = runner.invoke(app, ["workflow", "run", "unknown"])

    assert result.exit_code == 1
    assert "Unknown workflow" in result.output


def test_workflow_run_remote_requires_s3_bucket() -> None:
    result = runner.invoke(app, ["workflow", "run", "distill", "--remote"])

    assert result.exit_code == 1
    assert "--remote requires --s3-bucket" in result.output


def test_workflow_status_prints_status(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_run_status",
        return_value={"run_id": "run-1", "status": "success", "stages": {}},
    )

    result = runner.invoke(app, ["workflow", "status", "run-1"])

    assert result.exit_code == 0
    assert "run-1" in result.output
    assert "success" in result.output


def test_workflow_status_maps_distillation_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_run_status",
        side_effect=DistillationError("not found"),
    )

    result = runner.invoke(app, ["workflow", "status", "missing"])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_workflow_logs_prints_stage_logs(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_stage_logs",
        return_value="stage log text",
    )

    result = runner.invoke(app, ["workflow", "logs", "run-1", "convert"])

    assert result.exit_code == 0
    assert "stage log text" in result.output


def test_workflow_logs_maps_distillation_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill.get_stage_logs",
        side_effect=DistillationError("no logs"),
    )

    result = runner.invoke(app, ["workflow", "logs", "run-1", "bad"])

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
        ["workflow", "distill", "--student-policy", "bad"],
    )

    assert result.exit_code == 1
    assert "student-policy must be act, diffusion, or smolvla" in result.output


def test_workflow_distill_maps_two_vm_error(mocker) -> None:
    mocker.patch(
        "npa.workflows.distill_two_vm.distill",
        side_effect=TwoVMDistillError("infra failed"),
    )

    result = runner.invoke(app, ["workflow", "distill"])

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

    result = runner.invoke(app, ["workflow", "teardown"])

    assert result.exit_code == 0
    assert destroy_mock.call_count == 2
    assert remove_mock.call_count == 2


def test_workflow_teardown_errors_when_no_vms_registered(mocker) -> None:
    from npa.clients.config import ConfigError

    mocker.patch(
        "npa.clients.config.resolve_ssh_config",
        side_effect=ConfigError("missing"),
    )

    result = runner.invoke(app, ["workflow", "teardown"])

    assert result.exit_code == 1
    assert "No distill VMs found" in result.output
