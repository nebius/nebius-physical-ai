from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.workflow import app as workflow_app
from npa.cli.workflow_shim import workflow_shim_app
from npa.execution_preflight import ExecutionPreflightError
from npa.orchestration.skypilot.workflow import WorkflowResult


runner = CliRunner()


def test_workflow_shim_submit_matches_workbench_workflow(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nrun: echo hello\n", encoding="utf-8")
    preflight = mocker.patch(
        "npa.cli.workbench.workflow._raw_execution_preflight",
        return_value=(None, {}, {}),
    )
    submit = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=WorkflowResult(status="SUBMITTED", job_id="42", returncode=0),
    )

    workbench_result = runner.invoke(
        app,
        ["workbench", "workflow", "submit", str(yaml_path), "--run-id", "run-1"],
    )
    shim_result = runner.invoke(
        app,
        ["workflow", "submit", str(yaml_path), "--run-id", "run-1"],
    )

    assert workbench_result.exit_code == 0, workbench_result.output
    assert shim_result.exit_code == 0, shim_result.output
    assert "Warning: npa workflow is deprecated" in shim_result.output
    assert "npa workbench workflow <command>" in shim_result.output
    assert workbench_result.output in shim_result.output
    assert preflight.call_count == submit.call_count == 2
    assert preflight.call_args_list[0] == preflight.call_args_list[1]
    assert preflight.call_args.args[0] == [{"name": "demo", "run": "echo hello"}]


@pytest.mark.parametrize("command", [["workbench", "workflow"], ["workflow"]])
def test_workflow_shim_preserves_mandatory_preflight_rejection(mocker, tmp_path, command) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nrun: echo hello\n", encoding="utf-8")
    preflight = mocker.patch(
        "npa.cli.workbench.workflow._raw_execution_preflight",
        side_effect=ExecutionPreflightError("scope", "selected target disagrees"),
    )
    submit = mocker.patch("npa.orchestration.skypilot.workflow.submit_workflow")

    result = runner.invoke(app, [*command, "submit", str(yaml_path), "--run-id", "run-1", "--skip-preflight"])

    assert result.exit_code == 1
    assert "selected target disagrees" in result.output
    preflight.assert_called_once()
    submit.assert_not_called()


def test_workflow_shim_command_parity() -> None:
    canonical = {command.name for command in workflow_app.registered_commands}
    shimmed = {command.name for command in workflow_shim_app.registered_commands}

    assert shimmed == canonical


def test_workflow_shim_is_hidden_from_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert " workflow " not in result.output


def test_workflow_shim_help_prints_deprecation_warning() -> None:
    result = runner.invoke(app, ["workflow", "--help"])

    assert result.exit_code == 0
    assert "Warning: npa workflow is deprecated" in result.stderr
    assert "Usage: npa workflow" in result.stdout
