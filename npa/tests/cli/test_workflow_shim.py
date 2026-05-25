from __future__ import annotations

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.workflow import app as workflow_app
from npa.cli.workflow_shim import workflow_shim_app
from npa.orchestration.skypilot.workflow import WorkflowResult


runner = CliRunner()


def test_workflow_shim_submit_matches_workbench_workflow(mocker, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    mocker.patch(
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

    assert workbench_result.exit_code == 0
    assert shim_result.exit_code == 0
    assert "Warning: npa workflow is deprecated" in shim_result.output
    assert "npa workbench workflow <command>" in shim_result.output
    assert workbench_result.output in shim_result.output


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
