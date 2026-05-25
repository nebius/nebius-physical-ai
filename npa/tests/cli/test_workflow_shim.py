from __future__ import annotations

import warnings

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workflow import app as workflow_app
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim_result = runner.invoke(
            app,
            ["workflow", "submit", str(yaml_path), "--run-id", "run-1"],
        )

    assert workbench_result.exit_code == 0
    assert shim_result.exit_code == 0
    assert shim_result.output == workbench_result.output
    assert any(
        item.category is DeprecationWarning
        and "npa workbench workflow" in str(item.message)
        for item in caught
    )


def test_workflow_shim_command_parity() -> None:
    canonical = {command.name for command in workflow_app.registered_commands}
    shimmed = {command.name for command in workflow_shim_app.registered_commands}

    assert shimmed == canonical
