from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()

TOOLS = [
    ("cosmos", "npa.cli.cosmos"),
    ("groot", "npa.cli.groot"),
    ("isaac-lab", "npa.cli.isaac_lab"),
    ("fiftyone", "npa.cli.fiftyone"),
]


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_does_nothing_for_fresh_alias(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="fresh")
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")

    result = runner.invoke(app, ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial"])

    assert result.exit_code == 0
    assert "Nothing to clean up" in result.output
    destroy.assert_not_called()


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_refuses_fully_deployed_alias(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="fully_deployed")
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")

    result = runner.invoke(app, ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial"])

    assert result.exit_code == 1
    assert "Use `teardown` instead" in result.output
    destroy.assert_not_called()


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_skips_byovm(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="byovm")
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")

    result = runner.invoke(app, ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial"])

    assert result.exit_code == 0
    assert "BYOVM" in result.output
    destroy.assert_not_called()


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_destroys_partial_with_confirmation(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="partial")
    mocker.patch(
        f"{module}.list_terraform_managed_resources",
        return_value=["nebius_compute_instance.vm"],
    )
    mocker.patch(f"{module}.typer.confirm", return_value=True)
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")
    remove = mocker.patch(f"{module}.remove_partial_config_entry")

    result = runner.invoke(app, ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial"])

    assert result.exit_code == 0
    assert "Found orphaned resources" in result.output
    destroy.assert_called_once_with("proj", "alias")
    remove.assert_called_once_with("proj", "alias")


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_aborts_without_confirmation(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="partial")
    mocker.patch(
        f"{module}.list_terraform_managed_resources",
        return_value=["nebius_compute_instance.vm"],
    )
    mocker.patch(f"{module}.typer.confirm", return_value=False)
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")

    result = runner.invoke(app, ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial"])

    assert result.exit_code == 1
    assert "Aborted" in result.output
    destroy.assert_not_called()


@pytest.mark.parametrize(("tool", "module"), TOOLS)
def test_cleanup_partial_skips_confirmation_with_yes(tool: str, module: str, mocker) -> None:
    mocker.patch(f"{module}.classify_alias_state", return_value="partial")
    mocker.patch(
        f"{module}.list_terraform_managed_resources",
        return_value=["nebius_compute_instance.vm"],
    )
    confirm = mocker.patch(f"{module}.typer.confirm")
    destroy = mocker.patch(f"{module}.terraform_destroy_partial")
    remove = mocker.patch(f"{module}.remove_partial_config_entry")

    result = runner.invoke(
        app,
        ["workbench", tool, "-p", "proj", "-n", "alias", "cleanup-partial", "--yes"],
    )

    assert result.exit_code == 0
    confirm.assert_not_called()
    destroy.assert_called_once_with("proj", "alias")
    remove.assert_called_once_with("proj", "alias")
