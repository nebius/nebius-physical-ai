from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "Nebius Physical AI workbench CLI"),
        (["workbench", "--help"], "Physical AI workbench tools"),
        (["workbench", "lerobot", "--help"], "LeRobot policy training"),
        (["workbench", "genesis", "--help"], "Genesis simulation"),
        (["adapter", "--help"], "Convert simulation data"),
        (["network", "--help"], "Network operations"),
        (["viz", "--help"], "visualization"),
        (["workflow", "--help"], "Multi-stage training workflow"),
        (["configure", "--help"], "credential and config setup guidance"),
        (["init", "--help"], "credential and config setup guidance"),
    ],
)
def test_help_smoke(args: list[str], expected: str) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert expected in result.output


def test_no_args_shows_top_level_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Nebius Physical AI workbench CLI" in result.output
    assert "workbench" in result.output
    assert "adapter" in result.output
    assert "network" in result.output
    assert "viz" in result.output
    assert "workflow" in result.output
    assert "configure" in result.output
    assert "init" in result.output


@pytest.mark.parametrize("command", ["configure", "init"])
def test_setup_guidance_commands_show_credentials_path(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 0
    assert "~/.npa/credentials.yaml" in result.output
    assert "HF_TOKEN" in result.output
    assert "ngc:" in result.output
    assert "api_key" in result.output
    assert "chmod 600" in result.output
