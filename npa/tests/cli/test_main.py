from __future__ import annotations

import json
import re
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients.serverless import NotEnoughResourcesError


runner = CliRunner()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "Nebius Physical AI workbench CLI"),
        (["workbench", "--help"], "Physical AI workbench tools"),
        (["workbench", "lerobot", "--help"], "LeRobot policy training"),
        (["workbench", "genesis", "--help"], "Genesis simulation"),
        (["adapter", "--help"], "Convert simulation data"),
        (["convert", "--help"], "standalone formats"),
        (["demo", "--help"], "Demo artifact bootstrap"),
        (["network", "--help"], "Network operations"),
        (["rerun", "--help"], "Host and share Rerun"),
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
    assert "convert" in result.output
    assert "demo" in result.output
    assert "network" in result.output
    assert "rerun" in result.output
    assert "viz" in result.output
    assert "workflow" in result.output
    assert "configure" in result.output
    assert "init" in result.output


def test_version_flag_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert re.match(r"^npa \d+\.\d+(\.\d+)?", result.stdout)
    assert result.stdout.strip() == f"npa {version('npa')}"


@pytest.mark.parametrize("command", ["configure", "init"])
def test_setup_guidance_commands_show_credentials_path(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 0
    assert "~/.npa/credentials.yaml" in result.output
    assert "HF_TOKEN" in result.output
    assert "ngc:" in result.output
    assert "api_key" in result.output
    assert "chmod 600" in result.output


def test_app_entry_typed_error_exits_one_without_traceback(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise NotEnoughResourcesError(
            "capacity blocked",
            project_id="project-1",
            platform="gpu-h200-sxm",
            suggested_alternatives=["Retry in a few minutes"],
        )

    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Not enough resources" in err
    assert "Retry in a few minutes" in err
    assert "Traceback" not in err


def test_app_entry_typed_error_json_mode(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise NotEnoughResourcesError("capacity blocked", project_id="project-1")

    monkeypatch.setenv("NPA_ERROR_FORMAT", "json")
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "NotEnoughResources"
    assert payload["project_id"] == "project-1"


def test_app_entry_unexpected_error_no_stacktrace_by_default(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.delenv("NPA_DEBUG", raising=False)
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Unexpected error: boom" in err
    assert "NPA_DEBUG=1" in err
    assert "Traceback" not in err


def test_app_entry_unexpected_error_stacktrace_with_debug(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.setenv("NPA_DEBUG", "1")
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 2
    assert "Traceback" in capsys.readouterr().err
