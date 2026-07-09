"""CLI coverage for npa workbench byof."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench.byof import build_byof_argv
from npa.sdk.workbench import byof as byof_sdk

runner = CliRunner()


def test_byof_registered_in_workbench_help() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert re.search(r"│\s+byof\s+", result.output)


def test_byof_run_help() -> None:
    result = runner.invoke(app, ["workbench", "byof", "run", "--help"])
    assert result.exit_code == 0
    assert "--repo-url" in result.output
    assert "--workload" in result.output
    assert "--base-profile" in result.output


def test_byof_run_dry_run_json() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/example/repo.git",
            "--repo-ref",
            "main",
            "--workload",
            "container-verify",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["script"] == "npa/scripts/run_byof_repo.py"
    assert "--repo-url" in payload["argv"]
    assert "https://github.com/example/repo.git" in payload["argv"]
    assert payload["ladder"] == "docs/architecture/oss-onboarding-ladder.md"


def test_byof_ladder_and_status() -> None:
    ladder = runner.invoke(app, ["workbench", "byof", "ladder", "--output", "json"])
    assert ladder.exit_code == 0
    ladder_payload = json.loads(ladder.output)
    assert ladder_payload["tiers"][0]["tier"] == 0

    status = runner.invoke(app, ["workbench", "byof", "status", "--output", "json"])
    assert status.exit_code == 0
    status_payload = json.loads(status.output)
    assert status_payload["cli"] == "npa workbench byof"
    assert status_payload["sdk"] == "npa.sdk.workbench.byof"
    assert "workbench.byof.repo" in status_payload["tool_refs"]


def test_build_byof_argv_and_sdk_plan() -> None:
    argv = build_byof_argv(
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        workload="container-verify",
        skip_run=True,
    )
    assert "--skip-run" in argv
    assert byof_sdk.plan_argv(
        repo_url="https://github.com/example/repo.git",
        skip_run=True,
    ) == argv
