"""Tests for batch golden-eval runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.batch import iter_containers, run_all, run_container_eval


def test_iter_containers_excludes_blocked_by_default() -> None:
    names = iter_containers(include_foundation=True)
    assert "base-cuda13-b300" not in names
    assert "cosmos3-reason" not in names


def test_iter_containers_tools_only_matches_image_names() -> None:
    from npa.smoke.manifest import load_manifest

    names = iter_containers(tools_only=True, include_foundation=False)
    blocked = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status == "blocked-on-upstream"
    }
    expected = set(CONTAINER_IMAGE_NAMES) - blocked
    assert set(names) == expected


def test_run_container_eval_dry_run() -> None:
    result = run_container_eval("retargeting")
    assert result.ok
    assert result.mode == "dry-run"
    assert "test_retargeting_functional" in result.command


def test_run_all_dry_run_includes_every_tool() -> None:
    from npa.smoke.manifest import load_manifest

    blocked = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status == "blocked-on-upstream"
    }
    expected = set(CONTAINER_IMAGE_NAMES) - blocked
    batch = run_all(
        iter_containers(tools_only=True, include_foundation=False),
        execute=False,
        serverless=False,
    )
    assert {r.name for r in batch.results} == expected
    assert batch.ok


def test_run_all_execute_workflow_smoke() -> None:
    batch = run_all(["retargeting"], execute=True, parallel=1)
    assert len(batch.results) == 1
    assert batch.results[0].ok


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_all_serverless_parallel(mock_submit) -> None:
    mock_submit.side_effect = [
        {"ok": True, "job_id": "a", "status": "completed"},
        {"ok": False, "job_id": "b", "status": "failed"},
    ]
    batch = run_all(["retargeting", "fiftyone"], serverless=True, parallel=2)
    assert mock_submit.call_count == 2
    assert not batch.ok
    assert sum(1 for r in batch.ran if r.ok) == 1


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_container_eval_serverless_submit_error_continues(mock_submit) -> None:
    from npa.clients.serverless import ServerlessClientError

    mock_submit.side_effect = ServerlessClientError("job startup failed")
    result = run_container_eval("cosmos2-transfer", serverless=True)
    assert not result.ok
    assert result.detail.get("error") == "ServerlessClientError"
    assert "job startup failed" in result.detail.get("message", "")


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_all_serverless_submit_error_does_not_abort_fleet(mock_submit) -> None:
    from npa.clients.serverless import ServerlessClientError

    mock_submit.side_effect = [
        {"ok": True, "job_id": "a", "status": "completed"},
        ServerlessClientError("job startup failed"),
        {"ok": True, "job_id": "c", "status": "completed"},
    ]
    batch = run_all(
        ["retargeting", "cosmos2-transfer", "fiftyone"],
        serverless=True,
        parallel=1,
    )
    assert mock_submit.call_count == 3
    assert len(batch.results) == 3
    assert sum(1 for r in batch.ran if r.ok) == 2
    cosmos = next(r for r in batch.results if r.name == "cosmos2-transfer")
    assert not cosmos.ok


def test_run_all_cli_dry_run() -> None:
    from click.utils import strip_ansi
    from typer.testing import CliRunner

    from npa.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "golden-eval", "run-all", "--tools-only"],
    )
    assert result.exit_code == 0, strip_ansi(result.output)
    output = strip_ansi(result.output)
    assert "lerobot" in output
    assert '"passed"' in output


def test_run_all_script_dry_run() -> None:
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    python = repo / "npa" / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.skip("venv not present")
    proc = subprocess.run(
        [
            str(python),
            str(repo / "npa" / "scripts" / "run_golden_evals.py"),
            "run-all",
            "--tools-only",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "lerobot" in proc.stdout
