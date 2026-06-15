"""Shell smoke tests for golden-eval converge tmux launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERGE = REPO_ROOT / "npa" / "scripts" / "golden_eval_converge.sh"
START = REPO_ROOT / "npa" / "scripts" / "start_golden_evals_converge_tmux.sh"
AUTOFIX = REPO_ROOT / "npa" / "scripts" / "golden_eval_autofix.sh"


def test_converge_script_help() -> None:
    proc = subprocess.run(
        ["bash", str(CONVERGE), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--once" in proc.stdout
    assert "GOLDEN_EVAL_AUTO_PUSH" in proc.stdout


def test_converge_tmux_script_help() -> None:
    proc = subprocess.run(
        ["bash", str(START), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--watchdog" in proc.stdout
    assert "golden-evals-converge" in proc.stdout


def test_autofix_script_runs() -> None:
    proc = subprocess.run(
        ["bash", str(AUTOFIX), "test-smoke"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    subprocess.run(["bash", "-lc", "command -v tmux"], capture_output=True).returncode
    != 0,
    reason="tmux not installed",
)
def test_converge_tmux_launches_session() -> None:
    session = "golden-evals-converge-test-smoke"
    proc = subprocess.run(
        [
            "bash",
            str(START),
            "--if-dead",
            "--unit-only",
            "--once",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={
            **os.environ,
            "GOLDEN_EVAL_CONVERGE_SESSION": session,
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert f"TMUX_SESSION={session}" in proc.stdout
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)
