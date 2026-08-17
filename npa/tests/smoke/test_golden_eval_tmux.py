"""Shell smoke tests for golden-eval tmux launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "npa" / "scripts" / "start_golden_evals_tmux.sh"

# The script prefers npa/.venv and otherwise falls back to whatever python3 is on
# PATH, which will not have npa importable. Hand it the interpreter running this
# test so the smoke works from any venv location.
SCRIPT_ENV = {**os.environ, "GOLDEN_EVAL_PYTHON": sys.executable}


def test_tmux_script_help() -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        env=SCRIPT_ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--serverless" in proc.stdout
    assert "--max-in-flight" in proc.stdout


@pytest.mark.skipif(
    subprocess.run(["bash", "-lc", "command -v tmux"], capture_output=True).returncode
    != 0,
    reason="tmux not installed",
)
def test_tmux_script_dry_run_launches_session() -> None:
    session = "golden-evals-test-smoke"
    proc = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--tools-only",
            "--session",
            session,
            "retargeting",
        ],
        cwd=REPO_ROOT,
        env=SCRIPT_ENV,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"TMUX_SESSION={session}" in proc.stdout
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)
