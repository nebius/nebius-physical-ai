"""Shell smoke tests for golden-eval converge tmux launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERGE = REPO_ROOT / "npa" / "scripts" / "golden_eval_converge.sh"
START = REPO_ROOT / "npa" / "scripts" / "start_golden_evals_converge_tmux.sh"
AUTOFIX = REPO_ROOT / "npa" / "scripts" / "golden_eval_autofix.sh"
IN_CONVERGE_LOOP = os.environ.get("GOLDEN_EVAL_CONVERGE_LOOP") == "1"


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


def test_converge_script_declares_iam_block_guard() -> None:
    text = CONVERGE.read_text(encoding="utf-8")
    assert "PAUSED-IAM" in text
    assert "_fleet_iam_blocked" in text
    assert "_pause_for_iam_block" in text


def test_fleet_iam_block_detection_pattern(tmp_path: Path) -> None:
    log_root = tmp_path / "run"
    log_root.mkdir()
    (log_root / "cosmos.log").write_text(
        '"error": "SubnetResolutionError",\n'
        '"message": "PermissionDenied: service VPC API"\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            rf"""
            log_root="{log_root}"
            for f in "$log_root"/*.log; do
              if grep -qE 'SubnetResolutionError|PermissionDenied.*VPC|service VPC API' "$f" 2>/dev/null; then
                exit 0
              fi
            done
            exit 1
            """,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_converge_unit_gate_runs_when_paused_iam_marker_present(tmp_path: Path) -> None:
    if IN_CONVERGE_LOOP:
        pytest.skip("avoid recursive converge subprocess inside converge loop")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "PAUSED-IAM").write_text("blocked\n", encoding="utf-8")
    proc = subprocess.run(
        [
            "bash",
            str(CONVERGE),
            "--once",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={
            **os.environ,
            "GOLDEN_EVAL_STATE_DIR": str(state_dir),
            "GOLDEN_EVAL_AUTOFIX_SKIP_GIT": "1",
            "GOLDEN_EVAL_PYTHON": sys.executable,
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PAUSED-IAM" in proc.stdout + proc.stderr
    assert "unit gate pass" in proc.stdout + proc.stderr
    assert not (state_dir / "golden-evals-complete").exists()


def test_autofix_script_runs() -> None:
    proc = subprocess.run(
        ["bash", str(AUTOFIX), "test-smoke"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **os.environ,
            "GOLDEN_EVAL_AUTOFIX_SKIP_GIT": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    IN_CONVERGE_LOOP,
    reason="avoid recursive tmux launcher inside converge loop",
)
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
