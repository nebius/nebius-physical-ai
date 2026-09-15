"""Shell smoke tests for golden-eval converge tmux launcher."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERGE = REPO_ROOT / "npa" / "scripts" / "golden_eval_converge.sh"
START = REPO_ROOT / "npa" / "scripts" / "start_golden_evals_converge_tmux.sh"
AUTOFIX = REPO_ROOT / "npa" / "scripts" / "golden_eval_autofix.sh"
TMUX = shutil.which("tmux")

# Keep the real shell flow, but never install into the pytest runner's venv or
# recursively run its tests. Record every interpreter boundary, including cwd.
PYTHON_STUB = r"""#!/bin/sh
set -eu
root="$(cd "$(dirname "$0")/../../.." && pwd)"
{
  printf '%s\t' "$PWD" "$@"
  printf '\n'
} >> "$root/calls.log"
case "$*" in
  "-c import sys" | \
  "-m pip install -e $root/npa -q" | \
  "$root/npa/scripts/run_golden_evals.py validate" | \
  "npa/scripts/audit_workbench_image_tags.py") exit 0 ;;
  "-m pytest npa/tests/smoke/test_golden_eval_fixture.py -q")
    test "${GOLDEN_EVAL_CONVERGE_LOOP:-}" = 1
    exit "$(cat "$root/unit-exit")" ;;
  *) echo "unexpected interpreter invocation: $*" >&2; exit 97 ;;
esac
"""


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def golden_checkout(tmp_path: Path) -> Path:
    """Copy scripts so autofix and launcher chmod only touch fixture files."""
    root = tmp_path / "checkout"
    scripts = root / "npa/scripts"
    scripts.mkdir(parents=True)
    for source in (CONVERGE, START, AUTOFIX):
        shutil.copy2(source, scripts / source.name)
    interpreter = root / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    _write_executable(interpreter, PYTHON_STUB)
    tests = root / "npa/tests/smoke"
    tests.mkdir(parents=True)
    (tests / "test_golden_eval_fixture.py").write_text(
        'raise AssertionError("pytest must use the fixture boundary")\n',
        encoding="utf-8",
    )
    (root / "unit-exit").write_text("0\n", encoding="utf-8")
    (root / "state").mkdir()
    (root / "home").mkdir()
    (root / "bin").mkdir()
    # A mistaken fleet or git route must fail locally, never reach host tools.
    blocked = (
        '#!/bin/sh\nprintf "%s\\n" "$0 $*" '
        '>> "$GOLDEN_EVAL_STATE_DIR/unexpected.log"\nexit 97\n'
    )
    for path in (
        root / "bin/git",
        root / "bin/tmux",
        scripts / "start_golden_evals_tmux.sh",
    ):
        _write_executable(path, blocked)
    return root


@pytest.fixture
def golden_env(golden_checkout: Path) -> dict[str, str]:
    """Exclude ambient auto-push, interpreter, state and tmux settings."""
    root = golden_checkout
    path = f"{root / 'bin'}:{os.defpath}"
    bash_env = root / "bash-env"
    # The launcher uses login shells, which can otherwise reset the stub PATH.
    bash_env.write_text(f"export PATH={shlex.quote(path)}\n", encoding="utf-8")
    return {
        "PATH": path,
        "BASH_ENV": str(bash_env),
        "HOME": str(root / "home"),
        "SHELL": "/bin/bash",
        "TERM": "xterm",
        "GOLDEN_EVAL_STATE_DIR": str(root / "state"),
        "GOLDEN_EVAL_SOURCE_REF": "fixture",
        "GOLDEN_EVAL_AUTOFIX_SKIP_GIT": "1",
        "GOLDEN_EVAL_AUTO_COMMIT": "0",
        "GOLDEN_EVAL_AUTO_PUSH": "0",
        "GOLDEN_EVAL_PYTHON": str(root / "npa/.venv/bin/python"),
    }


def _assert_interpreter_calls(root: Path, *, unit_gate: bool) -> None:
    calls = [
        line.split("\t")[:-1] for line in (root / "calls.log").read_text().splitlines()
    ]
    expected = [
        [str(root), "-m", "pip", "install", "-e", str(root / "npa"), "-q"],
        [str(root), str(root / "npa/scripts/run_golden_evals.py"), "validate"],
    ]
    if unit_gate:
        expected.insert(0, [str(root), "-c", "import sys"])
        expected.extend(
            [
                [str(root), "npa/scripts/audit_workbench_image_tags.py"],
                [
                    str(root),
                    "-m",
                    "pytest",
                    "npa/tests/smoke/test_golden_eval_fixture.py",
                    "-q",
                ],
            ]
        )
    assert calls == expected
    assert not (root / "state/unexpected.log").exists()


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


@pytest.mark.parametrize("unit_exit", [0, 1], ids=["pass", "fail"])
def test_converge_unit_gate_runs_when_paused_iam_marker_present(
    golden_checkout: Path, golden_env: dict[str, str], unit_exit: int
) -> None:
    state_dir = golden_checkout / "state"
    (state_dir / "PAUSED-IAM").write_text("blocked\n", encoding="utf-8")
    (golden_checkout / "unit-exit").write_text(str(unit_exit), encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(golden_checkout / "npa/scripts" / CONVERGE.name), "--once"],
        cwd=golden_checkout,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env=golden_env,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == unit_exit, output
    assert "PAUSED-IAM" in output
    _assert_interpreter_calls(golden_checkout, unit_gate=True)
    if unit_exit == 0:
        assert "unit gate pass" in output
        assert output.index("unit tests:") < output.index("unit gate pass")
    else:
        assert "unit tests failed attempt=1" in output
        assert "giving up (--once)" in output
        assert "unit gate pass" not in output
        assert "unit gate ok" not in output
    assert (state_dir / "PAUSED-IAM").read_text(encoding="utf-8") == "blocked\n"
    assert not (state_dir / "golden-evals-complete").exists()


def test_autofix_script_runs(golden_checkout: Path, golden_env: dict[str, str]) -> None:
    proc = subprocess.run(
        ["bash", str(golden_checkout / "npa/scripts" / AUTOFIX.name), "test-smoke"],
        cwd=golden_checkout,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=golden_env,
    )
    assert proc.returncode == 0, proc.stderr
    _assert_interpreter_calls(golden_checkout, unit_gate=False)
    assert "run_id=test-smoke" in (golden_checkout / "state/autofix.log").read_text()


@pytest.fixture
def fixture_tmux(golden_checkout: Path, golden_env: dict[str, str]):
    """Route real tmux to a fixture socket, including cleanup after assertions."""
    wrapper = golden_checkout / "bin/tmux"
    # A relative socket path also works when pytest's temporary path exceeds
    # the Unix socket path length limit. Every client starts in this checkout.
    _write_executable(
        wrapper,
        f"#!/bin/sh\ncd {shlex.quote(str(golden_checkout))}\n"
        f'exec {shlex.quote(TMUX)} -S tmux.sock -f /dev/null "$@"\n',
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(wrapper), *args],
            cwd=golden_checkout,
            env=golden_env,
            capture_output=True,
            text=True,
            check=False,
        )

    try:
        yield run
    finally:
        run("kill-server")


def _wait_for_converge_exit(root: Path, deadline: float) -> str:
    log = root / "state/converge-tmux.log"
    while time.monotonic() < deadline:
        if log.exists():
            output = log.read_text(encoding="utf-8")
            if "converge_exit=" in output:
                return output
        time.sleep(0.05)
    pytest.fail(
        "fixture converge did not finish: "
        + (log.read_text() if log.exists() else "no log")
    )


@pytest.mark.skipif(TMUX is None, reason="tmux not installed")
def test_converge_tmux_launches_session(
    golden_checkout: Path, golden_env: dict[str, str], fixture_tmux
) -> None:
    session = f"golden-evals-converge-test-{uuid4().hex}"
    deadline = time.monotonic() + 30
    proc = subprocess.run(
        [
            "bash",
            str(golden_checkout / "npa/scripts" / START.name),
            "--if-dead",
            "--unit-only",
            "--once",
        ],
        cwd=golden_checkout,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={
            **golden_env,
            "GOLDEN_EVAL_CONVERGE_SESSION": session,
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert f"TMUX_SESSION={session}" in proc.stdout
    assert fixture_tmux("has-session", "-t", session).returncode == 0
    windows = fixture_tmux("list-windows", "-t", session, "-F", "#{window_name}")
    assert windows.returncode == 0, windows.stderr
    assert set(windows.stdout.splitlines()) == {"dashboard", "converge"}
    output = _wait_for_converge_exit(golden_checkout, deadline)
    assert "converge_exit=0" in output
    assert "golden-eval converge complete" in output
    assert (golden_checkout / "state/golden-evals-complete").exists()
    _assert_interpreter_calls(golden_checkout, unit_gate=True)
