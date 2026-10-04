"""Exercise the actual trusted live runner with call-phase traps and no network."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _blocked_plugin(tmp_path: Path) -> Path:
    reached = tmp_path / "call-phase-spy"
    (tmp_path / "blocked_live_caller.py").write_text(
        "from pathlib import Path\n"
        "import socket\n"
        "import pytest\n"
        "def pytest_configure(config):\n"
        "    def deny(*args, **kwargs):\n"
        "        raise AssertionError('NETWORK_BLOCKED')\n"
        "    socket.socket.connect = deny\n"
        "    socket.socket.connect_ex = deny\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_runtest_call(item):\n"
        f"    Path({str(reached)!r}).write_text('selected call phase reached')\n"
        "    raise AssertionError('CALL_PHASE_BLOCKED_BEFORE_PROVIDER')\n",
        encoding="utf-8",
    )
    return reached


def _caller_environment(root: Path, tmp_path: Path, key_present: bool) -> dict:
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root / "npa/src"))),
        "PYTEST_PLUGINS": "blocked_live_caller",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if key_present:
        env["NEBIUS_TOKEN_FACTORY_KEY"] = "synthetic-test-key"
    return env


def _assert_caller_receipt(receipt: dict, key_present: bool) -> None:
    assert receipt["required_live"] is True and receipt["passed"] is False
    assert receipt["credential_present"] is key_present
    if not key_present:
        assert receipt["pytest_exit_code"] == 2
        assert all(value == 0 for value in receipt["counts"].values())
        return
    assert receipt["pytest_exit_code"] == 1
    counts = receipt["counts"]
    assert counts["collected"] == counts["executed"] == counts["failed"] > 0
    assert counts["skipped"] == counts["collection_errors"] == counts["deselected"] == 0
    assert all(
        any(
            row["nodeid"].startswith(suite.removeprefix("npa/") + "::")
            for row in receipt["tests"]
        )
        for suite in receipt["suites"]
    )


@pytest.mark.parametrize("key_present", [False, True])
def test_actual_trusted_runner_opts_in_without_ambient_integration_flag(
    tmp_path: Path, key_present: bool
):
    root = Path(__file__).resolve().parents[3]
    reached = _blocked_plugin(tmp_path)
    target = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "npa/scripts/token_factory_live_recheck.py",
            "--evidence-dir",
            str(target),
        ],
        cwd=root,
        env=_caller_environment(root, tmp_path, key_present),
        capture_output=True,
        text=True,
    )
    (tmp_path / "caller-output.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 1
    assert reached.exists() is key_present
    _assert_caller_receipt(
        json.loads((target / "receipt.json").read_text()), key_present
    )
