"""Exercise overlapping precheck processes and fail-closed result handling."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
STUB = r"""
import os
from pathlib import Path
import sys
import time

root = Path(os.environ["PRECHECK_TEST_ROOT"])
assert "NPA_CI_SHARD_INDEX" not in os.environ
assert "NPA_CI_TOTAL_SHARDS" not in os.environ
if "ruff" in sys.argv:
    sys.exit(int(os.environ.get("LINT_STATUS", "0")))
if "--collect-only" in sys.argv:
    (root / "collection.pid").write_text(str(os.getpid()))
    (root / "collection.started").touch()
    while not (root / "tests.started").exists():
        time.sleep(0.01)
    if os.environ.get("TEST_STATUS"):
        while True:
            time.sleep(0.01)
    print("full collection result")
    sys.exit(int(os.environ.get("COLLECTION_STATUS", "0")))
(root / "tests.started").touch()
while not (root / "collection.started").exists():
    time.sleep(0.01)
print("smoke and guardrails result")
sys.exit(int(os.environ.get("TEST_STATUS", "0")))
"""


@pytest.fixture
def checkout(tmp_path):
    """Build an isolated checkout with a real subprocess-backed interpreter stub.

    Args:
        tmp_path: Private fixture directory.
    Returns:
        Checkout path and precheck command.
    Raises:
        OSError: Fixture files cannot be created.
    """
    script = tmp_path / "npa/scripts/ci_precheck.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "npa/scripts/ci_precheck.sh", script)
    interpreter = tmp_path / "npa/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!{sys.executable}\n{STUB}")
    interpreter.chmod(0o755)
    return tmp_path, ["bash", str(script)]


def _run(checkout, **changes):
    root, command = checkout
    environment = {
        **os.environ,
        "PRECHECK_TEST_ROOT": str(root),
        "TMPDIR": str(root),
        "NPA_CI_SHARD_INDEX": "1",
        "NPA_CI_TOTAL_SHARDS": "8",
        **changes,
    }
    return subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=10
    )


def test_collection_and_execution_overlap_and_both_must_pass(checkout):
    result = _run(checkout)
    assert result.returncode == 0, result.stderr
    assert "full collection result" in result.stdout
    assert "smoke and guardrails result" in result.stdout
    assert not list(checkout[0].glob("npa-precheck.*"))


def test_collection_failure_is_not_hidden_by_successful_guardrails(checkout):
    result = _run(checkout, COLLECTION_STATUS="2")
    assert result.returncode == 2
    assert "full collection result" in result.stdout


def test_failed_guardrails_stop_collection_and_preserve_failure(checkout):
    result = _run(checkout, TEST_STATUS="3")
    assert result.returncode == 3
    pid = int((checkout[0] / "collection.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not list(checkout[0].glob("npa-precheck.*"))


def test_lint_failure_prevents_expensive_work(checkout):
    result = _run(checkout, LINT_STATUS="4")
    assert result.returncode == 4
    assert not (checkout[0] / "tests.started").exists()
    assert not (checkout[0] / "collection.started").exists()
