"""Actual child/descendant cancellation must not leave verification locks held."""

import json
import signal
import subprocess
import sys
import time

import pytest

from npa.cluster.absence_process import (
    VerificationReadUnavailable,
    run_read,
    verification_deadline,
)


def _alive(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    )
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def _assert_stopped(path):
    pids = json.loads(path.read_text())
    for _ in range(100):
        if not any(_alive(pid) for pid in pids):
            return
        time.sleep(0.01)
    pytest.fail("owned verification descendant remained alive")


def _tree_script(path):
    return f"""
import os,subprocess,sys,time,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path({str(path)!r}).write_text(json.dumps([os.getpid(),child.pid]))
time.sleep(60)
"""


def test_real_timeout_kills_descendant_with_inherited_output(tmp_path):
    pids = tmp_path / "pids.json"
    with verification_deadline(0.5), pytest.raises(VerificationReadUnavailable):
        run_read([sys.executable, "-c", _tree_script(pids)])
    _assert_stopped(pids)


def test_zero_deadline_and_captured_binary_output():
    with verification_deadline(0):
        result = run_read(
            [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'\\x00ok')"],
            check=True,
        )
    assert result.stdout == b"\x00ok"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "2"])
def test_invalid_deadline_refuses_before_execution(value):
    with pytest.raises(ValueError), verification_deadline(value):
        pytest.fail("invalid deadline entered")


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM])
def test_real_parent_cancellation_joins_owned_descendants(tmp_path, cancel_signal):
    pids = tmp_path / "pids.json"
    script = (
        "from npa.cluster.absence_process import run_read; import sys; run_read([sys.executable, '-c', "
        + repr(_tree_script(pids))
        + "])"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(200):
            if pids.exists():
                break
            time.sleep(0.01)
        assert pids.exists()
        process.send_signal(cancel_signal)
        assert process.wait(timeout=5) != 0
        _assert_stopped(pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_nested_reader_helper_does_not_escape_outer_process_group(tmp_path):
    pids = tmp_path / "pids.json"
    nested = (
        "from npa.cluster.absence_process import run_read,reader_process_group; import sys\nwith reader_process_group(): run_read([sys.executable, '-c', "
        + repr(_tree_script(pids))
        + "])"
    )
    with verification_deadline(0.5), pytest.raises(VerificationReadUnavailable):
        run_read([sys.executable, "-c", nested])
    _assert_stopped(pids)
