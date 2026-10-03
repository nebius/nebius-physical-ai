"""Actual child/descendant cancellation must not leave verification locks held."""

import json
import os
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


def test_success_keeps_leader_unreaped_until_descendant_cleanup(tmp_path, monkeypatch):
    from npa.cluster import absence_process as module

    pids = tmp_path / "pids.json"
    seen = []
    stop = module._stop_group

    def observed(process, **kwargs):
        seen.append(process.returncode)
        return stop(process, **kwargs)

    monkeypatch.setattr(module, "_stop_group", observed)
    source = _tree_script(pids).rsplit("time.sleep(60)", 1)[0]
    assert run_read([sys.executable, "-c", source]).returncode == 0
    assert seen == [None]
    _assert_stopped(pids)


def test_already_reaped_process_never_signals_a_recycled_group(monkeypatch):
    from types import SimpleNamespace
    from npa.cluster import absence_process as module

    monkeypatch.setattr(module.os, "killpg", lambda *args: pytest.fail("unowned group"))
    monkeypatch.setattr(module.os, "kill", lambda *args: pytest.fail("unowned PID"))
    module._stop_group(SimpleNamespace(pid=1234, returncode=0))


@pytest.mark.parametrize("observation_error", [False, True])
def test_external_reap_without_popen_returncode_never_signals(
    monkeypatch, observation_error
):
    from npa.cluster import absence_process as module

    def externally_reaped(process, streams, limit):
        waited, _ = os.waitpid(process.pid, 0)
        assert waited == process.pid
        assert process.returncode is None
        if observation_error:
            raise ChildProcessError("another child reaper consumed exit status")

    monkeypatch.setattr(module, "_wait_for_read", externally_reaped)
    monkeypatch.setattr(module.os, "killpg", lambda *args: pytest.fail("unowned group"))
    monkeypatch.setattr(module.os, "kill", lambda *args: pytest.fail("unowned PID"))
    with pytest.raises(VerificationReadUnavailable, match="ownership unavailable"):
        run_read([sys.executable, "-c", "pass"])


def test_waitid_missing_child_never_authorizes_group_signal(monkeypatch):
    from types import SimpleNamespace
    from npa.cluster import absence_process as module

    def missing_child(*args):
        raise ChildProcessError("externally reaped")

    monkeypatch.setattr(module.os, "waitid", missing_child, raising=False)
    monkeypatch.setattr(module.os, "killpg", lambda *args: pytest.fail("unowned group"))
    with pytest.raises(VerificationReadUnavailable, match="ownership unavailable"):
        module._stop_group(SimpleNamespace(pid=1234, returncode=None))


def test_reader_group_refuses_a_parent_without_owned_session(monkeypatch):
    from npa.cluster.absence_process import reader_process_group

    monkeypatch.delenv("NPA_ABSENCE_READER_PARENT_PID", raising=False)
    with pytest.raises(VerificationReadUnavailable, match="run_read child session"):
        with reader_process_group():
            pytest.fail("unowned context entered")


def test_excess_output_refuses_and_stops_the_actual_child(monkeypatch):
    monkeypatch.setenv("NPA_ABSENCE_MAX_OUTPUT_BYTES", "1024")
    with pytest.raises(VerificationReadUnavailable, match="byte limit"):
        run_read(
            [
                sys.executable,
                "-c",
                "import os,time; os.write(1,b'x'*2048);time.sleep(60)",
            ]
        )


def test_output_limit_zero_keeps_explicit_large_read_possible(monkeypatch):
    monkeypatch.setenv("NPA_ABSENCE_MAX_OUTPUT_BYTES", "0")
    assert len(run_read([sys.executable, "-c", "print('x'*2048)"]).stdout) == 2049


@pytest.mark.parametrize(
    "output", ["1 9 S\n", "9 9 S\n", "9 9 Z\n10 9 S\n", "malformed", ""]
)
def test_denied_group_observation_never_proves_cleanup(monkeypatch, output):
    from npa.cluster import absence_process as module

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout=output),
    )
    assert not module._darwin_group_only_zombies(9)
