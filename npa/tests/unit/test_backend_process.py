from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from npa.cluster_backends.process import (
    BackendCommandError,
    require_bin,
    run_capture,
    run_stream,
    terraform_plugin_cache_lock,
    terraform_env,
)


def test_require_bin_rejects_missing_and_non_executable(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(BackendCommandError, match="Required executable not found"):
        require_bin(str(missing))
    inert = tmp_path / "inert"
    inert.write_text("not executable")
    with pytest.raises(BackendCommandError, match="Required executable not found"):
        require_bin(str(inert))


def test_capture_redacts_success_and_failure_output() -> None:
    success = run_capture(["/bin/sh", "-c", "printf 'access_token=secret-value'"])
    assert "secret-value" not in success.stdout
    assert "<redacted>" in success.stdout

    with pytest.raises(BackendCommandError) as raised:
        run_capture(["/bin/sh", "-c", "printf 'secret_access_key: hidden' >&2; exit 7"])
    assert "hidden" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


def test_stream_capture_redacts_returned_output(capsys) -> None:
    result = run_stream(
        [
            "/bin/sh",
            "-c",
            "printf 'iam_token=stream-secret\\n"
            "-----BEGIN PRIVATE KEY-----\\nprivate-material\\n"
            "-----END PRIVATE KEY-----\\n'",
        ],
        capture_output=True,
    )
    assert "stream-secret" not in result.stdout
    assert "private-material" not in result.stdout
    assert "<redacted>" in result.stdout
    assert "<redacted-private-key>" in result.stdout
    emitted = capsys.readouterr().out
    assert "stream-secret" not in emitted
    assert "private-material" not in emitted


def test_default_stream_redacts_environment_secret_and_private_key(capsys) -> None:
    env = {**os.environ, "TF_VAR_iam_token": "environment-secret"}
    result = run_stream(
        [
            "/bin/sh",
            "-c",
            "printf 'environment-secret\\n"
            "-----BEGIN PRIVATE KEY-----\\nprivate-material\\n"
            "-----END PRIVATE KEY-----\\n'",
        ],
        env=env,
    )
    assert result.stdout == ""
    assert result.stderr == ""
    emitted = capsys.readouterr().out
    assert "environment-secret" not in emitted
    assert "private-material" not in emitted
    assert "<redacted>" in emitted
    assert "<redacted-private-key>" in emitted


def test_cancellable_default_stream_is_also_redacted(capsys) -> None:
    env = {**os.environ, "TF_VAR_iam_token": "cancellable-secret"}
    run_stream(
        [
            "/bin/sh",
            "-c",
            "printf 'cancellable-secret\\n"
            "-----BEGIN PRIVATE KEY-----\\nprivate-material\\n"
            "-----END PRIVATE KEY-----\\n'",
        ],
        env=env,
        cancel=lambda: None,
    )
    emitted = capsys.readouterr().out
    assert "cancellable-secret" not in emitted
    assert "private-material" not in emitted
    assert "<redacted>" in emitted
    assert "<redacted-private-key>" in emitted


def test_capture_normalizes_launch_and_timeout_errors(tmp_path: Path) -> None:
    with pytest.raises(BackendCommandError, match="Could not start executable"):
        run_capture([str(tmp_path / "absent")])
    with pytest.raises(BackendCommandError, match="timed out"):
        run_capture(["/bin/sh", "-c", "sleep 2"], timeout=0.01)


def test_stream_timeout_stops_descendant_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    with pytest.raises(BackendCommandError, match="timed out"):
        run_stream(
            [
                "/bin/sh",
                "-c",
                f"sleep 30 & echo $! > {pid_path}; wait",
            ],
            timeout=0.2,
        )
    child_pid = int(pid_path.read_text())
    for _ in range(50):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        stat_path = Path(f"/proc/{child_pid}/stat")
        try:
            state = stat_path.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            # The descendant exited between kill(0) and procfs inspection.
            break
        if state == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail("stream timeout left a descendant process running")


def test_default_stream_interrupt_stops_process_group(monkeypatch) -> None:
    from io import StringIO

    import npa.cluster_backends.process as process_module

    class InterruptedProcess:
        pid = 123
        stdout = StringIO("")
        stderr = StringIO("")

        def wait(self, **_kwargs):  # noqa: ANN001
            raise KeyboardInterrupt

    process = InterruptedProcess()
    stopped = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(process_module, "_stop_process", stopped.append)

    with pytest.raises(KeyboardInterrupt):
        run_stream(["terraform", "apply"])
    assert stopped == [process]


def test_explicit_executable_path_wins_over_path_lookup(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    old_path = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = ""
        assert require_bin(str(executable)) == str(executable)
    finally:
        os.environ["PATH"] = old_path


def test_failed_nebius_token_exchange_is_redacted(tmp_path: Path) -> None:
    nebius = tmp_path / "nebius"
    nebius.write_text("#!/bin/sh\nprintf 'iam_token=provider-secret\\n' >&2\nexit 9\n")
    nebius.chmod(0o700)
    with pytest.raises(BackendCommandError) as raised:
        terraform_env(str(nebius))
    assert "provider-secret" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


def test_terraform_plugin_cache_lock_serializes_threads(tmp_path: Path) -> None:
    env = {"TF_PLUGIN_CACHE_DIR": str(tmp_path / ".providers")}
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with terraform_plugin_cache_lock(env):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second() -> None:
        assert first_entered.wait(timeout=2)
        with terraform_plugin_cache_lock(env):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
    assert (tmp_path / ".providers.npa-init.lock").stat().st_mode & 0o777 == 0o600
