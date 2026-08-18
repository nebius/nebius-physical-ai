from __future__ import annotations

import subprocess
from io import StringIO
from urllib.parse import quote

import pytest

from npa.clients.nebius_vm_auth import (
    VmAuthError,
    parse_auth_transcript,
    redact_auth_output,
    ssh_localhost_forward,
    verify_profile,
)
from npa.clients import nebius_vm_auth


def _transcript(port: int = 49152) -> str:
    callback = quote(f"http://127.0.0.1:{port}/callback", safe="")
    return f"Open https://auth.nebius.com/oauth?redirect_uri={callback}&state=public-state\n"


def _long_transcript(port: int = 49152, *, terminator: str = "\n") -> str:
    callback = quote(f"http://127.0.0.1:{port}/callback", safe="")
    padding = "a" * 320
    return (
        "Open https://auth.nebius.com/oauth?client_id=cli&scope=openid&state="
        f"{padding}&prompt=login&redirect_uri={callback}&response_type=code{terminator}"
    )


def test_parses_dynamic_loopback_callback_and_exact_forward() -> None:
    result = parse_auth_transcript(
        _transcript(52731),
        ssh_host="operator.example",
        ssh_user="ubuntu",
        identity_file="/local/key with space",
    )
    assert result.callback_port == 52731
    assert result.browser_url.startswith("https://auth.nebius.com/")
    assert result.ssh_command == (
        "ssh -N -L 52731:127.0.0.1:52731 -i '/local/key with space' "
        "ubuntu@operator.example"
    )


def test_duplicate_terminal_redraws_are_one_safe_candidate() -> None:
    transcript = _long_transcript(52731, terminator="\r") * 3
    result = parse_auth_transcript(transcript, ssh_host="operator.example")
    assert result.callback_port == 52731
    assert result.browser_url == _long_transcript(52731).removeprefix("Open ").strip()


def test_distinct_safe_browser_urls_remain_ambiguous() -> None:
    with pytest.raises(VmAuthError, match="exactly one safe browser URL"):
        parse_auth_transcript(
            _long_transcript(4040)
            + _long_transcript(4040).replace("state=", "state=other-"),
            ssh_host="operator.example",
        )


@pytest.mark.parametrize(
    "transcript",
    [
        "Open https://evil.example/auth?redirect_uri=http%3A%2F%2Flocalhost%3A4040%2Fcb",
        "Open https://auth.nebius.com/auth without callback",
        "Open https://auth.nebius.com/auth?redirect_uri=http%3A%2F%2F0.0.0.0%3A4040%2Fcb",
        _transcript(4040) + _transcript(5050),
        "Open https://auth.nebius.com/auth?access_token=secret&redirect_uri=http%3A%2F%2Flocalhost%3A4040%2Fcb",
    ],
)
def test_malformed_or_unsafe_transcript_fails_closed(transcript: str) -> None:
    with pytest.raises(VmAuthError):
        parse_auth_transcript(transcript, ssh_host="operator.example")


def test_forward_rejects_unsafe_host_or_port() -> None:
    with pytest.raises(VmAuthError):
        ssh_localhost_forward(0, ssh_host="operator.example")
    with pytest.raises(VmAuthError):
        ssh_localhost_forward(4040, ssh_host="bad host")


def test_redaction_removes_tokens_and_urls() -> None:
    raw = (
        "access_token: secret-value\n"
        "Authorization=Bearer hidden\n"
        "eyJabcdefghijk.abcdefghijk.signature\n"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n"
        + _transcript()
    )
    redacted = redact_auth_output(raw)
    assert "secret-value" not in redacted
    assert "Bearer hidden" not in redacted
    assert "eyJabcdefghijk" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "https://" not in redacted
    assert "[REDACTED]" in redacted


def test_profile_readiness_scrubs_ambient_tokens_and_discards_output() -> None:
    calls: list[tuple[list[str], dict]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="must-not-surface")

    result = verify_profile(
        "operator",
        env={
            "PATH": "/bin",
            "NEBIUS_IAM_TOKEN": "stale",
            "NEBIUS_IAM_TOKEN_FILE": "/tmp/stale",
        },
        runner=runner,
    )
    assert result.identity_verified is True
    assert result.iam_token_minted is True
    assert [call[0][-2:] for call in calls] == [
        ["iam", "whoami"],
        ["iam", "get-access-token"],
    ]
    assert all(call[1]["stdout"] is subprocess.DEVNULL for call in calls)
    assert all("NEBIUS_IAM_TOKEN" not in call[1]["env"] for call in calls)
    assert all("NEBIUS_IAM_TOKEN_FILE" not in call[1]["env"] for call in calls)


def test_profile_readiness_reports_failed_identity_without_token_text() -> None:
    returncodes = iter([1, 0])

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, next(returncodes))

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is True
    assert not hasattr(result, "token")


def test_already_authenticated_profile_never_starts_browser_flow(monkeypatch) -> None:
    ready = nebius_vm_auth.ProfileVerification("operator", True, True)
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(nebius_vm_auth, "verify_profile", lambda *args, **kwargs: ready)
    monkeypatch.setattr(
        nebius_vm_auth.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not launch")
        ),
    )
    output = StringIO()
    result = nebius_vm_auth.run_vm_profile_auth(
        ssh_host="operator.example", profile="operator", output=output
    )
    assert result == ready
    assert "already authenticated" in output.getvalue()


class _FakeProcess:
    def __init__(self, stdout) -> None:
        self.stdout = stdout
        self.returncode = None
        self.signals = []
        self.pid = 12345
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _ChunkProcess(_FakeProcess):
    def __init__(self, chunks: list[bytes], *, returncode: int = 0) -> None:
        super().__init__(object())
        self.chunks = list(chunks)
        self.final_returncode = returncode
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.returncode


def _drive_chunks(process: _ChunkProcess):
    def wait_for_output(_stream, _timeout):
        if process.chunks:
            return True
        process.returncode = process.final_returncode
        return False

    def read_output(_stream):
        return process.chunks.pop(0)

    return wait_for_output, read_output


@pytest.mark.parametrize("terminal_width", [80, 100, 120])
def test_vm_auth_uses_wide_pty_and_handles_long_late_redirect(
    monkeypatch, terminal_width: int
) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("operator", False, False)
    ready = nebius_vm_auth.ProfileVerification("operator", True, True)
    verifications = iter([not_ready, ready])
    monkeypatch.setattr(
        nebius_vm_auth, "verify_profile", lambda *args, **kwargs: next(verifications)
    )
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("COLUMNS", str(terminal_width))
    process = _ChunkProcess([_long_transcript(52731).encode()])
    wait_for_output, read_output = _drive_chunks(process)
    popen_calls = []

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    monkeypatch.setattr(nebius_vm_auth.subprocess, "Popen", popen)
    output = StringIO()
    result = nebius_vm_auth.run_vm_profile_auth(
        ssh_host="operator.example",
        ssh_user="ubuntu",
        profile="operator",
        output=output,
        _clock=lambda: 0.0,
        _wait_for_output=wait_for_output,
        _read_output=read_output,
    )
    assert result == ready
    assert "stty cols 4096 rows 24" in popen_calls[0][0][2]
    assert "profile create operator && exec" in popen_calls[0][0][2]
    assert "iam whoami" in popen_calls[0][0][2]
    assert popen_calls[0][1]["start_new_session"] is True
    assert "52731:127.0.0.1:52731" in output.getvalue()


def test_vm_auth_handles_cr_only_redraw_and_partial_reads_without_leaking_process_output(
    monkeypatch,
) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("operator", False, False)
    ready = nebius_vm_auth.ProfileVerification("operator", True, True)
    verifications = iter([not_ready, ready])
    monkeypatch.setattr(
        nebius_vm_auth, "verify_profile", lambda *args, **kwargs: next(verifications)
    )
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda name: f"/usr/bin/{name}")
    transcript = _long_transcript(52731, terminator="\r")
    raw = ("access_token: process-secret\r" + transcript + transcript).encode()
    process = _ChunkProcess([raw[:31], raw[31:173], raw[173:]])
    wait_for_output, read_output = _drive_chunks(process)
    monkeypatch.setattr(
        nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process
    )
    output = StringIO()
    result = nebius_vm_auth.run_vm_profile_auth(
        ssh_host="operator.example",
        output=output,
        _clock=lambda: 0.0,
        _wait_for_output=wait_for_output,
        _read_output=read_output,
    )
    rendered = output.getvalue()
    assert result == ready
    assert "process-secret" not in rendered
    assert rendered.count("Open locally:") == 1


def test_successful_process_with_malformed_output_fails_closed_and_is_reaped(
    monkeypatch,
) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("", False, False)
    monkeypatch.setattr(
        nebius_vm_auth, "verify_profile", lambda *args, **kwargs: not_ready
    )
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda name: f"/usr/bin/{name}")
    process = _ChunkProcess([b"unexpected terminal output\r"])
    wait_for_output, read_output = _drive_chunks(process)
    monkeypatch.setattr(
        nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process
    )
    with pytest.raises(VmAuthError, match="verifiable safe loopback"):
        nebius_vm_auth.run_vm_profile_auth(
            ssh_host="operator.example",
            output=StringIO(),
            _clock=lambda: 0.0,
            _wait_for_output=wait_for_output,
            _read_output=read_output,
        )
    assert process.wait_calls >= 1


def test_vm_auth_timeout_cancels_child(monkeypatch) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("", False, False)
    monkeypatch.setattr(
        nebius_vm_auth, "verify_profile", lambda *args, **kwargs: not_ready
    )
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        nebius_vm_auth.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = _FakeProcess(StringIO(""))
    monkeypatch.setattr(
        nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process
    )
    times = iter([0.0, 2.0])
    with pytest.raises(VmAuthError, match="timed out"):
        nebius_vm_auth.run_vm_profile_auth(
            ssh_host="operator.example",
            auth_timeout_seconds=1,
            output=StringIO(),
            _clock=lambda: next(times),
        )
    assert nebius_vm_auth.signal.SIGINT in process.signals


def test_vm_auth_keyboard_cancel_cancels_child(monkeypatch) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("", False, False)
    monkeypatch.setattr(
        nebius_vm_auth, "verify_profile", lambda *args, **kwargs: not_ready
    )
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        nebius_vm_auth.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process = _FakeProcess(object())
    monkeypatch.setattr(
        nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process
    )

    def cancel(_stream, _timeout):
        raise KeyboardInterrupt

    with pytest.raises(VmAuthError, match="cancelled"):
        nebius_vm_auth.run_vm_profile_auth(
            ssh_host="operator.example",
            auth_timeout_seconds=10,
            output=StringIO(),
            _clock=lambda: 0.0,
            _wait_for_output=cancel,
        )
    assert nebius_vm_auth.signal.SIGINT in process.signals
