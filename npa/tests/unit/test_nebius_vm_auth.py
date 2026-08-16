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
        env={"PATH": "/bin", "NEBIUS_IAM_TOKEN": "stale", "NEBIUS_IAM_TOKEN_FILE": "/tmp/stale"},
        runner=runner,
    )
    assert result.identity_verified is True
    assert result.iam_token_minted is True
    assert [call[0][-2:] for call in calls] == [["iam", "whoami"], ["iam", "get-access-token"]]
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
    monkeypatch.setattr(nebius_vm_auth, "verify_profile", lambda *args, **kwargs: ready)
    monkeypatch.setattr(
        nebius_vm_auth.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
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

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def kill(self):
        self.returncode = -9

    def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_vm_auth_timeout_cancels_child(monkeypatch) -> None:
    not_ready = nebius_vm_auth.ProfileVerification("", False, False)
    monkeypatch.setattr(nebius_vm_auth, "verify_profile", lambda *args, **kwargs: not_ready)
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda _name: "/usr/bin/tool")
    process = _FakeProcess(StringIO(""))
    monkeypatch.setattr(nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process)
    times = iter([0.0, 2.0])
    monkeypatch.setattr(nebius_vm_auth.time, "monotonic", lambda: next(times))
    with pytest.raises(VmAuthError, match="timed out"):
        nebius_vm_auth.run_vm_profile_auth(
            ssh_host="operator.example", auth_timeout_seconds=1, output=StringIO()
        )
    assert nebius_vm_auth.signal.SIGINT in process.signals


def test_vm_auth_keyboard_cancel_cancels_child(monkeypatch) -> None:
    class CancelStream(StringIO):
        def readline(self, *args, **kwargs):
            raise KeyboardInterrupt

    not_ready = nebius_vm_auth.ProfileVerification("", False, False)
    monkeypatch.setattr(nebius_vm_auth, "verify_profile", lambda *args, **kwargs: not_ready)
    monkeypatch.setattr(nebius_vm_auth.shutil, "which", lambda _name: "/usr/bin/tool")
    process = _FakeProcess(CancelStream(""))
    monkeypatch.setattr(nebius_vm_auth.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(nebius_vm_auth.select, "select", lambda *args, **kwargs: ([process.stdout], [], []))
    with pytest.raises(VmAuthError, match="cancelled"):
        nebius_vm_auth.run_vm_profile_auth(
            ssh_host="operator.example", auth_timeout_seconds=10, output=StringIO()
        )
    assert nebius_vm_auth.signal.SIGINT in process.signals
