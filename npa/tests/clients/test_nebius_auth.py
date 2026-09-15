"""Tests for Nebius profile selection and ambient-token scrubbing."""

from __future__ import annotations

import subprocess

import pytest

from npa.clients import nebius_auth
from npa.clients.nebius_auth import nebius_profile, verify_profile


def test_uses_profile_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "npa-mk8s")
    assert nebius_profile() == "npa-mk8s"


def test_strip_ambient_token_env_removes_token_vars() -> None:
    cleaned = nebius_auth.strip_ambient_token_env(
        {
            "NEBIUS_IAM_TOKEN": "x",
            "NEBIUS_IAM_TOKEN_FILE": "/f",
            "KUBECONFIG": "/k",
            "PATH": "/bin",
        }
    )
    assert "NEBIUS_IAM_TOKEN" not in cleaned
    assert "NEBIUS_IAM_TOKEN_FILE" not in cleaned
    assert cleaned["KUBECONFIG"] == "/k"
    assert cleaned["PATH"] == "/bin"


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
    assert result.failure_reason == ""
    assert [call[0] for call in calls] == [
        [
            "nebius",
            "--profile",
            "operator",
            "--no-browser",
            "--no-check-update",
            "iam",
            "whoami",
        ],
        [
            "nebius",
            "--profile",
            "operator",
            "--no-browser",
            "--no-check-update",
            "iam",
            "get-access-token",
        ],
    ]
    assert all(call[1]["stdin"] is subprocess.DEVNULL for call in calls)
    assert all(call[1]["stdout"] is subprocess.DEVNULL for call in calls)
    assert all(call[1]["stderr"] is subprocess.DEVNULL for call in calls)
    assert all("NEBIUS_IAM_TOKEN" not in call[1]["env"] for call in calls)
    assert all("NEBIUS_IAM_TOKEN_FILE" not in call[1]["env"] for call in calls)


def test_profile_readiness_reports_failed_identity_without_token_text() -> None:
    returncodes = iter([1, 0])

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, next(returncodes))

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is True
    assert result.failure_reason == "identity_failed"
    assert not hasattr(result, "token")


def test_profile_readiness_discards_failure_output() -> None:
    secret = "synthetic-provider-output"

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout=secret, stderr=secret)

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is False
    assert result.failure_reason == "identity_failed"
    assert secret not in repr(result)


def test_profile_readiness_reports_missing_cli() -> None:
    def runner(*args, **kwargs):
        raise FileNotFoundError

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is False
    assert result.failure_reason == "cli_unavailable"


def test_profile_readiness_reports_timeout() -> None:
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired("nebius", 30)

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is False
    assert result.failure_reason == "timeout"


def test_profile_readiness_reports_provider_execution_error() -> None:
    def runner(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nebius")

    result = verify_profile("operator", runner=runner)
    assert result.identity_verified is False
    assert result.iam_token_minted is False
    assert result.failure_reason == "probe_error"


def test_profile_readiness_reports_token_mint_failure() -> None:
    returncodes = iter([0, 1])

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, next(returncodes))

    result = verify_profile(runner=runner)
    assert result.identity_verified is True
    assert result.iam_token_minted is False
    assert result.failure_reason == "token_mint_failed"


@pytest.mark.parametrize("identity_returncode", [0, 1])
@pytest.mark.parametrize(
    ("error", "failure_reason"),
    [
        (FileNotFoundError("synthetic-provider-output"), "cli_unavailable"),
        (
            subprocess.TimeoutExpired(
                "nebius", 30, output="synthetic-provider-output", stderr="synthetic-provider-output"
            ),
            "timeout",
        ),
        (OSError("synthetic-provider-output"), "probe_error"),
        (
            subprocess.CalledProcessError(
                1, "nebius", output="synthetic-provider-output", stderr="synthetic-provider-output"
            ),
            "probe_error",
        ),
    ],
)
def test_token_probe_error_preserves_identity_result_without_output(
    identity_returncode: int, error: Exception, failure_reason: str, capsys
) -> None:
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[-1] == "whoami":
            return subprocess.CompletedProcess(command, identity_returncode)
        raise error

    result = verify_profile("operator", runner=runner)

    assert [command[-1] for command in calls] == ["whoami", "get-access-token"]
    assert result.identity_verified is (identity_returncode == 0)
    assert result.iam_token_minted is False
    assert result.failure_reason == failure_reason
    assert "synthetic-provider-output" not in repr(result)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
