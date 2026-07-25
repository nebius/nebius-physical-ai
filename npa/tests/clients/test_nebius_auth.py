"""Tests for the canonical Nebius IAM token helper (ambient-token robustness)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from npa.clients import nebius_auth
from npa.clients.nebius_auth import NebiusTokenError, mint_nebius_iam_token, nebius_profile


def _fake_run(captured: dict, *, returncode: int = 0, stdout: str = "fresh\n", stderr: str = ""):
    def run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env")
        return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def test_strips_ambient_token_for_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale ambient NEBIUS_IAM_TOKEN must not leak into the exchange env."""
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-ambient-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/stale")
    monkeypatch.delenv("NPA_NEBIUS_PROFILE", raising=False)
    monkeypatch.delenv("NEBIUS_PROFILE", raising=False)
    captured: dict = {}
    monkeypatch.setattr(nebius_auth.subprocess, "run", _fake_run(captured, stdout="fresh-token\n"))

    assert mint_nebius_iam_token() == "fresh-token"
    assert captured["cmd"] == ["nebius", "iam", "get-access-token"]
    assert "NEBIUS_IAM_TOKEN" not in captured["env"]
    assert "NEBIUS_IAM_TOKEN_FILE" not in captured["env"]


def test_uses_profile_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "npa-mk8s")
    captured: dict = {}
    monkeypatch.setattr(nebius_auth.subprocess, "run", _fake_run(captured))

    mint_nebius_iam_token()
    assert captured["cmd"] == ["nebius", "--profile", "npa-mk8s", "iam", "get-access-token"]
    assert nebius_profile() == "npa-mk8s"


def test_falls_back_to_ambient_token_when_cli_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "ambient-fallback-token")
    captured: dict = {}
    monkeypatch.setattr(
        nebius_auth.subprocess,
        "run",
        _fake_run(captured, returncode=1, stdout="", stderr="token from NEBIUS_IAM_TOKEN env is used"),
    )

    assert mint_nebius_iam_token() == "ambient-fallback-token"


def test_strip_ambient_token_env_removes_token_vars() -> None:
    cleaned = nebius_auth.strip_ambient_token_env(
        {"NEBIUS_IAM_TOKEN": "x", "NEBIUS_IAM_TOKEN_FILE": "/f", "KUBECONFIG": "/k", "PATH": "/bin"}
    )
    assert "NEBIUS_IAM_TOKEN" not in cleaned
    assert "NEBIUS_IAM_TOKEN_FILE" not in cleaned
    assert cleaned["KUBECONFIG"] == "/k"
    assert cleaned["PATH"] == "/bin"


def test_raises_when_no_token_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)
    captured: dict = {}
    monkeypatch.setattr(
        nebius_auth.subprocess,
        "run",
        _fake_run(captured, returncode=1, stdout="", stderr="boom"),
    )

    with pytest.raises(NebiusTokenError):
        mint_nebius_iam_token(allow_env_token=False)
