"""Tests for Nebius profile selection and ambient-token scrubbing."""

from __future__ import annotations

from npa.clients import nebius_auth
from npa.clients.nebius_auth import nebius_profile


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
