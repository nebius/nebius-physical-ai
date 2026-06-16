"""Tests for Nebius registry pull-secret refresh."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from npa.workflows.sim2real.registry_auth import (
    docker_config_json,
    ensure_nebius_registry_pull_secret,
    mint_nebius_registry_token,
)


def test_mint_nebius_registry_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.subprocess.run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="token-abc\n", stderr=""),
    )
    assert mint_nebius_registry_token() == "token-abc"


def test_docker_config_json_uses_iam_username() -> None:
    payload = docker_config_json(registry_server="cr.eu-north1.nebius.cloud", token="tok")
    entry = payload["auths"]["cr.eu-north1.nebius.cloud"]
    assert entry["username"] == "iam"
    assert entry["password"] == "tok"


def test_ensure_nebius_registry_pull_secret_applies_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured["input"] = kwargs.get("input", "")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("npa.workflows.sim2real.registry_auth.subprocess.run", fake_run)
    ensure_nebius_registry_pull_secret(
        registry_server="cr.eu-north1.nebius.cloud",
        k8s_context="demo-context",
    )
    payload = json.loads(captured["input"])
    assert payload["metadata"]["name"] == "npa-nebius-registry"
