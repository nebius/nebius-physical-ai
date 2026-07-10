"""Tests for host-side kubectl invocation robust to a stale ambient IAM token."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from npa.clients.kube import (
    IAM_TOKEN_ENV_KEYS,
    ambient_iam_token_present,
    looks_like_stale_iam_token_error,
    run_kubectl,
    strip_iam_token_env,
)


@dataclass
class _FakeProc:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _Recorder:
    """Injectable subprocess.run replacement returning scripted results."""

    def __init__(self, results: list[_FakeProc]) -> None:
        self._results = results
        self.calls: list[dict[str, str]] = []

    def __call__(self, cmd, *, env, **kwargs):  # noqa: ANN001 - test double
        self.calls.append(dict(env))
        return self._results[len(self.calls) - 1]


def test_strip_iam_token_env_removes_all_keys() -> None:
    env = {"PATH": "/usr/bin", "NEBIUS_IAM_TOKEN": "x", "NPA_NEBIUS_IAM_TOKEN": "y"}
    stripped = strip_iam_token_env(env)
    assert "PATH" in stripped
    for key in IAM_TOKEN_ENV_KEYS:
        assert key not in stripped


def test_ambient_iam_token_present() -> None:
    assert ambient_iam_token_present({"NEBIUS_IAM_TOKEN": "tok"})
    assert not ambient_iam_token_present({"NEBIUS_IAM_TOKEN": "  "})
    assert not ambient_iam_token_present({"PATH": "/bin"})


@pytest.mark.parametrize(
    "text",
    [
        "Service iam error Unauthenticated",
        "executable /home/ubuntu/.nebius/bin/nebius failed with exit code 7",
        "The NEBIUS_IAM_TOKEN environment variable likely contains an expired token",
        "getting credentials: exec: ...",
    ],
)
def test_stale_token_signatures_match(text: str) -> None:
    assert looks_like_stale_iam_token_error(text)


def test_no_false_positive_signature() -> None:
    assert not looks_like_stale_iam_token_error("error: pods 'foo' not found")


def test_first_call_success_no_retry() -> None:
    rec = _Recorder([_FakeProc(0, "prod-cluster")])
    result = run_kubectl(
        ["config", "current-context"],
        binary="kubectl",
        env={"NEBIUS_IAM_TOKEN": "stale"},
        runner=rec,
    )
    assert result.returncode == 0
    assert result.retried_without_iam_token is False
    assert len(rec.calls) == 1
    # token was left intact on the (only) successful call
    assert rec.calls[0].get("NEBIUS_IAM_TOKEN") == "stale"


def test_retries_without_token_on_stale_auth_error() -> None:
    rec = _Recorder(
        [
            _FakeProc(1, "", "Service iam error Unauthenticated; failed with exit code 7"),
            _FakeProc(0, "16"),
        ]
    )
    result = run_kubectl(
        ["get", "nodes"],
        binary="kubectl",
        kubeconfig="/tmp/kubeconfig",
        env={"NEBIUS_IAM_TOKEN": "stale", "PATH": "/bin"},
        runner=rec,
    )
    assert result.returncode == 0
    assert result.retried_without_iam_token is True
    assert len(rec.calls) == 2
    # first call kept the token; retry stripped it
    assert rec.calls[0].get("NEBIUS_IAM_TOKEN") == "stale"
    assert "NEBIUS_IAM_TOKEN" not in rec.calls[1]
    assert rec.calls[1].get("KUBECONFIG") == "/tmp/kubeconfig"


def test_no_retry_when_no_ambient_token() -> None:
    rec = _Recorder([_FakeProc(1, "", "Unauthenticated")])
    result = run_kubectl(["get", "nodes"], binary="kubectl", env={"PATH": "/bin"}, runner=rec)
    assert result.returncode == 1
    assert result.retried_without_iam_token is False
    assert len(rec.calls) == 1


def test_no_retry_on_non_auth_error() -> None:
    rec = _Recorder([_FakeProc(1, "", "error: pods not found")])
    result = run_kubectl(
        ["get", "pods"],
        binary="kubectl",
        env={"NEBIUS_IAM_TOKEN": "stale"},
        runner=rec,
    )
    assert result.returncode == 1
    assert result.retried_without_iam_token is False
    assert len(rec.calls) == 1


def test_missing_kubectl_binary_returns_127(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.clients.kube as kube

    monkeypatch.delenv("NPA_KUBECTL_BIN", raising=False)
    monkeypatch.setattr(kube.shutil, "which", lambda _name: None)
    result = run_kubectl(["get", "nodes"], binary="", env={"PATH": "/bin"})
    assert result.returncode == 127
