from __future__ import annotations

import stat
import subprocess

import pytest

from npa.workflows.byof import source_auth


def test_repository_url_rejects_embedded_credentials() -> None:
    with pytest.raises(source_auth.RepositoryAuthenticationError, match="must not contain"):
        source_auth.validate_repository_url(
            "https://token-value@github.com/example/private.git", private=True
        )


@pytest.mark.parametrize(
    "repo_url",
    [
        "git@github.com:example/private.git",
        "ssh://git@github.com/example/private.git",
        "https://gitlab.com/example/private.git",
    ],
)
def test_private_repository_auth_is_constrained_to_github_https(repo_url: str) -> None:
    with pytest.raises(
        source_auth.RepositoryAuthenticationError,
        match="must not contain credentials|requires an https",
    ):
        source_auth.validate_repository_url(repo_url, private=True)


def test_named_token_env_fails_closed_when_absent() -> None:
    with pytest.raises(source_auth.RepositoryAuthenticationError, match="unavailable"):
        with source_auth.private_repository_secrets(
            "https://github.com/example/private.git",
            "main",
            token_env="NPA_BYOF_GITHUB_TOKEN",
            environ={},
            preflight=False,
        ):
            pytest.fail("missing authentication must not yield secret files")


def test_private_secret_files_are_owner_only_and_repr_is_redacted() -> None:
    token = "github-private-token-canary"
    with source_auth.private_repository_secrets(
        "https://github.com/example/private.git",
        "main",
        token_env="NPA_BYOF_GITHUB_TOKEN",
        environ={"NPA_BYOF_GITHUB_TOKEN": token},
        preflight=False,
    ) as secrets:
        for path in (secrets.token, secrets.repo_url, secrets.repo_ref):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert token not in repr(secrets)
        assert secrets.redaction_values == (token,)


def test_private_access_preflight_keeps_token_out_of_argv_and_output(monkeypatch) -> None:
    token = "github-private-token-canary"
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(source_auth.subprocess, "run", fake_run)
    with source_auth.private_repository_secrets(
        "https://github.com/example/private.git",
        "main",
        token_env="NPA_BYOF_GITHUB_TOKEN",
        environ={"NPA_BYOF_GITHUB_TOKEN": token},
    ):
        pass

    assert token not in " ".join(seen["cmd"])
    assert token not in repr(seen["env"])
    assert seen["cmd"][:2] == ["git", "ls-remote"]


def test_existing_git_credential_fallback_supports_older_gh(monkeypatch) -> None:
    token = "existing-auth-token-canary"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["gh", "auth"]:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"old gh")
        assert cmd == ["git", "credential", "fill"]
        assert kwargs["input"] == b"protocol=https\nhost=github.com\n\n"
        kwargs["stdout"].write(f"protocol=https\nhost=github.com\npassword={token}\n".encode())
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(source_auth.subprocess, "run", fake_run)
    with source_auth.private_repository_secrets(
        "https://github.com/example/private.git",
        "main",
        environ={},
        preflight=False,
    ) as secrets:
        assert secrets.token.read_text(encoding="utf-8") == token
        assert token not in repr(secrets)

    assert calls == [
        ["gh", "auth", "token", "--hostname", "github.com"],
        ["git", "credential", "fill"],
    ]
