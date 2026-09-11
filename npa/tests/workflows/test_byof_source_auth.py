from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from npa.workflows.byof import source_auth


@pytest.mark.parametrize(
    "repo_url",
    [
        "https://token-value@github.com/example/private.git",
        "https://github.com/example/public.git?token=value",
        "https://github.com/example/public.git#branch",
    ],
)
def test_repository_url_hardening_applies_to_public_and_private_sources(
    repo_url: str,
) -> None:
    with pytest.raises(source_auth.RepositoryAuthenticationError, match="must not contain"):
        source_auth.validate_repository_url(repo_url, private=False)


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
    repo_url = "https://github.com/example/private.git"
    repo_ref = "main-private-canary"
    with source_auth.private_repository_secrets(
        repo_url,
        repo_ref,
        token_env="NPA_BYOF_GITHUB_TOKEN",
        environ={"NPA_BYOF_GITHUB_TOKEN": token},
        preflight=False,
    ) as secrets:
        for path in (secrets.token, secrets.repo_url, secrets.repo_ref):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert token not in repr(secrets)
        assert repo_url not in repr(secrets)
        assert repo_ref not in repr(secrets)
        assert secrets.redaction_values == (token, repo_url, repo_ref)


def test_private_access_preflight_checks_requested_ref_without_private_argv(
    monkeypatch,
) -> None:
    token = "github-private-token-canary"
    repo_url = "https://github.com/example/private.git"
    repo_ref = "private-ref-canary"
    seen: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(cmd, **kwargs):
        seen.append((list(cmd), kwargs))
        if cmd[:3] == ["git", "init", "--bare"]:
            repository = Path(cmd[-1])
            repository.mkdir()
            (repository / "config").write_text("[core]\n\tbare = true\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        assert cmd == ["git", "ls-remote", "origin"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"{'a' * 40}\trefs/heads/{repo_ref}\n",
            stderr="",
        )

    monkeypatch.setattr(source_auth.subprocess, "run", fake_run)
    with source_auth.private_repository_secrets(
        repo_url,
        repo_ref,
        token_env="NPA_BYOF_GITHUB_TOKEN",
        environ={"NPA_BYOF_GITHUB_TOKEN": token},
    ):
        pass

    commands = repr([cmd for cmd, _kwargs in seen])
    for private_value in (token, repo_url, repo_ref):
        assert private_value not in commands
    assert seen[-1][0] == ["git", "ls-remote", "origin"]


def test_private_access_preflight_rejects_missing_requested_ref(monkeypatch) -> None:
    token = "github-private-token-canary"
    repo_url = "https://github.com/example/private.git"
    missing_ref = "missing-private-ref-canary"
    seen_commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen_commands.append(list(cmd))
        if cmd[:3] == ["git", "init", "--bare"]:
            repository = Path(cmd[-1])
            repository.mkdir()
            (repository / "config").write_text("[core]\n\tbare = true\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"{'b' * 40}\trefs/heads/other\n",
            stderr=f"diagnostic {repo_url} {missing_ref} {token}",
        )

    monkeypatch.setattr(source_auth.subprocess, "run", fake_run)
    with pytest.raises(
        source_auth.RepositoryAuthenticationError,
        match="requested ref is unavailable",
    ) as exc_info:
        with source_auth.private_repository_secrets(
            repo_url,
            missing_ref,
            token_env="NPA_BYOF_GITHUB_TOKEN",
            environ={"NPA_BYOF_GITHUB_TOKEN": token},
        ):
            pytest.fail("a missing private ref must not yield secret files")

    published = str(exc_info.value) + repr(seen_commands)
    for private_value in (token, repo_url, missing_ref):
        assert private_value not in published


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
