"""Secret-safe authentication helpers for BYOF source repositories."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping
from urllib.parse import urlsplit


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})


class RepositoryAuthenticationError(RuntimeError):
    """A private source repository could not be accessed safely."""


@dataclass(frozen=True)
class RepositorySecretFiles:
    """Owner-only files passed to BuildKit as ephemeral secret mounts."""

    token: Path
    repo_url: Path
    repo_ref: Path
    repository_sha256: str
    ref_sha256: str
    _redaction_values: tuple[str, ...] = field(repr=False)

    @property
    def redaction_values(self) -> tuple[str, ...]:
        return self._redaction_values


def validate_repository_url(repo_url: str, *, private: bool) -> str:
    """Reject URL-carried credentials and constrain private auth to GitHub HTTPS."""

    value = str(repo_url or "").strip()
    if not value or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise RepositoryAuthenticationError(
            "repository URLs must be non-empty and contain no control characters"
        )
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise RepositoryAuthenticationError(
            "repository URLs must not contain credentials; use --repo-auth github"
        )
    if parsed.query or parsed.fragment:
        raise RepositoryAuthenticationError(
            "repository URLs must not contain query strings or fragments"
        )
    if private and (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() not in _GITHUB_HOSTS
        or parsed.port not in (None, 443)
        or len([part for part in parsed.path.split("/") if part]) < 2
    ):
        raise RepositoryAuthenticationError(
            "private BYOF source authentication requires an https://github.com/... URL"
        )
    return value


def _write_secret(path: Path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, value.encode("utf-8"))
    finally:
        os.close(fd)


def _token_from_github_cli(path: Path) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    os.close(fd)
    with path.open("wb") as output:
        proc = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    token = path.read_text(encoding="utf-8").strip() if proc.returncode == 0 else ""
    if not token:
        # Older gh releases have no `auth token` command. The credential helper
        # installed by `gh auth setup-git` exposes the same existing login via
        # Git's stdin/stdout credential protocol, without putting it in argv.
        with path.open("wb") as output:
            proc = subprocess.run(
                ["git", "credential", "fill"],
                input=b"protocol=https\nhost=github.com\n\n",
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
            )
        if proc.returncode == 0:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key == "password":
                    token = value.strip()
                    break
    if not token:
        raise RepositoryAuthenticationError(
            "private GitHub authentication is unavailable; set the requested token "
            "environment variable or refresh `gh auth login --hostname github.com`"
        )
    path.write_text(token, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return token


def _resolve_token(
    directory: Path,
    *,
    token_env: str,
    environ: Mapping[str, str],
) -> tuple[Path, str]:
    token_path = directory / "github-token"
    requested = token_env.strip()
    if requested:
        if _ENV_NAME.fullmatch(requested) is None:
            raise RepositoryAuthenticationError(
                "--repo-token-env must name an environment variable"
            )
        token = str(environ.get(requested) or "").strip()
        if not token:
            raise RepositoryAuthenticationError(
                f"private GitHub authentication is unavailable in environment variable {requested}"
            )
        _write_secret(token_path, token)
        return token_path, token

    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = str(environ.get(name) or "").strip()
        if token:
            _write_secret(token_path, token)
            return token_path, token
    token = _token_from_github_cli(token_path)
    return token_path, token


def _git_config_quote(value: str) -> str:
    """Quote one validated value for an owner-only Git config file."""

    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _remote_advertises_ref(output: str, requested_ref: str) -> bool:
    advertised: list[tuple[str, str]] = []
    for line in output.splitlines():
        oid, separator, name = line.partition("\t")
        if separator and oid and name:
            advertised.append((oid, name))

    names = {name for _oid, name in advertised}
    candidates = {
        requested_ref,
        f"refs/heads/{requested_ref}",
        f"refs/tags/{requested_ref}",
    }
    if requested_ref.startswith("origin/"):
        candidates.add(f"refs/heads/{requested_ref.removeprefix('origin/')}")
    if names.intersection(candidates):
        return True

    # The clone fallback accepts an advertised commit id (including an
    # unambiguous abbreviated id), so the access preflight does too.
    if re.fullmatch(r"[0-9a-fA-F]{4,40}", requested_ref):
        requested_oid = requested_ref.lower()
        return any(oid.lower().startswith(requested_oid) for oid, _name in advertised)
    return False


def _preflight_access(repo_url: str, repo_ref: str, token_path: Path) -> None:
    helper = token_path.parent / "git-askpass"
    helper.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        '  *) cat "$NPA_BYOF_GIT_TOKEN_FILE" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    env = dict(os.environ)
    env.update(
        {
            "GIT_ASKPASS": str(helper),
            "GIT_ASKPASS_REQUIRE": "force",
            "GIT_TERMINAL_PROMPT": "0",
            "NPA_BYOF_GIT_TOKEN_FILE": str(token_path),
        }
    )
    repository = token_path.parent / "preflight.git"
    initialized = subprocess.run(
        ["git", "init", "--bare", "--quiet", str(repository)],
        cwd=token_path.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if initialized.returncode != 0:
        raise RepositoryAuthenticationError(
            "private GitHub repository access preflight could not initialize safely"
        )
    config_path = repository / "config"
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with config_path.open("a", encoding="utf-8") as config:
        config.write(
            "\n[remote \"origin\"]\n"
            f"\turl = {_git_config_quote(repo_url)}\n"
        )
    proc = subprocess.run(
        ["git", "ls-remote", "origin"],
        cwd=repository,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RepositoryAuthenticationError(
            "private GitHub repository access failed for the requested repository/ref; "
            "verify repository permission, token scope, and ref name"
        )
    if not _remote_advertises_ref(proc.stdout or "", repo_ref):
        raise RepositoryAuthenticationError(
            "private GitHub repository access succeeded, but the requested ref is unavailable"
        )


@contextmanager
def private_repository_secrets(
    repo_url: str,
    repo_ref: str,
    *,
    token_env: str = "",
    environ: Mapping[str, str] | None = None,
    preflight: bool = True,
) -> Iterator[RepositorySecretFiles]:
    """Resolve GitHub auth without putting secret values in argv or metadata."""

    clean_url = validate_repository_url(repo_url, private=True)
    clean_ref = str(repo_ref or "").strip()
    if not clean_ref or any(
        ord(char) < 0x20 or ord(char) == 0x7F for char in clean_ref
    ):
        raise RepositoryAuthenticationError(
            "private repository ref must be non-empty and contain no control characters"
        )
    with tempfile.TemporaryDirectory(prefix="npa-byof-private-source-") as tmp:
        directory = Path(tmp)
        token_path, token = _resolve_token(
            directory,
            token_env=token_env,
            environ=environ if environ is not None else os.environ,
        )
        repo_url_path = directory / "repo-url"
        repo_ref_path = directory / "repo-ref"
        _write_secret(repo_url_path, clean_url)
        _write_secret(repo_ref_path, clean_ref)
        if preflight:
            _preflight_access(clean_url, clean_ref, token_path)
        yield RepositorySecretFiles(
            token=token_path,
            repo_url=repo_url_path,
            repo_ref=repo_ref_path,
            repository_sha256=hashlib.sha256(clean_url.encode("utf-8")).hexdigest(),
            ref_sha256=hashlib.sha256(clean_ref.encode("utf-8")).hexdigest(),
            _redaction_values=(token, clean_url, clean_ref),
        )
