"""User credential loading for NPA commands."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

CREDENTIALS_PATH = Path.home() / ".npa" / "credentials.yaml"
KNOWN_TOKEN_KEYS = ("HF_TOKEN", "NGC_API_KEY")
PERMISSIONS_WARNING = (
    "credentials.yaml is readable by other users. Run chmod 600 ~/.npa/credentials.yaml."
)
_TOKEN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class CredentialsConfig:
    tokens: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""

    @property
    def hf_token(self) -> str:
        return self.tokens.get("HF_TOKEN", "")


def _is_readable_by_other_users(path: Path) -> bool:
    mode = path.stat().st_mode
    return bool(mode & (stat.S_IRWXG | stat.S_IRWXO))


def _read_file_tokens(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}

    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        return {}

    cleaned: dict[str, str] = {}
    for key, value in tokens.items():
        name = str(key)
        if not _TOKEN_NAME_RE.fullmatch(name) or value is None:
            continue
        token = str(value)
        if token:
            cleaned[name] = token
    return cleaned


def _read_file_ssh(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}

    ssh = data.get("ssh", {})
    if not isinstance(ssh, dict):
        return {}

    return {
        "host": str(ssh.get("host", "") or ""),
        "user": str(ssh.get("user", "") or ""),
        "key_path": str(
            ssh.get("key_path", "")
            or ssh.get("ssh_key", "")
            or ssh.get("private_key", "")
            or ""
        ),
    }


def load_credentials(
    *,
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    warn: Callable[[str], None] | None = None,
    export_to_environment: bool = False,
) -> CredentialsConfig:
    """Load user credentials from env vars and ``~/.npa/credentials.yaml``.

    Environment variables take precedence over file values.  Missing files are
    treated as empty credentials.
    """
    credentials_path = path or CREDENTIALS_PATH
    env = environ if environ is not None else os.environ

    warnings: list[str] = []
    file_tokens: dict[str, str] = {}
    file_ssh: dict[str, str] = {}

    if credentials_path.exists():
        if _is_readable_by_other_users(credentials_path):
            warnings.append(PERMISSIONS_WARNING)
        file_tokens = _read_file_tokens(credentials_path)
        file_ssh = _read_file_ssh(credentials_path)

    keys = set(KNOWN_TOKEN_KEYS) | set(file_tokens)
    tokens: dict[str, str] = {}
    for key in sorted(keys):
        env_value = env.get(key)
        value = env_value if env_value else file_tokens.get(key, "")
        if value:
            tokens[key] = value

    for message in warnings:
        if warn is not None:
            warn(message)

    if export_to_environment and environ is None:
        for key, value in tokens.items():
            os.environ.setdefault(key, value)

    return CredentialsConfig(
        tokens=tokens,
        warnings=warnings,
        ssh_host=env.get("NPA_BYOVM_HOST") or env.get("NPA_SSH_HOST") or file_ssh.get("host", ""),
        ssh_user=env.get("NPA_BYOVM_SSH_USER") or env.get("NPA_SSH_USER") or file_ssh.get("user", ""),
        ssh_key_path=env.get("NPA_BYOVM_SSH_KEY") or env.get("NPA_SSH_KEY") or file_ssh.get("key_path", ""),
    )
