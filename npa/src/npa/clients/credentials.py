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
NGC_ENV_KEYS = ("NGC_API_KEY", "NGC_ORG", "NGC_TEAM")
AI_CLOUD_ENV_KEY = "NEBIUS_AI_CLOUD_KEY"
TOKEN_FACTORY_ENV_KEY = "NEBIUS_TOKEN_FACTORY_KEY"
KNOWN_TOKEN_KEYS = (
    "HF_TOKEN",
    AI_CLOUD_ENV_KEY,
    TOKEN_FACTORY_ENV_KEY,
    *NGC_ENV_KEYS,
)
HF_TOKEN_MISSING_WARNING = (
    "Warning: HF_TOKEN not found in ~/.npa/credentials.yaml. "
    "Gated model downloads will fail."
)
PERMISSIONS_WARNING = (
    "credentials.yaml is readable by other users. Run chmod 600 ~/.npa/credentials.yaml."
)
_TOKEN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UK_SOUTH1_STORAGE_ENDPOINT = "storage.uk-south1.nebius.cloud"
EU_NORTH1_STORAGE_ENDPOINT = "storage.eu-north1.nebius.cloud"
UK_SOUTH1_STORAGE_WARNING = (
    "Warning: using default storage endpoint storage.uk-south1.nebius.cloud.\n"
    "If your cluster is in eu-north1, pass --storage-endpoint "
    "storage.eu-north1.nebius.cloud\n"
    "or set NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud in your environment."
)


@dataclass
class CredentialsConfig:
    tokens: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ssh_host: str = ""
    ssh_user: str = ""
    ssh_key_path: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_endpoint: str = ""
    s3_bucket: str = ""

    @property
    def hf_token(self) -> str:
        return self.tokens.get("HF_TOKEN", "")

    @property
    def nebius_api_key(self) -> str:
        """Backward-compatible alias for the Nebius AI Cloud key."""
        return self.ai_cloud_api_key

    @property
    def token_factory_api_key(self) -> str:
        """Explicit alias for the Nebius Token Factory hosted-inference key."""
        return resolve_token_factory_key(self.tokens)

    @property
    def ai_cloud_api_key(self) -> str:
        """Nebius AI Cloud API key (``tokens.NEBIUS_AI_CLOUD_KEY``)."""
        return resolve_ai_cloud_key(self.tokens)

    @property
    def ngc_api_key(self) -> str:
        return self.tokens.get("NGC_API_KEY", "")

    @property
    def ngc_org(self) -> str:
        return self.tokens.get("NGC_ORG", "")

    @property
    def ngc_team(self) -> str:
        return self.tokens.get("NGC_TEAM", "")


def resolve_ai_cloud_key(tokens: Mapping[str, str]) -> str:
    """Return the Nebius AI Cloud key from a token map."""
    return tokens.get(AI_CLOUD_ENV_KEY, "")


def resolve_token_factory_key(tokens: Mapping[str, str]) -> str:
    """Return the Token Factory key from a token map."""
    return tokens.get(TOKEN_FACTORY_ENV_KEY, "")


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
        tokens = {}

    cleaned: dict[str, str] = {}
    for key in KNOWN_TOKEN_KEYS:
        value = data.get(key)
        if value:
            cleaned[key] = str(value)
    for key, value in tokens.items():
        name = str(key)
        if not _TOKEN_NAME_RE.fullmatch(name) or value is None:
            continue
        token = str(value)
        if token:
            cleaned[name] = token

    ngc = data.get("ngc", {})
    if isinstance(ngc, dict):
        ngc_fields = {
            "NGC_API_KEY": ("api_key", "apikey", "key", "token", "NGC_API_KEY"),
            "NGC_ORG": ("org", "organization", "NGC_ORG"),
            "NGC_TEAM": ("team", "NGC_TEAM"),
        }
        for env_key, field_names in ngc_fields.items():
            value = _first_nonempty(ngc, *field_names)
            if value:
                cleaned[env_key] = value
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


def _first_nonempty(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _read_file_storage(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}

    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}
    storage = data.get("storage", data.get("s3", data.get("object-storage", data.get("object_storage", {}))))
    if not isinstance(storage, dict):
        storage = {}

    merged: dict[str, Any] = {**tokens, **storage, **data}
    return {
        "access_key_id": _first_nonempty(
            merged,
            "AWS_ACCESS_KEY_ID",
            "aws_access_key_id",
            "access_key_id",
            "access_key",
            "nebius_api_key",
        ),
        "secret_access_key": _first_nonempty(
            merged,
            "AWS_SECRET_ACCESS_KEY",
            "aws_secret_access_key",
            "secret_access_key",
            "secret_key",
            "nebius_secret_key",
        ),
        "endpoint": _first_nonempty(
            merged,
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "endpoint_url",
            "endpoint",
            "s3_endpoint",
        ),
        "bucket": _first_nonempty(
            merged,
            "NEBIUS_S3_BUCKET",
            "NPA_CHECKPOINT_BUCKET",
            "bucket",
            "checkpoint_bucket",
            "s3_bucket",
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
    file_storage: dict[str, str] = {}

    if credentials_path.exists():
        if _is_readable_by_other_users(credentials_path):
            warnings.append(PERMISSIONS_WARNING)
        file_tokens = _read_file_tokens(credentials_path)
        file_ssh = _read_file_ssh(credentials_path)
        file_storage = _read_file_storage(credentials_path)

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
        s3_access_key_id=env.get("AWS_ACCESS_KEY_ID") or file_storage.get("access_key_id", ""),
        s3_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY") or file_storage.get("secret_access_key", ""),
        s3_endpoint=(
            env.get("AWS_ENDPOINT_URL")
            or env.get("NEBIUS_S3_ENDPOINT")
            or env.get("NPA_STORAGE_ENDPOINT")
            or file_storage.get("endpoint", "")
        ),
        s3_bucket=env.get("NPA_CHECKPOINT_BUCKET") or env.get("NEBIUS_S3_BUCKET") or file_storage.get("bucket", ""),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prune_empty(data: dict[str, Any]) -> dict[str, Any]:
    pruned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _prune_empty(value)
            if nested:
                pruned[key] = nested
        elif value not in ("", None):
            pruned[key] = value
    return pruned


def set_token_factory_api_key(api_key: str, *, path: Path | None = None) -> Path:
    """Persist the Nebius Token Factory key under ``tokens.NEBIUS_TOKEN_FACTORY_KEY``."""

    cleaned = api_key.strip()
    if not cleaned:
        raise ValueError("Token Factory API key must be non-empty")
    return write_credentials_file(
        {"tokens": {TOKEN_FACTORY_ENV_KEY: cleaned}},
        path=path,
    )


def write_credentials_file(
    data: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> Path:
    """Deep-merge *data* into ``~/.npa/credentials.yaml`` and write it 0600.

    Empty string / ``None`` values are dropped so an interactive setup that
    skips a field never clobbers an existing value with a blank one.
    """

    credentials_path = path or CREDENTIALS_PATH
    existing: dict[str, Any] = {}
    if credentials_path.exists():
        with credentials_path.open() as handle:
            loaded = yaml.safe_load(handle)
        if isinstance(loaded, dict):
            existing = loaded
    merged = _deep_merge(existing, _prune_empty(dict(data)))
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    with credentials_path.open("w") as handle:
        yaml.dump(merged, handle, default_flow_style=False, sort_keys=False)
    credentials_path.chmod(0o600)
    return credentials_path


def storage_endpoint_url(endpoint: str) -> str:
    """Return a URL form for a Nebius S3-compatible endpoint."""
    value = endpoint.strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")
    return f"https://{value.rstrip('/')}"


def storage_endpoint_host(endpoint: str) -> str:
    """Return the host part used for endpoint comparisons."""
    value = endpoint.strip().rstrip("/")
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0].lower()


def storage_endpoint_warning(endpoint: str) -> str:
    """Return an onboarding warning for the historical uk-south1 endpoint."""
    if storage_endpoint_host(endpoint) != UK_SOUTH1_STORAGE_ENDPOINT:
        return ""
    return UK_SOUTH1_STORAGE_WARNING


def shared_credential_env(credentials: CredentialsConfig) -> dict[str, str]:
    """Return service env vars that should be injected into every workbench."""
    env: dict[str, str] = {}
    hf_token = getattr(credentials, "hf_token", "")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    tokens = getattr(credentials, "tokens", {}) or {}
    ai_cloud_key = resolve_ai_cloud_key(tokens)
    if ai_cloud_key:
        env[AI_CLOUD_ENV_KEY] = ai_cloud_key
    token_factory_key = resolve_token_factory_key(tokens)
    if token_factory_key:
        env[TOKEN_FACTORY_ENV_KEY] = token_factory_key
    for key in NGC_ENV_KEYS:
        value = tokens.get(key, "")
        if value:
            env[key] = value
    s3_access_key_id = getattr(credentials, "s3_access_key_id", "")
    s3_secret_access_key = getattr(credentials, "s3_secret_access_key", "")
    s3_endpoint = getattr(credentials, "s3_endpoint", "")
    s3_bucket = getattr(credentials, "s3_bucket", "")
    if s3_access_key_id:
        env["AWS_ACCESS_KEY_ID"] = s3_access_key_id
    if s3_secret_access_key:
        env["AWS_SECRET_ACCESS_KEY"] = s3_secret_access_key
    if s3_endpoint:
        env["AWS_ENDPOINT_URL"] = s3_endpoint
        env["NEBIUS_S3_ENDPOINT"] = s3_endpoint
    if s3_bucket:
        env["NEBIUS_S3_BUCKET"] = s3_bucket
    return env


def apply_shared_credential_env(
    env: dict[str, str],
    credentials: CredentialsConfig,
    *,
    include: bool = True,
) -> dict[str, str]:
    if include:
        for key, value in shared_credential_env(credentials).items():
            if value:
                env[key] = value
    return env


def warn_if_hf_token_missing(
    credentials: CredentialsConfig,
    *,
    warn: Callable[[str], None],
) -> bool:
    if getattr(credentials, "hf_token", ""):
        return False
    warn(HF_TOKEN_MISSING_WARNING)
    return True
