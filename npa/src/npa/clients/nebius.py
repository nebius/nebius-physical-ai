"""Nebius CLI wrapper for authentication and resource management.

Calls the ``nebius`` binary to obtain IAM tokens, manage service accounts,
access keys, and S3 buckets — replacing the need to manually source
``environment.sh`` before running ``npa workbench lerobot deploy``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from npa.smoke._versions import supported_tool_version


class NebiusError(Exception):
    pass


# ── Low-level CLI runner ─────────────────────────────────────────────────

_NEBIUS_VERSION_CHECKED = False


def _parse_cli_version(output: str) -> str | None:
    match = re.search(r"\b(?:v)?(\d+\.\d+\.\d+)\b", output)
    if match is None:
        return None
    return match.group(1)


def _warn_if_nebius_version_mismatch(nebius_path: str) -> None:
    global _NEBIUS_VERSION_CHECKED

    if _NEBIUS_VERSION_CHECKED:
        return
    _NEBIUS_VERSION_CHECKED = True

    try:
        expected = supported_tool_version("nebius-cli", __file__)
        result = subprocess.run(
            [nebius_path, "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        warnings.warn(
            f"Could not check Nebius CLI version: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        warnings.warn(
            f"Could not check Nebius CLI version (exit {result.returncode}): {output}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    actual = _parse_cli_version(output)
    if actual is None:
        warnings.warn(
            f"Could not parse Nebius CLI version from output: {output}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if actual != expected:
        warnings.warn(
            f"Nebius CLI version mismatch: expected {expected}; found {actual}",
            RuntimeWarning,
            stacklevel=2,
        )


def _require_nebius() -> str:
    path = shutil.which("nebius")
    if path is None:
        raise NebiusError(
            "nebius CLI not found on PATH. "
            "Install it: https://docs.nebius.com/cli/install"
        )
    _warn_if_nebius_version_mismatch(path)
    return path


_REUSE_IAM_TOKEN_ENV = "NPA_REUSE_IAM_TOKEN"


def nebius_cli_env(base: "Mapping[str, str] | None" = None) -> dict[str, str]:
    """Return a sanitized environment for ``nebius`` CLI subprocesses.

    A stale ambient ``NEBIUS_IAM_TOKEN`` (an expired token, or one minted for a
    different tenant/project, left in the shell or a cloud-env) is used by the
    CLI in preference to the active profile's auto-refreshing exec-plugin
    credential. That shadows a perfectly good profile, so calls like
    ``storage bucket list`` return ``AccessDenied``/``Unauthenticated`` even
    though ``nebius iam get-access-token`` works. A stale ``NEBIUS_IAM_TOKEN_FILE``
    shadows the profile the same way (the CLI reads the token file and skips a
    real token exchange), so drop both. Unless the caller explicitly opts into
    reuse via ``NPA_REUSE_IAM_TOKEN`` (e.g. CI/VM injecting a short-lived token).
    This is the single source of truth for that behavior across the repo (mirrors
    ``npa.soperator.lifecycle._nebius_cli_env`` and the Terraform env builders).
    Python-level token resolution (``get_iam_token``) still reads the env vars
    directly, so token-only contexts keep working.

    *base* lets callers sanitize an already-customized environment; when omitted
    the current process environment is used.
    """
    env = dict(base) if base is not None else os.environ.copy()
    reuse = env.get(_REUSE_IAM_TOKEN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    if not reuse:
        env.pop("NEBIUS_IAM_TOKEN", None)
        env.pop("NEBIUS_IAM_TOKEN_FILE", None)
    return env


def _run(args: list[str], *, check: bool = True) -> str:
    """Run a nebius CLI command, return stdout."""
    nebius = _require_nebius()
    result = subprocess.run(
        [nebius] + args,
        capture_output=True,
        text=True,
        env=nebius_cli_env(),
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        raise NebiusError(
            f"nebius {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}"
        )
    return result.stdout.strip()


def _run_json(args: list[str], *, check: bool = True) -> dict[str, Any]:
    """Run a nebius CLI command with --format json, parse and return the result."""
    raw = _run(args + ["--format", "json"], check=check)
    if not raw:
        return {}
    return json.loads(raw)


# ── IAM token ────────────────────────────────────────────────────────────


_IAM_TOKEN_ENV_KEYS = ("NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN")
_IAM_TOKEN_FILE_ENV_KEYS = ("NPA_NEBIUS_IAM_TOKEN_FILE", "NEBIUS_IAM_TOKEN_FILE")
_DEFAULT_IAM_TOKEN_FILES = (
    "/mnt/cloud-metadata/token",
    "/run/secrets/nebius/iam_token",
)
_METADATA_SA_TOKEN_URL = "http://metadata.nebius.internal/v1/iam/sa/token/access_token"


def _env_iam_token() -> str:
    for key in _IAM_TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _candidate_iam_token_files() -> list[str]:
    candidates: list[str] = []
    for key in _IAM_TOKEN_FILE_ENV_KEYS:
        path = os.environ.get(key, "").strip()
        if path and path not in candidates:
            candidates.append(path)
    for path in _DEFAULT_IAM_TOKEN_FILES:
        if path not in candidates:
            candidates.append(path)
    return candidates


def _read_iam_token_file(path: str) -> str:
    candidate = path.strip()
    if not candidate:
        return ""
    try:
        return Path(candidate).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _metadata_iam_token(timeout_s: float = 2.0) -> str:
    req = urllib_request.Request(
        _METADATA_SA_TOKEN_URL,
        method="GET",
        headers={"Metadata": "true"},
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def get_iam_token() -> str:
    """Resolve an IAM token from CLI profile, env/file overrides, or VM metadata."""
    cli_error: str = ""
    try:
        token = _run(["iam", "get-access-token"])
    except NebiusError as exc:
        token = ""
        cli_error = str(exc)
    if token:
        return token

    env_token = _env_iam_token()
    if env_token:
        return env_token

    for candidate in _candidate_iam_token_files():
        file_token = _read_iam_token_file(candidate)
        if file_token:
            return file_token

    metadata_token = _metadata_iam_token()
    if metadata_token:
        return metadata_token

    detail = f" Last CLI error: {cli_error}" if cli_error else ""
    raise NebiusError(
        "Unable to resolve IAM token from Nebius CLI profile, environment, token files, "
        f"or metadata endpoint ({_METADATA_SA_TOKEN_URL}).{detail}"
    )


# ── Profile-derived defaults ─────────────────────────────────────────────


def _config_get(key: str) -> str:
    """Return ``nebius config get <key>`` output, or "" when unavailable.

    Best-effort: any failure (missing CLI, unauthenticated profile, unknown
    key) resolves to an empty string so callers can fall back to prompting.
    """
    try:
        return _run(["config", "get", key])
    except Exception:
        # Best-effort default lookup; never fail setup over a missing profile.
        return ""


def current_project_id() -> str:
    """Best-effort Nebius project id from the active CLI profile."""
    return _config_get("parent-id")


def current_tenant_id() -> str:
    """Best-effort Nebius tenant id from the active CLI profile."""
    return _config_get("tenant-id")


def set_profile_project(project_id: str, tenant_id: str = "") -> bool:
    """Point the active Nebius CLI profile at *project_id* / *tenant_id*.

    ``npa`` shells out to the Nebius CLI with the operator's active profile, so a
    profile whose ``parent-id``/``tenant-id`` are empty (or point somewhere else)
    silently disables project discovery and makes later commands target the wrong
    place. Writing the selected ids back onto the profile keeps the two in sync.

    Best-effort: returns ``False`` (never raises) when the CLI is missing or a
    ``nebius config set`` call fails.
    """
    project = str(project_id or "").strip()
    tenant = str(tenant_id or "").strip()
    if not project:
        return False
    updates = [("parent-id", project)]
    if tenant:
        updates.append(("tenant-id", tenant))
    try:
        for key, value in updates:
            _run(["config", "set", key, value])
    except Exception:
        return False
    return True


# ── Tenant / project discovery ───────────────────────────────────────────


def list_tenants() -> list[dict[str, str]]:
    """Return tenants the active profile can see: ``[{id, name, region}]``.

    Best-effort: returns ``[]`` when the CLI is missing/unauthenticated or the
    call fails, so callers can fall back to manual entry.
    """
    try:
        data = _run_json(["iam", "tenant", "list", "--all"])
    except Exception:
        return []
    tenants: list[dict[str, str]] = []
    for item in data.get("items", []):
        metadata = item.get("metadata", {}) or {}
        tenant_id = str(metadata.get("id", "") or "")
        if not tenant_id:
            continue
        status = item.get("status", {}) or {}
        spec = item.get("spec", {}) or {}
        tenants.append(
            {
                "id": tenant_id,
                "name": str(metadata.get("name", "") or ""),
                "region": str(status.get("region", "") or spec.get("region", "") or ""),
            }
        )
    return tenants


def list_projects_in_tenant(tenant_id: str) -> list[dict[str, str]]:
    """Return ACTIVE projects under *tenant_id*: ``[{id, name, tenant_id, region}]``.

    Best-effort: returns ``[]`` on any failure (e.g. the profile lacks list
    permission in this tenant) so discovery can skip it and continue.
    """
    tenant = str(tenant_id or "").strip()
    if not tenant:
        return []
    try:
        data = _run_json(
            ["iam", "project", "list", "--parent-id", tenant, "--all"]
        )
    except Exception:
        return []
    projects: list[dict[str, str]] = []
    for item in data.get("items", []):
        metadata = item.get("metadata", {}) or {}
        project_id = str(metadata.get("id", "") or "")
        if not project_id:
            continue
        status = item.get("status", {}) or {}
        spec = item.get("spec", {}) or {}
        # Skip suspended/deleting projects; only ACTIVE containers are usable.
        container_state = str(status.get("container_state", "") or "")
        if container_state and container_state != "ACTIVE":
            continue
        projects.append(
            {
                "id": project_id,
                "name": str(metadata.get("name", "") or ""),
                "tenant_id": tenant,
                "region": str(status.get("region", "") or spec.get("region", "") or ""),
            }
        )
    return projects


def list_accessible_projects() -> list[dict[str, str]]:
    """Return every ACTIVE project the active profile can reach.

    Enumerates tenants via :func:`list_tenants`, then projects per tenant via
    :func:`list_projects_in_tenant`. Best-effort and non-fatal: tenants whose
    project list is denied are simply skipped. Each entry is
    ``{id, name, tenant_id, region}``.
    """
    projects: list[dict[str, str]] = []
    for tenant in list_tenants():
        projects.extend(list_projects_in_tenant(tenant["id"]))
    return projects


def get_project_region(project_id: str) -> str:
    """Best-effort region for *project_id*, or "".

    A Nebius project belongs to exactly one region, and compute placement
    follows the project (the ``--region`` flag does not move a VM to a different
    region than its project). Resolving the real region lets callers check the
    right per-region quota and render accurate region-dependent config. Returns
    "" when the CLI is missing/unauthenticated or the lookup fails.
    """
    data = _get_project(project_id)
    status = data.get("status", {}) or {}
    spec = data.get("spec", {}) or {}
    return str(status.get("region", "") or spec.get("region", "") or "").strip()


def _get_project(project_id: str) -> dict[str, Any]:
    """Best-effort ``iam project get`` payload for *project_id*, or ``{}``."""
    pid = str(project_id or "").strip()
    if not pid:
        return {}
    try:
        return _run_json(["iam", "project", "get", "--id", pid]) or {}
    except Exception:
        return {}


def get_project_tenant_id(project_id: str) -> str:
    """Best-effort tenant (parent) id for *project_id*, or "".

    A Nebius CLI profile does not always carry ``tenant-id`` (federation
    profiles, and profiles created against a single project, often set only
    ``parent-id``). Project discovery needs a tenant, so recover it from the
    project itself rather than silently skipping discovery.
    """
    metadata = _get_project(project_id).get("metadata", {}) or {}
    return str(metadata.get("parent_id", "") or metadata.get("parentId", "") or "").strip()


def get_project_name(project_id: str) -> str:
    """Best-effort human-readable name for *project_id*, or "".

    Used to derive a local project alias (``tle-workbench``) instead of falling
    back to the region (``us-central1``), which reads like a region field rather
    than a project handle.
    """
    metadata = _get_project(project_id).get("metadata", {}) or {}
    return str(metadata.get("name", "") or "").strip()


def get_public_ipv4_quota(tenant_id: str, region: str) -> tuple[int | None, int | None]:
    """Return ``(usage, limit)`` for the tenant public IPv4 quota in *region*.

    Nebius meters public IPv4 addresses per (tenant, region) via the
    ``vpc.ipv4-address.public.count`` quota allowance. Best-effort: returns
    ``(None, None)`` when the CLI is missing/unauthenticated, the quota can't be
    read, or no matching per-region allowance exists — so callers never block a
    deploy on an unreadable quota.
    """
    tenant = str(tenant_id or "").strip()
    reg = str(region or "").strip()
    if not tenant or not reg:
        return (None, None)
    try:
        data = _run_json(
            ["quotas", "quota-allowance", "list", "--parent-id", tenant, "--all"]
        )
    except Exception:
        return (None, None)

    def _to_int(value: Any) -> int | None:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    for item in data.get("items", []):
        metadata = item.get("metadata", {}) or {}
        if metadata.get("name") != "vpc.ipv4-address.public.count":
            continue
        spec = item.get("spec", {}) or {}
        if str(spec.get("region", "") or "").strip() != reg:
            continue
        status = item.get("status", {}) or {}
        return (_to_int(status.get("usage", "0")), _to_int(spec.get("limit")))
    return (None, None)


def discover_container_registry(project_id: str, *, preferred_region: str = "eu-north1") -> str:
    """Best-effort container registry URL for *project_id*, or "".

    Returns ``<registry_fqdn>/<registry-id>`` (matching the
    ``DEFAULT_CONTAINER_REGISTRY`` format). A project can hold registries in
    several regions, and the API list order is not stable, so prefer a registry
    in *preferred_region* (``eu-north1``, the main registry region) and fall back
    to the first registry otherwise. Any failure resolves to "" so callers fall
    back to the default registry.
    """
    if not project_id:
        return ""
    try:
        data = _run_json(["registry", "list", "--parent-id", project_id])
    except Exception:
        return ""

    def _url(item: dict[str, Any]) -> str:
        fqdn = item.get("status", {}).get("registry_fqdn", "")
        registry_id = item.get("metadata", {}).get("id", "")
        if fqdn and registry_id:
            return f"{fqdn}/{registry_id.removeprefix('registry-')}"
        return ""

    items = data.get("items", [])
    if preferred_region:
        for item in items:
            fqdn = item.get("status", {}).get("registry_fqdn", "")
            if f".{preferred_region}." in fqdn:
                url = _url(item)
                if url:
                    return url
    for item in items:
        url = _url(item)
        if url:
            return url
    return ""


# ── Service account ──────────────────────────────────────────────────────


_SA_RESOURCE_ID_RE = re.compile(
    r"resource ID:\s*(serviceaccount-[a-z0-9]+)",
    re.IGNORECASE,
)


def _resource_id_from_nebius_error(message: str, *, prefix: str) -> str:
    """Best-effort parse of a Nebius resource id embedded in CLI stderr."""

    if prefix == "serviceaccount-":
        match = _SA_RESOURCE_ID_RE.search(message)
        return match.group(1) if match else ""
    match = re.search(rf"resource ID:\s*({re.escape(prefix)}[a-z0-9-]+)", message, re.I)
    return match.group(1) if match else ""


def _is_permission_denied(message: str) -> bool:
    lowered = message.lower()
    return (
        "permissiondenied" in lowered
        or "permission denied" in lowered
        or "no permission" in lowered
        # Nebius object storage reports authorization failures as AccessDenied.
        or "accessdenied" in lowered
        or "access denied" in lowered
    )


def is_permission_denied(message: str) -> bool:
    """Public predicate: does *message* look like a Nebius permission/access error?

    Lets callers (e.g. `npa configure`) render actionable IAM guidance instead of
    a raw rpc dump when provisioning is blocked by missing permissions.
    """
    return _is_permission_denied(message)


def _is_not_found(message: str) -> bool:
    lowered = message.lower()
    return "notfound" in lowered or "not found" in lowered or "resourcenotfound" in lowered


def _is_already_exists(message: str) -> bool:
    lowered = message.lower()
    return "alreadyexists" in lowered or "already exists" in lowered


def _normalize_bucket_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("s3://"):
        from urllib.parse import urlparse

        return urlparse(cleaned).netloc
    return cleaned.split("/", 1)[0]


def _saved_service_account_id() -> str:
    import os

    from npa.clients.credentials import CREDENTIALS_PATH

    env_value = os.environ.get("NPA_SERVICE_ACCOUNT_ID", "").strip()
    if env_value:
        return env_value
    if not CREDENTIALS_PATH.exists():
        return ""
    try:
        import yaml

        with CREDENTIALS_PATH.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except Exception:
        return ""
    if not isinstance(loaded, dict):
        return ""
    nebius = loaded.get("nebius", {})
    if isinstance(nebius, dict):
        return str(nebius.get("service_account_id", "") or "").strip()
    return ""


def _saved_storage_credentials(
    *,
    project_id: str,
    tenant_id: str,
    region: str,
    bucket_name: str | None,
    service_account_id: str = "",
) -> dict[str, str] | None:
    """Reuse configured object-storage credentials when IAM provisioning is blocked."""

    from npa.clients.credentials import load_credentials
    from npa.clients.config import resolve_project_storage

    creds = load_credentials()
    access_key = creds.s3_access_key_id.strip()
    secret_key = creds.s3_secret_access_key.strip()
    if not access_key or not secret_key:
        return None

    endpoint = creds.s3_endpoint.strip() or f"https://storage.{region}.nebius.cloud"

    bucket = _normalize_bucket_name(creds.s3_bucket)
    if not bucket:
        try:
            storage = resolve_project_storage(None)
            bucket = _normalize_bucket_name(getattr(storage, "checkpoint_bucket", ""))
        except Exception:
            bucket = ""
    if not bucket:
        bucket = _normalize_bucket_name(bucket_name or "")
    if not bucket:
        bucket = bucket_name_for(tenant_id, project_id)

    sa_id = service_account_id.strip() or resolve_service_account_id(project_id)
    if not sa_id:
        return None

    return {
        "iam_token": get_iam_token(),
        "service_account_id": sa_id,
        "nebius_api_key": access_key,
        "nebius_secret_key": secret_key,
        "s3_bucket": bucket,
        "s3_endpoint": endpoint,
        "nebius_project_id": project_id,
        "nebius_region": region,
    }


AGENT_SERVICE_ACCOUNT_NAME = "npa-agent"
AGENT_ACCESS_KEY_NAME = "npa-agent-access-key"
DEFAULT_SERVICE_ACCOUNT_NAME = "lerobot-training"
DEFAULT_ACCESS_KEY_NAME = "lerobot-access-key"


def ensure_service_account(
    project_id: str,
    name: str = DEFAULT_SERVICE_ACCOUNT_NAME,
    *,
    description: str = "Service account for LeRobot training on Nebius",
) -> str:
    """Get or create a service account, return its ID."""
    # Try to find existing.
    try:
        data = _run_json([
            "iam", "service-account", "get-by-name",
            "--parent-id", project_id,
            "--name", name,
        ])
        sa_id = data.get("metadata", {}).get("id", "")
        if sa_id:
            return sa_id
    except NebiusError as exc:
        message = str(exc)
        sa_id = _resource_id_from_nebius_error(message, prefix="serviceaccount-")
        if sa_id:
            return sa_id
        if _is_permission_denied(message):
            saved = _saved_service_account_id()
            if saved:
                return saved
            raise NebiusError(
                f"Cannot read or create service account {name!r}: {exc}. "
                "Set NPA_SERVICE_ACCOUNT_ID or nebius.service_account_id in "
                "~/.npa/credentials.yaml when IAM management is restricted."
            ) from exc
        if not _is_not_found(message):
            raise
        # Not found — create below.

    try:
        data = _run_json([
            "iam", "service-account", "create",
            "--parent-id", project_id,
            "--name", name,
            "--description", description,
        ])
    except NebiusError as exc:
        if _is_permission_denied(str(exc)):
            saved = _saved_service_account_id()
            if saved:
                return saved
        raise
    sa_id = data.get("metadata", {}).get("id", "")
    if not sa_id:
        raise NebiusError("Service account creation did not return an ID")
    return sa_id


# ── Editors group membership ─────────────────────────────────────────────


def ensure_editors_membership(tenant_id: str, sa_id: str) -> None:
    """Add the service account to the tenant's *editors* group."""
    group_data = _run_json([
        "iam", "group", "get-by-name",
        "--parent-id", tenant_id,
        "--name", "editors",
    ])
    group_id = group_data.get("metadata", {}).get("id", "")
    if not group_id:
        raise NebiusError(f"Could not find editors group in tenant {tenant_id}")

    # Check membership.
    members_data = _run_json([
        "iam", "group-membership", "list-members",
        "--parent-id", group_id,
        "--page-size", "1000",
    ])
    memberships = members_data.get("memberships", [])
    for m in memberships:
        if m.get("spec", {}).get("member_id") == sa_id:
            return  # Already a member.

    _run([
        "iam", "group-membership", "create",
        "--parent-id", group_id,
        "--member-id", sa_id,
    ])


# ── Access keys ──────────────────────────────────────────────────────────


def _find_active_access_key(
    project_id: str,
    sa_id: str,
    *,
    key_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the first ACTIVE access key for the given service account, or None."""
    data = _run_json([
        "iam", "v2", "access-key", "list",
        "--parent-id", project_id,
    ])
    for item in data.get("items", []):
        spec = item.get("spec", {})
        account = spec.get("account", {})
        # The SA ID can live under different JSON paths depending on API version.
        item_sa_id = (
            account.get("service_account", {}).get("id", "")
            or account.get("service_account_id", "")
        )
        if item_sa_id != sa_id:
            continue
        status = item.get("status", {})
        if status.get("state") != "ACTIVE":
            continue
        # Check expiry.
        expires_at = spec.get("expires_at", "")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                # Nebius uses the Unix epoch as a "no expiration" sentinel for
                # access keys created without an expiry.
                if exp_dt.year > 1971 and exp_dt < datetime.now(timezone.utc):
                    continue  # Expired.
            except ValueError:
                pass
        if key_name and item.get("metadata", {}).get("name") != key_name:
            continue
        return item
    return None


def ensure_access_key(
    project_id: str,
    sa_id: str,
    *,
    key_name: str = DEFAULT_ACCESS_KEY_NAME,
    description: str = "Access key for LeRobot S3 and API access",
) -> tuple[str, str]:
    """Ensure an active access key exists, return (aws_access_key_id, aws_secret_access_key).

    Reuses an existing key when possible; creates a new one otherwise.
    """
    existing = (
        _find_active_access_key(project_id, sa_id, key_name=key_name)
        or _find_active_access_key(project_id, sa_id)
    )
    if existing:
        key_id = existing["metadata"]["id"]
        # Retrieve the AWS access key ID.
        get_data = _run_json(["iam", "v2", "access-key", "get", "--id", key_id])
        aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")
        # Try to retrieve the secret (works for keys where the secret is stored).
        try:
            secret_data = _run_json(["iam", "v2", "access-key", "get-secret", "--id", key_id])
            aws_secret_key = secret_data.get("secret", "")
        except NebiusError:
            aws_secret_key = ""

        if aws_access_key and aws_secret_key:
            return aws_access_key, aws_secret_key
        # Secret not retrievable — fall through to create a new key.

    # Create a fresh key without deleting existing keys. Existing keys may own
    # Terraform remote-state objects for workbenches that still need destroy.
    create_name = key_name
    if existing:
        create_name = f"{key_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    create_data = _run_json([
        "iam", "v2", "access-key", "create",
        "--parent-id", project_id,
        "--name", create_name,
        "--account-service-account-id", sa_id,
        "--description", description,
    ])
    new_key_id = create_data.get("metadata", {}).get("id", "")
    if not new_key_id:
        raise NebiusError("Access key creation did not return an ID")

    # Fetch the AWS-compatible credentials.
    get_data = _run_json(["iam", "v2", "access-key", "get", "--id", new_key_id])
    aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")

    secret_data = _run_json(["iam", "v2", "access-key", "get-secret", "--id", new_key_id])
    aws_secret_key = secret_data.get("secret", "")

    if not aws_access_key or not aws_secret_key:
        raise NebiusError("Failed to retrieve AWS-compatible credentials from new access key")

    return aws_access_key, aws_secret_key


# ── S3 bucket ────────────────────────────────────────────────────────────

DEFAULT_BUCKET_BASENAME = "npa-bucket"
DEFAULT_BUCKET_STORAGE_CLASS = "standard"


def normalize_bucket_storage_class(value: str) -> str:
    """Map user-facing storage class labels to Nebius CLI values."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "standard", "std", "storage_class_unspecified"}:
        return DEFAULT_BUCKET_STORAGE_CLASS
    if normalized in {"enhanced", "enhanced_throughput", "enhancedthroughput"}:
        return "enhanced_throughput"
    if normalized == "intelligent":
        return "intelligent"
    return DEFAULT_BUCKET_STORAGE_CLASS


def bucket_name_for(tenant_id: str, project_id: str) -> str:
    """Derive a deterministic default bucket name from tenant + project IDs.

    Matches the logic in ``environment.sh`` so existing buckets are reused.
    """
    raw = f"{tenant_id}-{project_id}"
    suffix = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{DEFAULT_BUCKET_BASENAME}-{suffix}"


def _list_project_buckets(project_id: str) -> list[dict[str, Any]]:
    """Return every bucket in *project_id*.

    Uses ``--all`` so existing buckets are never missed behind the CLI's default
    pagination (matching the orphan-instance/tenant/project listers). Without it,
    a project with many buckets returned only the first page, so ``bucket_exists``
    reported ``False`` for a real bucket and ``npa configure`` wrongly prompted to
    create a new one.
    """
    data = _run_json([
        "storage", "bucket", "list",
        "--parent-id", project_id,
        "--all",
    ])
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def get_bucket_by_name(project_id: str, bucket_name: str) -> dict[str, Any] | None:
    """Return the bucket list item for *bucket_name*, or ``None``."""

    for item in _list_project_buckets(project_id):
        if item.get("metadata", {}).get("name") == bucket_name:
            return item
    return None


def delete_bucket(bucket_id: str, *, ttl: str = "") -> None:
    """Delete an object-storage bucket by resource id.

    A bucket that still holds objects (or non-current object *versions*, which
    ``aws s3 rb --force`` leaves behind) cannot be deleted immediately: the API
    answers ``BucketNotEmpty``. Passing *ttl* schedules the purge instead
    (``--ttl 1m``), which is how the platform empties and removes it.
    """

    if not bucket_id:
        return
    args = ["storage", "bucket", "delete", "--id", bucket_id]
    if str(ttl or "").strip():
        args.extend(["--ttl", str(ttl).strip()])
    _run(args)


def bucket_exists(project_id: str, bucket_name: str) -> bool:
    """Return True when *bucket_name* already exists in the project."""
    return any(
        item.get("metadata", {}).get("name") == bucket_name
        for item in _list_project_buckets(project_id)
    )


def ensure_bucket(
    project_id: str,
    bucket_name: str,
    *,
    max_size_bytes: int = 0,
    default_storage_class: str = DEFAULT_BUCKET_STORAGE_CLASS,
) -> str:
    """Get or create an S3 bucket, return its name.

    *max_size_bytes* caps a newly created bucket (0 = unlimited). It is only
    applied when the bucket is created; an existing bucket is reused unchanged.
    *default_storage_class* is applied only when the bucket is created.
    """
    if bucket_exists(project_id, bucket_name):
        return bucket_name

    storage_class = normalize_bucket_storage_class(default_storage_class)
    args = [
        "storage", "bucket", "create",
        "--name", bucket_name,
        "--parent-id", project_id,
        "--versioning-policy", "enabled",
        "--default-storage-class", storage_class,
    ]
    if max_size_bytes > 0:
        args += ["--max-size-bytes", str(max_size_bytes)]
    try:
        _run(args)
    except NebiusError as exc:
        # Bucket names are globally unique, so a create can race an existing
        # bucket (e.g. a prior run, or an existence check that missed it). Never
        # fail the flow on a name conflict: reuse the bucket when it turns out to
        # live in this project, and only surface a clear conflict when the name
        # is taken elsewhere and thus unusable here.
        if not _is_already_exists(str(exc)):
            raise
        if get_bucket_by_name(project_id, bucket_name) is not None:
            return bucket_name
        raise NebiusError(
            f"Object-storage bucket name '{bucket_name}' is already taken "
            "(bucket names are globally unique) and is not in project "
            f"{project_id}. Re-run `npa configure` and enter a different, "
            "unused bucket name."
        ) from exc
    return bucket_name


# ── Composite bootstrap ─────────────────────────────────────────────────


def bootstrap_environment(
    project_id: str,
    tenant_id: str,
    region: str,
    *,
    bucket_name: str | None = None,
    bucket_max_size_bytes: int = 0,
    bucket_storage_class: str = DEFAULT_BUCKET_STORAGE_CLASS,
    service_account_name: str = DEFAULT_SERVICE_ACCOUNT_NAME,
    access_key_name: str = DEFAULT_ACCESS_KEY_NAME,
    service_account_description: str = "Service account for LeRobot training on Nebius",
    access_key_description: str = "Access key for LeRobot S3 and API access",
    on_status: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Run the full environment bootstrap, return a dict of credentials.

    This is the Python equivalent of ``source environment.sh``.

    *bucket_name* selects the object-storage bucket; when omitted it falls back
    to the deterministic ``bucket_name_for`` name. *bucket_max_size_bytes* caps
    a newly created bucket (0 = unlimited); it is ignored when the bucket
    already exists. *bucket_storage_class* applies only when the bucket is
    created. *on_status* is an optional callback ``(message: str) -> None`` for
    progress reporting.
    """

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    _status("Getting IAM access token...")
    iam_token = get_iam_token()

    _status("Setting up service account...")
    sa_id = ensure_service_account(
        project_id,
        name=service_account_name,
        description=service_account_description,
    )

    _status("Configuring service account permissions...")
    try:
        ensure_editors_membership(tenant_id, sa_id)
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        # Non-fatal, but the operator must know: without the tenant `editors`
        # role the service account (e.g. the one attached to an agent VM) can
        # authenticate but is NOT authorized to manage Nebius AI Cloud resources
        # (clusters, buckets, access keys, registries). Surface it instead of
        # silently continuing.
        _status(
            "WARNING: could not add the service account to the tenant 'editors' "
            f"group (permission denied). Service account {sa_id} may lack "
            "permission to manage Nebius AI Cloud resources. Ask a tenant admin "
            f"to add {sa_id} to the 'editors' group (or grant an equivalent "
            "role), then re-run."
        )

    bucket_name = bucket_name or bucket_name_for(tenant_id, project_id)

    _status("Setting up S3 bucket...")
    try:
        ensure_bucket(
            project_id,
            bucket_name,
            max_size_bytes=bucket_max_size_bytes,
            default_storage_class=bucket_storage_class,
        )
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        fallback = _saved_storage_credentials(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            service_account_id=sa_id,
        )
        if fallback is None:
            raise
        _status("Reusing saved object-storage credentials (bucket provisioning skipped).")
        return fallback

    _status("Setting up access key for S3...")
    try:
        aws_access_key, aws_secret_key = ensure_access_key(
            project_id,
            sa_id,
            key_name=access_key_name,
            description=access_key_description,
        )
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        fallback = _saved_storage_credentials(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            service_account_id=sa_id,
        )
        if fallback is None:
            raise
        _status("Reusing saved object-storage credentials (access-key provisioning skipped).")
        return fallback

    s3_endpoint = f"https://storage.{region}.nebius.cloud"

    return {
        "iam_token": iam_token,
        "service_account_id": sa_id,
        "nebius_api_key": aws_access_key,
        "nebius_secret_key": aws_secret_key,
        "s3_bucket": bucket_name,
        "s3_endpoint": s3_endpoint,
        "nebius_project_id": project_id,
        "nebius_region": region,
    }


def resolve_service_account_id(
    project_id: str,
    *,
    names: tuple[str, ...] = (AGENT_SERVICE_ACCOUNT_NAME, DEFAULT_SERVICE_ACCOUNT_NAME),
) -> str:
    """Resolve a service-account id from config or best-effort IAM lookups."""

    saved = _saved_service_account_id()
    if saved:
        return saved
    project = str(project_id or "").strip()
    if not project:
        return ""
    for name in names:
        sa_id = get_service_account_id_by_name(project, name)
        if sa_id:
            return sa_id
    return ""


def get_service_account_id_by_name(project_id: str, name: str) -> str | None:
    """Return a service-account id when *name* exists, else ``None``."""

    try:
        data = _run_json([
            "iam", "service-account", "get-by-name",
            "--parent-id", project_id,
            "--name", name,
        ])
    except NebiusError as exc:
        message = str(exc)
        sa_id = _resource_id_from_nebius_error(message, prefix="serviceaccount-")
        if sa_id:
            return sa_id
        if "notfound" in message.lower() or "not found" in message.lower():
            return None
        if _is_permission_denied(message):
            return None
        raise
    sa_id = data.get("metadata", {}).get("id", "")
    return str(sa_id).strip() or None


def list_access_keys_for_service_account(project_id: str, sa_id: str) -> list[dict[str, str]]:
    """Return ``[{"id", "name", "state"}]`` for every access key owned by *sa_id*.

    Teardown needs the full list (not just the active one `ensure_access_key`
    reuses): an agent VM's long-lived key outlives its VM otherwise.
    """
    if not project_id or not sa_id:
        return []
    try:
        data = _run_json(["iam", "v2", "access-key", "list", "--parent-id", project_id])
    except NebiusError:
        return []
    keys: list[dict[str, str]] = []
    for item in data.get("items", []):
        account = (item.get("spec", {}) or {}).get("account", {}) or {}
        item_sa_id = (
            (account.get("service_account", {}) or {}).get("id", "")
            or account.get("service_account_id", "")
        )
        if item_sa_id != sa_id:
            continue
        metadata = item.get("metadata", {}) or {}
        keys.append(
            {
                "id": str(metadata.get("id", "") or ""),
                "name": str(metadata.get("name", "") or ""),
                "state": str((item.get("status", {}) or {}).get("state", "") or ""),
            }
        )
    return [key for key in keys if key["id"]]


def delete_access_key(access_key_id: str) -> None:
    """Delete an IAM access key by id."""
    if not access_key_id:
        return
    _run(["iam", "v2", "access-key", "delete", "--id", access_key_id])


def delete_service_account(sa_id: str) -> None:
    """Delete a service account by id."""
    if not sa_id:
        return
    _run(["iam", "service-account", "delete", "--id", sa_id])


def bootstrap_agent_environment(
    project_id: str,
    tenant_id: str,
    region: str,
    **kwargs: Any,
) -> dict[str, str]:
    """Bootstrap a long-lived ``npa-agent`` service account for agent VMs.

    When IAM provisioning is blocked, reuse saved or configured object-storage
    credentials instead of failing bootstrap.
    """

    on_status = kwargs.pop("on_status", None)
    bucket_name = kwargs.get("bucket_name")
    sa_id = get_service_account_id_by_name(project_id, AGENT_SERVICE_ACCOUNT_NAME)
    if sa_id and on_status:
        on_status(f"Reusing existing service account {AGENT_SERVICE_ACCOUNT_NAME!r}.")
    try:
        return bootstrap_environment(
            project_id,
            tenant_id,
            region,
            service_account_name=AGENT_SERVICE_ACCOUNT_NAME,
            access_key_name=AGENT_ACCESS_KEY_NAME,
            service_account_description="Long-lived service account for NPA agent VMs",
            access_key_description="Long-lived access key for NPA agent S3 and API access",
            on_status=on_status,
            **kwargs,
        )
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        fallback = _saved_storage_credentials(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            service_account_id=sa_id or resolve_service_account_id(project_id),
        )
        if fallback is None:
            raise
        if on_status:
            on_status("Reusing saved object-storage credentials (npa-agent provisioning skipped).")
        return fallback
