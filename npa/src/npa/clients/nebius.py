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
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from npa.smoke._versions import supported_tool_version


class NebiusError(Exception):
    pass


@dataclass(frozen=True)
class ServiceAccountIdentity:
    """Allowlisted provider identity used by guarded IAM reconciliation."""

    account_id: str
    name: str
    project_id: str
    tenant_id: str
    profile: str


@dataclass(frozen=True)
class ProjectIdentity:
    """Allowlisted immutable identity for exact project teardown."""

    project_id: str
    name: str
    tenant_id: str
    region: str
    profile: str = ""


@dataclass(frozen=True)
class ProjectDefaultNetworkIdentity:
    """Exact provider-created default topology in one disposable project."""

    network_id: str
    network_name: str
    subnet_id: str
    subnet_name: str
    security_group_id: str
    security_group_name: str
    project_id: str
    profile: str = ""


class IamBindingState(str, Enum):
    """Typed result of reconciling an effective provider IAM capability."""

    CREATED = "created"
    EXISTING = "existing"
    FAILED = "failed"


@dataclass(frozen=True)
class StorageIamBindingEvidence:
    state: IamBindingState
    role: str
    scope_id: str
    group_id: str
    group_name: str
    access_permit_id: str = ""
    compatibility_fallback: bool = False

    @property
    def propagation_eligible(self) -> bool:
        return self.state is IamBindingState.CREATED


class StorageIamBindingError(NebiusError):
    """Terminal IAM failure carrying a typed, non-secret capability outcome."""

    def __init__(self, message: str, evidence: StorageIamBindingEvidence):
        super().__init__(message)
        self.evidence = evidence


STORAGE_RUNTIME_ROLE = "storage.object-editor"
STORAGE_REQUIRED_S3_ACTIONS = (
    "GetObject",
    "HeadObject",
    "PutObject",
    "DeleteObject",
    "ListObjectsV2",
)
STORAGE_BINDING_GROUP_PREFIX = "npa-storage-object-editors"
# Compatibility export for callers that only display the group family name.
STORAGE_BINDING_GROUP_NAME = STORAGE_BINDING_GROUP_PREFIX


def storage_binding_group_name(project_id: str) -> str:
    """Give each exact project a distinct project IAM group capability boundary."""

    suffix = re.sub(r"[^a-z0-9-]", "-", str(project_id).lower()).strip("-")
    return f"{STORAGE_BINDING_GROUP_PREFIX}-{suffix}"


# ── Low-level CLI runner ─────────────────────────────────────────────────

_NEBIUS_VERSION_CHECKED = False
_TESTED_NEBIUS_CLI_VERSIONS = frozenset({"0.12.227", "0.12.254"})
_NEBIUS_CLI_INSTALL_URL = "https://storage.eu-north1.nebius.cloud/cli/install.sh"


def _nebius_cli_install_remedy(version: str) -> str:
    return f"curl -fsSL {_NEBIUS_CLI_INSTALL_URL} | NEBIUS_CLI_VERSION={version} bash"


def _parse_cli_version(output: str) -> str | None:
    match = re.search(r"\b(?:v)?(\d+\.\d+\.\d+)\b", output)
    if match is None:
        return None
    return match.group(1)


def _warn_if_nebius_version_mismatch(nebius_path: str) -> None:
    global _NEBIUS_VERSION_CHECKED

    if _NEBIUS_VERSION_CHECKED:
        return

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
        raise NebiusError(
            "Could not check the Nebius CLI version. Reinstall the tested version: "
            f"`{_nebius_cli_install_remedy(supported_tool_version('nebius-cli', __file__))}`"
        ) from exc

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        raise NebiusError(
            f"Could not check the Nebius CLI version (exit {result.returncode}). "
            "Reinstall the tested version: "
            f"`{_nebius_cli_install_remedy(expected)}`"
        )

    actual = _parse_cli_version(output)
    if actual is None:
        raise NebiusError(
            "Could not parse the Nebius CLI version. Reinstall the tested version: "
            f"`{_nebius_cli_install_remedy(expected)}`"
        )

    tested = set(_TESTED_NEBIUS_CLI_VERSIONS)
    tested.add(expected)
    if actual not in tested:
        supported = ", ".join(sorted(tested))
        raise NebiusError(
            f"Unsupported Nebius CLI {actual}; NPA has tested {supported}. "
            f"Install {expected}: `{_nebius_cli_install_remedy(expected)}`"
        )
    _NEBIUS_VERSION_CHECKED = True
    if actual != expected:
        warnings.warn(
            f"Nebius CLI {actual} is tested-compatible, but {expected} is the "
            f"recommended version. To align exactly: "
            f"`{_nebius_cli_install_remedy(expected)}`",
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
    reuse = env.get(_REUSE_IAM_TOKEN_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
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
        stderr = redact_nebius_output(result.stderr.strip())
        raise NebiusError(
            f"nebius {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}"
        )
    return result.stdout.strip()


def _run_json(args: list[str], *, check: bool = True) -> dict[str, Any]:
    """Run a nebius CLI command with --format json, parse and return the result."""
    raw = _run(args + ["--format", "json"], check=check)
    if not raw:
        return {}
    from npa.clients.json_output import parse_single_json_document

    parsed = parse_single_json_document(raw)
    if not isinstance(parsed, dict):
        # JSONDecodeError retains the complete source document on ``exc.doc``.
        # Never let malformed secret-bearing responses (for example get-secret)
        # escape through an exception, traceback, logger, or serialized error.
        raise NebiusError(
            f"nebius {' '.join(args[:3])} returned invalid JSON"
        ) from None
    return parsed


_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "awssecretaccesskey",
        "credential",
        "credentials",
        "iamtoken",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "secretaccesskey",
        "secretkey",
        "token",
    }
)
_SENSITIVE_FIELD_MARKERS = (
    "credential",
    "password",
    "passwd",
    "privatekey",
    "secret",
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?P<prefix>
      [\"']?
      (?:access[-_]?token|api[-_]?key|aws[-_]?secret[-_]?access[-_]?key|
         authorization|credential(?:s)?|iam[-_]?token|pass(?:word|wd)|
         private[-_]?key|secret(?:[-_]?access)?[-_]?key|secret|token)
      (?:[-_]?(?:data|material|value))?
      [\"']?\s*[:=]\s*
    )
    (?P<value>
      \"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,}\]]+
    )
    """
)
_AUTHORIZATION_ASSIGNMENT_RE = re.compile(
    r"(?im)(?P<prefix>[\"']?authorization[\"']?\s*[:=]\s*)[^,}\]\r\n]+"
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)


def _sensitive_field_name(value: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return (
        normalized in _SENSITIVE_FIELD_NAMES
        or any(marker in normalized for marker in _SENSITIVE_FIELD_MARKERS)
        or normalized.endswith("token")
    )


def _redact_sensitive_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if _sensitive_field_name(key)
            else _redact_sensitive_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_data(item) for item in value]
    return value


def redact_nebius_output(value: str) -> str:
    """Redact plausible credential fields from Nebius diagnostics.

    Access-key list output is prevented at the source by a JSONPath allowlist
    below. This is the second line of defence for CLI failures and other provider
    diagnostics: nested JSON is redacted structurally, while ordinary
    ``key=value`` / ``key: value`` messages use a conservative field-name match.
    """

    text = str(value or "")
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        redacted = _PRIVATE_KEY_BLOCK_RE.sub("<redacted>", text)
        redacted = _AUTHORIZATION_ASSIGNMENT_RE.sub(
            lambda match: f"{match.group('prefix')}<redacted>", redacted
        )
        return _SENSITIVE_ASSIGNMENT_RE.sub(
            lambda match: f"{match.group('prefix')}<redacted>", redacted
        )
    return json.dumps(_redact_sensitive_data(parsed), sort_keys=True)


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
        data = _run_json(["iam", "project", "list", "--parent-id", tenant, "--all"])
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
    return str(
        metadata.get("parent_id", "") or metadata.get("parentId", "") or ""
    ).strip()


def get_project_name(project_id: str) -> str:
    """Best-effort human-readable name for *project_id*, or "".

    Used to derive a local project alias (``tle-workbench``) instead of falling
    back to the region (``us-central1``), which reads like a region field rather
    than a project handle.
    """
    metadata = _get_project(project_id).get("metadata", {}) or {}
    return str(metadata.get("name", "") or "").strip()


def get_project_identity(
    project_id: str, *, tenant_id: str = "", profile: str | None = None
) -> ProjectIdentity | None:
    """Strictly get one project by immutable ID; exact NotFound is absence."""

    exact_id = str(project_id or "").strip()
    expected_tenant = str(tenant_id or "").strip()
    if not exact_id:
        raise NebiusError("exact project ID is required")
    profile_args, resolved_profile = _iam_profile_args(profile)
    try:
        payload = _run_json(
            [*profile_args, "iam", "v2", "project", "get", "--id", exact_id]
        )
    except NebiusError as exc:
        if _is_not_found(str(exc)):
            return None
        raise
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    spec = payload.get("spec") if isinstance(payload, dict) else None
    status = payload.get("status") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise NebiusError("Nebius returned schema-invalid project identity")
    returned_id = str(metadata.get("id") or "").strip()
    returned_tenant = str(
        metadata.get("parent_id") or metadata.get("parentId") or ""
    ).strip()
    name = str(metadata.get("name") or "").strip()
    region = (
        str((status or {}).get("region") or "").strip()
        if isinstance(status, dict)
        else ""
    ) or str(spec.get("region") or "").strip()
    if returned_id != exact_id or not returned_tenant or not name or not region:
        raise NebiusError("Nebius returned incomplete or mismatched project identity")
    if expected_tenant and returned_tenant != expected_tenant:
        raise NebiusError(
            f"Project {exact_id} belongs to tenant {returned_tenant}, not {expected_tenant}"
        )
    return ProjectIdentity(exact_id, name, returned_tenant, region, resolved_profile)


_PROJECT_CHILD_LIST_COMMANDS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("compute_instances", ("compute", "instance", "list"), True),
    ("compute_disks", ("compute", "disk", "list"), True),
    ("compute_filesystems", ("compute", "filesystem", "list"), True),
    ("vpc_networks", ("vpc", "network", "list"), True),
    ("vpc_subnets", ("vpc", "subnet", "list"), True),
    ("vpc_security_groups", ("vpc", "security-group", "list"), True),
    ("vpc_allocations", ("vpc", "allocation", "list"), True),
    ("mk8s_clusters", ("mk8s", "cluster", "list"), True),
    ("storage_buckets", ("storage", "bucket", "list"), True),
    ("registries", ("registry", "list"), True),
    ("service_accounts", ("iam", "service-account", "list"), True),
    # Workbench serverless training/inference creates these directly under a
    # project. The pinned CLI exposes complete list results but no --all flag.
    ("ai_endpoints", ("ai", "endpoint", "list"), False),
    ("ai_jobs", ("ai", "job", "list"), False),
)


def list_project_dependencies(
    project_id: str, *, profile: str | None = None
) -> dict[str, tuple[str, ...]]:
    """Authoritatively inventory NPA-managed project child resource classes.

    Every response must contain an ``items`` list with immutable IDs. Unsupported,
    unreadable, malformed, or partially identifiable inventory raises rather than
    being mistaken for an empty project.
    """

    exact_id = str(project_id or "").strip()
    if not exact_id:
        raise NebiusError("exact project ID is required for dependency inventory")
    profile_args, _resolved_profile = _iam_profile_args(profile)
    inventory: dict[str, tuple[str, ...]] = {}
    for kind, command, supports_all in _PROJECT_CHILD_LIST_COMMANDS:
        payload = _run_json(
            [
                *profile_args,
                *command,
                "--parent-id",
                exact_id,
                *(("--all",) if supports_all else ()),
            ]
        )
        if payload == {} or payload == {"items": None}:
            # Pinned Nebius list commands encode a valid empty collection in
            # either form (the serverless adapter already has this contract).
            # No other missing/null/extra-field shape is accepted as absence.
            items: Any = []
        else:
            items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise NebiusError(f"Nebius returned schema-invalid {kind} inventory")
        identities: list[str] = []
        for item in items:
            metadata = item.get("metadata") if isinstance(item, dict) else None
            child_id = str(
                metadata.get("id") if isinstance(metadata, dict) else ""
            ).strip()
            if not child_id:
                raise NebiusError(
                    f"Nebius returned a {kind} child without immutable identity"
                )
            identities.append(child_id)
        inventory[kind] = tuple(sorted(identities))
    # Access keys are separately enumerated through the safe allowlisted-field
    # adapter; never capture the provider's secret-bearing raw list response.
    keys = _list_access_key_metadata(exact_id, profile=profile)
    inventory["access_keys"] = tuple(
        sorted(
            str((item.get("metadata") or {}).get("id") or "").strip()
            for item in keys
            if str((item.get("metadata") or {}).get("id") or "").strip()
        )
    )
    if len(inventory["access_keys"]) != len(keys):
        raise NebiusError("Nebius returned an access key without immutable identity")
    return inventory


def delete_project(project_id: str, *, profile: str | None = None) -> None:
    """Delete one project through the supported exact-ID provider adapter."""

    exact_id = str(project_id or "").strip()
    if not exact_id:
        raise NebiusError("exact project ID is required")
    profile_args, _resolved_profile = _iam_profile_args(profile)
    try:
        _run([*profile_args, "iam", "v2", "project", "delete", "--id", exact_id])
    except NebiusError as exc:
        if _is_not_found(str(exc)):
            return
        raise


def get_project_default_network_identity(
    project_id: str, *, profile: str | None = None
) -> ProjectDefaultNetworkIdentity | None:
    """Return the unique provider default topology, rejecting mixed inventory."""

    exact_project = str(project_id or "").strip()
    if not exact_project:
        raise NebiusError("exact project ID is required")
    profile_args, resolved_profile = _iam_profile_args(profile)

    def _items(kind: str, command: list[str]) -> list[dict[str, Any]]:
        payload = _run_json(
            [*profile_args, *command, "--parent-id", exact_project, "--all"]
        )
        if payload == {} or payload == {"items": None}:
            return []
        rows = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise NebiusError(f"Nebius returned schema-invalid {kind} inventory")
        return rows

    networks = _items("network", ["vpc", "network", "list"])
    subnets = _items("subnet", ["vpc", "subnet", "list"])
    groups = _items("security-group", ["vpc", "security-group", "list"])
    if not networks and not subnets and not groups:
        return None
    if len(networks) != 1 or len(subnets) != 1 or len(groups) != 1:
        raise NebiusError(
            "project network inventory is not the unique provider default topology"
        )

    def _metadata(row: dict[str, Any], kind: str) -> tuple[str, str]:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise NebiusError(f"Nebius returned schema-invalid {kind} identity")
        resource_id = str(metadata.get("id") or "").strip()
        parent_id = str(
            metadata.get("parent_id") or metadata.get("parentId") or ""
        ).strip()
        name = str(metadata.get("name") or "").strip()
        if not resource_id or parent_id != exact_project or not name:
            raise NebiusError(f"Nebius returned mismatched {kind} identity")
        return resource_id, name

    network_id, network_name = _metadata(networks[0], "network")
    subnet_id, subnet_name = _metadata(subnets[0], "subnet")
    group_id, group_name = _metadata(groups[0], "security-group")
    subnet_spec = subnets[0].get("spec")
    group_spec = groups[0].get("spec")
    group_status = groups[0].get("status")
    if (
        not network_name.startswith("default-network")
        or not subnet_name.startswith("default-subnet-")
        or not group_name.startswith("default-security-group-")
        or not isinstance(subnet_spec, dict)
        or str(subnet_spec.get("network_id") or "") != network_id
        or not isinstance(group_spec, dict)
        or str(group_spec.get("network_id") or "") != network_id
        or not isinstance(group_status, dict)
        or group_status.get("default") is not True
    ):
        raise NebiusError(
            "project network inventory does not match the provider default topology"
        )
    return ProjectDefaultNetworkIdentity(
        network_id,
        network_name,
        subnet_id,
        subnet_name,
        group_id,
        group_name,
        exact_project,
        resolved_profile,
    )


def delete_project_default_network(
    identity: ProjectDefaultNetworkIdentity, *, profile: str | None = None
) -> None:
    """Delete one already-verified provider default subnet and parent network."""

    profile_args, _resolved_profile = _iam_profile_args(profile)
    for command, resource_id in (
        (("vpc", "subnet", "delete"), identity.subnet_id),
        (("vpc", "network", "delete"), identity.network_id),
    ):
        try:
            _run([*profile_args, *command, "--id", resource_id])
        except NebiusError as exc:
            if not _is_not_found(str(exc)):
                raise


def delete_subnet(subnet_id: str, *, profile: str | None = None) -> None:
    """Delete one exact subnet ID; no name discovery or creation is performed."""

    exact = str(subnet_id or "").strip()
    if not exact:
        raise NebiusError("exact subnet ID is required")
    profile_args, _resolved_profile = _iam_profile_args(profile)
    try:
        _run([*profile_args, "vpc", "subnet", "delete", "--id", exact])
    except NebiusError as exc:
        if not _is_not_found(str(exc)):
            raise


def delete_network(network_id: str, *, profile: str | None = None) -> None:
    """Delete one exact network ID; no name discovery or creation is performed."""

    exact = str(network_id or "").strip()
    if not exact:
        raise NebiusError("exact network ID is required")
    profile_args, _resolved_profile = _iam_profile_args(profile)
    try:
        _run([*profile_args, "vpc", "network", "delete", "--id", exact])
    except NebiusError as exc:
        if not _is_not_found(str(exc)):
            raise


def list_quota_allowances(
    tenant_id: str, *, profile: str | None = None
) -> dict[str, Any]:
    """Return one provider quota snapshot for *tenant_id*.

    Unlike the historical per-quota best-effort helpers, this API preserves
    provider/RBAC/malformed failures.  Mutation preflights must fail closed, and
    cannot distinguish "plenty of quota" from "the query was denied" if errors
    are normalized to ``(None, None)`` here.

    The query is profile-scoped like every other read here. Without that, a
    tenant only reachable through a non-default profile answers
    ``PermissionDenied``, and because this API fails closed that denial blocks a
    deploy the operator is fully entitled to make.
    """

    tenant = str(tenant_id or "").strip()
    if not tenant:
        raise NebiusError("tenant_id is required to list quota allowances")
    profile_args, _resolved = _iam_profile_args(profile)
    payload = _run_json(
        [*profile_args, "quotas", "quota-allowance", "list", "--parent-id", tenant, "--all"]
    )
    if not isinstance(payload.get("items"), list):
        raise NebiusError("quota allowance response is malformed: items is not a list")
    return payload


def get_public_ipv4_quota(
    tenant_id: str, region: str, *, profile: str | None = None
) -> tuple[int | None, int | None]:
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
    profile_args, _resolved = _iam_profile_args(profile)
    try:
        data = _run_json(
            [
                *profile_args,
                "quotas",
                "quota-allowance",
                "list",
                "--parent-id",
                tenant,
                "--all",
            ]
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


def get_compute_instance_quota(
    tenant_id: str, region: str, *, profile: str | None = None
) -> tuple[int | None, int | None]:
    """Return ``(usage, limit)`` for the tenant compute-instance quota in *region*.

    Nebius meters VMs via the ``compute.instance.count`` quota allowance; a tenant
    with ``limit 0`` (the reported failure) lets the agent VM's disk/network/SG
    create, then the instance create fails and the whole apply rolls back. Prefer
    an exact per-region allowance and fall back to a region-less (tenant-wide)
    one. Best-effort: ``(None, None)`` when unreadable, so callers never block a
    deploy on an unreadable quota. Nebius omits ``status.usage`` when nothing is
    allocated, so a missing usage reads as 0 (a real ``limit 0`` must gate).
    """
    tenant = str(tenant_id or "").strip()
    reg = str(region or "").strip()
    if not tenant or not reg:
        return (None, None)
    profile_args, _resolved = _iam_profile_args(profile)
    try:
        data = _run_json(
            [
                *profile_args,
                "quotas",
                "quota-allowance",
                "list",
                "--parent-id",
                tenant,
                "--all",
            ]
        )
    except Exception:
        return (None, None)

    def _to_int(value: Any) -> int | None:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    region_less: tuple[int | None, int | None] | None = None
    for item in data.get("items", []):
        metadata = item.get("metadata", {}) or {}
        if metadata.get("name") != "compute.instance.count":
            continue
        spec = item.get("spec", {}) or {}
        item_region = str(spec.get("region", "") or "").strip()
        status = item.get("status", {}) or {}
        pair = (_to_int(status.get("usage", "0")), _to_int(spec.get("limit")))
        if item_region == reg:
            return pair
        if not item_region and region_less is None:
            region_less = pair
    return region_less if region_less is not None else (None, None)


def discover_container_registry(
    project_id: str, *, preferred_region: str = ""
) -> str:
    """Compatibility seam for callers that previously discovered a registry.

    Official execution defaults to public GHCR and configuration no longer
    discovers or persists a provider registry. Customer BYOF registries must be
    selected explicitly.
    """

    del project_id, preferred_region
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
        # Compatibility for pre-typed storage probe summaries. New storage
        # callers route on StorageFailureKind instead of this text predicate.
        or ("s3 " in lowered and " was forbidden" in lowered)
    )


def is_permission_denied(message: str) -> bool:
    """Public predicate: does *message* look like a Nebius permission/access error?

    Lets callers (e.g. `npa configure`) render actionable IAM guidance instead of
    a raw rpc dump when provisioning is blocked by missing permissions.
    """
    return _is_permission_denied(message)


def _is_not_found(message: str) -> bool:
    lowered = message.lower()
    return (
        "notfound" in lowered or "not found" in lowered or "resourcenotfound" in lowered
    )


def is_not_found(message: str) -> bool:
    """Public predicate for idempotent teardown of already-absent resources."""

    return _is_not_found(message)


@dataclass(frozen=True)
class ComputeInstanceIdentity:
    instance_id: str
    name: str
    project_id: str
    labels: dict[str, str]
    profile: str = ""


def get_compute_instance_identity(
    instance_id: str,
    *,
    project_id: str,
    expected_name: str = "",
    profile: str | None = None,
) -> ComputeInstanceIdentity | None:
    """Strictly verify one immutable compute instance and its project scope."""

    exact_id = str(instance_id or "").strip()
    exact_project = str(project_id or "").strip()
    if not exact_id or not exact_project:
        raise NebiusError("Exact instance ID and project ID are required")
    profile_args, resolved_profile = _iam_profile_args(profile)
    try:
        payload = _run_json(
            [*profile_args, "compute", "instance", "get", "--id", exact_id]
        )
    except NebiusError as exc:
        if _is_not_found(str(exc)):
            return None
        raise
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise NebiusError("Nebius returned no compute-instance metadata")
    returned_id = str(metadata.get("id") or "").strip()
    returned_name = str(metadata.get("name") or "").strip()
    returned_project = str(
        metadata.get("parent_id") or metadata.get("parentId") or ""
    ).strip()
    if returned_id != exact_id or not returned_name or not returned_project:
        raise NebiusError("Nebius returned incomplete or mismatched compute identity")
    if returned_project != exact_project:
        raise NebiusError(
            f"Compute instance {exact_id} belongs to {returned_project}, not {exact_project}"
        )
    wanted_name = str(expected_name or "").strip()
    if wanted_name and returned_name != wanted_name:
        raise NebiusError(
            f"Compute instance {exact_id} has name {returned_name!r}, not {wanted_name!r}"
        )
    labels = metadata.get("labels")
    return ComputeInstanceIdentity(
        instance_id=returned_id,
        name=returned_name,
        project_id=returned_project,
        labels={str(key): str(value) for key, value in dict(labels or {}).items()},
        profile=resolved_profile,
    )


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


def _saved_service_account_id(project_id: str = "") -> str:
    import os

    from npa.clients.credentials import CREDENTIALS_PATH

    env_value = os.environ.get("NPA_SERVICE_ACCOUNT_ID", "").strip()
    if env_value:
        env_project = os.environ.get("NPA_SERVICE_ACCOUNT_PROJECT_ID", "").strip()
        return env_value if not project_id or env_project == project_id else ""
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
        saved_project = str(nebius.get("service_account_project_id", "") or "").strip()
        if project_id and saved_project != project_id:
            return ""
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
    on_created: Callable[[str], None] | None = None,
    allow_saved_fallback: bool = True,
) -> str:
    """Get or create a service account, return its ID.

    ``on_created`` is called only after this invocation successfully creates the
    account. Teardown uses that event to persist a narrow ownership record; an
    account found by name, recovered from an IAM error, or loaded from existing
    credentials is deliberately never claimed as NPA-owned.
    """
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_service_account")
    # Try to find existing.
    try:
        data = _run_json(
            [
                "iam",
                "service-account",
                "get-by-name",
                "--parent-id",
                project_id,
                "--name",
                name,
            ]
        )
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
            if saved and allow_saved_fallback:
                return saved
            raise NebiusError(
                f"Cannot read or create service account {name!r}: {exc}. "
                + (
                    "Set NPA_SERVICE_ACCOUNT_ID or nebius.service_account_id in "
                    "~/.npa/credentials.yaml when IAM management is restricted."
                    if allow_saved_fallback
                    else "The named service-account identity could not be verified; "
                    "a storage account cannot be substituted for the agent account."
                )
            ) from exc
        if not _is_not_found(message):
            raise
        # Not found — create below.

    try:
        data = _run_json(
            [
                "iam",
                "service-account",
                "create",
                "--parent-id",
                project_id,
                "--name",
                name,
                "--description",
                description,
            ]
        )
    except NebiusError as exc:
        if _is_permission_denied(str(exc)) and allow_saved_fallback:
            saved = _saved_service_account_id()
            if saved:
                return saved
        raise
    sa_id = data.get("metadata", {}).get("id", "")
    if not sa_id:
        raise NebiusError("Service account creation did not return an ID")
    if on_created:
        on_created(sa_id)
    return sa_id


# ── Editors group membership ─────────────────────────────────────────────


def _group_has_member(group_id: str, member_id: str) -> bool:
    members_data = _run_json(
        [
            "iam",
            "group-membership",
            "list-members",
            "--parent-id",
            group_id,
            "--page-size",
            "1000",
        ]
    )
    memberships = members_data.get("memberships", members_data.get("items", []))
    if not isinstance(memberships, list):
        raise NebiusError("IAM group membership inventory returned an invalid schema")
    return any(
        isinstance(item, dict)
        and str((item.get("spec") or {}).get("member_id") or "") == member_id
        for item in memberships
    )


def _ensure_editors_membership_impl(
    tenant_id: str, sa_id: str
) -> StorageIamBindingEvidence:
    """Explicit compatibility fallback for the tenant-wide editors group."""
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_editors_membership")
    group_data = _run_json(
        [
            "iam",
            "group",
            "get-by-name",
            "--parent-id",
            tenant_id,
            "--name",
            "editors",
        ]
    )
    group_id = group_data.get("metadata", {}).get("id", "")
    if not group_id:
        raise NebiusError(f"Could not find editors group in tenant {tenant_id}")

    if _group_has_member(group_id, sa_id):
        return StorageIamBindingEvidence(
            IamBindingState.EXISTING,
            "editor",
            tenant_id,
            group_id,
            "editors",
            compatibility_fallback=True,
        )

    _run(
        [
            "iam",
            "group-membership",
            "create",
            "--parent-id",
            group_id,
            "--member-id",
            sa_id,
        ]
    )
    return StorageIamBindingEvidence(
        IamBindingState.CREATED,
        "editor",
        tenant_id,
        group_id,
        "editors",
        compatibility_fallback=True,
    )


def ensure_editors_membership(tenant_id: str, sa_id: str) -> StorageIamBindingEvidence:
    """Reconcile the explicit broad fallback and type terminal failure evidence."""

    try:
        return _ensure_editors_membership_impl(tenant_id, sa_id)
    except NebiusError as exc:
        evidence = StorageIamBindingEvidence(
            IamBindingState.FAILED,
            "editor",
            tenant_id,
            "",
            "editors",
            compatibility_fallback=True,
        )
        raise StorageIamBindingError(
            "editors compatibility binding failed; required S3 actions: "
            + ", ".join(STORAGE_REQUIRED_S3_ACTIONS),
            evidence,
        ) from exc


def _existing_editors_binding(
    tenant_id: str, sa_id: str
) -> StorageIamBindingEvidence | None:
    """Read-only compatibility verification for existing installations."""

    group_data = _run_json(
        [
            "iam",
            "group",
            "get-by-name",
            "--parent-id",
            tenant_id,
            "--name",
            "editors",
        ]
    )
    group_id = str((group_data.get("metadata") or {}).get("id") or "")
    if not group_id:
        raise NebiusError(f"Could not find editors group in tenant {tenant_id}")
    if not _group_has_member(group_id, sa_id):
        return None
    return StorageIamBindingEvidence(
        IamBindingState.EXISTING,
        "editor",
        tenant_id,
        group_id,
        "editors",
        compatibility_fallback=True,
    )


def ensure_storage_capability_binding(
    *,
    project_id: str,
    tenant_id: str,
    bucket_id: str,
    service_account_id: str,
    allow_editors_fallback: bool = False,
    on_resource_created: Callable[[str, dict[str, str]], None] | None = None,
) -> StorageIamBindingEvidence:
    """Ensure the narrow provider-verified bucket object capability binding.

    Nebius assigns roles to groups. NPA therefore creates/reuses one project-
    scoped custom group, grants ``storage.object-editor`` on the exact bucket,
    and adds the storage service account. Existing tenant-wide editors members
    remain accepted for compatibility, but NPA only creates that broad binding
    when the operator explicitly opts into the fallback.
    """

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_storage_capability_binding")
    if not all((project_id, tenant_id, bucket_id, service_account_id)):
        raise NebiusError(
            "storage IAM verification requires exact project, tenant, bucket, and service-account IDs"
        )

    changed = False
    group_created = False
    group_name = storage_binding_group_name(project_id)
    try:
        group_data = _run_json(
            [
                "iam",
                "group",
                "get-by-name",
                "--parent-id",
                project_id,
                "--name",
                group_name,
            ]
        )
    except NebiusError as exc:
        if not _is_not_found(str(exc)):
            raise NebiusError(
                "Storage IAM inventory is unreadable; refusing to create a key or probe. "
                f"Required S3 actions: {', '.join(STORAGE_REQUIRED_S3_ACTIONS)}."
            ) from exc
        group_data = _run_json(
            [
                "iam",
                "group",
                "create",
                "--parent-id",
                project_id,
                "--name",
                group_name,
            ]
        )
        changed = group_created = True
    group_id = str((group_data.get("metadata") or {}).get("id") or "")
    if not group_id:
        raise NebiusError("Storage IAM group reconciliation did not return an exact ID")
    if group_created and on_resource_created:
        on_resource_created(
            "iam_group",
            {"id": group_id, "name": group_name, "project_id": project_id},
        )

    permits = _run_json(
        ["iam", "access-permit", "list", "--parent-id", group_id, "--all"]
    )
    items = permits.get("items", permits.get("access_permits", []))
    if not isinstance(items, list):
        raise NebiusError(
            "Storage IAM access-permit inventory returned an invalid schema"
        )
    matching = [
        item
        for item in items
        if isinstance(item, dict)
        and str((item.get("spec") or {}).get("resource_id") or "") == bucket_id
        and str((item.get("spec") or {}).get("role") or "") == STORAGE_RUNTIME_ROLE
    ]
    permit_id = ""
    if matching:
        permit_id = str((matching[0].get("metadata") or {}).get("id") or "")
        if not permit_id:
            raise NebiusError("Matching storage IAM permit is missing its exact ID")
    else:
        try:
            permit = _run_json(
                [
                    "iam",
                    "access-permit",
                    "create",
                    "--parent-id",
                    group_id,
                    "--resource-id",
                    bucket_id,
                    "--role",
                    STORAGE_RUNTIME_ROLE,
                ]
            )
        except NebiusError as exc:
            if allow_editors_fallback and re.search(
                r"(?i)unsupported|unimplemented|unknown role", str(exc)
            ):
                return ensure_editors_membership(tenant_id, service_account_id)
            raise NebiusError(
                "Cannot establish the supported narrow storage binding. Required S3 "
                f"actions: {', '.join(STORAGE_REQUIRED_S3_ACTIONS)}. Supported choices: "
                f"bucket-scoped {STORAGE_RUNTIME_ROLE}; an already-converged editors "
                "membership; or explicit editors compatibility fallback when the provider "
                "reports the narrow role unsupported."
            ) from exc
        permit_id = str((permit.get("metadata") or {}).get("id") or "")
        if not permit_id:
            raise NebiusError("Storage IAM permit creation did not return an exact ID")
        changed = True
        if on_resource_created:
            on_resource_created("iam_permit", {"id": permit_id, "group_id": group_id})

    if not _group_has_member(group_id, service_account_id):
        _run(
            [
                "iam",
                "group-membership",
                "create",
                "--parent-id",
                group_id,
                "--member-id",
                service_account_id,
            ]
        )
        changed = True
    return StorageIamBindingEvidence(
        IamBindingState.CREATED if changed else IamBindingState.EXISTING,
        STORAGE_RUNTIME_ROLE,
        bucket_id,
        group_id,
        group_name,
        permit_id,
    )


def delete_access_permit(permit_id: str, *, profile: str | None = None) -> None:
    if permit_id:
        profile_args, _resolved_profile = _iam_profile_args(profile)
        _run(["iam", "access-permit", "delete", "--id", permit_id, *profile_args])


def delete_group(group_id: str, *, profile: str | None = None) -> None:
    if group_id:
        profile_args, _resolved_profile = _iam_profile_args(profile)
        _run(["iam", "group", "delete", "--id", group_id, *profile_args])


# ── Access keys ──────────────────────────────────────────────────────────


# Nebius CLI list JSON currently contains secret-bearing status fields. The list
# call therefore projects *only* opaque resource IDs. Each ID is then inspected
# with one scalar JSONPath projection per allowlisted field. Keeping optional
# fields out of a shared range projection is important: the provider exits
# non-zero when even one heterogeneous list item omits a projected field.
_ACCESS_KEY_LIST_JSONPATH = 'jsonpath={range .items[*]}{.metadata.id}{"\\n"}{end}'

_ACCESS_KEY_METADATA_FIELDS = {
    "id": ".metadata.id",
    "name": ".metadata.name",
    "nested_service_account_id": ".spec.account.service_account.id",
    "direct_service_account_id": ".spec.account.service_account_id",
    "state": ".status.state",
    "expires_at": ".spec.expires_at",
}
_ACCESS_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _safe_access_key_field(value: str) -> str:
    field = str(value or "").strip()
    return "" if field == "<no value>" else field


def _validate_access_key_scalar(
    value: str,
    *,
    field_name: str,
    identifier: bool = False,
) -> str:
    """Validate a single provider-projected access-key metadata scalar.

    The original provider response is deliberately never included in errors.
    Structured/multiline output means the CLI did not honor the scalar
    allowlist and could contain credentials, so it is rejected at the command
    boundary before any caller can log it.
    """

    scalar = _safe_access_key_field(value)
    if not scalar:
        return ""
    if any(character in scalar for character in ("\n", "\r", "\t", "\x00")):
        raise NebiusError(
            f"Nebius returned malformed allowlisted access-key {field_name}; "
            "the provider response was discarded"
        )
    if scalar[:1] in {"{", "["} or scalar[-1:] in {"}", "]"}:
        raise NebiusError(
            f"Nebius returned non-scalar allowlisted access-key {field_name}; "
            "the potentially secret-bearing provider response was discarded"
        )
    if identifier and not _ACCESS_KEY_ID_RE.fullmatch(scalar):
        raise NebiusError(
            f"Nebius returned malformed allowlisted access-key {field_name}; "
            "the provider response was discarded"
        )
    return scalar


_EMPTY_ACCESS_KEY_LIST_ERRORS = (
    re.compile(r"\bitems\s+is\s+not\s+found\b", re.IGNORECASE),
    re.compile(
        r"(?:range|iterate).{0,80}\bitems\b.{0,80}\b(?:null|nil)\b", re.IGNORECASE
    ),
    re.compile(
        r"\bitems\b.{0,80}\b(?:null|nil)\b.{0,80}(?:range|iterate)", re.IGNORECASE
    ),
)


def _empty_access_key_list_error(message: str) -> bool:
    """Whether JSONPath failed solely because an empty response omitted/null-ed items."""

    safe = redact_nebius_output(str(message or ""))
    lowered = safe.lower()
    if any(
        marker in lowered
        for marker in (
            "accessdenied",
            "access denied",
            "permissiondenied",
            "permission denied",
            "unauthenticated",
            "unauthorized",
            "forbidden",
            "connection refused",
            "deadline exceeded",
            "timed out",
        )
    ):
        return False
    return any(pattern.search(safe) for pattern in _EMPTY_ACCESS_KEY_LIST_ERRORS)


def _missing_access_key_field_error(message: str, jsonpath: str) -> bool:
    """Whether a scalar projection failed solely because its optional field is absent."""

    safe = redact_nebius_output(str(message or ""))
    lowered = safe.lower()
    if any(
        marker in lowered
        for marker in (
            "accessdenied",
            "access denied",
            "permissiondenied",
            "permission denied",
            "unauthenticated",
            "unauthorized",
            "forbidden",
            "connection refused",
            "deadline exceeded",
            "timed out",
        )
    ):
        return False
    leaf = re.escape(jsonpath.rsplit(".", 1)[-1])
    return (
        re.search(
            rf"\b{leaf}\b.{{0,80}}\b(?:is\s+not\s+found|missing|null|nil|no\s+value)\b",
            safe,
            re.IGNORECASE,
        )
        is not None
    )


def _access_key_metadata_scalar(
    key_id: str,
    field_name: str,
    *,
    optional: bool,
    identifier: bool = False,
) -> str:
    """Read one allowlisted scalar for one access-key resource."""

    jsonpath = _ACCESS_KEY_METADATA_FIELDS[field_name]
    try:
        output = _run(
            [
                "iam",
                "v2",
                "access-key",
                "get",
                "--id",
                key_id,
                "--format",
                f"jsonpath={{{jsonpath}}}",
            ]
        )
    except NebiusError as exc:
        if optional and _missing_access_key_field_error(str(exc), jsonpath):
            return ""
        raise NebiusError(
            f"Unable to read allowlisted access-key {field_name} for {key_id}: {exc}"
        ) from exc
    value = _validate_access_key_scalar(
        output,
        field_name=field_name,
        identifier=identifier,
    )
    if not optional and not value:
        raise NebiusError(
            f"Nebius returned no allowlisted access-key {field_name} for {key_id}"
        )
    return value


def _list_access_key_metadata(
    project_id: str, *, profile: str | None = None
) -> list[dict[str, Any]]:
    """Return a strict allowlist of access-key metadata from the Nebius CLI.

    Do not replace this with ``_run_json(... access-key list ...)``: the upstream
    list response can include the access-key secret. JSONPath field selection is
    performed inside the CLI, so the raw secret-bearing object never reaches NPA
    stdout capture, parsing, exceptions, logs, or machine-readable state.
    """

    if not project_id:
        return []
    try:
        profile_args, _resolved_profile = _iam_profile_args(profile)
        output = _run(
            [
                *profile_args,
                "iam",
                "v2",
                "access-key",
                "list",
                "--parent-id",
                project_id,
                "--all",
                "--format",
                _ACCESS_KEY_LIST_JSONPATH,
            ]
        )
    except NebiusError as exc:
        # CLI 0.12.254 returns `{}` (or `items: null`) for an empty list and its
        # kubectl-style JSONPath formatter exits non-zero before emitting rows.
        # This narrow compatibility case is an empty inventory, not a provider
        # failure. Every other error remains strict, and only allowlisted fields
        # were requested from the CLI, so no secret-bearing JSON is captured.
        if _empty_access_key_list_error(str(exc)):
            return []
        raise
    key_ids: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        key_id = _validate_access_key_scalar(
            line,
            field_name="id",
            identifier=True,
        )
        # Some CLI builds render a null ranged item as ``<no value>`` instead
        # of failing the JSONPath expression. It is still an empty inventory,
        # not an access-key resource with an empty identity.
        if not key_id:
            continue
        if key_id in key_ids:
            raise NebiusError(
                "Nebius returned duplicate allowlisted access-key IDs; "
                "the ambiguous provider response was discarded"
            )
        key_ids.append(key_id)

    items: list[dict[str, Any]] = []
    for key_id in key_ids:
        returned_id = _access_key_metadata_scalar(
            key_id,
            "id",
            optional=False,
            identifier=True,
        )
        if returned_id != key_id:
            raise NebiusError(
                "Nebius returned a different access-key ID while inspecting an "
                "allowlisted resource; the ambiguous provider response was discarded"
            )
        name = _access_key_metadata_scalar(key_id, "name", optional=True)
        nested_sa_id = _access_key_metadata_scalar(
            key_id,
            "nested_service_account_id",
            optional=True,
            identifier=True,
        )
        direct_sa_id = _access_key_metadata_scalar(
            key_id,
            "direct_service_account_id",
            optional=True,
            identifier=True,
        )
        if nested_sa_id and direct_sa_id and nested_sa_id != direct_sa_id:
            raise NebiusError(
                "Nebius returned conflicting service-account identities for an "
                "access key; the ambiguous provider response was discarded"
            )
        state = _access_key_metadata_scalar(key_id, "state", optional=True)
        expires_at = _access_key_metadata_scalar(key_id, "expires_at", optional=True)
        items.append(
            {
                "metadata": {"id": key_id, "name": name},
                "spec": {
                    "account": {
                        "service_account": {"id": nested_sa_id},
                        "service_account_id": direct_sa_id,
                    },
                    "expires_at": expires_at,
                },
                "status": {"state": state},
            }
        )
    return items


def _find_active_access_key(
    project_id: str,
    sa_id: str,
    *,
    key_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the first ACTIVE access key for the given service account, or None."""
    for item in _list_access_key_metadata(project_id):
        spec = item.get("spec", {})
        account = spec.get("account", {})
        # The SA ID can live under different JSON paths depending on API version.
        item_sa_id = account.get("service_account", {}).get("id", "") or account.get(
            "service_account_id", ""
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
    on_created: Callable[[str, str], None] | None = None,
) -> tuple[str, str]:
    """Ensure an active access key exists, return (aws_access_key_id, aws_secret_access_key).

    Reuses an existing key when possible; creates a new one otherwise.
    """
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_access_key")
    existing = _find_active_access_key(
        project_id, sa_id, key_name=key_name
    ) or _find_active_access_key(project_id, sa_id)
    if existing:
        key_id = existing["metadata"]["id"]
        # Retrieve the AWS access key ID.
        get_data = _run_json(["iam", "v2", "access-key", "get", "--id", key_id])
        aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")
        # Try to retrieve the secret (works for keys where the secret is stored).
        try:
            secret_data = _run_json(
                ["iam", "v2", "access-key", "get-secret", "--id", key_id]
            )
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
        create_name = (
            f"{key_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
    create_data = _run_json(
        [
            "iam",
            "v2",
            "access-key",
            "create",
            "--parent-id",
            project_id,
            "--name",
            create_name,
            "--account-service-account-id",
            sa_id,
            "--description",
            description,
        ]
    )
    new_key_id = create_data.get("metadata", {}).get("id", "")
    if not new_key_id:
        raise NebiusError("Access key creation did not return an ID")
    if on_created:
        on_created(str(new_key_id), create_name)

    # Fetch the AWS-compatible credentials.
    get_data = _run_json(["iam", "v2", "access-key", "get", "--id", new_key_id])
    aws_access_key = get_data.get("status", {}).get("aws_access_key_id", "")

    secret_data = _run_json(
        ["iam", "v2", "access-key", "get-secret", "--id", new_key_id]
    )
    aws_secret_key = secret_data.get("secret", "")

    if not aws_access_key or not aws_secret_key:
        raise NebiusError(
            "Failed to retrieve AWS-compatible credentials from new access key"
        )

    return aws_access_key, aws_secret_key


# ── S3 bucket ────────────────────────────────────────────────────────────

DEFAULT_BUCKET_BASENAME = "npa-bucket"
DEFAULT_BUCKET_STORAGE_CLASS = "standard"
RERUN_BROWSER_CORS_RULE_ID = "npa-rerun-app"
RERUN_BROWSER_ORIGIN = "https://app.rerun.io"


@dataclass(frozen=True)
class BucketCorsPlan:
    """A lossless plan for reconciling the Rerun browser CORS rule."""

    bucket_id: str
    resource_version: str
    current_rules: tuple[dict[str, Any], ...]
    desired_rules: tuple[dict[str, Any], ...]
    changed: bool

    @property
    def preserved_rule_count(self) -> int:
        return sum(
            1
            for rule in self.current_rules
            if str(rule.get("id") or "") != RERUN_BROWSER_CORS_RULE_ID
        )


def rerun_browser_cors_rule() -> dict[str, Any]:
    """Return the least-privilege CORS rule needed by the Rerun web viewer."""

    return {
        "id": RERUN_BROWSER_CORS_RULE_ID,
        "allowed_origins": [RERUN_BROWSER_ORIGIN],
        "allowed_methods": ["GET"],
        "allowed_headers": ["Range"],
        "expose_headers": ["Accept-Ranges", "Content-Length", "Content-Range"],
        "max_age_seconds": 3600,
    }


def _bucket_cors_rules(item: dict[str, Any]) -> list[dict[str, Any]]:
    spec = item.get("spec")
    if not isinstance(spec, dict):
        raise NebiusError("exact bucket lookup returned no resource spec")
    cors = spec.get("cors")
    if cors in (None, {}):
        return []
    if not isinstance(cors, dict):
        raise NebiusError("exact bucket lookup returned schema-invalid CORS data")
    rules = cors.get("rules", [])
    if not isinstance(rules, list) or not all(isinstance(rule, dict) for rule in rules):
        raise NebiusError("exact bucket lookup returned schema-invalid CORS rules")
    return [dict(rule) for rule in rules]


def _rule_values(rule: Mapping[str, Any], field: str) -> set[str]:
    values = rule.get(field, [])
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def bucket_cors_supports_rerun(rule: Mapping[str, Any]) -> bool:
    """Return whether one provider rule satisfies the browser fetch contract."""

    origins = _rule_values(rule, "allowed_origins")
    methods = _rule_values(rule, "allowed_methods")
    headers = _rule_values(rule, "allowed_headers")
    exposed = _rule_values(rule, "expose_headers")
    required_exposed = {"accept-ranges", "content-length", "content-range"}
    return bool(
        (RERUN_BROWSER_ORIGIN.lower() in origins or "*" in origins)
        and "get" in methods
        and ("range" in headers or "*" in headers)
        and (required_exposed <= exposed or "*" in exposed)
    )


def plan_bucket_rerun_cors(project_id: str, bucket_name: str) -> BucketCorsPlan:
    """Read and merge a Rerun rule without discarding unrelated bucket CORS."""

    item = get_bucket_by_name(project_id, bucket_name)
    if item is None:
        raise NebiusError("the configured object-storage bucket does not exist")
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise NebiusError("exact bucket lookup returned no resource metadata")
    bucket_id = str(metadata.get("id") or "").strip()
    if not bucket_id:
        raise NebiusError("exact bucket lookup returned no resource ID")
    resource_version = str(metadata.get("resource_version") or "").strip()
    current = _bucket_cors_rules(item)
    if any(bucket_cors_supports_rerun(rule) for rule in current):
        desired = list(current)
    else:
        desired = [
            rule
            for rule in current
            if str(rule.get("id") or "") != RERUN_BROWSER_CORS_RULE_ID
        ]
        desired.append(rerun_browser_cors_rule())
    return BucketCorsPlan(
        bucket_id=bucket_id,
        resource_version=resource_version,
        current_rules=tuple(current),
        desired_rules=tuple(desired),
        changed=current != desired,
    )


def apply_bucket_rerun_cors(project_id: str, bucket_name: str) -> BucketCorsPlan:
    """Reconcile the browser rule through bucket-admin control-plane credentials."""

    plan = plan_bucket_rerun_cors(project_id, bucket_name)
    if not plan.changed:
        return plan
    args = [
        "storage",
        "bucket",
        "update",
        "--id",
        plan.bucket_id,
        "--cors-rules",
        json.dumps(list(plan.desired_rules), separators=(",", ":"), sort_keys=True),
    ]
    if plan.resource_version:
        args.extend(["--resource-version", plan.resource_version])
    _run_json(args)

    verified = plan_bucket_rerun_cors(project_id, bucket_name)
    if verified.changed:
        raise NebiusError("bucket CORS update completed but read-back verification failed")
    return BucketCorsPlan(
        bucket_id=verified.bucket_id,
        resource_version=verified.resource_version,
        current_rules=plan.current_rules,
        desired_rules=verified.desired_rules,
        changed=True,
    )


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
    """Return every bucket in *project_id* for explicit inventory commands.

    Exact-name configure checks use ``get_bucket_by_name`` and never enumerate
    unrelated buckets in the project.
    """
    data = _run_json(
        [
            "storage",
            "bucket",
            "list",
            "--parent-id",
            project_id,
            "--all",
        ]
    )
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def get_bucket_by_name(project_id: str, bucket_name: str) -> dict[str, Any] | None:
    """Return the exact parent-scoped bucket, or ``None`` on verified NotFound."""

    exact_project = str(project_id or "").strip()
    exact_name = str(bucket_name or "").strip()
    if not exact_project or not exact_name:
        return None
    try:
        item = _run_json(
            [
                "storage",
                "bucket",
                "get-by-name",
                "--parent-id",
                exact_project,
                "--name",
                exact_name,
            ]
        )
    except NebiusError as exc:
        if _is_exact_bucket_not_found(str(exc), exact_name):
            return None
        raise
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise NebiusError("exact bucket lookup returned no resource metadata")
    if str(metadata.get("name") or "").strip() != exact_name:
        raise NebiusError("exact bucket lookup returned an unexpected resource name")
    returned_parent = str(metadata.get("parent_id") or "").strip()
    if returned_parent and returned_parent != exact_project:
        raise NebiusError("exact bucket lookup returned an unexpected parent")
    return item


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
    return get_bucket_by_name(project_id, bucket_name) is not None


def _is_exact_bucket_not_found(message: str, bucket_name: str) -> bool:
    """Classify only a provider-confirmed absence of the requested bucket.

    The command wrapper itself contains ``storage bucket`` for every failure, so
    a broad ``"not found"`` substring check can misclassify a missing CLI
    profile or parent project as bucket absence. Restrict the decision to the
    provider detail line naming the exact bucket or explicitly saying that a
    bucket does not exist.
    """

    exact_name = str(bucket_name or "").strip().lower()
    for raw_line in str(message or "").splitlines():
        detail = raw_line.strip().lower()
        # The current CLI emits the bucket-specific provider detail before
        # trailing request/trace diagnostics, so inspecting only the final line
        # loses the authoritative NoSuchBucket result.
        if "nosuchbucket" in detail:
            return True
        if not _is_not_found(detail):
            continue
        if exact_name and exact_name in detail:
            return True
        if re.search(
            r"\bbucket\b\s+(?:does(?:n['’]?t| not)\s+exist|not found|is missing)\b",
            detail,
        ):
            return True
    return False


def ensure_bucket(
    project_id: str,
    bucket_name: str,
    *,
    max_size_bytes: int = 0,
    default_storage_class: str = DEFAULT_BUCKET_STORAGE_CLASS,
    on_created: Callable[[str], None] | None = None,
    allow_existing: bool = True,
) -> str:
    """Get or create an S3 bucket, return its name.

    *max_size_bytes* caps a newly created bucket (0 = unlimited). It is only
    applied when the bucket is created; an existing bucket is reused unchanged.
    *default_storage_class* is applied only when the bucket is created.
    Set *allow_existing* false when a generated name must never be adopted if it
    appears during either exact lookup race.
    """
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_bucket")
    if bucket_exists(project_id, bucket_name):
        if allow_existing:
            return bucket_name
        raise NebiusError(
            f"Object-storage bucket name '{bucket_name}' is already taken; "
            "refusing to adopt an existing bucket selected by a generated-name "
            "configure flow."
        )

    storage_class = normalize_bucket_storage_class(default_storage_class)
    args = [
        "storage",
        "bucket",
        "create",
        "--name",
        bucket_name,
        "--parent-id",
        project_id,
        "--versioning-policy",
        "enabled",
        "--default-storage-class",
        storage_class,
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
            if allow_existing:
                return bucket_name
            raise NebiusError(
                f"Object-storage bucket name '{bucket_name}' became already taken "
                "during creation; refusing to adopt an existing bucket selected "
                "by a generated-name configure flow."
            ) from exc
        raise NebiusError(
            f"Object-storage bucket name '{bucket_name}' is already taken "
            "(bucket names are globally unique) and is not in project "
            f"{project_id}. Re-run `npa configure` and enter a different, "
            "unused bucket name."
        ) from exc
    if on_created:
        on_created(bucket_name)
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
    on_resource_created: Callable[[str, dict[str, str]], None] | None = None,
    allow_editors_fallback: bool = False,
    allow_existing_bucket: bool = True,
) -> dict[str, str]:
    """Run the full environment bootstrap, return a dict of credentials.

    This is the Python equivalent of ``source environment.sh``.

    *bucket_name* selects the object-storage bucket; when omitted it falls back
    to the deterministic ``bucket_name_for`` name. *bucket_max_size_bytes* caps
    a newly created bucket (0 = unlimited); it is ignored when the bucket
    already exists. *bucket_storage_class* applies only when the bucket is
    created. Set *allow_existing_bucket* false for a generated configure name
    that must fail closed across a concurrent create. *on_status* is an optional
    callback ``(message: str) -> None`` for progress reporting.
    """

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("bootstrap_environment")

    def _status(msg: str) -> None:
        if on_status:
            on_status(msg)

    _status("Getting IAM access token...")
    iam_token = get_iam_token()

    _status("Setting up service account...")
    created_service_account_id = ""

    def _record_created_service_account(account_id: str) -> None:
        nonlocal created_service_account_id
        created_service_account_id = account_id
        if on_resource_created:
            on_resource_created(
                "service_account",
                {"id": account_id, "name": service_account_name},
            )

    sa_id = ensure_service_account(
        project_id,
        name=service_account_name,
        description=service_account_description,
        on_created=_record_created_service_account,
    )

    def _with_storage_account_ownership(payload: dict[str, str]) -> dict[str, str]:
        if created_service_account_id == sa_id:
            # This provenance is intentionally emitted only for the storage
            # account created in this call. Agent IAM has its own shared-account
            # teardown, and reused/user-managed accounts must never acquire it.
            payload = dict(payload)
            payload.update(
                {
                    "service_account_name": service_account_name,
                    "service_account_project_id": project_id,
                    "service_account_managed_by": "npa",
                }
            )
        return payload

    bucket_name = bucket_name or bucket_name_for(tenant_id, project_id)
    saved_storage: dict[str, str] | None = None
    created_bucket = False

    def _record_created_bucket(name: str) -> None:
        nonlocal created_bucket
        created_bucket = True
        if on_resource_created:
            on_resource_created("bucket", {"name": name})

    _status("Setting up S3 bucket...")
    try:
        ensure_bucket(
            project_id,
            bucket_name,
            max_size_bytes=bucket_max_size_bytes,
            default_storage_class=bucket_storage_class,
            on_created=_record_created_bucket,
            allow_existing=allow_existing_bucket,
        )
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        saved_storage = _saved_storage_credentials(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            service_account_id=sa_id,
        )
        if saved_storage is None:
            raise
        _status(
            "Bucket ensure was forbidden; validating the saved exact-project storage "
            "identity and IAM binding without provisioning."
        )

    bucket = get_bucket_by_name(project_id, bucket_name)
    bucket_id = str(((bucket or {}).get("metadata") or {}).get("id") or "")
    if not bucket_id:
        raise NebiusError(
            "Created/reused bucket could not be resolved to an exact provider ID; "
            "refusing IAM changes, key creation, and write probe."
        )

    _status("Verifying least-privilege storage capability binding...")
    try:
        try:
            binding = _existing_editors_binding(tenant_id, sa_id)
        except NebiusError:
            # A project-scoped administrator does not need tenant-wide group
            # inventory. Failure to read the legacy compatibility binding is not
            # evidence of capability, so continue to the exact-project binding
            # and still fail closed unless that narrower path is verified.
            binding = None
            _status(
                "Tenant-wide editors membership is not readable; verifying the "
                "exact-project storage binding instead."
            )
        if binding is None:
            binding = ensure_storage_capability_binding(
                project_id=project_id,
                tenant_id=tenant_id,
                bucket_id=bucket_id,
                service_account_id=sa_id,
                allow_editors_fallback=allow_editors_fallback,
                on_resource_created=on_resource_created,
            )
    except NebiusError as exc:
        failed = StorageIamBindingEvidence(
            IamBindingState.FAILED,
            STORAGE_RUNTIME_ROLE,
            bucket_id,
            "",
            storage_binding_group_name(project_id),
        )
        raise StorageIamBindingError(
            "Required storage IAM capability verification failed before access-key "
            "creation/probe. Required S3 actions: "
            f"{', '.join(STORAGE_REQUIRED_S3_ACTIONS)}. Supported binding choices: "
            f"bucket-scoped {STORAGE_RUNTIME_ROLE}; verified existing editors "
            "membership; or explicit editors compatibility fallback only when the "
            "provider reports the narrow role unsupported. The active profile needs "
            "project-scoped admin permission to manage the storage IAM group and its "
            "access permit; tenant-wide project listing or tenant-wide admin is not "
            "required.",
            failed,
        ) from exc

    _status("Setting up access key for S3...")
    try:
        aws_access_key, aws_secret_key = ensure_access_key(
            project_id,
            sa_id,
            key_name=access_key_name,
            description=access_key_description,
            **(
                {
                    "on_created": lambda key_id, name: on_resource_created(
                        "access_key",
                        {"id": key_id, "name": name, "service_account_id": sa_id},
                    )
                }
                if on_resource_created
                else {}
            ),
        )
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            raise
        saved_storage = saved_storage or _saved_storage_credentials(
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            bucket_name=bucket_name,
            service_account_id=sa_id,
        )
        if saved_storage is None:
            raise
        aws_access_key = str(saved_storage.get("nebius_api_key") or "")
        aws_secret_key = str(saved_storage.get("nebius_secret_key") or "")
        _status(
            "Reusing the saved exact-project active access key after IAM capability verification."
        )

    s3_endpoint = f"https://storage.{region}.nebius.cloud"

    result = {
        "iam_token": iam_token,
        "service_account_id": sa_id,
        "nebius_api_key": aws_access_key,
        "nebius_secret_key": aws_secret_key,
        "s3_bucket": bucket_name,
        "s3_endpoint": s3_endpoint,
        "bucket_disposition": "created" if created_bucket else "reused",
        "nebius_project_id": project_id,
        "nebius_region": region,
        "iam_binding_state": binding.state.value,
        "iam_binding_role": binding.role,
        "iam_binding_scope_id": binding.scope_id,
        "iam_binding_group_id": binding.group_id,
        "iam_binding_group_name": binding.group_name,
        "iam_binding_access_permit_id": binding.access_permit_id,
        "iam_binding_compatibility_fallback": str(
            binding.compatibility_fallback
        ).lower(),
    }
    _status(
        "Storage permission contract: "
        f"{binding.role} at {'tenant' if binding.compatibility_fallback else 'bucket'} scope; required S3 actions "
        f"{', '.join(STORAGE_REQUIRED_S3_ACTIONS)}; binding {binding.state.value}."
    )
    return _with_storage_account_ownership(result)


def resolve_service_account_id(
    project_id: str,
    *,
    names: tuple[str, ...] = (AGENT_SERVICE_ACCOUNT_NAME, DEFAULT_SERVICE_ACCOUNT_NAME),
) -> str:
    """Resolve a service-account id from config or best-effort IAM lookups."""

    project = str(project_id or "").strip()
    saved = _saved_service_account_id(project)
    if saved:
        return saved
    if not project:
        return ""
    for name in names:
        sa_id = get_service_account_id_by_name(project, name)
        if sa_id:
            return sa_id
    return ""


def get_service_account_id_by_name(
    project_id: str,
    name: str,
    *,
    strict: bool = False,
    profile: str | None = None,
) -> str | None:
    """Return a service-account id when *name* exists, else ``None``.

    ``strict`` preserves provider/auth failures so teardown can distinguish an
    account that is verified absent from one that could not be inspected.
    """

    try:
        profile_args, _resolved_profile = _iam_profile_args(profile)
        data = _run_json(
            [
                *profile_args,
                "iam",
                "service-account",
                "get-by-name",
                "--parent-id",
                project_id,
                "--name",
                name,
            ]
        )
    except NebiusError as exc:
        message = str(exc)
        if strict:
            if _is_not_found(message):
                return None
            raise
        sa_id = _resource_id_from_nebius_error(message, prefix="serviceaccount-")
        if sa_id:
            return sa_id
        if "notfound" in message.lower() or "not found" in message.lower():
            return None
        if _is_permission_denied(message):
            return None
        raise
    metadata = data.get("metadata") if isinstance(data, dict) else None
    sa_id = metadata.get("id", "") if isinstance(metadata, dict) else ""
    resolved = str(sa_id).strip()
    if strict and not resolved:
        raise NebiusError(
            "Nebius returned no service-account ID while verifying the named "
            "account; presence or absence could not be established"
        )
    return resolved or None


def _iam_profile_args(profile: str | None = None) -> tuple[list[str], str]:
    """Return global Nebius CLI profile args and the resolved profile label."""

    from npa.clients.nebius_auth import nebius_profile

    resolved = str(profile if profile is not None else nebius_profile()).strip()
    return (["--profile", resolved] if resolved else []), resolved


def get_service_account_identity(
    service_account_id: str,
    *,
    project_id: str,
    tenant_id: str = "",
    expected_name: str = "",
    profile: str | None = None,
) -> ServiceAccountIdentity | None:
    """Strictly verify one immutable service-account identity and its scope.

    ``None`` means the provider authoritatively returned NotFound for the exact
    ID.  Authentication errors, incomplete payloads, and scope/name mismatches
    raise instead of being collapsed into absence.
    """

    account_id = str(service_account_id or "").strip()
    expected_project = str(project_id or "").strip()
    expected_tenant = str(tenant_id or "").strip()
    wanted_name = str(expected_name or "").strip()
    if not account_id or not expected_project:
        raise NebiusError(
            "Exact service-account ID and provider project ID are required for IAM verification"
        )
    profile_args, resolved_profile = _iam_profile_args(profile)
    try:
        data = _run_json(
            [*profile_args, "iam", "service-account", "get", "--id", account_id]
        )
    except NebiusError as exc:
        if _is_not_found(str(exc)):
            return None
        raise
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict):
        raise NebiusError(
            "Nebius returned no service-account metadata; presence or scope could not be verified"
        )
    returned_id = str(metadata.get("id", "") or "").strip()
    returned_name = str(metadata.get("name", "") or "").strip()
    returned_project = str(
        metadata.get("parent_id", "") or metadata.get("parentId", "") or ""
    ).strip()
    if returned_id != account_id:
        raise NebiusError(
            "Nebius returned a different service-account ID while verifying the exact identity"
        )
    if not returned_name or not returned_project:
        raise NebiusError(
            "Nebius returned incomplete service-account identity fields; presence is not absence"
        )
    if returned_project != expected_project:
        raise NebiusError(
            f"Service account {account_id} belongs to project {returned_project}, not {expected_project}"
        )
    if wanted_name and returned_name != wanted_name:
        raise NebiusError(
            f"Service account {account_id} has name {returned_name!r}, not {wanted_name!r}"
        )

    try:
        project_data = _run_json(
            [*profile_args, "iam", "project", "get", "--id", expected_project]
        )
    except NebiusError as exc:
        raise NebiusError(
            f"Could not verify provider project scope for {account_id}: {exc}"
        ) from exc
    project_metadata = (
        project_data.get("metadata") if isinstance(project_data, dict) else None
    )
    if not isinstance(project_metadata, dict):
        raise NebiusError(
            f"Nebius returned no project metadata while verifying {account_id}"
        )
    returned_project_id = str(project_metadata.get("id", "") or "").strip()
    returned_tenant = str(
        project_metadata.get("parent_id", "")
        or project_metadata.get("parentId", "")
        or ""
    ).strip()
    if returned_project_id != expected_project:
        raise NebiusError(
            "Nebius returned a different project while verifying service-account scope"
        )
    if expected_tenant and not returned_tenant:
        raise NebiusError(
            f"Nebius returned no tenant for project {expected_project}; scope is ambiguous"
        )
    if expected_tenant and returned_tenant != expected_tenant:
        raise NebiusError(
            f"Project {expected_project} belongs to tenant {returned_tenant}, not {expected_tenant}"
        )
    return ServiceAccountIdentity(
        account_id=returned_id,
        name=returned_name,
        project_id=returned_project,
        tenant_id=returned_tenant,
        profile=resolved_profile,
    )


def service_account_exists(service_account_id: str) -> bool:
    """Verify whether an exact service-account ID exists.

    Provider/auth errors are never collapsed into absence.
    """

    account_id = str(service_account_id or "").strip()
    if not account_id:
        return False
    try:
        data = _run_json(["iam", "service-account", "get", "--id", account_id])
    except NebiusError as exc:
        if _is_not_found(str(exc)):
            return False
        raise
    metadata = data.get("metadata") if isinstance(data, dict) else None
    returned_id = str(
        metadata.get("id", "") if isinstance(metadata, dict) else ""
    ).strip()
    if not returned_id:
        raise NebiusError(
            "Nebius returned no service-account ID while verifying the exact "
            "NPA-owned account"
        )
    if returned_id != account_id:
        raise NebiusError(
            "Nebius returned a different service-account ID while verifying the "
            "NPA-owned account"
        )
    return True


def list_access_keys_for_service_account(
    project_id: str,
    sa_id: str,
    *,
    strict: bool = False,
    profile: str | None = None,
) -> list[dict[str, str]]:
    """Return safe metadata for every access key proven to be owned by *sa_id*.

    Teardown needs the full list (not just the active one `ensure_access_key`
    reuses): an agent VM's long-lived key outlives its VM otherwise.
    """
    if not project_id or not sa_id:
        return []
    try:
        items = _list_access_key_metadata(project_id, profile=profile)
    except NebiusError:
        if strict:
            raise
        return []
    keys: list[dict[str, str]] = []
    for item in items:
        account = (item.get("spec", {}) or {}).get("account", {}) or {}
        item_sa_id = (account.get("service_account", {}) or {}).get(
            "id", ""
        ) or account.get("service_account_id", "")
        if item_sa_id != sa_id:
            continue
        metadata = item.get("metadata", {}) or {}
        keys.append(
            {
                "id": str(metadata.get("id", "") or ""),
                "name": str(metadata.get("name", "") or ""),
                "state": str((item.get("status", {}) or {}).get("state", "") or ""),
                "service_account_id": str(item_sa_id),
            }
        )
    return [key for key in keys if key["id"]]


def delete_access_key(access_key_id: str, *, profile: str | None = None) -> None:
    """Delete an IAM access key by id."""
    if not access_key_id:
        return
    profile_args, _resolved_profile = _iam_profile_args(profile)
    _run([*profile_args, "iam", "v2", "access-key", "delete", "--id", access_key_id])


def delete_service_account(sa_id: str, *, profile: str | None = None) -> None:
    """Delete a service account by id."""
    if not sa_id:
        return
    profile_args, _resolved_profile = _iam_profile_args(profile)
    _run([*profile_args, "iam", "service-account", "delete", "--id", sa_id])


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
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("bootstrap_agent_environment")

    on_status = kwargs.pop("on_status", None)
    external_created = kwargs.pop("on_resource_created", None)
    reuse_storage_credentials = kwargs.pop("reuse_storage_credentials", None)
    bucket_name = kwargs.get("bucket_name")
    sa_id = get_service_account_id_by_name(project_id, AGENT_SERVICE_ACCOUNT_NAME)
    if sa_id and on_status:
        on_status(f"Reusing existing service account {AGENT_SERVICE_ACCOUNT_NAME!r}.")
    created_this_attempt: list[tuple[str, dict[str, str]]] = []

    def _record_agent_resource(kind: str, metadata: dict[str, str]) -> None:
        from npa.cli.agent_iam import record_agent_iam_resource

        created_this_attempt.append((kind, dict(metadata)))
        record_agent_iam_resource(project_id, kind, metadata)
        if external_created:
            external_created(kind, metadata)

    def _rollback_agent_resources() -> None:
        """Roll back this invocation's exact resources and preserve failures."""

        rollback_failed = False
        journal_resources_remain = False
        from npa.cli.agent_iam import mark_agent_iam_status, remove_agent_iam_resource

        for kind, metadata in reversed(created_this_attempt):
            try:
                if kind == "access_key":
                    delete_access_key(metadata.get("id", ""))
                elif kind == "service_account":
                    delete_service_account(metadata.get("id", ""))
            except NebiusError as rollback_exc:
                if not _is_not_found(str(rollback_exc)):
                    rollback_failed = True
                    continue
            try:
                journal_resources_remain = remove_agent_iam_resource(
                    project_id, kind, metadata.get("id", "")
                )
            except Exception:  # noqa: BLE001 - provider rollback succeeded; retain journal
                rollback_failed = True
        if not created_this_attempt:
            return
        if rollback_failed or journal_resources_remain:
            mark_agent_iam_status(project_id, "partial")

    try:
        if reuse_storage_credentials is not None:
            # ``npa configure`` already proved the standard data-plane profile in
            # the selected bucket. Agent provisioning only
            # needs the VM-attached service-account identity now; do not revisit
            # access-key inventory or create a second S3 key.
            if on_status:
                on_status(
                    "Reusing health-verified configured object-storage credentials."
                )
                on_status("Setting up the VM-attached npa-agent service account...")

            def _record_created_agent_account(account_id: str) -> None:
                _record_agent_resource(
                    "service_account",
                    {"id": account_id, "name": AGENT_SERVICE_ACCOUNT_NAME},
                )

            sa_id = ensure_service_account(
                project_id,
                name=AGENT_SERVICE_ACCOUNT_NAME,
                description="Long-lived service account for NPA agent VMs",
                on_created=_record_created_agent_account,
                allow_saved_fallback=False,
            )
            try:
                ensure_editors_membership(tenant_id, sa_id)
            except NebiusError as exc:
                raise NebiusError(
                    "Required agent IAM grant failed before deploy: service account "
                    f"{sa_id} must be a member of tenant {tenant_id}'s 'editors' "
                    "group, or the provider must prove an equivalent role."
                ) from exc
            result = dict(reuse_storage_credentials)
            result.update(
                {
                    "iam_token": get_iam_token(),
                    "service_account_id": sa_id,
                    "nebius_project_id": project_id,
                    "nebius_region": region,
                }
            )
        else:
            result = bootstrap_environment(
                project_id,
                tenant_id,
                region,
                service_account_name=AGENT_SERVICE_ACCOUNT_NAME,
                access_key_name=AGENT_ACCESS_KEY_NAME,
                service_account_description="Long-lived service account for NPA agent VMs",
                access_key_description="Long-lived access key for NPA agent S3 and API access",
                on_status=on_status,
                on_resource_created=_record_agent_resource,
                **kwargs,
            )
        from npa.cli.agent_iam import mark_agent_iam_status

        mark_agent_iam_status(project_id, "complete")
        return result
    except NebiusError as exc:
        if not _is_permission_denied(str(exc)):
            _rollback_agent_resources()
            raise
        fallback = (
            dict(reuse_storage_credentials)
            if reuse_storage_credentials is not None
            else _saved_storage_credentials(
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                bucket_name=bucket_name,
                service_account_id=sa_id or resolve_service_account_id(project_id),
            )
        )
        if fallback is None:
            _rollback_agent_resources()
            raise
        # Do not attach the configure/storage account as if it were the named
        # npa-agent account. Preserve only a separately verified identity.
        fallback["service_account_id"] = sa_id or ""
        from npa.cli.agent_iam import mark_agent_iam_status

        mark_agent_iam_status(project_id, "complete")
        if on_status:
            on_status(
                "Reusing saved object-storage credentials (npa-agent provisioning skipped)."
            )
        return fallback
