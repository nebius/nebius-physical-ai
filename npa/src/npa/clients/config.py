"""Configuration resolution: CLI flags → environment variables → NPA YAML files.

Config layout (SSH-config style — one stanza per Nebius project):

    projects:
      me-west1:                           # user-chosen alias
        project_id: project-...
        tenant_id: tenant-...
        region: me-west1
        workbenches:
          b200:
            gpu_platform: gpu-b200-sxm
            gpu_preset: 8gpu-160vcpu-1792gb
            endpoint: http://...
            ssh: {host: ..., user: ubuntu, key_path: ~/.ssh/id_ed25519}
            storage: {checkpoint_bucket: s3://..., endpoint_url: https://...}

    default_project: me-west1
    default_workbench: b200

User secrets that are not tied to a single workbench live in
``~/.npa/credentials.yaml`` and are resolved by ``npa.clients.credentials``.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from npa.clients.credentials import CredentialsConfig, load_credentials
from npa.config_schema import STRICT_CONFIG_SECTION_KEYS, unknown_config_keys
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

NPA_CONFIG_DIR = Path(
    os.environ.get("NPA_CONFIG_DIR", "").strip() or (Path.home() / ".npa")
)
CONFIG_PATH = NPA_CONFIG_DIR / "config.yaml"

APP_STATUS_PROVISIONED = "provisioned"
APP_STATUS_INSTALLING = "installing"
APP_STATUS_HEALTHY = "healthy"
APP_STATUS_INSTALL_FAILED = "install_failed"

# Non-secret, project-scoped evidence that cloud storage IAM still needs an
# authoritative verification or guarded deletion.  Keeping this beside the
# project stanza prevents a local cleanup pass from erasing the only remaining
# route back to a legacy service account.
STORAGE_IAM_RESIDUE_KEY = "storage_iam_verification_required"
STORAGE_IAM_RESIDUE_SCHEMA = "npa.storage-iam-residue.v2"


def _normalize_storage_iam_residue(marker: dict[str, Any]) -> dict[str, Any]:
    """Migrate a secret-free IAM cleanup marker to the current state machine."""

    normalized = dict(marker)
    status = str(normalized.get("status", "") or "").strip()
    ownership = str(normalized.get("ownership", "") or "").strip()
    ownership_state = str(normalized.get("ownership_state", "") or "").strip()
    if ownership_state not in {
        "owned",
        "pending-verification",
        "verified-deleted",
        "pre-existing",
        "unknown",
    }:
        if status in {"verified_absent", "verified_deleted"}:
            ownership_state = "verified-deleted"
        elif status == "pre_existing":
            ownership_state = "pre-existing"
        elif ownership in {"npa", "npa-recovery-attested"} or status in {
            "present_owned",
            "reconciled_pending_delete",
        }:
            ownership_state = "owned"
        elif status == "present_unverified_ownership":
            ownership_state = "pending-verification"
        else:
            ownership_state = "unknown"
    for key in ("access_key_ids", "candidate_service_account_ids"):
        values = normalized.get(key)
        if isinstance(values, (list, tuple, set)):
            normalized[key] = sorted(
                {str(item).strip() for item in values if str(item).strip()}
            )
        else:
            normalized.pop(key, None)
    normalized["ownership_state"] = ownership_state
    normalized["schema_version"] = STORAGE_IAM_RESIDUE_SCHEMA
    return normalized


ENV_MAP = {
    "endpoint": "NPA_WORKBENCH_ENDPOINT",
    "endpoint_strategy": "NPA_ENDPOINT_STRATEGY",
    "service_port": "NPA_SERVICE_PORT",
    "container_registry": "NPA_REGISTRY",
    "ssh_host": "NPA_SSH_HOST",
    "ssh_user": "NPA_SSH_USER",
    "ssh_key": "NPA_SSH_KEY",
    "checkpoint_bucket": "NPA_CHECKPOINT_BUCKET",
    "storage_endpoint_url": "AWS_ENDPOINT_URL",
    "hf_token": "HF_TOKEN",
    "ngc_api_key": "NGC_API_KEY",
    "ngc_org": "NGC_ORG",
    "ngc_team": "NGC_TEAM",
    "aws_access_key_id": "AWS_ACCESS_KEY_ID",
    "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
}


@dataclass
class SSHConfig:
    host: str
    user: str
    key_path: str
    tokens: dict[str, str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    checkpoint_bucket: str
    endpoint_url: str
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""


@dataclass
class EnvironmentConfig:
    project_id: str
    tenant_id: str
    region: str


@dataclass
class TerraformStateConfig:
    bucket: str = ""
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    session_token: str = ""
    region: str = ""
    addressing_style: str = "path"


@dataclass
class ServerlessConfig:
    resource_type: str = ""
    endpoint_id: str = ""
    endpoint_name: str = ""
    project_id: str = ""
    url: str = ""
    image: str = ""
    platform: str = ""
    preset: str = ""
    container_port: int = 0
    auth: str = "none"


@dataclass
class ServerlessJobConfig:
    resource_type: str = "job"
    job_id: str = ""
    job_name: str = ""
    project_id: str = ""
    image: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    # DEPRECATED (W7-subnet-refactor): Tool CLIs no longer read this field.
    # Subnet resolution is centralized in npa.serverless_common.resolve_subnet.
    # Use --subnet-id at command time for overrides. This field is kept for a
    # migration grace window and should be removed in a future cleanup release.
    subnet_id: str = ""
    output_path: str = ""
    last_status: str = ""
    last_submitted_at: str = ""


@dataclass
class WorkbenchConfig:
    endpoint: str
    ssh: SSHConfig
    storage: StorageConfig
    endpoint_strategy: str = "public"
    service_port: int = 0
    endpoint_strategy_configured: bool = False
    service_port_configured: bool = False
    project: str = ""
    name: str = ""
    hf_token: str = ""
    tf_instance_name: str = ""
    app_status: str = ""
    runtime: str = "vm"
    container_registry: str = DEFAULT_CONTAINER_REGISTRY
    instance_id: str = ""
    project_id: str = ""
    security_group_id: str = ""
    gpu_platform: str = ""
    gpu_count: int = 0
    detected_gpu_count: int = 0
    cuda_visible_devices: str = ""
    workbench_type: str = ""
    serverless: ServerlessConfig = field(default_factory=ServerlessConfig)
    serverless_job: ServerlessJobConfig = field(default_factory=ServerlessJobConfig)


class ConfigError(Exception):
    pass


# Accepted ``workbench_type`` values per tool. A workbench alias written by a
# newer client records the tool it belongs to; this lets a tool refuse to act
# on another tool's alias (e.g. ``npa workbench cosmos status`` on a LeRobot
# alias) instead of SSHing in and mis-reporting a foreign VM.
WORKBENCH_TYPE_ALIASES: dict[str, set[str]] = {
    "cosmos": {"cosmos"},
    "fiftyone": {"fiftyone"},
    "genesis": {"genesis"},
    "groot": {"groot", "groot-container"},
    "isaac-lab": {"isaac-lab"},
    "lancedb": {"lancedb"},
    "lerobot": {"lerobot", "lerobot-container"},
    "sonic": {"sonic", "sonic-container"},
}


def _guard_workbench_type(
    wb: dict[str, Any],
    expected: str | None,
    *,
    name: str,
) -> None:
    """Raise if a resolved alias belongs to a different tool.

    Only enforced when the entry records a ``workbench_type`` (aliases written
    by older clients omit it, so those stay permissive to avoid false
    negatives). Unknown ``expected`` tools are treated as no-ops.
    """
    if not expected:
        return
    accepted = WORKBENCH_TYPE_ALIASES.get(expected)
    if not accepted:
        return
    actual = str(wb.get("workbench_type", "") or "").strip().lower()
    if not actual or actual in accepted:
        return
    raise ConfigError(
        f"Workbench '{name}' is a '{actual}' workbench, not {expected}. "
        f"Use the '{actual}' commands for it, or pass -n/--name to select a "
        f"{expected} workbench."
    )


# ── YAML helpers ─────────────────────────────────────────────────────────


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_yaml() -> dict[str, Any]:
    return _load_yaml_file(CONFIG_PATH)


def _deep_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)  # type: ignore[assignment]
    return d


def _require(value: str | None, name: str, env_var: str) -> str:
    if value:
        return value
    raise ConfigError(
        f"{name} is not configured. "
        f"Set it via CLI flag, {env_var} env var, or in {CONFIG_PATH}"
    )


def _normalize_endpoint_strategy(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    return "ssh_fallback" if normalized in {"ssh", "ssh_fallback"} else "public"


def _endpoint_port(endpoint: str) -> int:
    from urllib.parse import urlparse

    try:
        return int(urlparse(endpoint or "").port or 0)
    except ValueError:
        return 0


def _has_path(d: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return False
        d = d[key]
    return True


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Project / workbench resolution ───────────────────────────────────────


def _resolve_project_section(
    yml: dict[str, Any],
    project: str | None,
) -> dict[str, Any]:
    """Return the YAML dict for the requested project.

    Falls back through: ``projects.<name>``, ``projects.<default_project>``,
    legacy ``workbenches`` (treated as a single unnamed project), legacy
    ``workbench``.
    """
    projects = yml.get("projects")
    if project and not (isinstance(projects, dict) and projects):
        # An explicit alias against an empty/absent `projects` map used to fall
        # through to the legacy branches, which silently ignored it: the command
        # then failed downstream complaining it could not tell which Nebius
        # project to use, never mentioning the alias the operator had passed.
        # This is what an alias removed by teardown looks like.
        raise ConfigError(
            f"Project '{project}' is not configured in ~/.npa/config.yaml "
            "(no projects are). Pass an explicit --project-id, or run "
            "`npa configure` to recreate the project entry."
        )
    if isinstance(projects, dict) and projects:
        if project:
            proj = projects.get(project, {})
            if not proj:
                raise ConfigError(
                    f"Project '{project}' not found. "
                    f"Available: {', '.join(projects.keys())}"
                )
            return proj
        default_name = yml.get("default_project", "default")
        if default_name in projects:
            return projects[default_name]
        if len(projects) == 1:
            return next(iter(projects.values()))
        # No usable default and several candidates: guessing here sends
        # commands at a stale or arbitrary endpoint, so ask for the alias.
        raise ConfigError(
            "No project selected and no default_project configured. "
            f"Pass -p/--project (available: {', '.join(projects.keys())}) "
            "or set default_project in ~/.npa/config.yaml."
        )

    # ── Legacy compat: flat workbenches → synthetic project ──────────
    workbenches = yml.get("workbenches")
    if isinstance(workbenches, dict) and workbenches:
        # Hoist environment from the first workbench that has one.
        env: dict[str, Any] = {}
        for wb in workbenches.values():
            if isinstance(wb, dict) and "environment" in wb:
                env = wb["environment"]
                break
        return {**env, "workbenches": workbenches}

    wb = yml.get("workbench")
    if isinstance(wb, dict) and wb:
        return {"workbenches": {"default": wb}}

    return {}


def _resolve_workbench_in_project(
    proj: dict[str, Any],
    name: str | None,
    yml: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the workbench config dict within a project section."""
    workbenches = proj.get("workbenches", {})
    if not isinstance(workbenches, dict):
        workbenches = {}

    if name:
        wb = workbenches.get(name, {})
        if not wb:
            available = ", ".join(workbenches.keys()) if workbenches else "(none)"
            raise ConfigError(
                f"Workbench '{name}' not found. Available: {available}"
            )
        return wb

    # Fall back to default_workbench, then a sole unambiguous entry.
    default_name = (yml or {}).get("default_workbench", "default")
    if default_name in workbenches:
        return workbenches[default_name]
    if len(workbenches) == 1:
        return next(iter(workbenches.values()))
    if workbenches:
        raise ConfigError(
            "No workbench selected and no default_workbench configured. "
            f"Pass -n/--name (available: {', '.join(workbenches.keys())}) "
            "or set default_workbench in ~/.npa/config.yaml."
        )
    return {}


def _resolved_project_name(yml: dict[str, Any], project: str | None) -> str:
    projects = yml.get("projects")
    if isinstance(projects, dict) and projects:
        if project:
            return project
        default_name = str(yml.get("default_project", "default") or "default")
        if default_name in projects:
            return default_name
        return str(next(iter(projects.keys())))
    return project or str(yml.get("default_project", "default") or "default")


def _resolved_workbench_name(
    proj: dict[str, Any],
    name: str | None,
    yml: dict[str, Any],
) -> str:
    workbenches = proj.get("workbenches", {})
    if not isinstance(workbenches, dict):
        workbenches = {}
    if name:
        return name
    default_name = str(yml.get("default_workbench", "default") or "default")
    if default_name in workbenches:
        return default_name
    if workbenches:
        return str(next(iter(workbenches.keys())))
    return default_name


# ── Public query helpers ─────────────────────────────────────────────────


def list_projects() -> dict[str, dict[str, Any]]:
    """Return all projects as ``{alias: config_dict}``."""
    yml = _load_yaml()
    projects = yml.get("projects")
    if isinstance(projects, dict) and projects:
        return projects
    # Legacy: synthesize from flat workbenches
    proj = _resolve_project_section(yml, None)
    if proj:
        return {"default": proj}
    return {}


def default_project_name() -> str:
    yml = _load_yaml()
    return yml.get("default_project", "default")


def default_workbench_name() -> str:
    yml = _load_yaml()
    return yml.get("default_workbench", "default")


def resolve_environment(
    project: str | None = None,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
    region: str | None = None,
) -> EnvironmentConfig | None:
    """Read Nebius environment fields from a project's saved config.

    CLI args override stored values.  Returns ``None`` if nothing is
    available.
    """
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}

    pid = project_id or proj.get("project_id", "")
    tid = tenant_id or proj.get("tenant_id", "")
    reg = region or proj.get("region", "")

    if not pid and not tid and not reg:
        return None
    return EnvironmentConfig(project_id=pid, tenant_id=tid, region=reg)


def resolve_credentials() -> CredentialsConfig:
    """Resolve user-level credentials from env vars and credentials.yaml."""
    return load_credentials()


def resolve_container_registry(project: str | None = None) -> str:
    """Resolve an image registry override, then fall back to public GHCR."""
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}

    # Explicit environment configuration wins over legacy saved values, matching
    # the repository-wide explicit > env > config precedence contract. Honor both
    # supported registry env vars so NPA_REGISTRY_ID behaves consistently across
    # tool-deploy paths (lerobot/fiftyone/sonic/detection-training/sim2real).
    from npa.deploy.images import registry_from_env

    value = registry_from_env()
    if not value and isinstance(proj, dict):
        value = str(proj.get("container_registry", "") or "")
    if not value:
        value = str(yml.get("container_registry", "") or "")
    return value.rstrip("/") if value else DEFAULT_CONTAINER_REGISTRY


def resolve_workflow_src_s3_uri(project: str | None = None) -> str:
    """Return the staged ``npa`` source prefix used when no workbench image is pinned.

    Staging the source is a one-off setup step, but the URI was previously only
    readable from the environment, so the next shell failed preflight even though
    the objects were already in the bucket. Persisting it under the project makes
    it survive the shell that staged it.
    """

    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}
    value = ""
    if isinstance(proj, dict):
        value = str(proj.get("src_s3_uri", "") or "")
    if not value:
        value = str(yml.get("src_s3_uri", "") or "")
    return value.strip()


def persist_workflow_src_s3_uri(uri: str, project: str | None = None) -> Path:
    """Persist a verified, non-secret staged-source URI for future shells.

    The URI contains no credentials.  An explicit project is honored; otherwise
    the configured default (or sole project) is used.  Legacy single-project
    configs retain a top-level value for compatibility.
    """

    value = str(uri or "").strip()
    if not value.startswith("s3://"):
        raise ConfigError(f"workflow source URI must use s3://, got {value or '<empty>'}")
    yml = _load_yaml()
    projects = yml.get("projects")
    if isinstance(projects, dict) and projects:
        alias = str(project or yml.get("default_project", "") or "").strip()
        if not alias and len(projects) == 1:
            alias = str(next(iter(projects)))
        if not alias or alias not in projects:
            available = ", ".join(str(item) for item in projects)
            raise ConfigError(
                "Cannot persist workflow source without a configured project alias. "
                f"Pass --project (available: {available})."
            )
        return write_config({"projects": {alias: {"src_s3_uri": value}}})
    return write_config({"src_s3_uri": value})


# ── Read / write ─────────────────────────────────────────────────────────


CONFIG_PERMISSIONS_WARNING = (
    "config.yaml is readable by other users. It holds Terraform backend S3 keys "
    "under projects.<alias>.terraform_state. Run chmod 600 ~/.npa/config.yaml."
)


def config_permissions_warning(path: Path | None = None) -> str:
    """Return a warning when ``~/.npa/config.yaml`` is group/world-readable.

    ``config.yaml`` is usually thought of as non-secret machine config, but the
    deploy paths persist Terraform remote-state S3 access keys into it
    (``projects.<alias>.terraform_state``). ``credentials.yaml`` already warns on
    loose permissions; this gives config.yaml the same treatment for a file that
    predates the ``terraform_state`` block or was copied between machines.
    """
    target = path or CONFIG_PATH
    try:
        if not target.exists():
            return ""
        mode = target.stat().st_mode
    except OSError:
        return ""
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return CONFIG_PERMISSIONS_WARNING
    return ""


def write_config(data: dict[str, Any]) -> Path:
    """Deep-merge *data* into ``~/.npa/config.yaml`` and write."""
    return update_config_document(lambda existing: _deep_merge(existing, data))


def _validate_strict_config_sections(data: dict[str, Any]) -> None:
    """Keep NPA-owned writes compatible with every strict section reader."""

    for section, allowed in STRICT_CONFIG_SECTION_KEYS.items():
        value = data.get(section)
        if value in (None, ""):
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"NPA config {section} section must be a mapping")
        unknown = unknown_config_keys(section, value)
        if unknown:
            keys = ", ".join(unknown)
            valid = ", ".join(sorted(allowed))
            raise ConfigError(
                f"Unrecognized {section} config key(s): {keys}. Valid keys: {valid}"
            )


def update_config_document(
    updater: Callable[[dict[str, Any]], dict[str, Any]], *, path: Path | None = None
) -> Path:
    """Atomically mutate config after enforcing all strict section schemas."""

    from npa.clients.credentials import update_private_yaml

    target = path or CONFIG_PATH

    def validated(existing: dict[str, Any]) -> dict[str, Any]:
        updated = updater(existing)
        if not isinstance(updated, dict):
            raise ConfigError("NPA config must be a mapping")
        _validate_strict_config_sections(updated)
        return updated

    update_private_yaml(target, validated)
    return target


def _write_config_replace(data: dict[str, Any]) -> Path:
    """Write *data* to ``config.yaml`` verbatim (replacing), 0600.

    ``write_config`` deep-merges and so cannot *drop* a key; teardown paths that
    remove a stanza (a deleted bucket's ``terraform_state``, an uninstalled
    SkyPilot ``sky_bin``, a forgotten project) rewrite the whole file instead.
    """
    return update_config_document(lambda _existing: data)


def _bucket_key(value: str) -> str:
    """Normalize a bucket URI/name to its bare bucket name for comparison."""
    return str(value or "").strip().removeprefix("s3://").strip("/").split("/", 1)[0]


def clear_terraform_state_for_bucket(bucket_name: str) -> list[str]:
    """Drop ``projects.<alias>.terraform_state`` entries backed by *bucket_name*.

    The Terraform remote-state block persists S3 access/secret keys for the
    backend bucket. When that bucket is deleted, those secrets are dead weight
    (and a leak) on disk; remove them for every project whose state bucket
    matches. Returns the aliases whose state was cleared.
    """
    target = _bucket_key(bucket_name)
    if not target:
        return []
    yml = _load_yaml()
    projects = yml.get("projects")
    if not isinstance(projects, dict):
        return []
    cleared: list[str] = []
    for alias, proj in projects.items():
        if not isinstance(proj, dict):
            continue
        state = proj.get("terraform_state")
        if not isinstance(state, dict):
            continue
        if _bucket_key(str(state.get("bucket", "") or "")) == target:
            del proj["terraform_state"]
            cleared.append(str(alias))
    if cleared:
        yml["projects"] = projects
        _write_config_replace(yml)
    return cleared


def clear_skypilot_bin() -> bool:
    """Remove ``skypilot.sky_bin`` from ``config.yaml``; return whether present.

    Paired with removing ``~/.npa/skypilot-venv`` so an uninstalled SkyPilot
    runtime does not leave a dangling persisted binary path behind.
    """
    yml = _load_yaml()
    sky = yml.get("skypilot")
    if not isinstance(sky, dict) or "sky_bin" not in sky:
        return False
    del sky["sky_bin"]
    if sky:
        yml["skypilot"] = sky
    else:
        yml.pop("skypilot", None)
    _write_config_replace(yml)
    return True


def storage_iam_residue(alias: str) -> dict[str, Any]:
    """Return the non-secret unresolved storage-IAM marker for *alias*."""

    cleaned = str(alias or "").strip()
    if not cleaned:
        return {}
    projects = _load_yaml().get("projects")
    project = projects.get(cleaned) if isinstance(projects, dict) else None
    marker = project.get(STORAGE_IAM_RESIDUE_KEY) if isinstance(project, dict) else None
    return _normalize_storage_iam_residue(marker) if isinstance(marker, dict) else {}


def storage_iam_residues() -> dict[str, dict[str, Any]]:
    """Return every project alias with unresolved storage-IAM evidence."""

    projects = _load_yaml().get("projects")
    if not isinstance(projects, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for alias, project in projects.items():
        marker = project.get(STORAGE_IAM_RESIDUE_KEY) if isinstance(project, dict) else None
        if isinstance(marker, dict) and marker:
            result[str(alias)] = _normalize_storage_iam_residue(marker)
    return result


def project_alias_for_id(project_id: str) -> str:
    """Return the unique configured alias for *project_id*, else ``""``."""

    wanted = str(project_id or "").strip()
    if not wanted:
        return ""
    projects = _load_yaml().get("projects")
    if not isinstance(projects, dict):
        return ""
    aliases = [
        str(alias)
        for alias, project in projects.items()
        if isinstance(project, dict)
        and str(project.get("project_id", "") or "").strip() == wanted
    ]
    return aliases[0] if len(aliases) == 1 else ""


def mark_storage_iam_residue(alias: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Persist non-secret unresolved IAM evidence, preserving prior facts.

    A later provider/auth failure must not overwrite a prior positive presence
    observation.  Callers therefore provide only fresh, allowlisted metadata;
    this helper merges it into the existing journal and never deletes fields.
    """

    cleaned = str(alias or "").strip()
    if not cleaned:
        raise ConfigError("A configured project alias is required to save IAM residue.")
    yml = _load_yaml()
    projects = yml.get("projects")
    if not isinstance(projects, dict) or cleaned not in projects:
        raise ConfigError(
            f"Project '{cleaned}' must remain configured while storage IAM is unresolved."
        )
    project = projects[cleaned]
    if not isinstance(project, dict):
        raise ConfigError(f"Project '{cleaned}' has an invalid configuration stanza.")
    current = project.get(STORAGE_IAM_RESIDUE_KEY)
    merged = (
        _normalize_storage_iam_residue(current) if isinstance(current, dict) else {}
    )
    status_rank = {
        "verification_failed": 0,
        "present_unverified_ownership": 1,
        "present_owned": 2,
        "reconciled_pending_delete": 3,
    }
    current_status = str(merged.get("status", "") or "")
    incoming_status = str(evidence.get("status", "") or "")
    incoming = dict(evidence)
    if any(key in evidence for key in ("status", "ownership", "ownership_state")):
        incoming["ownership_state"] = _normalize_storage_iam_residue(evidence)[
            "ownership_state"
        ]
    for key, value in incoming.items():
        if value not in (None, "", [], {}):
            if (
                key == "status"
                and status_rank.get(incoming_status, -1)
                < status_rank.get(current_status, -1)
            ):
                continue
            merged[str(key)] = value
    merged = _normalize_storage_iam_residue(merged)
    project[STORAGE_IAM_RESIDUE_KEY] = merged
    projects[cleaned] = project
    yml["projects"] = projects
    _write_config_replace(yml)
    return merged


def clear_storage_iam_residue(alias: str, *, account_id: str = "") -> bool:
    """Clear a residue marker after provider-verified absence/deletion only."""

    cleaned = str(alias or "").strip()
    if not cleaned:
        return False
    yml = _load_yaml()
    projects = yml.get("projects")
    project_map: dict[str, Any] = projects if isinstance(projects, dict) else {}
    project = project_map.get(cleaned)
    if not isinstance(project, dict):
        return False
    marker = project.get(STORAGE_IAM_RESIDUE_KEY)
    if not isinstance(marker, dict):
        return False
    expected = str(account_id or "").strip()
    recorded = str(marker.get("service_account_id", "") or "").strip()
    if expected and recorded and recorded != expected:
        raise ConfigError(
            "Refusing to clear storage-IAM residue for a different service account."
        )
    project.pop(STORAGE_IAM_RESIDUE_KEY, None)
    project_map[cleaned] = project
    yml["projects"] = project_map
    _write_config_replace(yml)
    return True


def forget_project(alias: str) -> bool:
    """Remove ``projects.<alias>`` (stanza + ``terraform_state``) from config.

    The inverse of the project stanza ``npa configure`` writes. Fixes
    ``default_project`` when the forgotten alias was the default. Returns whether
    a stanza was removed.
    """
    cleaned = str(alias or "").strip()
    if not cleaned:
        return False
    yml = _load_yaml()
    projects = yml.get("projects")
    if not isinstance(projects, dict) or cleaned not in projects:
        return False
    project = projects.get(cleaned)
    marker = project.get(STORAGE_IAM_RESIDUE_KEY) if isinstance(project, dict) else None
    if isinstance(marker, dict) and marker:
        account_id = str(marker.get("service_account_id", "") or "").strip()
        suffix = f" ({account_id})" if account_id else ""
        raise ConfigError(
            f"Project '{cleaned}' has unresolved storage IAM{suffix}. Reconcile it "
            "with `npa storage service-account reconcile --project "
            f"{cleaned} --dry-run`, complete guarded deletion/verification, then "
            "retry forgetting the project."
        )
    project_id = (
        str(project.get("project_id", "") or "")
        if isinstance(project, dict)
        else ""
    )
    skypilot = yml.get("skypilot")
    skypilot_map: dict[str, Any] = skypilot if isinstance(skypilot, dict) else {}
    owner = skypilot_map.get("controller_owner")
    if isinstance(owner, dict):
        owner_project_id = str(owner.get("project_id", "") or "")
        owner_alias = str(owner.get("project_alias", "") or "")
        if (project_id and owner_project_id == project_id) or (
            not owner_project_id and owner_alias == cleaned
        ):
            skypilot_map.pop("controller_owner", None)
    del projects[cleaned]
    yml["projects"] = projects
    if yml.get("default_project") == cleaned:
        remaining = list(projects.keys())
        if remaining:
            yml["default_project"] = remaining[0]
        else:
            # Pointing `default_project` at the literal "default" once the last
            # project is gone leaves a dangling alias that resolves to nothing;
            # an absent key is the honest "no project configured".
            yml.pop("default_project", None)
    _write_config_replace(yml)
    return True


def remove_workbench_config(
    project: str,
    name: str,
) -> None:
    """Remove ``projects.<project>.workbenches.<name>``."""
    def remove(existing: dict[str, Any]) -> dict[str, Any]:
        projects = existing.get("projects", {})
        proj = projects.get(project, {}) if isinstance(projects, dict) else {}
        workbenches = proj.get("workbenches", {}) if isinstance(proj, dict) else {}
        if not isinstance(workbenches, dict) or name not in workbenches:
            return existing
        del workbenches[name]
        if workbenches:
            proj["workbenches"] = workbenches
        else:
            proj.pop("workbenches", None)
        agents = proj.get("agents", {}) if isinstance(proj, dict) else {}
        if not workbenches and not (isinstance(agents, dict) and agents):
            del projects[project]
            if existing.get("default_project") == project:
                remaining = list(projects.keys())
                if remaining:
                    existing["default_project"] = remaining[0]
                else:
                    existing.pop("default_project", None)
        else:
            projects[project] = proj
        existing["projects"] = projects
        return existing

    update_config_document(remove)


def remove_agent_config(project: str, name: str) -> None:
    """Atomically remove one agent while preserving its project ownership stanza."""

    def remove(existing: dict[str, Any]) -> dict[str, Any]:
        projects = existing.get("projects", {})
        proj = projects.get(project, {}) if isinstance(projects, dict) else {}
        agents = proj.get("agents", {}) if isinstance(proj, dict) else {}
        if not isinstance(agents, dict) or name not in agents:
            return existing
        del agents[name]
        if agents:
            proj["agents"] = agents
        else:
            proj.pop("agents", None)
        projects[project] = proj
        existing["projects"] = projects
        return existing

    update_config_document(remove)


def workbench_entry(project: str | None, name: str | None) -> dict[str, Any]:
    """Return the raw configured workbench entry, or an empty dict."""
    if not project or not name:
        return {}
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        return {}
    workbenches = proj.get("workbenches", {}) if isinstance(proj, dict) else {}
    if not isinstance(workbenches, dict):
        return {}
    wb = workbenches.get(name, {})
    return wb if isinstance(wb, dict) else {}


def workbench_is_byovm(project: str | None, name: str | None) -> bool:
    """Return True when the saved alias is a BYOVM registration."""
    wb = workbench_entry(project, name)
    return str(wb.get("runtime", "") or "").lower() == "byovm"


def alias_has_terraform_state(project: str | None, name: str | None) -> bool:
    """Return True if a saved alias should use managed Terraform state.

    Terraform backend credentials are project-level in config.yaml, while the
    state object key is per workbench alias. BYOVM aliases are never treated as
    Terraform-managed.
    """
    if not project or not name or workbench_is_byovm(project, name):
        return False
    if not workbench_entry(project, name):
        return False

    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        return False
    state = proj.get("terraform_state", {}) if isinstance(proj, dict) else {}
    return isinstance(state, dict) and any(bool(value) for value in state.values())


def update_workbench_app_status(project: str, name: str, app_status: str) -> Path:
    """Set ``projects.<project>.workbenches.<name>.app_status``."""
    return write_config({
        "projects": {
            project: {
                "workbenches": {
                    name: {
                        "app_status": app_status,
                    },
                },
            },
        },
    })


def update_workbench_endpoint_strategy(
    project: str,
    name: str,
    endpoint_strategy: str,
    service_port: int,
) -> Path:
    """Persist live-command endpoint routing for a workbench alias."""
    return write_config({
        "projects": {
            project: {
                "workbenches": {
                    name: {
                        "endpoint_strategy": _normalize_endpoint_strategy(endpoint_strategy),
                        "service_port": int(service_port),
                    },
                },
            },
        },
    })


def update_workbench_serverless_endpoint(
    project: str,
    name: str,
    *,
    endpoint_id: str,
    endpoint_name: str,
    project_id: str,
    url: str,
    image: str,
    platform: str,
    preset: str,
    container_port: int,
    auth: str = "none",
) -> Path:
    """Persist Nebius Serverless AI endpoint metadata for a workbench alias."""
    return write_config({
        "projects": {
            project: {
                "workbenches": {
                    name: {
                        "endpoint": url,
                        "endpoint_strategy": "public",
                        "service_port": int(container_port),
                        "runtime": "serverless",
                        "app_status": APP_STATUS_PROVISIONED,
                        "serverless": {
                            "resource_type": "endpoint",
                            "endpoint_id": endpoint_id,
                            "endpoint_name": endpoint_name,
                            "project_id": project_id,
                            "url": url,
                            "image": image,
                            "platform": platform,
                            "preset": preset,
                            "container_port": int(container_port),
                            "auth": auth,
                        },
                    },
                },
            },
        },
    })


def update_workbench_serverless_job(
    project: str,
    name: str,
    *,
    job_id: str,
    job_name: str,
    project_id: str,
    image: str,
    gpu_type: str,
    gpu_count: int,
    subnet_id: str,
    output_path: str,
    last_status: str,
    last_submitted_at: str,
) -> Path:
    """Persist Nebius Serverless AI Job metadata for a workbench alias."""
    return write_config({
        "projects": {
            project: {
                "workbenches": {
                    name: {
                        "runtime": "serverless",
                        "app_status": APP_STATUS_PROVISIONED,
                        "serverless_job": {
                            "resource_type": "job",
                            "job_id": job_id,
                            "job_name": job_name,
                            "project_id": project_id,
                            "image": image,
                            "gpu_type": gpu_type,
                            "gpu_count": int(gpu_count),
                            "subnet_id": subnet_id,
                            "output_path": output_path,
                            "last_status": last_status,
                            "last_submitted_at": last_submitted_at,
                        },
                    },
                },
            },
        },
    })


def _serverless_config(wb: dict[str, Any]) -> ServerlessConfig:
    raw = wb.get("serverless", {})
    if not isinstance(raw, dict):
        raw = {}

    def pick(*keys: str) -> str:
        for key in keys:
            value = raw.get(key)
            if value is not None and value != "":
                return str(value)
        return ""

    port_raw = pick("container_port", "port")
    return ServerlessConfig(
        resource_type=pick("resource_type") or "endpoint",
        endpoint_id=pick("endpoint_id", "id"),
        endpoint_name=pick("endpoint_name", "name"),
        project_id=pick("project_id"),
        url=pick("url", "endpoint_url"),
        image=pick("image"),
        platform=pick("platform"),
        preset=pick("preset"),
        container_port=int(port_raw) if port_raw.isdigit() else 0,
        auth=pick("auth") or "none",
    )


def _serverless_job_config(wb: dict[str, Any]) -> ServerlessJobConfig:
    raw = wb.get("serverless_job", {})
    if not isinstance(raw, dict):
        raw = {}
    legacy = wb.get("serverless", {})
    if not raw and isinstance(legacy, dict) and legacy.get("resource_type") == "job":
        raw = legacy

    def pick(*keys: str) -> str:
        for key in keys:
            value = raw.get(key)
            if value is not None and value != "":
                return str(value)
        return ""

    gpu_count_raw = pick("gpu_count", "gpus")
    return ServerlessJobConfig(
        resource_type=pick("resource_type") or "job",
        job_id=pick("job_id", "id"),
        job_name=pick("job_name", "name"),
        project_id=pick("project_id"),
        image=pick("image"),
        gpu_type=pick("gpu_type", "platform"),
        gpu_count=int(gpu_count_raw) if gpu_count_raw.isdigit() else 0,
        subnet_id=pick("subnet_id", "vpc_subnet_id", "subnet"),
        output_path=pick("output_path", "output_uri"),
        last_status=pick("last_status", "status"),
        last_submitted_at=pick("last_submitted_at", "submitted_at"),
    )


# ── Config resolution ────────────────────────────────────────────────────


def resolve_config(
    *,
    project: str | None = None,
    name: str | None = None,
    endpoint: str | None = None,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    checkpoint_bucket: str | None = None,
    storage_endpoint_url: str | None = None,
    hf_token: str | None = None,
    expected_workbench_type: str | None = None,
) -> WorkbenchConfig:
    """Resolve configuration with precedence: explicit args > env > credentials > yaml."""
    yml = _load_yaml()
    proj = _resolve_project_section(yml, project)
    wb = _resolve_workbench_in_project(proj, name, yml)
    credentials = resolve_credentials()
    resolved_project = _resolved_project_name(yml, project)
    resolved_name = _resolved_workbench_name(proj, name, yml)
    _guard_workbench_type(wb, expected_workbench_type, name=resolved_name)

    def pick(cli_val: str | None, env_key: str, *yaml_path: str) -> str:
        if cli_val:
            return cli_val
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        yaml_val = _deep_get(wb, *yaml_path)
        return str(yaml_val) if yaml_val is not None else ""

    runtime = pick(None, "", "runtime") or "vm"
    serverless = _serverless_config(wb)
    serverless_job = _serverless_job_config(wb)
    ep = (
        pick(endpoint, "NPA_WORKBENCH_ENDPOINT", "endpoint")
        or serverless.url
    )
    s_host = pick(ssh_host, "NPA_SSH_HOST", "ssh", "host")
    s_user = pick(ssh_user, "NPA_SSH_USER", "ssh", "user")
    s_key = pick(ssh_key, "NPA_SSH_KEY", "ssh", "key_path")
    cb = pick(checkpoint_bucket, "NPA_CHECKPOINT_BUCKET", "storage", "checkpoint_bucket")
    se = (
        pick(storage_endpoint_url, "AWS_ENDPOINT_URL", "storage", "endpoint_url")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
        or os.environ.get("NPA_STORAGE_ENDPOINT", "")
    )
    ht = hf_token or credentials.hf_token
    ak = pick(None, "AWS_ACCESS_KEY_ID", "storage", "aws_access_key_id")
    sk = pick(None, "AWS_SECRET_ACCESS_KEY", "storage", "aws_secret_access_key")

    tin = pick(None, "", "tf_instance_name")
    app_status = pick(None, "", "app_status")
    endpoint_strategy = pick(None, "NPA_ENDPOINT_STRATEGY", "endpoint_strategy")
    endpoint_strategy_configured = (
        "NPA_ENDPOINT_STRATEGY" in os.environ
        or _has_path(wb, "endpoint_strategy")
    )
    service_port_raw = (
        pick(None, "NPA_SERVICE_PORT", "service_port")
        or pick(None, "", "app_port")
        or str(_endpoint_port(ep) or "")
    )
    service_port_configured = (
        "NPA_SERVICE_PORT" in os.environ
        or _has_path(wb, "service_port")
    )
    container_registry = resolve_container_registry(project)
    instance_id = pick(None, "", "instance_id")
    alias_project_id = pick(None, "", "project_id")
    security_group_id = pick(None, "", "security_group_id")
    gpu_platform = pick(None, "", "gpu_platform")
    gpu_count_raw = pick(None, "", "gpu_count")
    detected_gpu_count_raw = pick(None, "", "detected_gpu_count")
    cuda_visible_devices = pick(None, "", "cuda_visible_devices")
    workbench_type = pick(None, "", "workbench_type")

    if runtime != "serverless":
        _require(ep, "Workbench endpoint", "NPA_WORKBENCH_ENDPOINT")
    if runtime != "serverless":
        _require(s_host, "SSH host", "NPA_SSH_HOST")
        _require(s_user, "SSH user", "NPA_SSH_USER")
        _require(s_key, "SSH key path", "NPA_SSH_KEY")

    return WorkbenchConfig(
        endpoint=ep,
        ssh=SSHConfig(host=s_host, user=s_user, key_path=s_key, tokens=credentials.tokens),
        storage=StorageConfig(
            checkpoint_bucket=cb,
            endpoint_url=se,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        ),
        endpoint_strategy=_normalize_endpoint_strategy(endpoint_strategy),
        service_port=int(service_port_raw) if str(service_port_raw).isdigit() else 0,
        endpoint_strategy_configured=endpoint_strategy_configured,
        service_port_configured=service_port_configured,
        project=resolved_project,
        name=resolved_name,
        hf_token=ht,
        tf_instance_name=tin,
        app_status=app_status,
        runtime=runtime,
        container_registry=container_registry,
        instance_id=instance_id,
        project_id=alias_project_id,
        security_group_id=security_group_id,
        gpu_platform=gpu_platform,
        gpu_count=int(gpu_count_raw) if str(gpu_count_raw).isdigit() else 0,
        detected_gpu_count=int(detected_gpu_count_raw) if str(detected_gpu_count_raw).isdigit() else 0,
        cuda_visible_devices=cuda_visible_devices,
        workbench_type=workbench_type,
        serverless=serverless,
        serverless_job=serverless_job,
    )


def resolve_ssh_config(
    *,
    project: str | None = None,
    name: str | None = None,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    expected_workbench_type: str | None = None,
) -> WorkbenchConfig:
    """Resolve workbench config requiring only SSH fields (no endpoint).

    Use this for commands that interact with a VM only via SSH — e.g. the
    Genesis workbench which has no HTTP server.
    """
    yml = _load_yaml()
    proj = _resolve_project_section(yml, project)
    wb = _resolve_workbench_in_project(proj, name, yml)
    credentials = resolve_credentials()
    resolved_project = _resolved_project_name(yml, project)
    resolved_name = _resolved_workbench_name(proj, name, yml)
    _guard_workbench_type(wb, expected_workbench_type, name=resolved_name)

    def pick(cli_val: str | None, env_key: str, *yaml_path: str) -> str:
        if cli_val:
            return cli_val
        if env_key:
            env_val = os.environ.get(env_key)
            if env_val:
                return env_val
        yaml_val = _deep_get(wb, *yaml_path)
        return str(yaml_val) if yaml_val is not None else ""

    ep = pick(None, "NPA_WORKBENCH_ENDPOINT", "endpoint")
    s_host = pick(ssh_host, "NPA_SSH_HOST", "ssh", "host")
    s_user = pick(ssh_user, "NPA_SSH_USER", "ssh", "user")
    s_key = pick(ssh_key, "NPA_SSH_KEY", "ssh", "key_path")
    cb = pick(None, "NPA_CHECKPOINT_BUCKET", "storage", "checkpoint_bucket")
    se = (
        pick(None, "AWS_ENDPOINT_URL", "storage", "endpoint_url")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
        or os.environ.get("NPA_STORAGE_ENDPOINT", "")
    )
    ht = credentials.hf_token
    ak = pick(None, "AWS_ACCESS_KEY_ID", "storage", "aws_access_key_id")
    sk = pick(None, "AWS_SECRET_ACCESS_KEY", "storage", "aws_secret_access_key")
    tin = pick(None, "", "tf_instance_name")
    app_status = pick(None, "", "app_status")
    runtime = pick(None, "", "runtime") or "vm"
    endpoint_strategy = pick(None, "NPA_ENDPOINT_STRATEGY", "endpoint_strategy")
    endpoint_strategy_configured = (
        "NPA_ENDPOINT_STRATEGY" in os.environ
        or _has_path(wb, "endpoint_strategy")
    )
    service_port_raw = (
        pick(None, "NPA_SERVICE_PORT", "service_port")
        or pick(None, "", "app_port")
        or str(_endpoint_port(ep) or "")
    )
    service_port_configured = (
        "NPA_SERVICE_PORT" in os.environ
        or _has_path(wb, "service_port")
    )
    container_registry = resolve_container_registry(project)
    instance_id = pick(None, "", "instance_id")
    alias_project_id = pick(None, "", "project_id")
    security_group_id = pick(None, "", "security_group_id")
    gpu_platform = pick(None, "", "gpu_platform")
    gpu_count_raw = pick(None, "", "gpu_count")
    detected_gpu_count_raw = pick(None, "", "detected_gpu_count")
    cuda_visible_devices = pick(None, "", "cuda_visible_devices")
    workbench_type = pick(None, "", "workbench_type")

    _require(s_host, "SSH host", "NPA_SSH_HOST")
    _require(s_user, "SSH user", "NPA_SSH_USER")
    _require(s_key, "SSH key path", "NPA_SSH_KEY")

    return WorkbenchConfig(
        endpoint=ep,
        ssh=SSHConfig(host=s_host, user=s_user, key_path=s_key, tokens=credentials.tokens),
        storage=StorageConfig(
            checkpoint_bucket=cb,
            endpoint_url=se,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        ),
        endpoint_strategy=_normalize_endpoint_strategy(endpoint_strategy),
        service_port=int(service_port_raw) if str(service_port_raw).isdigit() else 0,
        endpoint_strategy_configured=endpoint_strategy_configured,
        service_port_configured=service_port_configured,
        project=resolved_project,
        name=resolved_name,
        hf_token=ht,
        tf_instance_name=tin,
        app_status=app_status,
        runtime=runtime,
        container_registry=container_registry,
        instance_id=instance_id,
        project_id=alias_project_id,
        security_group_id=security_group_id,
        gpu_platform=gpu_platform,
        gpu_count=int(gpu_count_raw) if str(gpu_count_raw).isdigit() else 0,
        detected_gpu_count=int(detected_gpu_count_raw) if str(detected_gpu_count_raw).isdigit() else 0,
        cuda_visible_devices=cuda_visible_devices,
        workbench_type=workbench_type,
    )


def resolve_terraform_state(project: str | None = None) -> TerraformStateConfig:
    """Resolve the saved Terraform remote-state backend credentials for a project.

    Terraform's S3 backend must use the same S3 principal that can update the
    existing state object. Persisting the backend access key used during apply
    prevents a later destroy from accidentally reconfiguring the backend with a
    different active access key.
    """
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}

    state = proj.get("terraform_state", {}) if isinstance(proj, dict) else {}
    if not isinstance(state, dict):
        state = {}
    project_id = str(proj.get("project_id", "") or "") if isinstance(proj, dict) else ""
    if project_id:
        from npa.clients.project_credential_store import project_credential_record

        from npa.lifecycle_intent import OperationIntent, current_intent

        read_only = current_intent() in {
            OperationIntent.DESTROY,
            OperationIntent.OBSERVE,
        }
        record = project_credential_record(
            project_id,
            alias="" if read_only else str(project or ""),
            migrate_legacy=not read_only,
        )
        saved = record.get("terraform_state")
        if isinstance(saved, dict):
            state = {**state, **saved}

    return TerraformStateConfig(
        bucket=str(state.get("bucket", "") or ""),
        endpoint=str(state.get("endpoint", "") or ""),
        access_key=str(state.get("access_key", "") or ""),
        secret_key=str(state.get("secret_key", "") or ""),
        session_token=str(state.get("session_token", "") or ""),
        region=str(state.get("region", "") or ""),
        addressing_style=str(state.get("addressing_style", "path") or "path"),
    )


def resolve_project_storage(
    project: str | None = None,
    *,
    include_shared_credentials: bool = True,
) -> StorageConfig:
    """Resolve project-level object storage settings.

    Accepts the newer project ``object-storage``/``object_storage``/``storage``
    blocks and falls back to ``terraform_state`` for older configs. When
    ``include_shared_credentials`` is true, host-scoped credentials from
    ``~/.npa/credentials.yaml`` are used as a final fallback for operator
    workflows that only need a writable default bucket. Exact-project
    credential-store records are selected atomically: a partial record is
    ignored rather than mixed with routing or key fields from another source.
    """
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}
    if not isinstance(proj, dict):
        proj = {}

    storage = (
        proj.get("object-storage")
        or proj.get("object_storage")
        or proj.get("storage")
        or {}
    )
    if not isinstance(storage, dict):
        storage = {}

    state = proj.get("terraform_state", {})
    if not isinstance(state, dict):
        state = {}

    credentials = load_credentials()
    project_id = str(proj.get("project_id", "") or "")
    project_storage_credentials: dict[str, Any] = {}
    if project_id:
        from npa.clients.project_credential_store import project_credential_record

        from npa.lifecycle_intent import OperationIntent, current_intent

        read_only = current_intent() in {
            OperationIntent.DESTROY,
            OperationIntent.OBSERVE,
        }
        record = project_credential_record(
            project_id,
            alias="" if read_only else str(project or ""),
            migrate_legacy=not read_only,
        )
        saved_storage = record.get("storage")
        if isinstance(saved_storage, dict):
            project_storage_credentials = saved_storage

    def source_value(source: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = source.get(key)
            if value:
                return str(value)
        return ""

    # A committed exact-project storage record is one validated identity
    # generation: bucket, endpoint, access key, and secret key were proved and
    # written together by StorageSetupTransaction.commit(). Select that record
    # only as a complete unit. A hand-edited, legacy, or interrupted partial
    # record must not combine its key pair with routing fields from a different
    # config generation (or vice versa). When the exact-project record is
    # absent or partial, it contributes no fields at all; the pre-existing
    # project/config resolution path below remains the fallback.
    project_storage = StorageConfig(
        checkpoint_bucket=source_value(
            project_storage_credentials,
            "checkpoint_bucket",
            "bucket",
            "s3_bucket",
        ),
        endpoint_url=source_value(
            project_storage_credentials,
            "endpoint_url",
            "endpoint",
            "s3_endpoint",
        ),
        aws_access_key_id=source_value(
            project_storage_credentials,
            "aws_access_key_id",
            "access_key",
            "nebius_api_key",
        ),
        aws_secret_access_key=source_value(
            project_storage_credentials,
            "aws_secret_access_key",
            "secret_key",
            "nebius_secret_key",
        ),
    )
    if project_storage_credentials and all(
        (
            project_storage.checkpoint_bucket,
            project_storage.endpoint_url,
            project_storage.aws_access_key_id,
            project_storage.aws_secret_access_key,
        )
    ):
        return project_storage

    def pick(*keys: str, default: str = "") -> str:
        value = source_value(storage, *keys)
        if value:
            return value
        return default

    env_bucket = os.environ.get("NPA_CHECKPOINT_BUCKET", "") or os.environ.get(
        "NEBIUS_S3_BUCKET", ""
    )
    env_endpoint = (
        os.environ.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
        or os.environ.get("NPA_STORAGE_ENDPOINT", "")
    )
    env_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    env_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    # Shared credentials are host-scoped. The scoped config stanza remains
    # primary; shared values are only a fallback for projects without an exact
    # provider project ID.
    credentials_bucket = ""
    credentials_endpoint = ""
    credentials_access_key = ""
    credentials_secret_key = ""
    if include_shared_credentials and not project_id:
        credentials_bucket = credentials_bucket or credentials.s3_bucket
        credentials_endpoint = credentials_endpoint or credentials.s3_endpoint
        credentials_access_key = credentials_access_key or credentials.s3_access_key_id
        credentials_secret_key = (
            credentials_secret_key or credentials.s3_secret_access_key
        )

    bucket = pick(
        "checkpoint_bucket",
        "bucket",
        "s3_bucket",
        default=(
            str(state.get("bucket", "") or "") or env_bucket or credentials_bucket
        ),
    )
    endpoint = pick(
        "endpoint_url",
        "endpoint",
        "s3_endpoint",
        default=(
            str(state.get("endpoint", "") or "") or env_endpoint or credentials_endpoint
        ),
    )
    access_key = pick(
        "aws_access_key_id",
        "access_key",
        "nebius_api_key",
        default=(
            str(state.get("access_key", "") or "")
            or env_access_key
            or credentials_access_key
        ),
    )
    secret_key = pick(
        "aws_secret_access_key",
        "secret_key",
        "nebius_secret_key",
        default=(
            str(state.get("secret_key", "") or "")
            or env_secret_key
            or credentials_secret_key
        ),
    )
    return StorageConfig(
        checkpoint_bucket=bucket,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
