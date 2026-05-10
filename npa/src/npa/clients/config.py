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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from npa.clients.credentials import CredentialsConfig, load_credentials
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY

CONFIG_PATH = Path.home() / ".npa" / "config.yaml"

APP_STATUS_PROVISIONED = "provisioned"
APP_STATUS_INSTALLING = "installing"
APP_STATUS_HEALTHY = "healthy"
APP_STATUS_INSTALL_FAILED = "install_failed"

ENV_MAP = {
    "endpoint": "NPA_WORKBENCH_ENDPOINT",
    "endpoint_strategy": "NPA_ENDPOINT_STRATEGY",
    "service_port": "NPA_SERVICE_PORT",
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


class ConfigError(Exception):
    pass


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
    normalized = str(value or "").strip().lower()
    return "ssh" if normalized == "ssh" else "public"


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
        return projects.get(default_name, {})

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

    # Fall back to default_workbench, then first entry.
    default_name = (yml or {}).get("default_workbench", "default")
    if default_name in workbenches:
        return workbenches[default_name]
    if workbenches:
        return next(iter(workbenches.values()))
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
    """Return the project-level container registry override, or the default."""
    yml = _load_yaml()
    try:
        proj = _resolve_project_section(yml, project)
    except ConfigError:
        proj = {}

    value = ""
    if isinstance(proj, dict):
        value = str(proj.get("container_registry", "") or "")
    if not value:
        value = str(yml.get("container_registry", "") or "")
    return value.rstrip("/") if value else DEFAULT_CONTAINER_REGISTRY


# ── Read / write ─────────────────────────────────────────────────────────


def write_config(data: dict[str, Any]) -> Path:
    """Deep-merge *data* into ``~/.npa/config.yaml`` and write."""
    existing = _load_yaml()
    merged = _deep_merge(existing, data)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)
    CONFIG_PATH.chmod(0o600)
    return CONFIG_PATH


def remove_workbench_config(
    project: str,
    name: str,
) -> None:
    """Remove ``projects.<project>.workbenches.<name>``."""
    existing = _load_yaml()
    projects = existing.get("projects", {})
    proj = projects.get(project, {})
    workbenches = proj.get("workbenches", {})
    if name in workbenches:
        del workbenches[name]
        proj["workbenches"] = workbenches
        if not workbenches:
            del projects[project]
            if existing.get("default_project") == project:
                remaining = list(projects.keys())
                existing["default_project"] = remaining[0] if remaining else "default"
        existing["projects"] = projects
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
        CONFIG_PATH.chmod(0o600)


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
) -> WorkbenchConfig:
    """Resolve configuration with precedence: explicit args > env > credentials > yaml."""
    yml = _load_yaml()
    proj = _resolve_project_section(yml, project)
    wb = _resolve_workbench_in_project(proj, name, yml)
    credentials = resolve_credentials()
    resolved_project = _resolved_project_name(yml, project)
    resolved_name = _resolved_workbench_name(proj, name, yml)

    def pick(cli_val: str | None, env_key: str, *yaml_path: str) -> str:
        if cli_val:
            return cli_val
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        yaml_val = _deep_get(wb, *yaml_path)
        return str(yaml_val) if yaml_val is not None else ""

    ep = pick(endpoint, "NPA_WORKBENCH_ENDPOINT", "endpoint")
    s_host = pick(ssh_host, "NPA_SSH_HOST", "ssh", "host")
    s_user = pick(ssh_user, "NPA_SSH_USER", "ssh", "user")
    s_key = pick(ssh_key, "NPA_SSH_KEY", "ssh", "key_path")
    cb = pick(checkpoint_bucket, "NPA_CHECKPOINT_BUCKET", "storage", "checkpoint_bucket")
    se = pick(storage_endpoint_url, "AWS_ENDPOINT_URL", "storage", "endpoint_url")
    ht = hf_token or credentials.hf_token
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

    _require(ep, "Workbench endpoint", "NPA_WORKBENCH_ENDPOINT")
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
    )


def resolve_ssh_config(
    *,
    project: str | None = None,
    name: str | None = None,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
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
    se = pick(None, "AWS_ENDPOINT_URL", "storage", "endpoint_url")
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

    return TerraformStateConfig(
        bucket=str(state.get("bucket", "") or ""),
        endpoint=str(state.get("endpoint", "") or ""),
        access_key=str(state.get("access_key", "") or ""),
        secret_key=str(state.get("secret_key", "") or ""),
    )


def resolve_project_storage(project: str | None = None) -> StorageConfig:
    """Resolve project-level object storage settings.

    Accepts the newer project ``object-storage``/``object_storage``/``storage``
    blocks and falls back to ``terraform_state`` for older configs.
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

    def pick(*keys: str, default: str = "") -> str:
        for key in keys:
            value = storage.get(key)
            if value:
                return str(value)
        return default

    bucket = pick(
        "checkpoint_bucket",
        "bucket",
        "s3_bucket",
        default=str(state.get("bucket", "") or ""),
    )
    endpoint = pick(
        "endpoint_url",
        "endpoint",
        "s3_endpoint",
        default=str(state.get("endpoint", "") or ""),
    )
    access_key = pick(
        "aws_access_key_id",
        "access_key",
        "nebius_api_key",
        default=str(state.get("access_key", "") or ""),
    )
    secret_key = pick(
        "aws_secret_access_key",
        "secret_key",
        "nebius_secret_key",
        default=str(state.get("secret_key", "") or ""),
    )
    return StorageConfig(
        checkpoint_bucket=bucket,
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
