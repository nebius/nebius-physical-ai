"""Shared helpers for workbench ingress subcommands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from npa.clients.config import (
    ConfigError,
    default_project_name,
    default_workbench_name,
    list_projects,
    write_config,
)
from npa.clients.network import (
    EnsureIngressResult,
    InstanceNetworkContext,
    ensure_ingress,
    resolve_instance_network_context,
)


@dataclass(frozen=True)
class AliasRecord:
    project_alias: str
    name: str
    data: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.data.get("instance_id", "") or "")


@dataclass(frozen=True)
class RegisterByovmResult:
    project_alias: str
    alias: str
    instance_id: str
    project_id: str
    security_group_id: str
    port: int
    ingress: EnsureIngressResult | None


def resolve_alias_record(project_alias: str | None, name: str | None) -> AliasRecord:
    """Resolve a workbench alias to its raw config dictionary."""
    projects = list_projects()
    if not projects:
        raise ConfigError("No projects configured")

    resolved_project = project_alias or default_project_name()
    if resolved_project not in projects:
        if project_alias:
            available = ", ".join(projects.keys())
            raise ConfigError(f"Project '{project_alias}' not found. Available: {available}")
        resolved_project = next(iter(projects.keys()))

    project_config = projects[resolved_project]
    workbenches = project_config.get("workbenches", {})
    if not isinstance(workbenches, dict) or not workbenches:
        raise ConfigError(f"No workbenches configured in project '{resolved_project}'")

    resolved_name = name or default_workbench_name()
    if resolved_name not in workbenches:
        if name:
            available = ", ".join(workbenches.keys()) if workbenches else "(none)"
            raise ConfigError(f"Workbench '{name}' not found. Available: {available}")
        resolved_name = next(iter(workbenches.keys()))

    alias_data = workbenches[resolved_name]
    if not isinstance(alias_data, dict):
        raise ConfigError(f"Workbench '{resolved_name}' is not a valid alias config")
    return AliasRecord(project_alias=resolved_project, name=resolved_name, data=alias_data)


def register_byovm_alias(
    *,
    tool: str,
    alias: str,
    instance_id: str,
    port: int,
    project_alias: str | None,
    source: str = "0.0.0.0/0",
    warn,
) -> RegisterByovmResult:
    """Register a BYOVM workbench alias and best-effort ingress for its HTTP port."""
    context = resolve_instance_network_context(instance_id)
    resolved_project = _registration_project_alias(project_alias)
    projects = list_projects()
    project_config = projects.get(resolved_project, {})
    workbenches = project_config.get("workbenches", {}) if isinstance(project_config, dict) else {}
    existing = workbenches.get(alias, {}) if isinstance(workbenches, dict) else {}
    existing_alias = existing if isinstance(existing, dict) else {}
    if existing_alias:
        warn(f"Warning: alias '{alias}' already exists; overwriting BYOVM registration fields.")

    alias_config = _byovm_alias_config(
        existing=existing_alias,
        tool=tool,
        alias=alias,
        context=context,
        port=port,
    )
    write_config({
        "projects": {
            resolved_project: {
                "workbenches": {
                    alias: alias_config,
                },
            },
        },
    })
    warn(f"Registered {tool} BYOVM alias '{alias}' in project '{resolved_project}'.")
    warn(f"  instance_id: {context.instance_id}")
    warn(f"  project_id: {context.project_id}")
    warn(f"  security_group_id: {context.security_group_id}")
    ingress = ensure_deploy_ingress(
        tool=tool,
        port=port,
        alias=alias,
        instance_id=context.instance_id,
        source=source,
        warn=warn,
    )
    return RegisterByovmResult(
        project_alias=resolved_project,
        alias=alias,
        instance_id=context.instance_id,
        project_id=context.project_id,
        security_group_id=context.security_group_id,
        port=port,
        ingress=ingress,
    )


def ensure_alias_ingress(
    *,
    tool: str,
    port: int,
    project_alias: str | None,
    name: str | None,
    source: str = "0.0.0.0/0",
) -> EnsureIngressResult:
    """Ensure ingress for a saved BYOVM alias that carries an instance ID."""
    alias = resolve_alias_record(project_alias, name)
    instance_id = alias.instance_id
    if not instance_id:
        raise ConfigError(
            f"alias '{alias.name}' has no instance_id; re-register with "
            f"'npa workbench {tool} register-byovm'"
        )
    return ensure_ingress(
        vm_id=instance_id,
        ports=(int(port),),
        source=source,
        tool=tool,
    )


def resolve_deploy_instance_id(
    *,
    tf_outputs: dict[str, Any],
    project_alias: str | None,
    name: str | None,
) -> str:
    """Resolve a deploy target instance ID from Terraform outputs or saved alias config."""
    instance_id = str(tf_outputs.get("instance_id", "") or "")
    if instance_id:
        return instance_id
    try:
        return resolve_alias_record(project_alias, name).instance_id
    except ConfigError:
        return ""


def ensure_deploy_ingress(
    *,
    tool: str,
    port: int,
    alias: str,
    instance_id: str,
    source: str = "0.0.0.0/0",
    warn,
) -> EnsureIngressResult | None:
    """Best-effort deploy-time ingress management.

    Deploy already completed by the time this runs, so ingress failures are
    reported as operator-actionable warnings instead of aborting the deploy.
    """
    if not instance_id:
        warn(f"Debug: skipping network ingress for port {port}: instance_id unavailable.")
        return None
    try:
        result = ensure_ingress(
            vm_id=instance_id,
            ports=(int(port),),
            source=source,
            tool=tool,
        )
    except Exception as exc:
        warn(
            f"Warning: could not ensure network ingress for port {port}: {exc}. "
            f"Run 'npa workbench {tool} ensure-ingress -n {alias}' to retry after fixing permissions."
        )
        return None
    warn(f"Network ingress confirmed for port {port}: {ingress_summary(result, port)}")
    return result


def ingress_summary(result: EnsureIngressResult, port: int) -> str:
    """Return a concise user-facing ingress status string."""
    if result.changed:
        return f"ingress rule created for port {port}"
    return f"ingress already covered for port {port}"


def _registration_project_alias(project_alias: str | None) -> str:
    if project_alias:
        return project_alias
    return default_project_name()


def _byovm_alias_config(
    *,
    existing: dict[str, Any],
    tool: str,
    alias: str,
    context: InstanceNetworkContext,
    port: int,
) -> dict[str, Any]:
    host = _strip_cidr(context.public_ip)
    existing_ssh = existing.get("ssh", {}) if isinstance(existing.get("ssh"), dict) else {}
    existing_storage = existing.get("storage", {}) if isinstance(existing.get("storage"), dict) else {}

    # Existing aliases are overwritten for the fields resolved from the VM so
    # re-registration repairs drift, while unrelated storage/custom fields remain.
    alias_config = {
        **existing,
        "alias": alias,
        "endpoint": f"http://{host}:{port}",
        "workbench_type": tool,
        "runtime": "byovm",
        "endpoint_strategy": "public",
        "service_port": int(port),
        "instance_id": context.instance_id,
        "project_id": context.project_id,
        "security_group_id": context.security_group_id,
        "ssh": {
            "host": host,
            "user": str(existing_ssh.get("user", "") or "ubuntu"),
            "key_path": str(existing_ssh.get("key_path", "") or "~/.ssh/id_ed25519"),
        },
        "storage": {
            "checkpoint_bucket": str(existing_storage.get("checkpoint_bucket", "") or ""),
            "endpoint_url": str(existing_storage.get("endpoint_url", "") or ""),
        },
    }
    if tool == "fiftyone":
        alias_config["app_port"] = int(port)
    return alias_config


def _strip_cidr(value: str) -> str:
    return value.split("/", 1)[0]
