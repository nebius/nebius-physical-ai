"""Shared helpers for workbench ingress subcommands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from npa.clients.config import (
    ConfigError,
    default_project_name,
    default_workbench_name,
    list_projects,
)
from npa.clients.network import EnsureIngressResult, ensure_ingress


@dataclass(frozen=True)
class AliasRecord:
    project_alias: str
    name: str
    data: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.data.get("instance_id", "") or "")


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


def ingress_summary(result: EnsureIngressResult, port: int) -> str:
    """Return a concise user-facing ingress status string."""
    if result.changed:
        return f"ingress rule created for port {port}"
    return f"ingress already covered for port {port}"
