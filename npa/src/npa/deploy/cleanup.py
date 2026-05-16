"""Cleanup helpers for interrupted Terraform-managed workbench deploys."""

from __future__ import annotations

from typing import Literal

from npa.clients.config import (
    ConfigError,
    list_projects,
    remove_workbench_config,
    resolve_environment,
    resolve_terraform_state,
)
from npa.deploy import provisioner
from npa.deploy.provisioner import ProvisionerError

AliasCleanupState = Literal["fresh", "partial", "fully_deployed", "byovm"]


class CleanupPartialError(Exception):
    pass


def _has_terraform_state(project_cfg: dict) -> bool:
    state = project_cfg.get("terraform_state", {})
    return isinstance(state, dict) and any(bool(value) for value in state.values())


def _has_host(workbench_cfg: dict) -> bool:
    ssh = workbench_cfg.get("ssh", {})
    if isinstance(ssh, dict) and ssh.get("host"):
        return True
    return bool(workbench_cfg.get("host") or workbench_cfg.get("endpoint"))


def classify_alias_state(project: str, name: str) -> AliasCleanupState:
    """Classify an alias for conservative partial-cleanup handling."""
    projects = list_projects()
    project_cfg = projects.get(project, {})
    if not isinstance(project_cfg, dict):
        return "fresh"
    workbenches = project_cfg.get("workbenches", {})
    if not isinstance(workbenches, dict) or name not in workbenches:
        return "fresh"
    workbench_cfg = workbenches.get(name, {})
    if not isinstance(workbench_cfg, dict):
        return "fully_deployed"

    if str(workbench_cfg.get("runtime", "") or "").lower() == "byovm":
        return "byovm"

    has_state = _has_terraform_state(project_cfg)
    has_host = _has_host(workbench_cfg)
    if has_state and not has_host:
        return "partial"
    if not has_state and not has_host:
        return "partial"
    return "fully_deployed"


def _remote_state_working_dir(project: str, name: str) -> str:
    state = resolve_terraform_state(project)
    if not (state.bucket and state.endpoint and state.access_key and state.secret_key):
        raise CleanupPartialError(
            f"No complete Terraform state backend is configured for project {project!r}."
        )
    env = resolve_environment(project)
    if env is None or not env.region:
        raise CleanupPartialError(
            f"No region is configured for project {project!r}; cannot initialize Terraform state."
        )
    return str(
        provisioner.prepare_working_dir(
            project,
            name,
            bucket=state.bucket,
            region=env.region,
            endpoint=state.endpoint,
        )
    )


def _init_remote_state(project: str, name: str) -> str:
    state = resolve_terraform_state(project)
    work_dir = _remote_state_working_dir(project, name)
    provisioner.init(
        tf_dir=work_dir,
        backend_config={
            "access_key": state.access_key,
            "secret_key": state.secret_key,
        },
    )
    return work_dir


def list_terraform_managed_resources(project: str, name: str) -> list[str]:
    """Return Terraform resource addresses for a partial workbench alias."""
    try:
        work_dir = _init_remote_state(project, name)
        return provisioner.state_list(tf_dir=work_dir)
    except (ConfigError, ProvisionerError) as exc:
        raise CleanupPartialError(str(exc)) from exc


def terraform_destroy_partial(project: str, name: str) -> None:
    """Run Terraform destroy for a partial workbench alias."""
    try:
        work_dir = _init_remote_state(project, name)
        provisioner.destroy(tf_dir=work_dir, tf_vars={})
    except (ConfigError, ProvisionerError) as exc:
        raise CleanupPartialError(str(exc)) from exc


def remove_partial_config_entry(project: str, name: str) -> None:
    remove_workbench_config(project, name)
