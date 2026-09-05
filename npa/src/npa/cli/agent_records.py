"""Local configuration records for NPA agent deployments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from npa.clients.config import list_projects, update_config_document, write_config


def resolve_project_agents(project_alias: str) -> dict[str, Any]:
    projects = list_projects()
    project = projects.get(project_alias, {})
    agents = project.get("agents", {}) if isinstance(project, dict) else {}
    return agents if isinstance(agents, dict) else {}


def agent_record(project_alias: str, name: str) -> dict[str, Any]:
    record = resolve_project_agents(project_alias).get(name, {})
    return record if isinstance(record, dict) else {}


def leisaac_ui_enabled(record: object) -> bool:
    """Require an explicit YAML boolean in the selected agent's UI settings."""
    ui = record.get("ui") if isinstance(record, dict) else None
    return isinstance(ui, dict) and ui.get("leisaac_enabled") is True


def store_agent_record(project_alias: str, name: str, payload: dict[str, Any]) -> None:
    write_config({"projects": {project_alias: {"agents": {name: payload}}}})


def remove_agent_record(project_alias: str, name: str) -> None:
    def remove(current: dict[str, Any]) -> dict[str, Any]:
        data = deepcopy(current)
        projects = data.get("projects", {})
        if not isinstance(projects, dict):
            return data
        project = projects.get(project_alias, {})
        if not isinstance(project, dict):
            return data
        agents = project.get("agents", {})
        if not isinstance(agents, dict) or name not in agents:
            return data
        del agents[name]
        if agents:
            project["agents"] = agents
        else:
            project.pop("agents", None)
        projects[project_alias] = project
        data["projects"] = projects
        return data

    update_config_document(remove)


__all__ = [
    "agent_record",
    "leisaac_ui_enabled",
    "remove_agent_record",
    "resolve_project_agents",
    "store_agent_record",
]
