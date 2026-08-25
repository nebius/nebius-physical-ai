"""Local configuration records for NPA agent deployments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from npa.clients.config import list_projects, update_config_document, write_config


AGENT_RECORD_SCHEMA_VERSION = 1


class AgentRecordError(RuntimeError):
    """Saved agent state is present but cannot be interpreted safely."""


class AgentRecordState(str, Enum):
    ABSENT = "absent"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class DecodedAgentRecord:
    state: AgentRecordState
    present: bool
    record: dict[str, Any]
    detail: str = ""


def resolve_project_agents(project_alias: str) -> dict[str, Any]:
    projects = list_projects()
    if project_alias not in projects:
        return {}
    project = projects[project_alias]
    if not isinstance(project, Mapping):
        raise AgentRecordError(
            f"project {project_alias!r} is present but schema-invalid"
        )
    if "agents" not in project:
        return {}
    agents = project["agents"]
    if not isinstance(agents, Mapping):
        raise AgentRecordError(
            f"project {project_alias!r} agents container is present but schema-invalid"
        )
    return dict(agents)


def decode_agent_record(project_alias: str, name: str) -> DecodedAgentRecord:
    """Decode one record without collapsing invalid presence into absence."""

    records = resolve_project_agents(project_alias)
    if name not in records:
        return DecodedAgentRecord(AgentRecordState.ABSENT, False, {})
    raw = records[name]
    if not isinstance(raw, Mapping):
        return DecodedAgentRecord(
            AgentRecordState.INVALID,
            True,
            {},
            "saved value is not an object",
        )
    record = dict(raw)
    if not record:
        return DecodedAgentRecord(
            AgentRecordState.INCOMPLETE,
            True,
            record,
            "saved object is empty",
        )
    version = record.get("schema_version", AGENT_RECORD_SCHEMA_VERSION)
    if type(version) is not int or version != AGENT_RECORD_SCHEMA_VERSION:
        return DecodedAgentRecord(
            AgentRecordState.INVALID,
            True,
            record,
            "saved schema_version is missing a supported integer value",
        )
    for field in ("project_id", "instance_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            return DecodedAgentRecord(
                AgentRecordState.INCOMPLETE,
                True,
                record,
                f"required immutable field {field!r} is missing or invalid",
            )
    contextual = (("project_alias", project_alias), ("name", name))
    for field, expected in contextual:
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            return DecodedAgentRecord(
                AgentRecordState.INVALID,
                True,
                record,
                f"immutable field {field!r} has an invalid type/value",
            )
        if value.strip() != expected:
            return DecodedAgentRecord(
                AgentRecordState.CONFLICTING,
                True,
                record,
                f"immutable field {field!r} conflicts with the selected key",
            )
    for field in ("tenant_id", "region", "service_account_id", "operation_id"):
        if field in record and not isinstance(record[field], str):
            return DecodedAgentRecord(
                AgentRecordState.INVALID,
                True,
                record,
                f"immutable field {field!r} has an invalid type",
            )
    return DecodedAgentRecord(AgentRecordState.COMPLETE, True, record)


def agent_record(project_alias: str, name: str) -> dict[str, Any]:
    decoded = decode_agent_record(project_alias, name)
    if decoded.state in {AgentRecordState.INVALID, AgentRecordState.CONFLICTING}:
        raise AgentRecordError(
            f"saved agent record {name!r} is {decoded.state.value}: {decoded.detail}"
        )
    return decoded.record


def store_agent_record(project_alias: str, name: str, payload: dict[str, Any]) -> None:
    stored = dict(payload)
    stored["schema_version"] = AGENT_RECORD_SCHEMA_VERSION
    write_config({"projects": {project_alias: {"agents": {name: stored}}}})


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
    "AGENT_RECORD_SCHEMA_VERSION",
    "AgentRecordError",
    "AgentRecordState",
    "DecodedAgentRecord",
    "agent_record",
    "decode_agent_record",
    "remove_agent_record",
    "resolve_project_agents",
    "store_agent_record",
]
