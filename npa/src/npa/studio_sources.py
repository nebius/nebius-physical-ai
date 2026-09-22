"""Discover tenant artifact buckets without changing infrastructure or credentials."""

from __future__ import annotations

from npa.clients.config import list_projects, resolve_environment
from npa.clients.nebius import NebiusError, _run_json


def _items(arguments: list[str]) -> list[dict]:
    response = _run_json([*arguments, "--all"])
    if response == {}:
        return []
    items = response.get("items")
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError("Provider inventory has no valid items array")
    if response.get("next_page_token"):
        raise ValueError("Provider returned incomplete inventory despite --all")
    return items


def _project_sources(
    item: dict, fallback: str | None, aliases: dict[str, str]
) -> list[dict]:
    metadata = item.get("metadata", {})
    project_id = metadata.get("id")
    region = item.get("status", {}).get("region") or item.get("spec", {}).get("region")
    if (
        not isinstance(project_id, str)
        or not project_id
        or not isinstance(region, str)
        or not region
    ):
        raise ValueError("Project inventory lacks an identity or region")
    if not all(character.isalnum() or character == "-" for character in region):
        raise ValueError("Project inventory has an invalid region")
    rows = []
    for bucket in _items(["storage", "bucket", "list", "--parent-id", project_id]):
        name = bucket.get("metadata", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Bucket inventory lacks a name")
        rows.append(
            {
                "resource_project_id": project_id,
                "bucket": name,
                "project": aliases.get(project_id, fallback),
                "endpoint": f"https://storage.{region}.nebius.cloud",
            }
        )
    return rows


def tenant_sources(project: str | None) -> tuple[list[dict], list[dict]]:
    """Inventory a selected tenant and bind each bucket to a credential context.

    Args:
        project: Selected configured alias, or the default configuration.
    Returns:
        Source tuples and explicit inventory failures. Known project aliases use
        their own credentials; other buckets are probed with the selected context.
    Raises:
        ValueError: The selected configuration does not identify a tenant.
    """
    environment = resolve_environment(project)
    if environment is None or not environment.tenant_id:
        raise ValueError("--discover-tenant requires a configured tenant")
    aliases = {
        value.get("project_id"): name
        for name, value in sorted(list_projects().items())
        if value.get("project_id") and value.get("tenant_id") == environment.tenant_id
    }
    try:
        projects = _items(
            ["iam", "project", "list", "--parent-id", environment.tenant_id]
        )
    except (NebiusError, ValueError, OSError) as error:
        return [], [{"operation": "list_tenant_projects", "code": type(error).__name__}]
    rows, errors = [], []
    for item in projects:
        try:
            rows.extend(_project_sources(item, project, aliases))
        except (NebiusError, ValueError, OSError) as error:
            errors.append(
                {
                    "operation": "list_project_buckets",
                    "code": type(error).__name__,
                    "resource_project_id": item.get("metadata", {}).get("id"),
                }
            )
    return rows, errors
