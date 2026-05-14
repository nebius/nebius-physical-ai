"""Subnet resolution for Nebius Serverless Job submissions."""

from __future__ import annotations

import json
import subprocess
from typing import Any


class SubnetResolutionError(RuntimeError):
    """Raised when a Serverless Job subnet cannot be resolved unambiguously."""


def resolve_subnet(
    project_id: str,
    explicit_subnet_id: str | None = None,
    default_network_name: str = "default-network",
    default_subnet_prefix: str = "default-subnet",
) -> str:
    """Resolve the subnet ID for a Serverless Job submission.

    Precedence:
      1. Explicit subnet ID from ``--subnet-id``.
      2. Unique READY ``default_subnet_prefix*`` subnet under ``default_network_name``.
      3. The only READY subnet in the project.
      4. Raise ``SubnetResolutionError`` with an actionable message.
    """

    explicit = str(explicit_subnet_id or "").strip()
    if explicit:
        return explicit

    project = str(project_id or "").strip()
    if not project:
        raise SubnetResolutionError("Project ID is required to resolve a Serverless Job subnet.")

    networks = _list_vpc_resources(project, "network")
    subnets = _list_vpc_resources(project, "subnet")

    default_networks = [
        network
        for network in networks
        if _resource_name(network) == default_network_name and _is_ready(network)
    ]
    if len(default_networks) > 1:
        raise SubnetResolutionError(
            f"Found multiple READY networks named {default_network_name!r} in project {project}; "
            "specify --subnet-id."
        )

    ready_subnets = [subnet for subnet in subnets if _is_ready(subnet)]
    if not ready_subnets:
        raise SubnetResolutionError(
            f"No READY subnets found in project {project}; specify --subnet-id."
        )

    if len(default_networks) == 1:
        default_network_id = _resource_id(default_networks[0])
        default_subnets = [
            subnet
            for subnet in ready_subnets
            if _subnet_network_id(subnet) == default_network_id
            and _resource_name(subnet).startswith(default_subnet_prefix)
        ]
        if len(default_subnets) > 1:
            raise SubnetResolutionError(
                f"Found multiple READY subnets named {default_subnet_prefix!r}* under "
                f"network {default_network_name!r} in project {project}; specify --subnet-id."
            )
        if len(default_subnets) == 1:
            subnet_id = _resource_id(default_subnets[0])
            if subnet_id:
                return subnet_id
            raise SubnetResolutionError(
                f"Default subnet under network {default_network_name!r} in project {project} "
                "has no metadata.id; specify --subnet-id."
            )

    if len(ready_subnets) == 1:
        subnet_id = _resource_id(ready_subnets[0])
        if subnet_id:
            return subnet_id
        raise SubnetResolutionError(
            f"The only READY subnet in project {project} has no metadata.id; specify --subnet-id."
        )

    raise SubnetResolutionError(
        f"Found {len(ready_subnets)} READY subnets in project {project}, but no unique "
        f"{default_subnet_prefix!r}* subnet under network {default_network_name!r}; "
        "specify --subnet-id."
    )


def _list_vpc_resources(project_id: str, resource: str) -> list[dict[str, Any]]:
    command = [
        "nebius",
        "vpc",
        resource,
        "list",
        "--parent-id",
        project_id,
        "--format",
        "json",
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubnetResolutionError(
            f"Timed out while listing Nebius VPC {resource}s for project {project_id}."
        ) from exc
    except OSError as exc:
        raise SubnetResolutionError(
            f"Unable to run Nebius CLI while listing VPC {resource}s for project {project_id}: {exc}"
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else f" (exit code {result.returncode})"
        raise SubnetResolutionError(
            f"Unable to list Nebius VPC {resource}s for project {project_id}{suffix}"
        )

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SubnetResolutionError(
            f"Unable to parse Nebius VPC {resource} list JSON for project {project_id}: {exc}"
        ) from exc

    items: Any
    if isinstance(payload, dict):
        items = payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        raise SubnetResolutionError(
            f"Unexpected Nebius VPC {resource} list response for project {project_id}; "
            "specify --subnet-id."
        )
    return [item for item in items if isinstance(item, dict)]


def _resource_id(resource: dict[str, Any]) -> str:
    metadata = resource.get("metadata") if isinstance(resource, dict) else {}
    return str((metadata or {}).get("id") or "")


def _resource_name(resource: dict[str, Any]) -> str:
    metadata = resource.get("metadata") if isinstance(resource, dict) else {}
    return str((metadata or {}).get("name") or "")


def _is_ready(resource: dict[str, Any]) -> bool:
    status = resource.get("status") if isinstance(resource, dict) else {}
    return str((status or {}).get("state") or "").upper() == "READY"


def _subnet_network_id(subnet: dict[str, Any]) -> str:
    spec = subnet.get("spec") if isinstance(subnet, dict) else {}
    return str((spec or {}).get("network_id") or "")
