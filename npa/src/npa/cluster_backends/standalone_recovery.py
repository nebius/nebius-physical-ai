"""Recover partial standalone mk8s ownership from its journal and native state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npa.cluster.state import metadata_file
from npa.provisioning_journal import list_operations, load_operation


def _operation_for_context(context: str, project_id: str, operation_id: str):
    operations = (
        [load_operation(operation_id)]
        if operation_id
        else list_operations(project_id=project_id, resource_type="cluster")
    )
    matches = []
    for operation in operations:
        payload = operation.read()
        parameters = payload.get("parameters") or {}
        recorded_context = parameters.get("context") or payload.get("requested_name")
        if payload.get("project_id") == project_id and recorded_context == context:
            matches.append(payload)
    if len(matches) > 1:
        raise ValueError("Partial standalone recovery needs an exact --operation-id")
    return matches[0] if matches else None


def _read_private_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "Partial standalone recovery requires regular owned state files"
        )
    if path.stat().st_mode & 0o077:
        raise ValueError("Partial standalone recovery state must be owner-only")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Partial standalone recovery state must be a JSON object")
    return payload


def _recorded_cluster_id(workdir: Path, project_id: str, name: str) -> str:
    payload = _read_private_json(workdir / "terraform.tfstate")
    clusters = [
        instance.get("attributes", {})
        for resource in payload.get("resources", [])
        if resource.get("mode") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_cluster"
        and not resource.get("module")
        for instance in resource.get("instances", [])
        if not instance.get("deposed")
    ]
    if len(clusters) != 1:
        raise ValueError(
            "Partial standalone recovery requires one exact cluster in state"
        )
    cluster = clusters[0]
    if cluster.get("parent_id") != project_id or cluster.get("name") != name:
        raise ValueError("Partial standalone Terraform cluster identity does not match")
    cluster_id = str(cluster.get("id") or "")
    if not cluster_id:
        raise ValueError(
            "Partial standalone Terraform state has no immutable cluster ID"
        )
    return cluster_id


def partial_backend_metadata(
    *,
    context: str,
    project_id: str,
    tenant_id: str,
    region: str,
    operation_id: str = "",
) -> dict[str, Any]:
    """Resolve one interrupted native backend without rewriting ownership files.

    Args:
        context: Exact local context to remove.
        project_id: Selected provider project.
        tenant_id: Selected provider tenant.
        region: Selected provider region.
        operation_id: Optional exact provisioning journal ID.
    Returns:
        Verified native backend metadata, or an empty mapping if absent.
    Raises:
        ValueError: Ambiguous, unsafe, or contradictory recovery evidence.
    """
    operation = _operation_for_context(context, project_id, operation_id)
    if operation is None:
        return {}
    name = str(operation.get("requested_name") or "")
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Partial standalone journal has an unsafe cluster name")
    root = metadata_file(context).parent / "backend-state"
    install = root / project_id / name
    sidecar_path = install / ".npa-fleet-env.json"
    if not sidecar_path.exists():
        return {}
    return {
        **_metadata_from_install(install, root, name, project_id, tenant_id, region),
        "backend_recovery_operation_id": str(operation.get("operation_id") or ""),
    }


def _metadata_from_install(
    install: Path, root: Path, name: str, project_id: str, tenant_id: str, region: str
) -> dict[str, Any]:
    if install.resolve() != root.resolve() / project_id / name:
        raise ValueError("Partial standalone backend state is non-canonical")
    sidecar = _read_private_json(install / ".npa-fleet-env.json")
    expected = {
        "backend": "mk8s",
        "project_id": project_id,
        "tenant_id": tenant_id,
        "region": region,
        "cluster_name": name,
        "context": f"fleet-standalone-{project_id}-{name}",
    }
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Partial standalone backend identity does not match the journal"
        )
    return {
        "managed_by": "npa cluster shared-mk8s-backend",
        "backend": "mk8s",
        "backend_state_root": str(root.resolve()),
        "backend_fleet_name": "standalone",
        "backend_project_key": project_id,
        "backend_cluster_name": name,
        "backend_project_id": project_id,
        "backend_cluster_id": _recorded_cluster_id(
            install / "k8s-training", project_id, name
        ),
        "backend_tenant_id": tenant_id,
        "backend_region": region,
        "backend_profile": str(sidecar.get("profile") or ""),
    }
