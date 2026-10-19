"""Preserve cumulative Fleet demand when tenant inventory is denied by IAM."""

from dataclasses import dataclass, field

from npa.cluster_backends.project_preflight import (
    ProjectCapacityReader,
    TenantCapacityAccessDenied,
)
from npa.cluster_backends.quotas import (
    preflight_quota_requirements,
    preflight_region,
    required_quotas,
)


@dataclass
class ProjectDemand:
    """Cumulative incremental resources for one provider-verified project."""

    project_id: str
    clusters: list = field(default_factory=list)
    object_storage: list = field(default_factory=list)


def group_project_demands(projects, project_ids, clusters, storage):
    """Group all selected aliases by immutable project identity and region.

    Args:
        projects: Selected project specifications with resolved regions.
        project_ids: Provider-verified IDs keyed by project alias.
        clusters: Incremental cluster demand keyed by project alias.
        storage: Incremental bucket demand keyed by project alias.
    Returns:
        Regional project demand, combining aliases of the same project.
    Raises:
        None.
    """
    grouped = {}
    for project, region in projects:
        project_id = project_ids.get(project.key())
        if not project_id:
            continue
        regional = grouped.setdefault(region, {})
        demand = regional.setdefault(project_id, ProjectDemand(project_id))
        demand.clusters.extend(clusters.get(project.key(), []))
        demand.object_storage.extend(storage.get(project.key(), []))
    return {region: list(demands.values()) for region, demands in grouped.items()}


def preflight_fleet_region(*, project_demands=(), **request):
    """Use exact project constraints only after a typed tenant inventory denial.

    Args:
        project_demands: Verified cumulative demands for existing projects.
        request: Standard regional preflight arguments, with explicit identity.
    Returns:
        Verified project and any still-readable aggregate quota shortfalls.
    Raises:
        ValueError: Capacity evidence is unavailable or incompatible.
    """
    try:
        return preflight_region(**request)
    except TenantCapacityAccessDenied:
        if request.get("new_projects") or not project_demands:
            raise
    shortfalls = []
    for demand in project_demands:
        scoped = {
            **request,
            "project_id": demand.project_id,
            "clusters": demand.clusters,
            "object_storage": demand.object_storage,
        }
        shortfalls.extend(preflight_region(**scoped))
    return shortfalls + _readable_tenant_quota_shortfalls(request)


def _readable_tenant_quota_shortfalls(request):
    # Reservation inventory and quota permissions are independent. A denied
    # reservation read must never suppress a readable aggregate quota deficit.
    reader = ProjectCapacityReader(
        request["nebius_bin"],
        request["tenant_id"],
        "",
        request["profile"],
        request["env"],
        request["run_capture"],
        request["nebius_argv"],
    )
    needed = required_quotas(
        request["clusters"], object_storage=request.get("object_storage", ())
    )
    try:
        return preflight_quota_requirements(
            reader, request["region"], needed, request.get("on_status")
        )
    except TenantCapacityAccessDenied:
        return []
