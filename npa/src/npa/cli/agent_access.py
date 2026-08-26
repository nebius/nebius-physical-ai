"""Project-aware effective access model for the NPA agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


ACCESS_SCHEMA = "npa.agent.access/v1"
ACCESS_STATES = frozenset({"available", "partial", "denied", "unavailable", "unverified"})
ACCESS_DISCOVERY_MAX_WORKERS = 8


def consistent_agent_service_account_id(existing: str, refreshed: str) -> str:
    """Keep bootstrap identity deterministic across credential refreshes."""
    expected = str(existing or "").strip()
    candidate = str(refreshed or "").strip()
    if expected and candidate and expected != candidate:
        raise ValueError(
            "credential refresh resolved a different service account than the existing agent record"
        )
    return candidate or expected


class AccessProbeError(RuntimeError):
    """A classified, public-safe access probe failure."""

    def __init__(self, status: str, operation: str):
        normalized = status if status in ACCESS_STATES else "unavailable"
        self.status = normalized
        self.operation = str(operation or "access probe")
        super().__init__(f"{self.operation} is {self.status}")


@dataclass(frozen=True)
class CapabilityAccess:
    status: str
    reason: str
    scope: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"status": self.status, "reason": self.reason}
        if self.scope:
            payload["scope"] = self.scope
        return payload


@dataclass(frozen=True)
class BucketProbe:
    list_status: str
    read_status: str
    reason: str = ""


@dataclass(frozen=True)
class StorageResourceAccess:
    resource_id: str
    name: str
    project_id: str
    capabilities: dict[str, CapabilityAccess]
    source: str = "project_inventory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "object_storage_bucket",
            "id": self.resource_id,
            "name": self.name,
            "project_id": self.project_id,
            "source": self.source,
            "capabilities": {
                key: self.capabilities[key].to_dict() for key in sorted(self.capabilities)
            },
        }


@dataclass(frozen=True)
class ProjectAccess:
    project_id: str
    name: str
    deployment_project: bool
    status: str
    capabilities: dict[str, CapabilityAccess]
    resources: tuple[StorageResourceAccess, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.project_id,
            "name": self.name,
            "deployment_project": self.deployment_project,
            "status": self.status,
            "capabilities": {
                key: self.capabilities[key].to_dict() for key in sorted(self.capabilities)
            },
            "resources": [item.to_dict() for item in self.resources],
        }


@dataclass(frozen=True)
class AgentAccessReport:
    tenant_id: str
    deployment_project_id: str
    deployment_project_name: str
    status: str
    scope: str
    capabilities: dict[str, CapabilityAccess]
    projects: tuple[ProjectAccess, ...] = ()
    errors: tuple[dict[str, str], ...] = ()
    refreshed_at: str = ""
    service_account_id: str = ""
    credential_source: str = ""
    credential_profile: str = ""
    credential_config: str = ""
    schema: str = ACCESS_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "apiVersion": self.schema,
            "identity": {
                "tenant_id": self.tenant_id,
                "deployment_project_id": self.deployment_project_id,
                "deployment_project_name": self.deployment_project_name,
                "service_account_id": self.service_account_id,
                "credential_source": self.credential_source,
                "credential_profile": self.credential_profile,
                "credential_config": self.credential_config,
            },
            "status": self.status,
            "scope": self.scope,
            "capabilities": {
                key: self.capabilities[key].to_dict() for key in sorted(self.capabilities)
            },
            "projects": [item.to_dict() for item in self.projects],
            "errors": [dict(item) for item in self.errors],
            "refreshed_at": self.refreshed_at,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classified_failure(exc: Exception, operation: str) -> tuple[str, str, dict[str, str]]:
    """Return a safe status/reason/error without copying cloud error text."""
    status = exc.status if isinstance(exc, AccessProbeError) else "unavailable"
    if status == "denied":
        reason = f"Permission denied while attempting to {operation}."
        code = "permission_denied"
    else:
        reason = f"Unable to {operation}."
        code = "unavailable"
    return status, reason, {"scope": operation, "code": code, "message": reason}


def _metadata(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    raw = item.get("metadata")
    return raw if isinstance(raw, dict) else item


def _project_identity(item: Any) -> tuple[str, str]:
    meta = _metadata(item)
    return str(meta.get("id") or "").strip(), str(meta.get("name") or "").strip()


def _bucket_identity(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        value = item.strip()
        return "", value
    meta = _metadata(item)
    return str(meta.get("id") or "").strip(), str(meta.get("name") or "").strip()


def _normalize_probe(value: Any) -> BucketProbe:
    if isinstance(value, BucketProbe):
        return value
    if isinstance(value, dict):
        return BucketProbe(
            list_status=str(value.get("list_status") or "unavailable"),
            read_status=str(value.get("read_status") or "unavailable"),
            reason=str(value.get("reason") or ""),
        )
    raise AccessProbeError("unavailable", "probe object storage bucket")


def _aggregate_status(statuses: list[str]) -> str:
    values = [item for item in statuses if item]
    if not values:
        return "unavailable"
    if all(item == "available" for item in values):
        return "available"
    if "available" in values or "partial" in values or "unverified" in values:
        return "partial"
    if "denied" in values:
        return "denied"
    return "unavailable"


def _artifact_status(resources: list[StorageResourceAccess], capability: str) -> str:
    states = [
        item.capabilities[capability].status
        for item in resources
        if capability in item.capabilities
    ]
    return _aggregate_status(states)


def discover_agent_access(
    *,
    tenant_id: str,
    deployment_project_id: str,
    deployment_project_name: str = "",
    fallback_buckets: tuple[str, ...] | list[str] = (),
    list_projects: Callable[[str], list[Any]],
    list_buckets: Callable[[str], list[Any]],
    probe_bucket: Callable[[str], BucketProbe | dict[str, str]],
    service_account_id: str = "",
    credential_source: str = "",
    credential_profile: str = "",
    credential_config: str = "",
    now: Callable[[], str] = _now_iso,
) -> AgentAccessReport:
    """Discover effective access with injected Nebius and S3 clients.

    Tenant/project inventory failures are retained in the report. The deployment
    project and explicitly configured storage are still probed for backward
    compatibility, but an inaccessible project never suppresses accessible ones.
    """
    tenant = str(tenant_id or "").strip()
    deployment_id = str(deployment_project_id or "").strip()
    deployment_name = str(deployment_project_name or "").strip()
    errors: list[dict[str, str]] = []

    project_listing_status = "unavailable"
    project_listing_reason = "Tenant identity is not configured."
    project_items: list[Any] = []
    if tenant:
        try:
            project_items = list(list_projects(tenant) or [])
            project_listing_status = "available"
            project_listing_reason = "Projects visible to the running agent were listed from the tenant."
        except Exception as exc:  # noqa: BLE001 - converted to a public-safe classified result
            project_listing_status, project_listing_reason, error = _classified_failure(
                exc, "list tenant projects"
            )
            errors.append(error)

    identities: dict[str, str] = {}
    for item in project_items:
        project_id, name = _project_identity(item)
        if project_id:
            identities[project_id] = name or project_id
    listed_project_ids = frozenset(identities)
    if deployment_id:
        identities.setdefault(deployment_id, deployment_name or deployment_id)

    ordered_project_ids = sorted(
        identities,
        key=lambda value: (value != deployment_id, identities[value].lower(), value),
    )
    configured_fallbacks: list[str] = []
    for bucket in fallback_buckets:
        value = str(bucket or "").strip()
        if value and value not in configured_fallbacks:
            configured_fallbacks.append(value)

    def discover_project_inventory(project_id: str) -> dict[str, Any]:
        is_deployment = bool(deployment_id and project_id == deployment_id)
        bucket_listing_status = "unavailable"
        bucket_listing_reason = "Object storage resource listing was not attempted."
        bucket_items: list[tuple[str, str, str]] = []
        error: dict[str, str] | None = None
        try:
            raw_buckets = list(list_buckets(project_id) or [])
            bucket_listing_status = "available"
            bucket_listing_reason = "Object storage resources visible in this project were listed."
            for raw in raw_buckets:
                resource_id, name = _bucket_identity(raw)
                if name:
                    bucket_items.append((resource_id, name, "project_inventory"))
        except Exception as exc:  # noqa: BLE001 - retain this project and continue with others
            bucket_listing_status, bucket_listing_reason, error = _classified_failure(
                exc, f"list object storage resources for project {project_id}"
            )

        if is_deployment:
            known = {name for _resource_id, name, _source in bucket_items}
            for name in configured_fallbacks:
                if name not in known:
                    bucket_items.append(("", name, "agent_configuration"))
                    known.add(name)
        return {
            "project_id": project_id,
            "is_deployment": is_deployment,
            "bucket_listing_status": bucket_listing_status,
            "bucket_listing_reason": bucket_listing_reason,
            "bucket_items": sorted(bucket_items, key=lambda item: item[1]),
            "error": error,
        }

    inventory_workers = min(len(ordered_project_ids), ACCESS_DISCOVERY_MAX_WORKERS)
    if inventory_workers <= 1:
        inventories = (
            [discover_project_inventory(ordered_project_ids[0])]
            if ordered_project_ids
            else []
        )
    else:
        with ThreadPoolExecutor(max_workers=inventory_workers) as pool:
            inventories = list(pool.map(discover_project_inventory, ordered_project_ids))
    for inventory in inventories:
        if inventory["error"] is not None:
            errors.append(inventory["error"])

    probe_jobs = [
        (inventory["project_id"], resource_id, bucket_name, source)
        for inventory in inventories
        for resource_id, bucket_name, source in inventory["bucket_items"]
    ]

    def probe_storage_resource(
        job: tuple[str, str, str, str],
    ) -> tuple[str, StorageResourceAccess, dict[str, str] | None]:
        project_id, resource_id, bucket_name, source = job
        is_deployment = bool(deployment_id and project_id == deployment_id)
        error: dict[str, str] | None = None
        try:
            probe = _normalize_probe(probe_bucket(bucket_name))
            list_reason = probe.reason or (
                "The running agent can list objects in this bucket."
                if probe.list_status == "available"
                else "The running agent cannot list objects in this bucket."
            )
            read_reason = probe.reason or (
                "The running agent can read an object from this bucket."
                if probe.read_status == "available"
                else "Object read access could not be verified."
            )
        except Exception as exc:  # noqa: BLE001 - classify one resource without hiding siblings
            probe_status, probe_reason, error = _classified_failure(
                exc, f"probe object storage bucket {bucket_name}"
            )
            probe = BucketProbe(
                list_status=probe_status,
                read_status=probe_status,
                reason=probe_reason,
            )
            list_reason = probe_reason
            read_reason = probe_reason
        return (
            project_id,
            StorageResourceAccess(
                resource_id=resource_id,
                name=bucket_name,
                project_id=project_id,
                source=source,
                capabilities={
                    "artifact_discovery": CapabilityAccess(
                        probe.list_status, list_reason, "read_only"
                    ),
                    "artifact_read": CapabilityAccess(
                        probe.read_status, read_reason, "read_only"
                    ),
                    "artifact_write": CapabilityAccess(
                        "unverified" if is_deployment else "unavailable",
                        (
                            "Writes remain scoped to the deployment project's configured workflow paths."
                            if is_deployment
                            else "Cross-project artifact writes are not enabled by tenant-wide discovery."
                        ),
                        "deployment_project",
                    ),
                    "artifact_delete": CapabilityAccess(
                        "unavailable",
                        "The agent access surface does not enable artifact deletion.",
                        "none",
                    ),
                },
            ),
            error,
        )

    probe_workers = min(len(probe_jobs), ACCESS_DISCOVERY_MAX_WORKERS)
    if probe_workers <= 1:
        probe_results = [probe_storage_resource(probe_jobs[0])] if probe_jobs else []
    else:
        with ThreadPoolExecutor(max_workers=probe_workers) as pool:
            probe_results = list(pool.map(probe_storage_resource, probe_jobs))
    resources_by_project: dict[str, list[StorageResourceAccess]] = {
        project_id: [] for project_id in ordered_project_ids
    }
    for project_id, resource, _error in probe_results:
        resources_by_project[project_id].append(resource)
        # A bucket-level S3 denial is an effective capability result for that
        # resource, not a failure of tenant/project discovery.  Keep it on the
        # resource's list/read capabilities so selecting that bucket remains
        # visibly blocked, but do not promote every inaccessible sibling into
        # the report-wide error banner.  Structural inventory failures above
        # still remain global because they prevent the affected scope from
        # being enumerated at all.

    projects: list[ProjectAccess] = []
    for inventory in inventories:
        project_id = str(inventory["project_id"])
        is_deployment = bool(inventory["is_deployment"])
        bucket_listing_status = str(inventory["bucket_listing_status"])
        bucket_listing_reason = str(inventory["bucket_listing_reason"])
        resources = resources_by_project[project_id]
        discovery_status = _artifact_status(resources, "artifact_discovery")
        read_status = _artifact_status(resources, "artifact_read")
        if not resources and bucket_listing_status in {"denied", "unavailable"}:
            discovery_status = bucket_listing_status
            read_status = bucket_listing_status
        metadata_status = (
            "available"
            if project_id in listed_project_ids
            else "unverified"
            if is_deployment
            else "unavailable"
        )
        metadata_reason = (
            "Project identity was returned by tenant project discovery."
            if metadata_status == "available"
            else "Project identity comes from deployment configuration and was not verified by tenant discovery."
        )
        project_status = _aggregate_status(
            [bucket_listing_status, discovery_status, read_status]
        )
        capabilities = {
            "project_metadata": CapabilityAccess(
                metadata_status,
                metadata_reason,
                "project",
            ),
            "storage_resource_discovery": CapabilityAccess(
                bucket_listing_status, bucket_listing_reason, "project"
            ),
            "artifact_discovery": CapabilityAccess(
                discovery_status,
                (
                    "At least one project bucket is searchable."
                    if discovery_status in {"available", "partial"}
                    else "No searchable object storage bucket was verified for this project."
                ),
                "read_only",
            ),
            "artifact_read": CapabilityAccess(
                read_status,
                (
                    "Artifact object reads were verified."
                    if read_status == "available"
                    else "Artifact object reads were not fully verified."
                ),
                "read_only",
            ),
            "artifact_write": CapabilityAccess(
                "unverified" if is_deployment else "unavailable",
                (
                    "Artifact writes remain scoped to configured workflow paths in the deployment project."
                    if is_deployment
                    else "Tenant-wide discovery does not enable artifact writes in this project."
                ),
                "deployment_project",
            ),
            "artifact_delete": CapabilityAccess(
                "unavailable",
                "Tenant-wide discovery does not enable artifact deletion.",
                "none",
            ),
            "workflow_submission": CapabilityAccess(
                "available" if is_deployment else "unavailable",
                (
                    "Workflow submission remains scoped to the deployment project."
                    if is_deployment
                    else "Tenant-wide discovery does not enable workflow submission in this project."
                ),
                "deployment_project",
            ),
        }
        projects.append(
            ProjectAccess(
                project_id=project_id,
                name=identities[project_id],
                deployment_project=is_deployment,
                status=project_status,
                capabilities=capabilities,
                resources=tuple(resources),
            )
        )

    project_discovery_states = [
        item.capabilities["artifact_discovery"].status for item in projects
    ]
    artifact_status = _aggregate_status(project_discovery_states)
    if project_listing_status != "available" and artifact_status == "available":
        artifact_status = "partial"
    overall = _aggregate_status([project_listing_status, artifact_status])
    if project_listing_status != "available" and artifact_status in {"available", "partial"}:
        overall = "partial"
    # A tenant-configured agent whose tenant inventory failed is *not* a healthy
    # single-project agent. The deployment-project fallback stays usable, but
    # its scope remains explicitly partial so operators see the failed promise.
    if tenant and project_listing_status != "available":
        scope = "partial_tenant"
    elif len(projects) <= 1:
        scope = "single_project"
    elif all(
        item.capabilities["storage_resource_discovery"].status == "available"
        for item in projects
    ):
        scope = "tenant"
    else:
        scope = "partial_tenant"

    tenant_capabilities = {
        "project_discovery": CapabilityAccess(
            project_listing_status, project_listing_reason, "tenant"
        ),
        "artifact_discovery": CapabilityAccess(
            artifact_status,
            (
                "Artifact discovery searches every project bucket with verified list access."
                if artifact_status == "available"
                else "Artifact discovery searches accessible projects and reports inaccessible ones."
            ),
            "tenant_read_only",
        ),
        "workflow_submission": CapabilityAccess(
            "available" if deployment_id else "unavailable",
            "Workflow submission remains scoped to the deployment project.",
            "deployment_project",
        ),
        "artifact_write": CapabilityAccess(
            "unverified" if deployment_id else "unavailable",
            "Tenant-wide access does not broaden artifact write targets.",
            "deployment_project",
        ),
        "artifact_delete": CapabilityAccess(
            "unavailable",
            "Tenant-wide access does not enable artifact deletion.",
            "none",
        ),
        "arbitrary_s3_uri": CapabilityAccess(
            "unavailable",
            "Caller-supplied S3 URIs remain restricted to configured buckets; discovered artifacts are authorized by run membership.",
            "configured_resources",
        ),
    }
    return AgentAccessReport(
        tenant_id=tenant,
        deployment_project_id=deployment_id,
        deployment_project_name=deployment_name,
        status=overall,
        scope=scope,
        capabilities=tenant_capabilities,
        projects=tuple(projects),
        errors=tuple(errors),
        refreshed_at=now(),
        service_account_id=str(service_account_id or "").strip(),
        credential_source=str(credential_source or "").strip(),
        credential_profile=str(credential_profile or "").strip(),
        credential_config=str(credential_config or "").strip(),
    )


def accessible_artifact_buckets(report: AgentAccessReport | dict[str, Any]) -> list[str]:
    """Return searchable bucket names, deployment project first."""
    payload = report.to_dict() if isinstance(report, AgentAccessReport) else report
    projects = payload.get("projects", []) if isinstance(payload, dict) else []
    ordered: list[str] = []
    for project in sorted(
        (item for item in projects if isinstance(item, dict)),
        key=lambda item: (not bool(item.get("deployment_project")), str(item.get("name") or "")),
    ):
        for resource in project.get("resources", []) or []:
            if not isinstance(resource, dict) or resource.get("type") != "object_storage_bucket":
                continue
            capabilities = resource.get("capabilities") or {}
            discovery = capabilities.get("artifact_discovery") or {}
            name = str(resource.get("name") or "").strip()
            if discovery.get("status") == "available" and name and name not in ordered:
                ordered.append(name)
    return ordered


def artifact_bucket_projects(report: AgentAccessReport | dict[str, Any]) -> dict[str, str]:
    """Return accessible bucket name -> owning project id."""
    payload = report.to_dict() if isinstance(report, AgentAccessReport) else report
    mapping: dict[str, str] = {}
    for project in payload.get("projects", []) if isinstance(payload, dict) else []:
        if not isinstance(project, dict):
            continue
        project_id = str(project.get("id") or "")
        for resource in project.get("resources", []) or []:
            if not isinstance(resource, dict):
                continue
            capabilities = resource.get("capabilities") or {}
            if (capabilities.get("artifact_discovery") or {}).get("status") != "available":
                continue
            name = str(resource.get("name") or "").strip()
            if name:
                mapping[name] = project_id
    return mapping


def scoped_artifact_buckets(
    report: AgentAccessReport | dict[str, Any],
    *,
    resource_bucket: str = "",
    project_id: str = "",
) -> list[str]:
    """Resolve a caller-selected list scope against effective read-only access.

    Empty selectors preserve tenant-wide discovery. A selected bucket must be a
    first-class resource with verified artifact-discovery access, and an
    accompanying project id must match that resource's owner. This keeps UI
    filtering useful without turning a caller-supplied bucket name into an
    authorization mechanism.
    """
    bucket = str(resource_bucket or "").strip()
    project = str(project_id or "").strip()
    mapping = artifact_bucket_projects(report)
    if bucket:
        if bucket not in mapping:
            raise ValueError("artifact bucket is outside effective agent access")
        if project and mapping[bucket] != project:
            raise ValueError("artifact bucket does not belong to the selected project")
        return [bucket]
    if project:
        selected = [name for name, owner in mapping.items() if owner == project]
        if not selected:
            raise ValueError("selected project has no searchable artifact bucket")
        return selected
    return accessible_artifact_buckets(report)
