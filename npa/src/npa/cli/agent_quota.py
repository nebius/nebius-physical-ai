"""Public-IPv4 quota checks for ``npa agent`` deploy/preflight.

Extracted from ``npa.cli.agent`` (a monolith under a size ratchet) so the two
cohesive, independently-tested quota helpers live in one small module. They are
re-exported from ``npa.cli.agent`` for the existing call sites and tests.

The agent VM always needs exactly one public IP, and compute placement follows
the *project's* region (not the ``--region`` flag). Both helpers therefore
resolve the project's real region and check the tenant's per-region
``vpc.ipv4-address.public.count`` allowance. A project-scoped administrator may
not be allowed to inspect the tenant aggregate, so the whole-path check falls
back to the same quota catalog under the exact deployment project. A real finite
project allowance remains a hard gate; only the unavailable tenant-wide view is
reported as advisory. The provider remains authoritative for the tenant
aggregate during apply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


_QUOTA_SCOPE_CHECK = "quota_evidence_scope"
_TENANT_QUOTA_RBAC_FALLBACK = (
    "tenant-wide quota query unavailable due to RBAC; "
    "project-scoped quota allowances verified"
)

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult


def _exact_owned_cluster_name(project_id: str, fallback: str) -> str:
    """Return one durable NPA-owned cluster identity, never a name-only guess."""

    from npa.provisioning_journal import list_operations

    matches: list[str] = []
    for operation in list_operations(project_id=project_id, resource_type="cluster"):
        payload = operation.read()
        if payload.get("phase") != "committed":
            continue
        requested_name = str(payload.get("requested_name") or "").strip()
        resources = payload.get("resources")
        if not requested_name or not isinstance(resources, list):
            continue
        owns_exact_cluster = any(
            isinstance(resource, dict)
            and resource.get("resource_type") == "managed_kubernetes_cluster"
            and resource.get("ownership") == "created_by_this_operation"
            and resource.get("project_id") == project_id
            and resource.get("requested_name") == requested_name
            and bool(str(resource.get("provider_id") or "").strip())
            for resource in resources
        )
        if owns_exact_cluster and requested_name not in matches:
            matches.append(requested_name)
    return matches[0] if len(matches) == 1 else fallback


def _project_quota_observations(
    project_id: str, region: str, names: tuple[str, ...]
):
    """Read project-local quota constraints after a tenant RBAC denial.

    Project quota allowances only subdivide the tenant allowance. Missing or
    explicitly unbounded project entries therefore mean "no project-local
    restriction"; they do not claim that the unreadable tenant aggregate has
    headroom. The caller surfaces that remaining uncertainty as a warning.
    """

    from copy import deepcopy

    from npa.clients.nebius import list_quota_allowances
    from npa.provisioning_preflight import QuotaObservation, parse_quota_allowances

    payload = deepcopy(list_quota_allowances(project_id))
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("project quota response contains a non-mapping item")
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    "project quota response contains malformed metadata"
                )
            if str(metadata.get("name") or "") not in names:
                continue
            spec = item.get("spec")
            if not isinstance(spec, dict):
                raise ValueError("project quota response contains a malformed spec")
            if not str(spec.get("limit") or "").strip():
                spec["limit"] = "unbounded"
    parsed = parse_quota_allowances(payload, region=region, names=names)
    return {
        name: (
            QuotaObservation(
                name=name,
                state="unbounded",
                reason="no project-specific quota restriction is configured",
            )
            if observation.state == "unsupported"
            else observation
        )
        for name, observation in parsed.items()
    }


def _agent_check_whole_path_capacity(
    project_id: str,
    tenant_id: str,
    fallback_region: str,
    *,
    agent_exists: bool = False,
    include_paidf: bool = True,
):
    """Apply the shared VM+disk+public-IP plan before agent mutation."""

    from npa.clients import nebius as nebius_client
    from npa.provisioning_preflight import (
        ExistingCapacity,
        PreflightBlockedError,
        PreflightCheck,
        QuotaObservation,
        build_whole_path_plan,
        discover_existing_capacity,
        read_provider_quotas,
        resolve_topology,
    )

    region = (
        nebius_client.get_project_region(project_id) or str(fallback_region or "")
    ).strip()
    requested = resolve_topology(
        agent_requested=True,
        agent_exists=agent_exists,
        cpu_nodes=-1 if include_paidf else 0,
        gpu_nodes=-1 if include_paidf else 0,
    )
    # An agent-only plan reserves no cluster nodes, so there is nothing for the
    # existing-cluster inventory to deduct. Probing mk8s anyway turns an
    # unrelated `resource.mk8scluster.list` denial into an unverified mutation
    # prerequisite that blocks a CPU-only agent VM the operator can create.
    reserves_cluster_nodes = requested.cpu_nodes > 0 or requested.gpu_nodes > 0
    if reserves_cluster_nodes:
        cluster_name = _exact_owned_cluster_name(project_id, requested.cluster_name)
        existing = discover_existing_capacity(
            project_id=project_id,
            cluster_name=cluster_name,
            cpu_platform=requested.cpu_platform,
            cpu_preset=requested.cpu_preset,
            gpu_platform=requested.gpu_platform,
            gpu_preset=requested.gpu_preset,
        )
    else:
        existing = ExistingCapacity()
    topology = resolve_topology(
        agent_requested=True,
        agent_exists=agent_exists,
        cpu_nodes=requested.cpu_nodes,
        existing_cpu_nodes=min(requested.cpu_nodes, existing.cpu_nodes),
        cpu_platform=requested.cpu_platform,
        cpu_preset=requested.cpu_preset,
        cpu_disk_gib=requested.cpu_disk_gib,
        gpu_nodes=requested.gpu_nodes,
        existing_gpu_nodes=min(requested.gpu_nodes, existing.gpu_nodes),
        gpu_platform=requested.gpu_platform,
        gpu_preset=requested.gpu_preset,
        gpu_disk_gib=requested.gpu_disk_gib,
    )
    quota_names = tuple(topology.quota_requirements())
    quota_scope = PreflightCheck(
        name=_QUOTA_SCOPE_CHECK,
        status="ready",
        reason="tenant-wide quota allowances verified",
    )
    try:
        observations = read_provider_quotas(tenant_id, region, quota_names)
    except Exception as exc:  # noqa: BLE001 - classify before choosing fallback
        if nebius_client.is_permission_denied(str(exc)):
            try:
                observations = _project_quota_observations(
                    project_id, region, quota_names
                )
            except Exception as project_exc:  # noqa: BLE001 - fail closed, sanitized
                observations = {
                    name: QuotaObservation(
                        name=name,
                        state="unknown",
                        reason=(
                            "tenant-wide quota query is unavailable due to RBAC and "
                            "the project-scoped quota query failed "
                            f"({type(project_exc).__name__})"
                        ),
                    )
                    for name in quota_names
                }
            else:
                quota_scope = PreflightCheck(
                    name=_QUOTA_SCOPE_CHECK,
                    status="ready",
                    reason=_TENANT_QUOTA_RBAC_FALLBACK,
                )
        else:
            observations = {
                name: QuotaObservation(
                    name=name,
                    state="unknown",
                    reason=(
                        "tenant-wide quota query failed for a reason other than "
                        f"an RBAC scope limitation ({type(exc).__name__})"
                    ),
                )
                for name in quota_names
            }

    plan = build_whole_path_plan(
        project_alias="",
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        # Agent deploy is the first mutating step in the README whole-path flow.
        # Reserve the same canonical PAIDF cluster shape here so the account is
        # not left with a paid VM that makes the immediately-following cluster
        # impossible. Existing resources are deducted by the shared planner at
        # the provisioning entrypoint; an already-present agent is deducted here.
        topology=topology,
        quota_reader=lambda _parent, _region, _names: observations,
        checks=[existing.check, quota_scope],
        mutation=True,
    )
    try:
        plan.assert_mutation_ready()
    except PreflightBlockedError as exc:
        if quota_scope.reason == _TENANT_QUOTA_RBAC_FALLBACK:
            blocked = any(item.status == "blocked" for item in plan.quotas)
            description = (
                "Project-scoped quota evidence denies the requested capacity"
                if blocked
                else "Project-scoped quota evidence could not verify the requested capacity"
            )
            raise PreflightBlockedError(f"{description}: {exc}") from exc
        raise
    return plan


def _agent_whole_path_capacity_result(
    project_id: str,
    tenant_id: str,
    fallback_region: str,
    *,
    agent_exists: bool = False,
    include_paidf: bool = True,
) -> "CheckResult":
    """Render the deploy gate through the health/preflight result contract."""

    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS, WARN

    try:
        plan = _agent_check_whole_path_capacity(
            project_id,
            tenant_id,
            fallback_region,
            agent_exists=agent_exists,
            include_paidf=include_paidf,
        )
    except Exception as exc:  # noqa: BLE001 - same fail-closed resolver as deploy
        return CheckResult(
            name="whole_path_capacity",
            status=FAIL,
            summary=f"Whole-path agent capacity is blocked: {exc}",
            remedy=(
                "Resolve the exact project/tenant/region or quota diagnostic, then "
                "rerun `npa agent preflight`."
            ),
        )
    topology = plan.topology
    quota_scope = next(
        (item for item in plan.checks if item.name == _QUOTA_SCOPE_CHECK), None
    )
    if quota_scope and quota_scope.reason == _TENANT_QUOTA_RBAC_FALLBACK:
        return CheckResult(
            name="whole_path_capacity",
            status=WARN,
            summary=(
                "Project-scoped whole-path capacity is ready; tenant-wide quota "
                "visibility is unavailable due to RBAC."
            ),
            remedy=(
                "No tenant-wide read grant is required for agent deploy. If apply "
                "reports a tenant capacity denial, ask a tenant administrator to "
                "review the named quota."
            ),
            details=(
                "Project allowances were verified. The provider will still enforce "
                "the tenant aggregate during apply.",
            ),
        )
    return CheckResult(
        name="whole_path_capacity",
        status=PASS,
        summary=(
            "Whole-path agent capacity is ready: "
            f"instances={topology.required_instances}, disks={topology.required_disks}, "
            f"network_ssd_bytes={topology.required_network_ssd_bytes}, "
            f"public_ips={topology.required_public_ips}."
        ),
    )


def _agent_public_ip_quota_result() -> "CheckResult":
    """Public-IPv4 quota check (FAIL): deploy needs one free public IP.

    Mirrors the gate `deploy_cmd` enforces so an exhausted quota is caught in
    preflight rather than after provisioning starts. Best-effort: resolves the
    configured default project; skips (PASS with note) when the project/region
    or quota can't be resolved so preflight never false-fails.
    """
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        from npa.clients.config import default_project_name, list_projects

        projects = list_projects()
        alias = default_project_name()
        stanza = projects.get(alias) or (
            next(iter(projects.values())) if projects else {}
        )
        project_id = str((stanza or {}).get("project_id", "")).strip()
        tenant_id = str((stanza or {}).get("tenant_id", "")).strip()
        fallback_region = str((stanza or {}).get("region", "")).strip()
    except Exception:  # noqa: BLE001
        project_id = tenant_id = fallback_region = ""

    if not project_id or not tenant_id:
        return CheckResult(
            name="public_ipv4_quota",
            status=PASS,
            summary="Public IPv4 quota check skipped (no configured project).",
        )

    from npa.clients.nebius import get_project_region, get_public_ipv4_quota

    region = (get_project_region(project_id) or fallback_region).strip()
    if not region:
        return CheckResult(
            name="public_ipv4_quota",
            status=PASS,
            summary="Public IPv4 quota check skipped (region unresolved).",
        )
    usage, limit = get_public_ipv4_quota(tenant_id, region)
    if usage is None or limit is None:
        return CheckResult(
            name="public_ipv4_quota",
            status=PASS,
            summary=f"Public IPv4 quota not readable for region {region!r}; skipping.",
        )
    if usage >= limit:
        return CheckResult(
            name="public_ipv4_quota",
            status=FAIL,
            summary=(
                f"Public IPv4 quota is exhausted in region {region} "
                f"({usage}/{limit}); agent deploy needs one public IP."
            ),
            remedy=(
                "Free a public IP (`npa agent destroy` an unused agent, or delete "
                "an idle VM/allocation), or ask a tenant admin to raise the "
                "vpc.ipv4-address.public.count quota, then re-run."
            ),
        )
    return CheckResult(
        name="public_ipv4_quota",
        status=PASS,
        summary=f"Public IPv4 quota OK in {region} ({usage}/{limit}).",
    )


def _agent_compute_instance_quota_result() -> "CheckResult":
    """Compute-instance quota check (FAIL): deploy needs one VM.

    Mirrors `_agent_public_ip_quota_result` for ``compute.instance.count``. This
    is the quota the audit reported at ``limit 0``: preflight passed on public
    IPv4 while the deploy then created the disk/network/SG and failed to attach a
    VM, rolling back. Best-effort: skips (PASS) when the project/region/quota
    can't be resolved so preflight never false-fails.
    """
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        from npa.clients.config import default_project_name, list_projects

        projects = list_projects()
        alias = default_project_name()
        stanza = projects.get(alias) or (
            next(iter(projects.values())) if projects else {}
        )
        project_id = str((stanza or {}).get("project_id", "")).strip()
        tenant_id = str((stanza or {}).get("tenant_id", "")).strip()
        fallback_region = str((stanza or {}).get("region", "")).strip()
    except Exception:  # noqa: BLE001
        project_id = tenant_id = fallback_region = ""

    if not project_id or not tenant_id:
        return CheckResult(
            name="compute_instance_quota",
            status=PASS,
            summary="Compute instance quota check skipped (no configured project).",
        )

    from npa.clients.nebius import get_compute_instance_quota, get_project_region

    region = (get_project_region(project_id) or fallback_region).strip()
    if not region:
        return CheckResult(
            name="compute_instance_quota",
            status=PASS,
            summary="Compute instance quota check skipped (region unresolved).",
        )
    usage, limit = get_compute_instance_quota(tenant_id, region)
    if usage is None or limit is None:
        return CheckResult(
            name="compute_instance_quota",
            status=PASS,
            summary=f"Compute instance quota not readable for region {region!r}; skipping.",
        )
    if usage >= limit:
        return CheckResult(
            name="compute_instance_quota",
            status=FAIL,
            summary=(
                f"Compute instance quota is exhausted in region {region} "
                f"({usage}/{limit}); agent deploy needs one VM."
            ),
            remedy=(
                "Free a VM (`npa agent destroy` an unused agent, or delete an idle "
                "instance), or ask a tenant admin to raise the "
                "compute.instance.count quota for this region, then re-run."
            ),
        )
    return CheckResult(
        name="compute_instance_quota",
        status=PASS,
        summary=f"Compute instance quota OK in {region} ({usage}/{limit}).",
    )


def _agent_check_public_ip_quota(
    project_id: str, tenant_id: str, fallback_region: str, *, agent_exists: bool = False
) -> None:
    """Fail fast when the deploy region has no public-IPv4 quota headroom.

    Placement follows the project's region (not ``--region``), so resolve the
    project's real region and check the tenant's per-region
    ``vpc.ipv4-address.public.count`` allowance. Best-effort: any unresolved
    region or unreadable quota is a no-op so a healthy deploy is never blocked.

    ``agent_exists`` skips the gate entirely: ``npa agent deploy`` is also the
    update path, and re-applying an agent that already holds its address needs no
    headroom. Without this, a fully-used allowance (used up by that very agent)
    made every re-deploy abort with advice to destroy the thing being updated.
    """
    if agent_exists:
        return
    from npa.clients.nebius import get_project_region, get_public_ipv4_quota

    region = (get_project_region(project_id) or (fallback_region or "").strip()).strip()
    if not region:
        return
    usage, limit = get_public_ipv4_quota(tenant_id, region)
    if usage is None or limit is None:
        return
    if usage >= limit:
        # Imported lazily to avoid a circular import at module load
        # (npa.cli.agent imports this module).
        from npa.cli.agent import _fail

        _fail(
            f"Nebius public IPv4 quota is exhausted in region {region!r} "
            f"(in use {usage}/{limit}); the agent VM needs one public IP. "
            "Free a public IP (e.g. `npa agent destroy` an unused agent, or delete "
            "an idle VM/allocation), or ask a tenant admin to raise the "
            "vpc.ipv4-address.public.count quota for this region, then re-run."
        )


def _agent_check_compute_instance_quota(
    project_id: str, tenant_id: str, fallback_region: str, *, agent_exists: bool = False
) -> None:
    """Fail fast when the deploy region has no compute-instance quota headroom.

    The audit's tenant had ``compute.instance.count`` at ``limit 0``: preflight
    passed on public IPv4, then deploy created the disk/network/SG and the VM
    create failed, rolling everything back. Gate on it before any Terraform side
    effect, mirroring `_agent_check_public_ip_quota`. Best-effort (unreadable
    quota is a no-op); ``agent_exists`` skips it (a re-deploy reuses the VM).
    """
    if agent_exists:
        return
    from npa.clients.nebius import get_compute_instance_quota, get_project_region

    region = (get_project_region(project_id) or (fallback_region or "").strip()).strip()
    if not region:
        return
    usage, limit = get_compute_instance_quota(tenant_id, region)
    if usage is None or limit is None:
        return
    if usage >= limit:
        from npa.cli.agent import _fail

        _fail(
            f"Nebius compute instance quota is exhausted in region {region!r} "
            f"(in use {usage}/{limit}); the agent VM needs one instance. "
            "Free a VM (e.g. `npa agent destroy` an unused agent, or delete an "
            "idle instance), or ask a tenant admin to raise the "
            "compute.instance.count quota for this region, then re-run."
        )
