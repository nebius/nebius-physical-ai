"""Deploy, reconcile, destroy, and repair npa-managed Soperator clusters.

NPA wraps an immutable, runtime-asserted ``nebius-solutions-library`` Soperator
Terraform contract. It applies the monitoring prerequisites, stalled-dashboard
repair, ``ncclInspectorPreConf`` CRD compatibility patch, prefixed Slurm scripts
configmap, Ubuntu user-namespace configuration, best-effort worker recovery,
and direct creation-time CUDA checks needed by that contract. REST is explicit
at the NPA surface, while runtime validation accounts for the pinned operator's
remaining REST/accounting limitation.

Uses backend-neutral subprocess helpers shared by cluster lifecycle implementations.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from npa.cluster_backends.process import (
    BackendCommandError,
    require_bin as _require_bin,
    run_capture as _run_capture,
    run_stream as _run_stream,
    terraform_env as _terraform_env,
)
from npa.clients.config import resolve_environment
from npa.soperator.spec import (
    DEFAULT_SOLUTIONS_LIBRARY_REF,
    DEFAULT_SLURM_OPERATOR_VERSION,
    SoperatorSpec,
    WorkerPoolSpec,
    validate_ssh_public_key_record,
)
from npa.soperator.tfvars import render_tfvars

_SOLUTIONS_LIBRARY_REPO = "https://github.com/nebius/nebius-solutions-library.git"
_IMMUTABLE_GIT_REF_RE = re.compile(r"^[0-9a-f]{40}$")
_PROMETHEUS_CRD_BASE = (
    "https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/"
    "v0.76.0/example/prometheus-operator-crd"
)
_PROMETHEUS_CRDS = (
    "monitoring.coreos.com_servicemonitors.yaml",
    "monitoring.coreos.com_podmonitors.yaml",
    "monitoring.coreos.com_probes.yaml",
)
_MONITORING_NAMESPACE = "monitoring-system"
_MONITORING_RELEASE_SUFFIX = "-monitoring-dashboards"
_ACTIVECHECKS_RELEASE_SUFFIX = "-soperator-activechecks"
DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS = 30 * 60
_GPU_CHECK_CLEANUP_TIMEOUT_SECONDS = 30
_SAFE_LOCAL_RECONCILIATION_REPLACEMENTS = {
    (
        "module.k8s.terraform_data.kubectl_cluster_context",
        "terraform.io/builtin/terraform",
    ),
    (
        "module.login_script.terraform_data.lb_service_ip",
        "terraform.io/builtin/terraform",
    ),
    (
        "module.login_script.local_file.this",
        "registry.terraform.io/hashicorp/local",
    ),
}

# Sidecar written next to the generated tfvars so ``destroy`` can rebuild the
# same TF_VAR_* env the recipe requires. region/tenant/project/subnet/o11y are
# passed as env vars at apply time (not persisted in terraform.tfvars), so a
# later ``terraform destroy`` would fail on "No value for required variable"
# without these.
_ENV_SIDECAR = ".npa-soperator-env.json"


class UpstreamContractError(ValueError):
    """Raised before provider mutation when the pinned recipe contract differs."""


class SolutionsLibraryReconciliationError(UpstreamContractError):
    """Raised when an existing source checkout cannot be reconciled safely."""


class ProviderReplacementPlanError(RuntimeError):
    """Raised when a deploy plan contains an unsafe destructive action."""

    def __init__(self, replacements: list[str]) -> None:
        self.replacements = list(replacements)
        super().__init__(
            "Terraform planned "
            f"{len(self.replacements)} provider or unexpected destructive action(s); "
            "refusing to apply. The saved installation identity, Terraform state, "
            "and live cluster were preserved"
        )


@dataclass(frozen=True)
class GuardedTerraformPlan:
    """One inspected plan plus its exact non-cloud local refresh replacements."""

    path: Path
    safe_local_replacements: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeploymentValidationFailure:
    """Public-safe details for a mandatory post-apply validation failure."""

    code: str
    message: str
    check: str
    pool: str | None = None
    phase: str | None = None
    cleanup_confirmed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "failed",
            "code": self.code,
            "check": self.check,
            "message": self.message,
        }
        if self.pool is not None:
            result["pool"] = self.pool
        if self.phase is not None:
            result["phase"] = self.phase
        if self.cleanup_confirmed is not None:
            result["cleanup_confirmed"] = self.cleanup_confirmed
        return result


class SoperatorDeploymentValidationError(RuntimeError):
    """A cluster was applied but a mandatory validation gate failed.

    Successful SDK calls keep returning the historical dictionary. Failed
    post-apply validation raises this typed exception and retains the same
    deployment metadata in :attr:`result`, clearly marked as degraded rather
    than deployed successfully.
    """

    def __init__(
        self,
        deployment: dict[str, Any],
        failure: DeploymentValidationFailure,
    ) -> None:
        self.deployment = dict(deployment)
        self.failure = failure
        self.result = {
            **self.deployment,
            "status": "degraded-validation",
            "deployment_status": "applied",
            "validation": failure.to_dict(),
        }
        super().__init__(failure.message)


class SoperatorStateCaptureError(RuntimeError):
    """Apply succeeded, but authoritative ownership could not be checkpointed."""

    def __init__(self, deployment: dict[str, Any], message: str) -> None:
        self.result = {
            **deployment,
            "status": "deployed-state-capture-failed",
            "deployment_status": "applied",
            "error": message,
            "recovery": (
                "Authoritative Terraform state and the pre-apply ownership sidecar "
                "were retained. Retry the identical deploy after restoring state "
                "readability; do not delete resources by name."
            ),
        }
        super().__init__(message)


class GPUCreationCheckError(RuntimeError):
    """Internal typed failure from the mandatory direct Slurm/CUDA gate."""

    def __init__(
        self,
        message: str,
        *,
        pool: str | None,
        phase: str,
        cleanup_confirmed: bool | None = None,
        completed_checks: list[dict[str, Any]] | None = None,
    ) -> None:
        self.pool = pool
        self.phase = phase
        self.cleanup_confirmed = cleanup_confirmed
        self.completed_checks = list(completed_checks or [])
        super().__init__(message)


@dataclass(frozen=True)
class ResolvedRootLoginSSHKey:
    """One validated public key that explicitly grants login-node root access."""

    value: str
    source: str
    fingerprint: str


def _write_env_sidecar(
    install_dir: Path,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
    o11y_profile: str,
    auth_profile: str = "",
    cluster_id: str = "",
    owned_filesystem_ids: list[str] | None = None,
    owned_allocation_ids: list[str] | None = None,
    cluster_name: str = "",
    provider_cluster_name: str = "",
    owned_auxiliary_resources: list[dict[str, str]] | None = None,
) -> None:
    path = install_dir / _ENV_SIDECAR
    temporary = install_dir / f".{_ENV_SIDECAR}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(
                {
                    "region": region,
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "subnet_id": subnet_id,
                    "o11y_profile": o11y_profile,
                    # Authentication and observability are independent Nebius
                    # profiles. Never use the telemetry profile to authenticate
                    # a later status or destroy operation.
                    "auth_profile": auth_profile,
                    "backend": "soperator",
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "provider_cluster_name": provider_cluster_name,
                    "owned_filesystem_ids": list(owned_filesystem_ids or []),
                    "owned_allocation_ids": list(owned_allocation_ids or []),
                    "owned_auxiliary_resources": list(owned_auxiliary_resources or []),
                },
                indent=2,
            )
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_env_sidecar(install_dir: Path) -> dict[str, Any] | None:
    path = install_dir / _ENV_SIDECAR
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"persisted Soperator installation identity at {path} is unreadable; "
            "refusing to fall back to an ambient project"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"persisted Soperator installation identity at {path} is not an object; "
            "refusing to fall back to an ambient project"
        )
    backend = str(data.get("backend") or "")
    if backend and backend != "soperator":
        raise ValueError(
            f"persisted installation at {path} belongs to backend {backend!r}; "
            "refusing Soperator lifecycle action"
        )
    return data


def _resolve_deploy_environment(
    spec: SoperatorSpec,
    recipe_dir: Path,
    *,
    project: str | None,
) -> tuple[str, str, str, str | None, dict[str, str] | None]:
    """Resolve deploy identity without drifting an existing Terraform install.

    The persisted apply-time sidecar is authoritative for omitted identity
    fields. Ambient/default project configuration is only a fallback for a new
    install. Re-resolving an existing install against a changed default project
    would otherwise produce a Terraform replacement plan for the whole cluster.
    """

    install_dir = recipe_dir / "installations" / spec.name
    saved = _load_env_sidecar(install_dir)
    explicit = {
        "region": spec.region,
        "tenant_id": spec.tenant_id,
        "project_id": spec.project_id,
        "subnet_id": spec.subnet_id,
    }
    if saved is not None:
        persisted = {
            field: str(saved.get(field) or "").strip()
            for field in ("region", "tenant_id", "project_id", "subnet_id")
        }
        missing = [field for field, value in persisted.items() if not value]
        if missing:
            raise ValueError(
                f"persisted identity for existing installation {spec.name!r} is "
                f"incomplete (missing {', '.join(missing)}); refusing to resolve "
                "replacement-sensitive fields from an ambient project"
            )
        for field, requested in explicit.items():
            if requested and requested != persisted[field]:
                raise ValueError(
                    f"existing installation {spec.name!r} is pinned to its persisted "
                    f"{field}; refusing a provider-replacement deploy. Restore the "
                    "original value or deploy a distinct cluster name"
                )
        if project is not None:
            configured = resolve_environment(project=project)
            for field in ("region", "tenant_id", "project_id"):
                requested = str(getattr(configured, field) or "")
                if requested and requested != persisted[field]:
                    raise ValueError(
                        f"project {project!r} does not match the persisted {field} for "
                        f"existing installation {spec.name!r}; refusing a provider-"
                        "replacement deploy"
                    )
        return (
            persisted["region"],
            persisted["tenant_id"],
            persisted["project_id"],
            persisted["subnet_id"],
            saved,
        )

    configured = resolve_environment(
        project=project,
        project_id=spec.project_id or None,
        tenant_id=spec.tenant_id or None,
        region=spec.region or None,
    )
    region = spec.region or configured.region
    tenant_id = spec.tenant_id or configured.tenant_id
    project_id = spec.project_id or configured.project_id
    if not (region and tenant_id and project_id):
        raise ValueError(
            "region, tenant_id and project_id must be resolvable from the spec, "
            "the existing installation sidecar, or ~/.npa config"
        )
    subnet_id = spec.subnet_id or None
    return region, tenant_id, project_id, subnet_id, None


def worker_capacity_summary(
    spec: SoperatorSpec, *, reservations_verified: bool = False
) -> list[dict[str, Any]]:
    """Return public-safe worker capacity metadata for plans/results/status."""

    return [
        {
            "name": pool.name,
            "platform": pool.platform,
            "preset": pool.preset,
            "size": pool.size,
            "capacity_mode": pool.capacity_mode(),
            "reservation_selector": pool.reservation_selector_kind() or None,
            "reservation_verified": (
                reservations_verified if pool.capacity_mode() == "reserved" else None
            ),
        }
        for pool in spec.workers
    ]


def plan_cluster(spec: SoperatorSpec) -> dict[str, Any]:
    """Return a provider-free public plan for one declarative Soperator spec."""

    spec.validate()
    return {
        "name": spec.name,
        "region": spec.region or "(resolve-at-deploy)",
        "control_plane": {
            "system_min_size": spec.system_min_size,
            "system_max_size": spec.effective_system_max_size(),
        },
        "workers": worker_capacity_summary(spec),
        "reservation_preflight": (
            "required"
            if any(pool.capacity_mode() == "reserved" for pool in spec.workers)
            else "not-required"
        ),
        "provider_mutation": False,
    }


def _status_installation_dir(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
) -> Path | None:
    """Find one existing installation without cloning or reconciling source."""

    if terraform_dir is not None:
        candidate = terraform_dir.expanduser().resolve() / "installations" / name
        return candidate if candidate.is_dir() else None
    root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    candidates = [
        root / "nebius-solutions-library" / "soperator" / "installations" / name,
        *sorted(
            root.glob(f"nebius-solutions-library-*/soperator/installations/{name}")
        ),
    ]
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) > 1:
        raise ValueError(
            f"multiple local installations exist for cluster {name!r}; pass "
            "--terraform-dir to select the authoritative one"
        )
    return existing[0] if existing else None


def worker_capacity_status(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Read applied worker capacity modes from local Terraform state."""

    install_dir = _status_installation_dir(
        name, terraform_dir=terraform_dir, work_root=work_root
    )
    if install_dir is None:
        return []
    state_path = install_dir / "terraform.tfstate"
    if not state_path.is_file():
        return []
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            "local Terraform state is unreadable; worker capacity mode is unknown"
        ) from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise ValueError(
            "local Terraform state has no valid resource inventory; worker "
            "capacity mode is unknown"
        )
    pools: dict[str, dict[str, Any]] = {}
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("type") != "nebius_mk8s_v1_node_group"
        ):
            continue
        if resource.get("name") != "worker_v2":
            continue
        for instance in resource.get("instances") or []:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            if not isinstance(attributes, dict):
                continue
            node_group_name = str(attributes.get("name") or "")
            pool_name = re.sub(r"-[0-9]+$", "", node_group_name) or node_group_name
            template = attributes.get("template") or {}
            policy = template.get("reservation_policy") or {}
            reservation_ids = (
                policy.get("reservation_ids") if isinstance(policy, dict) else []
            ) or []
            if (
                isinstance(policy, dict)
                and policy.get("policy") == "STRICT"
                and isinstance(reservation_ids, list)
                and reservation_ids
            ):
                mode = "reserved"
            elif isinstance(template, dict) and template.get("preemptible") is not None:
                mode = "preemptible"
            else:
                mode = "on-demand"
            existing = pools.get(pool_name)
            if existing and existing["capacity_mode"] != mode:
                raise ValueError(
                    f"worker pool {pool_name!r} has mixed applied capacity modes"
                )
            entry = existing or {
                "name": pool_name,
                "capacity_mode": mode,
                "node_groups": 0,
                "nodes": 0,
            }
            entry["node_groups"] += 1
            entry["nodes"] += int(attributes.get("fixed_node_count") or 0)
            pools[pool_name] = entry
    return [pools[key] for key in sorted(pools)]


def cluster_status(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Query the real Slurm controller and augment it with local capacity state."""

    context = f"nebius-{name}-slurm"
    kubectl = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
    proc = _run_capture(
        [
            kubectl,
            "--context",
            context,
            "exec",
            "-n",
            "soperator",
            "controller-0",
            "-c",
            "slurmctld",
            "--",
            "sinfo",
        ],
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"Could not query Slurm on {name!r}: {detail}")
    workers = worker_capacity_status(
        name, terraform_dir=terraform_dir, work_root=work_root
    )
    return {
        "name": name,
        "context": context,
        "sinfo": proc.stdout,
        "workers": workers,
        "capacity_status": "applied" if workers else "unknown",
        "status": "running",
    }


def _provider_json(
    command: list[str],
    *,
    env: dict[str, str],
    description: str,
) -> dict[str, Any]:
    """Run one read-only provider query and require an object-shaped response."""

    try:
        completed = _run_capture(
            command,
            env=env,
            check=False,
            timeout=120,
        )
    except (BackendCommandError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            f"{description} timed out; refusing provider mutation"
        ) from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"{description} failed; refusing provider mutation")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{description} returned invalid JSON; refusing provider mutation"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{description} returned an incompatible response; refusing provider mutation"
        )
    return payload


def _current_reserved_gpu_usage(install_dir: Path) -> dict[tuple[str, str], int]:
    """Read already-applied STRICT GPU usage so deploy preflight is idempotent."""

    state_path = install_dir / "terraform.tfstate"
    if not state_path.is_file():
        return {}
    try:
        state = json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            "existing Terraform state is unreadable; reservation availability "
            "cannot be checked safely"
        ) from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise ValueError(
            "existing Terraform state has no valid resource inventory; reservation "
            "availability cannot be checked safely"
        )
    usage: dict[tuple[str, str], int] = {}
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("type") != "nebius_mk8s_v1_node_group"
        ):
            continue
        if resource.get("name") != "worker_v2":
            continue
        for instance in resource.get("instances") or []:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            if not isinstance(attributes, dict):
                continue
            template = attributes.get("template") or {}
            if not isinstance(template, dict):
                continue
            policy = template.get("reservation_policy") or {}
            if not isinstance(policy, dict) or policy.get("policy") != "STRICT":
                continue
            reservation_ids = policy.get("reservation_ids") or []
            if not isinstance(reservation_ids, list) or len(reservation_ids) != 1:
                raise ValueError(
                    "existing worker reservation policy is ambiguous; refusing to "
                    "guess available reserved capacity"
                )
            reservation_id = str(reservation_ids[0] or "").strip()
            preset = str((template.get("resources") or {}).get("preset") or "")
            try:
                node_count = int(attributes.get("fixed_node_count") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "existing reserved worker count is invalid; refusing provider mutation"
                ) from exc
            if not reservation_id or node_count < 0:
                raise ValueError(
                    "existing worker reservation policy is incomplete; refusing "
                    "provider mutation"
                )
            key = (reservation_id, preset)
            usage[key] = usage.get(key, 0) + (
                node_count * _gpu_count_from_preset(preset)
            )
    return usage


def _resolve_reserved_worker_capacity(
    spec: SoperatorSpec,
    *,
    install_dir: Path,
    nebius_bin: str,
    tenant_id: str,
    project_id: str,
    region: str,
    env: dict[str, str],
    on_status: Callable[[str], None] | None = None,
) -> tuple[SoperatorSpec, list[dict[str, Any]]]:
    """Resolve and verify every explicit STRICT capacity-block selector.

    The selected project is first verified to belong to the selected tenant and
    region. Group IDs/names are then resolved only within that tenant, checked
    for a unique active region/platform/fabric match, and capacity-checked after
    crediting already-applied STRICT workers from the same installation.
    """

    reserved_pools = [
        pool for pool in spec.workers if pool.capacity_mode() == "reserved"
    ]
    if not reserved_pools:
        return spec, worker_capacity_summary(spec)

    project = _provider_json(
        [nebius_bin, "iam", "project", "get", "--id", project_id, "--format", "json"],
        env=env,
        description="selected-project identity preflight",
    )
    project_metadata = project.get("metadata") or {}
    project_status = project.get("status") or {}
    project_spec = project.get("spec") or {}
    if (
        not isinstance(project_metadata, dict)
        or str(project_metadata.get("id") or "") != project_id
        or str(project_metadata.get("parent_id") or "") != tenant_id
    ):
        raise ValueError(
            "selected project does not belong to the selected tenant; refusing "
            "reserved-capacity provider mutation"
        )
    project_region = str(
        (project_status if isinstance(project_status, dict) else {}).get("region")
        or (project_spec if isinstance(project_spec, dict) else {}).get("region")
        or ""
    )
    if project_region != region:
        raise ValueError(
            "selected project region does not match the Soperator region; refusing "
            "reserved-capacity provider mutation"
        )

    inventory = _provider_json(
        [
            nebius_bin,
            "capacity",
            "capacity-block-group",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ],
        env=env,
        description="capacity-block inventory preflight",
    )
    items = inventory.get("items")
    if not isinstance(items, list):
        raise ValueError(
            "capacity-block inventory omitted a valid items list; refusing provider mutation"
        )
    from npa.cluster_backends.quotas import parse_capacity_blocks

    blocks = parse_capacity_blocks(json.dumps(inventory))
    advice_inventory = _provider_json(
        [
            nebius_bin,
            "capacity",
            "resource-advice",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ],
        env=env,
        description="reserved preset-availability preflight",
    )
    advice_items = advice_inventory.get("items")
    if not isinstance(advice_items, list):
        raise ValueError(
            "reserved preset-availability preflight omitted a valid items list; "
            "refusing provider mutation"
        )
    existing_usage = _current_reserved_gpu_usage(install_dir)
    resolved_workers: list[WorkerPoolSpec] = []
    requirements: dict[str, dict[str, Any]] = {}

    for pool in spec.workers:
        if pool.capacity_mode() != "reserved":
            resolved_workers.append(pool)
            continue
        selector_kind = pool.reservation_selector_kind()
        selector = (
            pool.capacity_block_group
            if selector_kind == "id"
            else pool.capacity_block_group_name
        )
        if selector_kind == "id":
            matches = [
                item
                for item in items
                if isinstance(item, dict)
                and str((item.get("metadata") or {}).get("id") or "") == selector
            ]
            if not matches:
                # Distinguish a wrong-tenant ID from a missing/unreadable one
                # without echoing the private selector in public diagnostics.
                try:
                    foreign = _provider_json(
                        [
                            nebius_bin,
                            "capacity",
                            "capacity-block-group",
                            "get",
                            "--id",
                            selector,
                            "--format",
                            "json",
                        ],
                        env=env,
                        description="capacity-block selector lookup",
                    )
                except ValueError:
                    foreign = {}
                foreign_parent = str(
                    (foreign.get("metadata") or {}).get("parent_id") or ""
                )
                reason = (
                    "belongs to another tenant"
                    if foreign_parent and foreign_parent != tenant_id
                    else "was not found in the selected tenant"
                )
                raise ValueError(
                    f"worker pool {pool.name!r} capacity-block ID {reason}; "
                    "refusing provider mutation"
                )
        else:
            matches = [
                item
                for item in items
                if isinstance(item, dict)
                and str((item.get("metadata") or {}).get("name") or "") == selector
            ]
            if not matches:
                raise ValueError(
                    f"worker pool {pool.name!r} capacity-block name was not found "
                    "in the selected tenant; refusing provider mutation"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"worker pool {pool.name!r} capacity-block name is ambiguous "
                    "in the selected tenant; use its immutable ID"
                )
        if len(matches) != 1:
            raise ValueError(
                f"worker pool {pool.name!r} capacity-block selector is ambiguous; "
                "refusing provider mutation"
            )
        metadata = matches[0].get("metadata") or {}
        reservation_id = str(metadata.get("id") or "")
        block = blocks.get(reservation_id)
        if not reservation_id or block is None:
            raise ValueError(
                f"worker pool {pool.name!r} capacity-block response is incomplete; "
                "refusing provider mutation"
            )
        if block["parent_id"] != tenant_id:
            raise ValueError(
                f"worker pool {pool.name!r} capacity block belongs to another tenant"
            )
        if block["state"] != "STATE_ACTIVE":
            raise ValueError(f"worker pool {pool.name!r} capacity block is not active")
        if block["region"] != region:
            raise ValueError(
                f"worker pool {pool.name!r} capacity-block region does not match {region}"
            )
        if block["platform"] != pool.platform:
            raise ValueError(
                f"worker pool {pool.name!r} capacity-block platform does not match "
                f"{pool.platform}"
            )
        if block["fabric"] != pool.fabric:
            raise ValueError(
                f"worker pool {pool.name!r} capacity-block fabric does not match "
                f"{pool.fabric}"
            )
        required_gpus = pool.size * _gpu_count_from_preset(pool.preset)
        advice_matches = [
            item
            for item in advice_items
            if isinstance(item, dict)
            and str(
                ((item.get("spec") or {}).get("compute_instance") or {}).get("platform")
                or ""
            )
            == pool.platform
            and str(
                (
                    (
                        ((item.get("spec") or {}).get("compute_instance") or {}).get(
                            "preset"
                        )
                        or {}
                    ).get("name")
                )
                or ""
            )
            == pool.preset
            and str((item.get("spec") or {}).get("region") or "") == region
            and str((item.get("spec") or {}).get("fabric") or "") == pool.fabric
        ]
        if len(advice_matches) != 1:
            raise ValueError(
                f"worker pool {pool.name!r} reserved preset availability is "
                "missing or ambiguous; refusing provider mutation"
            )
        reserved_advice = (advice_matches[0].get("status") or {}).get("reserved") or {}
        advice_is_fresh = (
            isinstance(reserved_advice, dict)
            and reserved_advice.get("data_state") == "DATA_STATE_FRESH"
            and reserved_advice.get("available") is not None
        )
        available_nodes: int | None = None
        if advice_is_fresh:
            try:
                available_nodes = int(reserved_advice["available"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"worker pool {pool.name!r} reserved preset availability is invalid; "
                    "refusing provider mutation"
                ) from exc
        previous = requirements.get(reservation_id)
        if previous and (
            previous["platform"] != pool.platform
            or previous["fabric"] != pool.fabric
            or previous["preset"] != pool.preset
        ):
            raise ValueError(
                "one capacity block cannot back incompatible Soperator worker pools"
            )
        requirement = previous or {
            "platform": pool.platform,
            "fabric": pool.fabric,
            "preset": pool.preset,
            "required_gpus": 0,
            "required_nodes": 0,
            "available_gpus": block["available_gpus"],
            "available_nodes": available_nodes,
            "pools": [],
        }
        requirement["required_gpus"] += required_gpus
        requirement["required_nodes"] += pool.size
        if requirement["available_nodes"] is None or available_nodes is None:
            requirement["available_nodes"] = None
        else:
            requirement["available_nodes"] = min(
                requirement["available_nodes"], available_nodes
            )
        requirement["pools"].append(pool.name)
        requirements[reservation_id] = requirement
        resolved_workers.append(
            replace(pool, resolved_capacity_block_group_id=reservation_id)
        )

    for reservation_id, requirement in requirements.items():
        available = requirement["available_gpus"]
        if available is None:
            raise ValueError(
                "reserved GPU availability is unavailable; refusing provider mutation"
            )
        already_applied = existing_usage.get((reservation_id, requirement["preset"]), 0)
        additional = max(0, requirement["required_gpus"] - already_applied)
        if available < additional:
            pools = ", ".join(sorted(requirement["pools"]))
            raise ValueError(
                f"reserved capacity is insufficient for worker pool(s) {pools}: "
                f"needs {additional} additional GPUs, only {available} are available; "
                "STRICT reservation-backed workers cannot fall back to on-demand "
                "or preemptible capacity"
            )
        gpus_per_node = _gpu_count_from_preset(requirement["preset"])
        already_applied_nodes = already_applied // gpus_per_node
        additional_nodes = max(0, requirement["required_nodes"] - already_applied_nodes)
        if additional_nodes and requirement["available_nodes"] is None:
            pools = ", ".join(sorted(requirement["pools"]))
            raise ValueError(
                f"worker pool(s) {pools} reserved preset availability is stale "
                "or unavailable; refusing provider mutation"
            )
        if additional_nodes and requirement["available_nodes"] < additional_nodes:
            pools = ", ".join(sorted(requirement["pools"]))
            raise ValueError(
                f"reserved preset capacity is insufficient for worker pool(s) "
                f"{pools}: needs {additional_nodes} additional node(s), only "
                f"{requirement['available_nodes']} are available for preset "
                f"{requirement['preset']}; STRICT reservation-backed workers "
                "cannot fall back to on-demand or preemptible capacity"
            )
    resolved = replace(spec, workers=resolved_workers)
    _log(
        on_status,
        f"reserved-capacity preflight passed: {len(reserved_pools)} worker pool(s), "
        f"{len(requirements)} active capacity block group(s)",
    )
    return resolved, worker_capacity_summary(resolved, reservations_verified=True)


def _log(on_status: Callable[[str], None] | None, message: str) -> None:
    if on_status is not None:
        on_status(message)


def _api_domain(region: str) -> str:
    """Nebius API domain for a region (the recipe hardcodes the EU domain)."""

    return (
        "api.eu.nebius.cloud:443" if region.startswith("eu") else "api.nebius.cloud:443"
    )


def _root_login_key_fingerprint(value: str) -> str:
    blob = base64.b64decode(value.split(maxsplit=2)[1], validate=True)
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def _read_root_login_key_file(path: Path, *, source: str) -> ResolvedRootLoginSSHKey:
    try:
        value = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"could not read {source} root-login SSH public-key file"
        ) from exc
    normalized = validate_ssh_public_key_record(value)
    return ResolvedRootLoginSSHKey(
        value=normalized,
        source=source,
        fingerprint=_root_login_key_fingerprint(normalized),
    )


def _resolve_root_login_ssh_public_key(
    spec: SoperatorSpec,
    *,
    explicit_file: Path | None = None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> ResolvedRootLoginSSHKey:
    """Resolve the one key granting root access to the public login node.

    Precedence is: explicit CLI/SDK file, canonical or legacy spec field,
    Soperator-specific inline environment value, Soperator-specific environment
    file, generic ``NPA_SSH_PUBLIC_KEY`` file, then conventional operator-home
    key discovery. Only a single OpenSSH public-key record is accepted.
    """

    if explicit_file is not None:
        return _read_root_login_key_file(explicit_file, source="explicit argument")

    explicit = spec.explicit_root_login_ssh_public_key()
    if explicit:
        normalized = validate_ssh_public_key_record(explicit)
        source = (
            "spec root_login_ssh_public_key"
            if spec.root_login_ssh_public_key
            else "legacy spec ssh_public_keys"
        )
        return ResolvedRootLoginSSHKey(
            value=normalized,
            source=source,
            fingerprint=_root_login_key_fingerprint(normalized),
        )

    env = os.environ if environ is None else environ
    inline = env.get("NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY", "").strip()
    if inline:
        normalized = validate_ssh_public_key_record(inline)
        return ResolvedRootLoginSSHKey(
            value=normalized,
            source="NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY",
            fingerprint=_root_login_key_fingerprint(normalized),
        )

    for env_name in (
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY_FILE",
        "NPA_SSH_PUBLIC_KEY",
    ):
        configured_path = env.get(env_name, "").strip()
        if configured_path:
            return _read_root_login_key_file(Path(configured_path), source=env_name)

    ssh_dir = (home or Path.home()).expanduser() / ".ssh"
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        candidate = ssh_dir / name
        if candidate.is_file():
            return _read_root_login_key_file(
                candidate, source=f"operator default {name}"
            )
    raise ValueError(
        "soperator login-node root access requires one SSH public key: set "
        "root_login_ssh_public_key in the spec, pass "
        "--root-login-ssh-public-key-file, set "
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY[_FILE], or create "
        "~/.ssh/id_ed25519.pub"
    )


def _validate_immutable_solutions_library_ref(ref: str) -> str:
    normalized = ref.strip().lower()
    if not _IMMUTABLE_GIT_REF_RE.fullmatch(normalized):
        raise ValueError(
            "solutions_library_ref must be an immutable 40-character commit SHA; "
            "branches and moving tags are not accepted"
        )
    return normalized


@contextmanager
def _solutions_library_lock(work_root: Path):
    """Serialize source reconciliation without locking Terraform operations."""

    work_root.mkdir(parents=True, exist_ok=True)
    lock_path = work_root / ".solutions-library.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except BaseException:
        # ``os.fdopen`` owns the descriptor once entered; close it only when an
        # exception happened before ownership transferred.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _git_capture(
    git: str,
    repo_root: Path,
    args: list[str],
    *,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git, "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _reconcile_existing_solutions_library_checkout(
    repo_root: Path,
    *,
    ref: str,
    git: str,
) -> None:
    """Move a clean legacy checkout to *ref* without touching untracked state."""

    inside = _git_capture(git, repo_root, ["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise SolutionsLibraryReconciliationError(
            f"existing solutions-library path {repo_root} is not a Git checkout; "
            "move it aside only after preserving its installations/Terraform state"
        )

    head = _git_capture(git, repo_root, ["rev-parse", "HEAD"])
    actual = head.stdout.strip().lower() if head.returncode == 0 else ""
    symbolic_ref = _git_capture(git, repo_root, ["symbolic-ref", "-q", "HEAD"])
    if actual == ref:
        if symbolic_ref.returncode == 0:
            detached = _git_capture(git, repo_root, ["checkout", "--detach", ref])
            if detached.returncode != 0:
                raise SolutionsLibraryReconciliationError(
                    "the solutions-library checkout is at the pinned commit but could "
                    "not detach from its moving branch; preserve its installation state, "
                    "resolve the reported Git conflict, and retry"
                )
        # The later contract assertion accepts only pristine source plus NPA's
        # exact idempotent patches, and rejects every other tracked mutation.
        return

    if symbolic_ref.returncode != 0 and actual:
        raise SolutionsLibraryReconciliationError(
            "existing solutions-library checkout is already detached at a different "
            f"immutable commit ({actual[:12]}); retry with that installation's ref or "
            "use a separate work root. The checkout and Terraform state were unchanged"
        )

    tracked = _git_capture(
        git,
        repo_root,
        ["status", "--porcelain", "--untracked-files=no"],
    )
    if tracked.returncode != 0:
        raise SolutionsLibraryReconciliationError(
            "could not inspect the existing solutions-library checkout; no files were changed"
        )
    if tracked.stdout.strip():
        raise SolutionsLibraryReconciliationError(
            "existing solutions-library checkout has tracked changes and cannot be migrated "
            "safely; commit, stash, or revert the tracked changes and retry. Untracked "
            "installations/Terraform state were preserved"
        )

    present = _git_capture(git, repo_root, ["cat-file", "-e", f"{ref}^{{commit}}"])
    if present.returncode != 0:
        fetched = _git_capture(git, repo_root, ["fetch", "--no-tags", "origin", ref])
        if fetched.returncode != 0:
            raise SolutionsLibraryReconciliationError(
                "the pinned solutions-library commit is missing from this checkout and "
                f"`git fetch origin {ref}` failed; restore network/remote access and retry. "
                "The checkout and all installation state were left unchanged"
            )
        present = _git_capture(git, repo_root, ["cat-file", "-e", f"{ref}^{{commit}}"])
        if present.returncode != 0:
            raise SolutionsLibraryReconciliationError(
                "the solutions-library fetch completed but the pinned commit is still "
                "unavailable; verify the origin remote and retry. No state was removed"
            )

    checked_out = _git_capture(git, repo_root, ["checkout", "--detach", ref])
    if checked_out.returncode != 0:
        raise SolutionsLibraryReconciliationError(
            "could not check out the pinned solutions-library commit, commonly because an "
            "untracked file conflicts with the pinned tree; move only the reported conflict "
            "aside and retry. Installations/Terraform state were not deleted"
        )
    verified = _git_capture(git, repo_root, ["rev-parse", "HEAD"])
    if verified.returncode != 0 or verified.stdout.strip().lower() != ref:
        raise SolutionsLibraryReconciliationError(
            "solutions-library checkout did not reach the requested immutable commit"
        )


def _clone_solutions_library_atomically(
    clone_dir: Path,
    *,
    ref: str,
    git: str,
) -> None:
    """Publish a complete checkout atomically; never expose a partial clone."""

    with tempfile.TemporaryDirectory(
        prefix=f".{clone_dir.name}.clone-",
        dir=clone_dir.parent,
    ) as temp_root:
        candidate = Path(temp_root) / "checkout"
        cloned = subprocess.run(
            [
                git,
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                _SOLUTIONS_LIBRARY_REPO,
                str(candidate),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False,
        )
        if cloned.returncode != 0:
            raise SolutionsLibraryReconciliationError(
                "could not clone the solutions library; restore GitHub/network access and retry"
            )
        present = _git_capture(git, candidate, ["cat-file", "-e", f"{ref}^{{commit}}"])
        if present.returncode != 0:
            fetched = _git_capture(
                git, candidate, ["fetch", "--no-tags", "origin", ref]
            )
            if fetched.returncode != 0:
                raise SolutionsLibraryReconciliationError(
                    "the fresh clone could not fetch the pinned solutions-library commit"
                )
        checked_out = _git_capture(git, candidate, ["checkout", "--detach", ref])
        if checked_out.returncode != 0:
            raise SolutionsLibraryReconciliationError(
                "the fresh clone could not check out the pinned solutions-library commit"
            )
        if not (candidate / "soperator" / "installations" / "example").is_dir():
            raise SolutionsLibraryReconciliationError(
                "the pinned solutions-library checkout is missing soperator/installations/example"
            )
        os.replace(candidate, clone_dir)


def _resolve_solutions_library(
    terraform_dir: Path | None, work_root: Path, ref: str
) -> Path:
    """Resolve and safely reconcile one immutable solutions-library commit."""

    ref = _validate_immutable_solutions_library_ref(ref)

    if terraform_dir is not None:
        path = terraform_dir.expanduser().resolve()
        if not (path / "installations" / "example").exists():
            raise ValueError(
                f"{path} is not a soperator recipe dir (missing installations/example)"
            )
        return path
    git = _require_bin("git")
    with _solutions_library_lock(work_root):
        # Preserve and migrate the historical default location in place so its
        # untracked installations, Terraform state, and operator files retain
        # both path identity and bytes.
        legacy_clone = work_root / "nebius-solutions-library"
        if legacy_clone.exists():
            _reconcile_existing_solutions_library_checkout(
                legacy_clone,
                ref=ref,
                git=git,
            )
            recipe = legacy_clone / "soperator"
            if not (recipe / "installations" / "example").is_dir():
                raise SolutionsLibraryReconciliationError(
                    "the reconciled legacy solutions-library checkout is missing "
                    "soperator/installations/example; no state was removed"
                )
            return recipe

        clone_dir = work_root / f"nebius-solutions-library-{ref[:12]}"
        if clone_dir.exists():
            _reconcile_existing_solutions_library_checkout(
                clone_dir,
                ref=ref,
                git=git,
            )
        else:
            _clone_solutions_library_atomically(clone_dir, ref=ref, git=git)
        recipe = clone_dir / "soperator"
        if not (recipe / "installations" / "example").is_dir():
            raise SolutionsLibraryReconciliationError(
                "the pinned solutions-library checkout is incomplete; no files were deleted"
            )
        return recipe


def _nebius_cli_env(profile: str = "") -> dict[str, str]:
    """Environment for direct ``nebius`` CLI calls (pre-flight / cleanup).

    A stale ambient ``NEBIUS_IAM_TOKEN`` (e.g. an expired cloud-env token left in
    the parent process) is used by the CLI in preference to the active profile's
    exec-plugin, so pre-flight calls like ``vpc subnet list`` fail Unauthenticated
    even though the profile can mint a fresh token. Drop it so the CLI falls back
    to the auto-refreshing profile credential -- unless the caller explicitly opts
    into reuse (NPA_REUSE_IAM_TOKEN, e.g. CI injecting a short-lived token).
    """

    env = os.environ.copy()
    if profile.strip():
        env["NEBIUS_PROFILE"] = profile.strip()
    elif env.get("NPA_NEBIUS_PROFILE", "").strip():
        env["NEBIUS_PROFILE"] = env["NPA_NEBIUS_PROFILE"].strip()
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not reuse:
        env.pop("NEBIUS_IAM_TOKEN", None)
    return env


def _resolve_subnet(
    nebius_bin: str,
    project_id: str,
    env: dict[str, str],
    *,
    timeout: int | float | None = None,
) -> str:
    result = _run_capture(
        [
            nebius_bin,
            "vpc",
            "subnet",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
        timeout=timeout,
    )
    payload = json.loads(result.stdout or "{}")
    items = payload.get("items") or []
    if not items:
        raise ValueError(f"no VPC subnet found in project {project_id}")
    return str(items[0].get("metadata", {}).get("id") or "")


_ESSENTIAL_HEALTHY_NODES_MARKER = "# npa: CPU-only clusters disable the GPU checks"
_ESSENTIAL_HEALTHY_NODES_OVERRIDE = (
    "      " + _ESSENTIAL_HEALTHY_NODES_MARKER + "\n"
    "      # that ensure-healthy-nodes dependsOn, so its creation run never fires\n"
    "      # and wait-for-active-checks (which gates the activechecks HelmRelease,\n"
    "      # and thus terraform apply) deadlocks. Skip it at creation time.\n"
    "      ensure-healthy-nodes = {\n"
    "        runAfterCreation = false\n"
    "      }\n"
)
_NODECONFIGURATOR_USERNS_MARKER = "# npa: allow Enroot user namespaces on Ubuntu hosts"
_NODECONFIGURATOR_USERNS_VALUES = (
    "              " + _NODECONFIGURATOR_USERNS_MARKER + "\n"
    "              # Preserve the chart's complete default init-container list; Helm\n"
    "              # replaces arrays rather than merging individual list entries.\n"
    "              initContainers:\n"
    "                - name: node-sysctl-params\n"
    "                  image: cr.eu-north1.nebius.cloud/soperator/busybox\n"
    "                  securityContext:\n"
    "                    privileged: true\n"
    "                    runAsUser: 0\n"
    "                    runAsGroup: 0\n"
    "                    readOnlyRootFilesystem: false\n"
    "                    allowPrivilegeEscalation: true\n"
    "                  command:\n"
    "                    - /bin/sh\n"
    "                    - -c\n"
    "                    - |-\n"
    "                      sysctl -w kernel.unprivileged_userns_clone=1\n"
    '                      if [ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ] && [ "${apparmor_enabled}" = "false" ]; then\n'
    "                        sysctl -w kernel.apparmor_restrict_unprivileged_userns=0\n"
    "                      fi\n"
    "                      sysctl -w net.core.rmem_max=536870912\n"
    "                      sysctl -w net.core.wmem_max=536870912\n"
    '                      sysctl -w net.ipv4.tcp_rmem="4096 131072 536870912"\n'
    '                      sysctl -w net.ipv4.tcp_wmem="4096 16384 536870912"\n'
)
_STABLE_KUBECTL_CONTEXT_MARKER = (
    "# npa: refresh credentials only when the cluster id changes"
)
_STABLE_LOGIN_IP_MARKER = (
    "# npa: regenerate the login script only when its public IP changes"
)


def _patch_kubectl_context_trigger_text(text: str) -> tuple[str, bool]:
    """Remove the pinned recipe's perpetual timestamp replacement trigger."""

    if _STABLE_KUBECTL_CONTEXT_MARKER in text:
        return text, False
    original = (
        "  triggers_replace = [\n"
        "    nebius_mk8s_v1_cluster.this.id,\n"
        "    timestamp(),\n"
        "  ]\n"
    )
    if text.count(original) != 1:
        raise UpstreamContractError(
            "pinned kubectl context trigger does not match the verified timestamp contract"
        )
    replacement = (
        "  " + _STABLE_KUBECTL_CONTEXT_MARKER + "\n"
        "  triggers_replace = [\n"
        "    nebius_mk8s_v1_cluster.this.id,\n"
        "  ]\n"
    )
    return text.replace(original, replacement, 1), True


def _patch_login_ip_trigger_text(text: str) -> tuple[str, bool]:
    """Key the local login script refresh to its IP, not Service metadata churn."""

    if _STABLE_LOGIN_IP_MARKER in text:
        return text, False
    original = (
        "  triggers_replace = [\n"
        "    one(data.kubernetes_service_v1.slurm_login.metadata).resource_version\n"
        "  ]\n"
    )
    if text.count(original) != 1:
        raise UpstreamContractError(
            "pinned login-script trigger does not match the verified Service contract"
        )
    replacement = (
        "  " + _STABLE_LOGIN_IP_MARKER + "\n"
        "  triggers_replace = [\n"
        "    one(one(one(data.kubernetes_service_v1.slurm_login.status).load_balancer).ingress).ip\n"
        "  ]\n"
    )
    return text.replace(original, replacement, 1), True


def _patch_stable_local_reconciliation_triggers(recipe_dir: Path) -> bool:
    """Stabilize two local-only resources that otherwise replace every reconcile."""

    targets = (
        (
            recipe_dir / "modules" / "k8s" / "k8s_cluster.tf",
            _patch_kubectl_context_trigger_text,
        ),
        (
            recipe_dir / "modules" / "login" / "main.tf",
            _patch_login_ip_trigger_text,
        ),
    )
    staged: list[tuple[Path, str]] = []
    for path, transform in targets:
        if not path.is_file():
            raise UpstreamContractError(
                f"missing pinned local reconciliation target {path}"
            )
        patched, target_changed = transform(path.read_text())
        if target_changed:
            staged.append((path, patched))
    for path, patched in staged:
        path.write_text(patched)
    return bool(staged)


def _patch_active_checks_text(text: str) -> tuple[str, bool]:
    if _ESSENTIAL_HEALTHY_NODES_MARKER in text:
        return text, False
    marker = "    essential = {\n"
    idx = text.find(marker)
    if idx == -1:
        raise UpstreamContractError(
            "pinned active-checks contract lacks the essential scope"
        )
    insert_at = idx + len(marker)
    return (
        text[:insert_at] + _ESSENTIAL_HEALTHY_NODES_OVERRIDE + text[insert_at:],
        True,
    )


def _patch_active_checks_locals(recipe_dir: Path) -> bool:
    """Ensure the ``essential`` active-checks scope skips ``ensure-healthy-nodes``.

    On a CPU-only cluster npa selects the ``essential`` scope, which sets
    ``runAfterCreation = false`` on every GPU/NCCL/IB/perf check. But
    ``ensure-healthy-nodes`` (a slurmJob check) ``dependsOn`` those very checks,
    so with them disabled its creation run never triggers and its status stays
    empty. ``wait-for-active-checks`` -- the Helm hook that gates the activechecks
    HelmRelease, and therefore ``terraform apply`` -- waits for every
    ``runAfterCreation = true`` check to reach a terminal state, so it hangs until
    the 2h Helm timeout. Add ``ensure-healthy-nodes = { runAfterCreation = false }``
    to the ``essential`` scope (mirroring the recipe's own gb300 handling) so the
    hook no longer waits on it. Idempotent; returns True if a patch was applied.
    """

    locals_tf = recipe_dir / "modules" / "slurm" / "locals_active_checks.tf"
    if not locals_tf.exists():
        return False
    text = locals_tf.read_text()
    patched, changed = _patch_active_checks_text(text)
    if changed:
        locals_tf.write_text(patched)
    return changed


def _yaml_mapping_block_bounds(text: str, key: str) -> tuple[int, int, int]:
    """Return byte bounds and indentation for one YAML mapping block.

    The Terraform template is YAML with interpolation expressions; parsing it as
    ordinary YAML is unreliable. Indentation is nevertheless structural, so a
    bounded mapping walk prevents a missing child from matching a later chart.
    """

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    matches: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        offsets.append(offset)
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped.rstrip("\r\n") == f"{key}:":
            matches.append((index, indent))
        offset += len(line)
    if len(matches) != 1:
        raise UpstreamContractError(
            f"pinned Helm template must contain exactly one {key!r} mapping; "
            f"found {len(matches)}"
        )
    start_index, indent = matches[0]
    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%{"):
            continue
        candidate = lines[index].lstrip(" ")
        candidate_indent = len(lines[index]) - len(candidate)
        if candidate_indent <= indent:
            end_index = index
            break
    start = offsets[start_index]
    end = offsets[end_index] if end_index < len(offsets) else len(text)
    return start, end, indent


def _patch_nodeconfigurator_text(text: str) -> tuple[str, bool]:
    section_start, section_end, section_indent = _yaml_mapping_block_bounds(
        text, "nodeConfigurator"
    )
    section = text[section_start:section_end]
    marker_inside = _NODECONFIGURATOR_USERNS_MARKER in section
    marker_anywhere = _NODECONFIGURATOR_USERNS_MARKER in text
    if marker_anywhere and not marker_inside:
        raise UpstreamContractError(
            "nodeConfigurator user-namespace marker exists outside its chart block"
        )
    if marker_inside:
        return text, False

    values_line = " " * (section_indent + 2) + "values:\n"
    matches = [match.start() for match in re.finditer(re.escape(values_line), section)]
    if len(matches) != 1:
        raise UpstreamContractError(
            "pinned nodeConfigurator block must contain its own values: mapping; "
            "refusing to inject into a sibling chart"
        )
    insert_at = section_start + matches[0] + len(values_line)
    return (
        text[:insert_at] + _NODECONFIGURATOR_USERNS_VALUES + text[insert_at:],
        True,
    )


def _patch_nodeconfigurator_userns(recipe_dir: Path) -> bool:
    """Teach the upstream node configurator about Ubuntu's AppArmor userns gate.

    The verified chart enables ``kernel.unprivileged_userns_clone`` but Ubuntu's
    newer host images independently deny unprivileged user namespaces through
    ``kernel.apparmor_restrict_unprivileged_userns``. Enroot/Pyxis image startup
    then fails even though the worker container itself is AppArmor-unconfined.
    Override the chart's full init-container list and disable the second gate
    only when Soperator's default AppArmor profile is intentionally disabled.
    """

    template = (
        recipe_dir
        / "modules"
        / "slurm"
        / "templates"
        / "helm_values"
        / "terraform_fluxcd_values.yaml.tftpl"
    )
    if not template.exists():
        return False
    text = template.read_text()
    patched, changed = _patch_nodeconfigurator_text(text)
    if changed:
        template.write_text(patched)
    return changed


def _git_checkout_text(repo_root: Path, ref: str, relative_path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{relative_path}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise UpstreamContractError(
            f"could not read {relative_path} from solutions-library {ref[:12]}"
        )
    return proc.stdout


def _assert_solutions_library_contract(
    recipe_dir: Path,
    *,
    ref: str,
) -> None:
    """Assert the complete pinned source/mutation contract before cloud writes."""

    ref = _validate_immutable_solutions_library_ref(ref)
    repo_root = recipe_dir.parent
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or head.stdout.strip().lower() != ref:
        actual = (
            head.stdout.strip()[:12] if head.returncode == 0 else "not-a-git-checkout"
        )
        raise UpstreamContractError(
            f"solutions-library checkout mismatch: expected {ref[:12]}, got {actual}"
        )

    critical_fragments = {
        "soperator/installations/example/terraform.tfvars": (
            f'slurm_operator_version = "{DEFAULT_SLURM_OPERATOR_VERSION}"',
            "k8s_version = 1.34",
            "node_group_version = 72",
        ),
        "soperator/installations/example/main.tf": (
            "rest_enabled                    = var.slurm_rest_enabled",
            "accounting_enabled              = var.accounting_enabled",
            "sizing_tier_override = module.sizing.sizing_tier",
        ),
        "soperator/installations/example/variables.tf": (
            'variable "slurm_rest_enabled"',
            'variable "accounting_enabled"',
            'variable "sizing_tier_override"',
        ),
        "soperator/modules/sizing_tier/main.tf": (
            'var.worker_count < 10 ? "XS"',
            'var.worker_count < 100 ? "S"',
            'var.worker_count < 500 ? "M"',
            'var.worker_count < 2000 ? "L" : "XL"',
            'L  = "32vcpu-128gb"',
            'XL = "64vcpu-256gb"',
        ),
    }
    for relative, fragments in critical_fragments.items():
        pristine = _git_checkout_text(repo_root, ref, relative)
        current_path = repo_root / relative
        try:
            current = current_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UpstreamContractError(
                f"missing pinned contract file {relative}"
            ) from exc
        if current != pristine:
            raise UpstreamContractError(
                f"unexpected local mutation in pinned contract file {relative}"
            )
        missing = [fragment for fragment in fragments if fragment not in current]
        if missing:
            raise UpstreamContractError(
                f"pinned runtime contract is incompatible in {relative}: "
                f"missing {missing[0]!r}"
            )

    patch_contracts = (
        (
            "soperator/modules/k8s/k8s_cluster.tf",
            _patch_kubectl_context_trigger_text,
        ),
        (
            "soperator/modules/login/main.tf",
            _patch_login_ip_trigger_text,
        ),
        (
            "soperator/modules/slurm/locals_active_checks.tf",
            _patch_active_checks_text,
        ),
        (
            "soperator/modules/slurm/templates/helm_values/"
            "terraform_fluxcd_values.yaml.tftpl",
            _patch_nodeconfigurator_text,
        ),
    )
    for relative, transform in patch_contracts:
        pristine = _git_checkout_text(repo_root, ref, relative)
        expected_patched, _ = transform(pristine)
        current_path = repo_root / relative
        try:
            current = current_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UpstreamContractError(
                f"missing pinned patch target {relative}"
            ) from exc
        if current not in (pristine, expected_patched):
            raise UpstreamContractError(
                f"unexpected local mutation in pinned patch target {relative}"
            )


def _prepare_installation(recipe_dir: Path, spec: SoperatorSpec, region: str) -> Path:
    """Create installations/<name> with the recipe files + generated tfvars."""

    _patch_active_checks_locals(recipe_dir)
    _patch_stable_local_reconciliation_triggers(recipe_dir)
    if not spec.use_default_apparmor_profile:
        _patch_nodeconfigurator_userns(recipe_dir)
    example = recipe_dir / "installations" / "example"
    install_dir = recipe_dir / "installations" / spec.name
    install_dir.mkdir(parents=True, exist_ok=True)
    for item in ("main.tf", "variables.tf", "terraform.tf", "driver_presets.tf"):
        src = example / item
        if src.exists():
            shutil.copy2(src, install_dir / item)
    assets = example / "assets"
    if assets.exists():
        shutil.copytree(assets, install_dir / "assets", dirs_exist_ok=True)

    # Patch the hardcoded provider domain for the target region.
    terraform_tf = install_dir / "terraform.tf"
    if terraform_tf.exists():
        text = terraform_tf.read_text()
        text = text.replace("api.eu.nebius.cloud:443", _api_domain(region))
        terraform_tf.write_text(text)

    (install_dir / "terraform.tfvars").write_text(render_tfvars(spec))
    return install_dir


def _soperator_tf_env(
    nebius_bin: str,
    *,
    region: str,
    tenant_id: str,
    project_id: str,
    subnet_id: str,
    profile: str = "",
    timeout: int | float | None = None,
) -> dict[str, str]:
    profile = profile.strip() or (
        os.environ.get("NPA_NEBIUS_PROFILE", "").strip()
        or os.environ.get("NEBIUS_PROFILE", "").strip()
    )
    terraform_env_kwargs: dict[str, Any] = {"profile": profile}
    if timeout is not None:
        terraform_env_kwargs["timeout"] = timeout
    env = _terraform_env(nebius_bin, **terraform_env_kwargs)
    if profile:
        # Terraform local-exec and kubeconfig generation invoke the bare CLI;
        # keep them on the same explicitly selected cross-tenant principal.
        env["NEBIUS_PROFILE"] = profile
    env["TF_VAR_region"] = region
    env["TF_VAR_iam_tenant_id"] = tenant_id
    env["TF_VAR_iam_project_id"] = project_id
    # o11y is disabled in tfvars, but the variables are required to parse.
    env["TF_VAR_o11y_iam_tenant_id"] = tenant_id
    env["TF_VAR_o11y_profile"] = profile or "default"
    env["TF_VAR_vpc_subnet_id"] = subnet_id
    return env


def _terraform_cluster_identity(
    terraform_bin: str,
    install_dir: Path,
    env: dict[str, str],
    *,
    timeout: int | float | None = None,
) -> tuple[str, str]:
    """Return the exact mk8s ID/name pair from authoritative Terraform state."""

    result = _run_capture(
        [terraform_bin, "state", "pull"],
        cwd=install_dir,
        env=env,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "", ""
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "", ""
    for resource in state.get("resources", []):
        if resource.get("type") != "nebius_mk8s_v1_cluster":
            continue
        for instance in resource.get("instances", []):
            attributes = instance.get("attributes", {})
            cid = attributes.get("id")
            if cid:
                return str(cid), str(attributes.get("name") or "")
    return "", ""


def _terraform_cluster_id(
    terraform_bin: str,
    install_dir: Path,
    env: dict[str, str],
    *,
    timeout: int | float | None = None,
) -> str:
    """Return the mk8s cluster ID from Terraform state (empty if not found)."""

    return _terraform_cluster_identity(
        terraform_bin, install_dir, env, timeout=timeout
    )[0]


def _terraform_cluster_name(
    terraform_bin: str,
    install_dir: Path,
    env: dict[str, str],
    *,
    timeout: int | float | None = None,
) -> str:
    """Return the provider mk8s name from Terraform state (empty if not found)."""

    return _terraform_cluster_identity(
        terraform_bin, install_dir, env, timeout=timeout
    )[1]


_OWNED_AUXILIARY_RESOURCE_TYPES = {
    "nebius_compute_v1_filesystem": "filesystem",
    # Verified against the pinned upstream recipe at
    # soperator/modules/k8s/k8s_ng_login.tf: the login LoadBalancer allocation
    # is a real nebius_vpc_v1_allocation Terraform resource.
    "nebius_vpc_v1_allocation": "allocation",
}


def _terraform_owned_auxiliary_resources(
    terraform_bin: str, install_dir: Path, env: dict[str, str]
) -> list[dict[str, str]]:
    """Snapshot typed IDs and exact names from authoritative Terraform state."""

    result = _run_capture(
        [terraform_bin, "state", "pull"], cwd=install_dir, env=env, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "applied Terraform state could not be read; exact auxiliary ownership "
            "was not persisted"
        )
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "applied Terraform state returned invalid JSON; exact auxiliary "
            "ownership was not persisted"
        ) from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if not isinstance(resources, list):
        raise RuntimeError(
            "applied Terraform state has no valid resource inventory; exact "
            "auxiliary ownership was not persisted"
        )
    owned: list[dict[str, str]] = []
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("mode", "managed") != "managed"
        ):
            continue
        provider_type = str(resource.get("type") or "")
        kind = _OWNED_AUXILIARY_RESOURCE_TYPES.get(provider_type)
        if kind is None:
            continue
        instances = resource.get("instances")
        if not isinstance(instances, list):
            raise RuntimeError(
                f"applied Terraform {kind} state is malformed; exact ownership "
                "was not persisted"
            )
        for instance in instances:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            resource_id = (
                str(attributes.get("id") or "") if isinstance(attributes, dict) else ""
            )
            resource_name = (
                str(attributes.get("name") or "")
                if isinstance(attributes, dict)
                else ""
            )
            if not resource_id or not resource_name:
                raise RuntimeError(
                    f"applied Terraform {kind} state is missing an exact ID/name; "
                    "exact ownership was not persisted"
                )
            owned.append(
                {
                    "kind": kind,
                    "provider_type": provider_type,
                    "id": resource_id,
                    "name": resource_name,
                }
            )
    return sorted(owned, key=lambda item: (item["kind"], item["name"], item["id"]))


def _terraform_owned_auxiliary_ids(
    terraform_bin: str, install_dir: Path, env: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Snapshot exact managed auxiliary IDs from the applied Terraform state.

    An unreadable or structurally ambiguous state is not durable ownership
    evidence. Refuse to report a successful deployment rather than writing an
    empty ownership set that would make later cleanup unverifiable.
    """

    records = _terraform_owned_auxiliary_resources(terraform_bin, install_dir, env)
    owned: dict[str, set[str]] = {"filesystem": set(), "allocation": set()}
    for record in records:
        owned[record["kind"]].add(record["id"])
    return sorted(owned["filesystem"]), sorted(owned["allocation"])


def _require_applied_cluster_id(
    terraform_bin: str, install_dir: Path, env: dict[str, str]
) -> str:
    cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
    if not cluster_id:
        raise RuntimeError(
            "applied Terraform state has no exact Managed Kubernetes cluster ID; "
            "refusing to overwrite durable ownership or report success"
        )
    return cluster_id


def _find_cluster_id_by_name(
    nebius_bin: str, project_id: str, cluster_name: str, env: dict[str, str]
) -> str:
    """Return the mk8s cluster id matching *cluster_name* (empty if none)."""

    result = _run_capture(
        [
            nebius_bin,
            "mk8s",
            "cluster",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return ""
    try:
        items = json.loads(result.stdout).get("items", [])
    except json.JSONDecodeError:
        return ""
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("name") == cluster_name and meta.get("id"):
            return str(meta["id"])
    return ""


def _is_not_found_result(result: Any) -> bool:
    text = f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".casefold()
    return "not found" in text or "not_found" in text or "does not exist" in text


def _deadline_timeout(deadline: float, maximum: int = 120) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("configured destroy deadline has expired")
    return min(float(maximum), remaining)


def _provider_inventory_ids(
    command: list[str],
    *,
    env: dict[str, str],
    description: str,
    deadline: float | None = None,
) -> set[str]:
    """Read a provider list response without treating ambiguity as absence."""

    timeout = 120 if deadline is None else _deadline_timeout(deadline)
    listed = _run_capture(command, env=env, check=False, timeout=timeout)
    if listed.returncode != 0 or not listed.stdout.strip():
        raise RuntimeError(f"{description} inventory could not be read")
    try:
        payload = json.loads(listed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{description} inventory returned invalid JSON") from exc
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError(f"{description} inventory has no valid items list")
    ids: set[str] = set()
    for item in items:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        resource_id = (
            str(metadata.get("id") or "") if isinstance(metadata, dict) else ""
        )
        if not resource_id:
            raise RuntimeError(
                f"{description} inventory contains an item without an ID"
            )
        ids.add(resource_id)
    return ids


def _provider_read_terminal(result: Any) -> bool:
    detail = (
        f"{getattr(result, 'stdout', '')} {getattr(result, 'stderr', '')}".casefold()
    )
    return any(
        marker in detail
        for marker in (
            "permission denied",
            "permissiondenied",
            "unauthenticated",
            "unauthorized",
            "forbidden",
            "invalid argument",
        )
    )


def _wait_for_provider_id_absence(
    *,
    command: list[str],
    env: dict[str, str],
    description: str,
    deadline: float,
) -> str | None:
    """Poll exact provider identity; retry transient reads until the deadline."""

    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = _run_capture(
                command,
                env=env,
                check=False,
                timeout=_deadline_timeout(deadline),
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            last_error = type(exc).__name__
            time.sleep(min(2, max(0, deadline - time.monotonic())))
            continue
        if _is_not_found_result(result):
            return None
        if result.returncode == 0 and result.stdout.strip():
            last_error = "resource is still present"
        elif _provider_read_terminal(result):
            return f"{description} absence check failed terminally"
        else:
            last_error = "provider read was transient or unreadable"
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return f"{description} absence was not confirmed before the configured deadline: {last_error}"


def _resolve_exact_cluster_presence(
    *,
    nebius_bin: str,
    cluster_id: str,
    cluster_name: str,
    project_id: str,
    env: dict[str, str],
    deadline: float,
) -> tuple[bool | None, str]:
    """Resolve an exact cluster as present/absent, retrying transient reads."""

    last_error = ""
    while time.monotonic() < deadline:
        try:
            result = _run_capture(
                [
                    nebius_bin,
                    "mk8s",
                    "cluster",
                    "get",
                    "--id",
                    cluster_id,
                    "--format",
                    "json",
                ],
                env=env,
                check=False,
                timeout=_deadline_timeout(deadline),
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            last_error = type(exc).__name__
            time.sleep(min(2, max(0, deadline - time.monotonic())))
            continue
        if _is_not_found_result(result):
            return False, ""
        if result.returncode != 0 or not result.stdout.strip():
            if _provider_read_terminal(result):
                return None, "exact Managed Kubernetes identity check failed terminally"
            last_error = "transient or unreadable provider response"
            time.sleep(min(2, max(0, deadline - time.monotonic())))
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None, "exact Managed Kubernetes identity returned invalid JSON"
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        parent_id = (
            str(metadata.get("parent_id") or metadata.get("parentId") or "")
            if isinstance(metadata, dict)
            else ""
        )
        if (
            not isinstance(metadata, dict)
            or str(metadata.get("id") or "") != cluster_id
            or str(metadata.get("name") or "") != cluster_name
            or parent_id != project_id
        ):
            return None, "exact Managed Kubernetes ownership evidence is ambiguous"
        return True, ""
    return (
        None,
        "exact Managed Kubernetes identity was not resolved before the configured "
        f"deadline: {last_error}",
    )


def _cleanup_owned_provider_ids(
    *,
    nebius_bin: str,
    project_id: str,
    env: dict[str, str],
    owned_ids: set[str],
    service: tuple[str, ...],
    description: str,
    on_status: Callable[[str], None] | None,
    deadline: float,
) -> list[str]:
    """Delete only persisted exact IDs and prove each one is absent."""

    if not owned_ids:
        return []
    list_command = [
        nebius_bin,
        *service,
        "list",
        "--parent-id",
        project_id,
        "--format",
        "json",
    ]
    try:
        inventory_ids = _provider_inventory_ids(
            list_command, env=env, description=description, deadline=deadline
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return [str(exc)]

    errors: list[str] = []
    for resource_id in sorted(owned_ids):
        present = resource_id in inventory_ids
        if not present:
            try:
                initial = _run_capture(
                    [
                        nebius_bin,
                        *service,
                        "get",
                        "--id",
                        resource_id,
                        "--format",
                        "json",
                    ],
                    env=env,
                    check=False,
                    timeout=_deadline_timeout(deadline),
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(
                    f"owned {description} {resource_id} presence check failed: "
                    f"{type(exc).__name__}"
                )
                continue
            if _is_not_found_result(initial):
                continue
            if initial.returncode != 0 or not initial.stdout.strip():
                errors.append(
                    f"owned {description} {resource_id} presence check was unreadable"
                )
                continue
            present = True
        if present:
            _log(on_status, f"deleting owned {description} {resource_id}")
            try:
                deleted = _run_capture(
                    [nebius_bin, *service, "delete", "--id", resource_id],
                    env=env,
                    check=False,
                    timeout=_deadline_timeout(deadline),
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                errors.append(
                    f"owned {description} {resource_id} delete failed: "
                    f"{type(exc).__name__}"
                )
                continue
            if deleted.returncode != 0 and not _is_not_found_result(deleted):
                errors.append(f"owned {description} {resource_id} delete failed")
                continue
        absence_error = _wait_for_provider_id_absence(
            command=[
                nebius_bin,
                *service,
                "get",
                "--id",
                resource_id,
                "--format",
                "json",
            ],
            env=env,
            description=f"owned {description} {resource_id}",
            deadline=deadline,
        )
        if absence_error:
            errors.append(absence_error)
    return errors


def _reconcile_recreated_auxiliary_resources(
    *,
    nebius_bin: str,
    project_id: str,
    cluster_name: str,
    env: dict[str, str],
    records: list[dict[str, str]],
    deadline: float,
    on_status: Callable[[str], None] | None,
) -> list[str]:
    """Delete uniquely proven same-name replacements created by CCM teardown races."""

    if not records:
        return []
    if not cluster_name:
        return [
            "persisted cluster name is missing; recreated auxiliary ownership "
            "cannot be proven"
        ]
    services = {
        "filesystem": ("compute", "filesystem"),
        "allocation": ("vpc", "allocation"),
    }
    errors: list[str] = []
    for kind in ("filesystem", "allocation"):
        expected = [record for record in records if record.get("kind") == kind]
        if not expected:
            continue
        service = services[kind]
        listed = _run_capture(
            [
                nebius_bin,
                *service,
                "list",
                "--parent-id",
                project_id,
                "--format",
                "json",
            ],
            env=env,
            check=False,
            timeout=_deadline_timeout(deadline),
        )
        if listed.returncode != 0 or not listed.stdout.strip():
            errors.append(f"{kind} name reconciliation inventory could not be read")
            continue
        try:
            payload = json.loads(listed.stdout)
        except json.JSONDecodeError:
            errors.append(f"{kind} name reconciliation inventory returned invalid JSON")
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            errors.append(
                f"{kind} name reconciliation inventory has no valid items list"
            )
            continue
        for record in expected:
            expected_type = next(
                (
                    provider_type
                    for provider_type, mapped_kind in _OWNED_AUXILIARY_RESOURCE_TYPES.items()
                    if mapped_kind == kind
                ),
                "",
            )
            if record.get("provider_type") != expected_type:
                errors.append(f"persisted {kind} provider type is invalid")
                continue
            exact_name = str(record.get("name") or "")
            # The exact Terraform-derived name must remain tied to this cluster;
            # never infer ownership from a prefix-only match.
            if not exact_name or cluster_name not in exact_name:
                errors.append(f"persisted {kind} exact name is not tied to the cluster")
                continue
            matches = []
            malformed = False
            for item in items:
                metadata = item.get("metadata") if isinstance(item, dict) else None
                if not isinstance(metadata, dict):
                    malformed = True
                    continue
                if str(metadata.get("name") or "") != exact_name:
                    continue
                candidate_id = str(metadata.get("id") or "")
                candidate_parent = str(
                    metadata.get("parent_id") or metadata.get("parentId") or ""
                )
                if not candidate_id or candidate_parent != project_id:
                    malformed = True
                    continue
                matches.append(candidate_id)
            if malformed:
                errors.append(
                    f"{kind} exact-name ownership evidence is malformed or cross-project"
                )
                continue
            if len(matches) > 1:
                errors.append(f"{kind} exact-name ownership evidence is ambiguous")
                continue
            if not matches:
                continue
            replacement_id = matches[0]
            _log(on_status, f"deleting recreated owned {kind} {replacement_id}")
            deleted = _run_capture(
                [nebius_bin, *service, "delete", "--id", replacement_id],
                env=env,
                check=False,
                timeout=_deadline_timeout(deadline),
            )
            if deleted.returncode != 0 and not _is_not_found_result(deleted):
                errors.append(f"recreated owned {kind} delete failed")
                continue
            absence_error = _wait_for_provider_id_absence(
                command=[
                    nebius_bin,
                    *service,
                    "get",
                    "--id",
                    replacement_id,
                    "--format",
                    "json",
                ],
                env=env,
                description=f"recreated owned {kind} {replacement_id}",
                deadline=deadline,
            )
            if absence_error:
                errors.append(absence_error)
    return errors


def _refresh_kube_credentials(
    nebius_bin: str,
    cluster_id: str,
    context: str,
    env: dict[str, str],
    *,
    timeout: int | float | None = None,
) -> None:
    """Write an admin kubeconfig context for the cluster (recipe writes a limited SA)."""

    argv = [nebius_bin]
    profile = (
        env.get("NPA_NEBIUS_PROFILE", "").strip()
        or env.get("NEBIUS_PROFILE", "").strip()
    )
    if profile:
        argv.extend(["--profile", profile])
    _run_capture(
        [
            *argv,
            "mk8s",
            "cluster",
            "get-credentials",
            "--id",
            cluster_id,
            "--external",
            "--force",
            "--context-name",
            context,
        ],
        env=env,
        check=False,
        timeout=timeout,
    )


def _install_monitoring_crds(
    kubectl_bin: str, context: str, *, on_status: Callable[[str], None] | None = None
) -> None:
    """Install prometheus-operator CRDs the soperator operator chart requires.

    The operator chart creates a ServiceMonitor unconditionally; with telemetry
    off the recipe never installs its CRD, so the operator HelmRelease cannot
    install. These must be present before the operator reconciles.

    kubectl runs with the ambient ``NEBIUS_IAM_TOKEN`` stripped (via
    ``_nebius_cli_env``): a stale token shadows the kubeconfig exec-plugin and
    makes the apply fail Unauthenticated. Each apply is retried, and the
    ServiceMonitor CRD is confirmed registered before returning -- swallowing a
    failure here otherwise surfaces only ~an hour later as an operator
    HelmRelease InstallFailed and a ``wait_for_slurm_cluster_hr`` timeout.
    """

    kube_env = _nebius_cli_env()
    _ensure_monitoring_namespace(kubectl_bin, context, env=kube_env)
    _log(
        on_status,
        "installing prometheus-operator CRDs (ServiceMonitor/PodMonitor/Probe)",
    )
    for crd in _PROMETHEUS_CRDS:
        last: subprocess.CompletedProcess[str] | None = None
        for _attempt in range(3):
            last = _run_capture(
                [
                    kubectl_bin,
                    "--context",
                    context,
                    "apply",
                    "--server-side",
                    "-f",
                    f"{_PROMETHEUS_CRD_BASE}/{crd}",
                ],
                env=kube_env,
                check=False,
            )
            if last.returncode == 0:
                break
            time.sleep(5)
        if last is None or last.returncode != 0:
            detail = (last.stderr or last.stdout).strip() if last else ""
            raise RuntimeError(
                f"failed to install prometheus-operator CRD {crd} after 3 attempts"
                + (f": {detail}" if detail else "")
            )
    # Confirm the ServiceMonitor CRD is actually registered: the operator chart
    # renders a ServiceMonitor and cannot install without it, so a no-op apply
    # (wrong context / swallowed auth error) must fail loudly here, not later.
    check = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "crd",
            "servicemonitors.monitoring.coreos.com",
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if check.returncode != 0 or not check.stdout.strip():
        detail = (check.stderr or check.stdout).strip()
        raise RuntimeError(
            "prometheus-operator ServiceMonitor CRD not present after install"
            + (f": {detail}" if detail else "")
        )
    reset = _reset_stalled_monitoring_releases(kubectl_bin, context, env=kube_env)
    if reset:
        _log(on_status, f"reset {reset} stalled monitoring HelmRelease(s)")


def _ensure_monitoring_namespace(
    kubectl_bin: str, context: str, *, env: dict[str, str] | None = None
) -> None:
    """Ensure the namespace required by the unconditional dashboards chart.

    The pinned Soperator contract still reconciles monitoring dashboards when observability is
    disabled, but that mode does not create ``monitoring-system``.  Creating the
    namespace is idempotent and lets Flux install the chart instead of leaving a
    permanently failed HelmRelease in an otherwise healthy cluster.
    """

    kube_env = env or _nebius_cli_env()
    get = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "namespace",
            _MONITORING_NAMESPACE,
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if get.returncode == 0 and get.stdout.strip():
        return
    create = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "create",
            "namespace",
            _MONITORING_NAMESPACE,
        ],
        env=kube_env,
        check=False,
    )
    if create.returncode == 0:
        return

    # Another reconciler may create the namespace between our read and write.
    # Confirm the desired end state before treating a failed create as fatal.
    confirm = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "namespace",
            _MONITORING_NAMESPACE,
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if confirm.returncode == 0 and confirm.stdout.strip():
        return

    detail = (create.stderr or create.stdout).strip()
    raise RuntimeError(
        f"failed to ensure {_MONITORING_NAMESPACE} namespace"
        + (f": {detail}" if detail else "")
    )


def _reset_stalled_monitoring_releases(
    kubectl_bin: str, context: str, *, env: dict[str, str] | None = None
) -> int:
    """Reset failed dashboards releases after repairing their prerequisites.

    Flux stops retrying a HelmRelease after its remediation budget is exhausted.
    A rerun of NPA can therefore repair ``monitoring-system`` and the CRDs yet
    remain blocked on the old failure. Flux's paired ``requestedAt``/``resetAt``
    annotations reset that counter. Clean installs have no HelmRelease at this
    point, and healthy releases are left untouched.
    """

    kube_env = env or _nebius_cli_env()
    listed = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "-n",
            "flux-system",
            "get",
            "helmreleases",
            "-o",
            "json",
        ],
        env=kube_env,
        check=False,
    )
    if listed.returncode != 0:
        raw_detail = (listed.stderr or listed.stdout).strip()
        detail = raw_detail.lower()
        if (
            "not found" in detail
            or "doesn't have a resource type" in detail
            or "the server could not find the requested resource" in detail
        ):
            return 0
        raise RuntimeError(
            "failed to inspect monitoring HelmReleases"
            + (f": {raw_detail}" if raw_detail else "")
        )

    try:
        payload = json.loads(listed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "failed to inspect monitoring HelmReleases: invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "failed to inspect monitoring HelmReleases: invalid JSON object"
        )

    reset = 0
    for item in payload.get("items") or []:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        if not name.endswith(_MONITORING_RELEASE_SUFFIX):
            continue
        conditions = (item.get("status") or {}).get("conditions") or []
        stalled = any(
            condition.get("status") == "True" and condition.get("type") == "Stalled"
            for condition in conditions
        )
        retries_exhausted = any(
            condition.get("type") == "Ready"
            and condition.get("status") == "False"
            and condition.get("reason") == "RetriesExceeded"
            for condition in conditions
        )
        if not (stalled or retries_exhausted):
            continue
        token = str(time.time_ns())
        annotated = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                "flux-system",
                "annotate",
                "helmrelease",
                name,
                f"reconcile.fluxcd.io/requestedAt={token}",
                f"reconcile.fluxcd.io/resetAt={token}",
                "--overwrite",
            ],
            env=kube_env,
            check=False,
        )
        if annotated.returncode != 0:
            detail = (annotated.stderr or annotated.stdout).strip()
            raise RuntimeError(
                f"failed to reset monitoring HelmRelease {name}"
                + (f": {detail}" if detail else "")
            )
        reset += 1
    return reset


def _abort_superseded_activechecks_upgrade(
    kubectl_bin: str,
    context: str,
    *,
    namespace: str = "soperator",
    env: dict[str, str] | None = None,
) -> list[str]:
    """Unblock a newer ActiveChecks generation from an older Helm action.

    An upgrade can remain in its generated ``wait-for-active-checks`` hook when
    an old REST-backed generation cannot observe Slurm job status. Flux cannot
    start an already-rendered newer generation until that action exits. Only
    when ``lastAttemptedGeneration`` is older than metadata.generation and the
    release is actively Progressing, delete the Helm-owned hook Job and request
    a reset/reconcile. Current-generation installs are never interrupted.
    """

    kube_env = env or _nebius_cli_env()
    listed = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "-n",
            "flux-system",
            "get",
            "helmreleases",
            "-o",
            "json",
        ],
        env=kube_env,
        check=False,
    )
    if listed.returncode != 0:
        return []
    try:
        items = json.loads(listed.stdout or "{}").get("items", [])
    except (AttributeError, json.JSONDecodeError):
        return []

    reset: list[str] = []
    for item in items:
        metadata = item.get("metadata") or {}
        status = item.get("status") or {}
        name = str(metadata.get("name") or "")
        generation = int(metadata.get("generation") or 0)
        attempted = int(status.get("lastAttemptedGeneration") or 0)
        progressing = any(
            condition.get("type") == "Reconciling"
            and condition.get("status") == "True"
            and condition.get("reason") == "Progressing"
            for condition in (status.get("conditions") or [])
        )
        if (
            not name.endswith(_ACTIVECHECKS_RELEASE_SUFFIX)
            or not progressing
            or attempted <= 0
            or attempted >= generation
        ):
            continue
        deleted = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                namespace,
                "delete",
                "job",
                "wait-for-active-checks",
                "--ignore-not-found=true",
                "--wait=false",
            ],
            env=kube_env,
            check=False,
        )
        if deleted.returncode != 0:
            continue
        token = str(time.time_ns())
        annotated = _run_capture(
            [
                kubectl_bin,
                "--context",
                context,
                "-n",
                "flux-system",
                "annotate",
                "helmrelease",
                name,
                f"reconcile.fluxcd.io/requestedAt={token}",
                f"reconcile.fluxcd.io/resetAt={token}",
                "--overwrite",
            ],
            env=kube_env,
            check=False,
        )
        if annotated.returncode == 0:
            reset.append(name)
    return reset


def _patch_slurmcluster_crd(kubectl_bin: str, context: str) -> bool:
    """Patch the SlurmCluster CRD to accept plugStackConfig.ncclInspectorPreConf.

    Idempotent. Returns True once the CRD exists and the patch is applied. The
    CRD is created by the operator, so this only succeeds after the operator
    installs -- callers should retry until it returns True.
    """

    kube_env = _nebius_cli_env()
    got = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "crd",
            "slurmclusters.slurm.nebius.ai",
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if got.returncode != 0 or not got.stdout.strip():
        return False
    patched = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "patch",
            "crd",
            "slurmclusters.slurm.nebius.ai",
            "--type=json",
            "-p",
            '[{"op":"add","path":"/spec/versions/0/schema/openAPIV3Schema/'
            "properties/spec/properties/plugStackConfig/"
            'x-kubernetes-preserve-unknown-fields","value":true}]',
        ],
        env=kube_env,
        check=False,
    )
    return patched.returncode == 0


def _ensure_scripts_configmap(kubectl_bin: str, context: str, namespace: str) -> bool:
    """Create the cluster-name-prefixed <ns>-slurm-scripts configmap.

    The nodesets chart mounts ``<ns>-slurm-scripts`` while the slurm-cluster
    chart creates the unprefixed ``slurm-scripts`` (a chart naming skew).
    Idempotent; returns True once the prefixed copy exists.
    """

    kube_env = _nebius_cli_env()
    target = f"{namespace}-slurm-scripts"
    exists = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "cm",
            target,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        env=kube_env,
        check=False,
    )
    if exists.returncode == 0 and exists.stdout.strip():
        return True
    src = _run_capture(
        [
            kubectl_bin,
            "--context",
            context,
            "get",
            "cm",
            "slurm-scripts",
            "-n",
            namespace,
            "-o",
            "json",
        ],
        env=kube_env,
        check=False,
    )
    if src.returncode != 0 or not src.stdout.strip():
        return False
    try:
        cm = json.loads(src.stdout)
    except json.JSONDecodeError:
        return False
    cm["metadata"] = {"name": target, "namespace": namespace}
    applied = subprocess.run(
        [kubectl_bin, "--context", context, "apply", "-f", "-"],
        input=json.dumps(cm),
        text=True,
        env=kube_env,
        check=False,
    )
    return applied.returncode == 0


def _mid_apply_fix_loop(
    kubectl_bin: str,
    context: str,
    name: str,
    *,
    namespace: str = "soperator",
    stop: "threading.Event | None" = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Apply mid-apply fixes while phase 2 blocks on the slurm-cluster HelmRelease.

    The operator creates the SlurmCluster CRD and the slurm-scripts configmap
    *during* phase 2, and the slurm-cluster / nodesets HelmReleases then block on
    the CRD patch + the prefixed configmap. Poll and apply both as soon as they
    appear so phase 2 can converge unattended.
    """

    crd_done = False
    cm_done = False
    logged_crd = False
    logged_cm = False
    while stop is None or not stop.is_set():
        if not crd_done and _patch_slurmcluster_crd(kubectl_bin, context):
            crd_done = True
            if not logged_crd:
                _log(
                    on_status,
                    "mid-apply: patched SlurmCluster CRD (ncclInspectorPreConf)",
                )
                logged_crd = True
        if not cm_done and _ensure_scripts_configmap(kubectl_bin, context, namespace):
            cm_done = True
            if not logged_cm:
                _log(
                    on_status, f"mid-apply: ensured {namespace}-slurm-scripts configmap"
                )
                logged_cm = True
        if crd_done and cm_done:
            return
        if stop is not None:
            stop.wait(15)
        else:
            time.sleep(15)


def _run_terraform_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    stream_output: bool,
) -> None:
    """Run Terraform while preserving a machine-readable caller's stdout."""

    if stream_output:
        _run_stream(args, cwd=cwd, env=env, timeout=timeout)
        return
    completed = _run_capture(
        args,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        from npa.clients.nebius import redact_nebius_output

        detail = redact_nebius_output(
            _tail_diagnostic(completed.stderr, completed.stdout)
        )
        raise RuntimeError(
            f"Terraform command failed ({completed.returncode}): {' '.join(args[:2])}"
            + (f": {detail}" if detail else "")
        )


@contextmanager
def _terraform_plan_without_unsafe_replacements(
    terraform_bin: str,
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    targets: tuple[str, ...] = (),
    on_status: Callable[[str], None] | None = None,
) -> Iterator[GuardedTerraformPlan]:
    """Yield a saved plan after rejecting provider/unexpected destruction.

    Applying the exact inspected plan closes the gap between a safety preflight
    and Terraform's provider mutation. Plan files can contain sensitive values,
    so they live in an owner-only temporary directory inside the installation
    and are removed on every exit path. The pinned recipe has three audited
    local-only refresh resources; their exact replacement addresses/providers
    are allowed once while every other replacement or pure delete fails closed.
    """

    phase = ", ".join(targets) if targets else "full deployment"
    _log(on_status, f"terraform replacement guard: planning {phase}")
    with tempfile.TemporaryDirectory(prefix=".npa-plan-", dir=cwd) as temporary:
        temporary_path = Path(temporary)
        temporary_path.chmod(0o700)
        plan_path = temporary_path / "deploy.tfplan"
        plan_command = [
            terraform_bin,
            "plan",
            "-input=false",
            "-detailed-exitcode",
            f"-out={plan_path}",
            *(f"-target={target}" for target in targets),
        ]
        try:
            planned = _run_capture(
                plan_command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                check=False,
            )
        except (BackendCommandError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Terraform replacement-guard plan timed out before provider mutation"
            ) from exc
        if planned.returncode not in (0, 2):
            from npa.clients.nebius import redact_nebius_output

            detail = redact_nebius_output(
                _tail_diagnostic(planned.stderr, planned.stdout)
            )
            raise RuntimeError(
                f"Terraform replacement-guard plan failed ({planned.returncode}) "
                "before provider mutation" + (f": {detail}" if detail else "")
            )
        if not plan_path.is_file():
            raise RuntimeError(
                "Terraform replacement-guard plan did not produce a saved plan; "
                "no provider mutation was attempted"
            )
        plan_path.chmod(0o600)

        try:
            shown = _run_capture(
                [terraform_bin, "show", "-json", str(plan_path)],
                cwd=cwd,
                env=env,
                timeout=timeout,
                check=False,
            )
        except (BackendCommandError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Terraform replacement-guard plan inspection timed out before "
                "provider mutation"
            ) from exc
        if shown.returncode != 0:
            from npa.clients.nebius import redact_nebius_output

            detail = redact_nebius_output(_tail_diagnostic(shown.stderr, shown.stdout))
            raise RuntimeError(
                "Terraform replacement-guard plan inspection failed before provider mutation"
                + (f": {detail}" if detail else "")
            )
        try:
            plan = json.loads(shown.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Terraform replacement-guard plan inspection returned invalid JSON; "
                "no provider mutation was attempted"
            ) from exc
        resource_changes = plan.get("resource_changes", [])
        if not isinstance(resource_changes, list):
            raise RuntimeError(
                "Terraform replacement-guard plan omitted a valid resource_changes list; "
                "no provider mutation was attempted"
            )
        safe_local_replacements: list[str] = []
        unsafe_destructive_actions: list[str] = []
        for change in resource_changes:
            if not isinstance(change, dict) or not isinstance(
                change.get("change"), dict
            ):
                continue
            actions = change["change"].get("actions")
            if not isinstance(actions, list) or "delete" not in actions:
                continue
            address = str(change.get("address") or "<unknown-resource>")
            provider = str(change.get("provider_name") or "<unknown-provider>")
            replacement = "create" in actions
            if (
                replacement
                and (address, provider) in _SAFE_LOCAL_RECONCILIATION_REPLACEMENTS
            ):
                safe_local_replacements.append(address)
            else:
                unsafe_destructive_actions.append(address)
        if unsafe_destructive_actions:
            raise ProviderReplacementPlanError(sorted(unsafe_destructive_actions))
        _log(
            on_status,
            "terraform replacement guard passed: 0 provider/unexpected destructive "
            f"actions; safe local refresh replacements={len(safe_local_replacements)}",
        )
        yield GuardedTerraformPlan(
            path=plan_path,
            safe_local_replacements=tuple(sorted(safe_local_replacements)),
        )


def deploy_cluster(
    spec: SoperatorSpec,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF,
    root_login_ssh_public_key_file: Path | None = None,
    project: str | None = None,
    timeout_minutes: int = 90,
    gpu_creation_check_timeout_seconds: int = DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS,
    apply_fixes: bool = True,
    source_preflight_only: bool = False,
    stream_terraform_output: bool = True,
    on_status: Callable[[str], None] | None = None,
    profile: str = "",
) -> dict[str, Any]:
    """Deploy or reconcile *spec* after pinned-contract and key preflight."""

    spec.validate()
    _validate_gpu_creation_check_timeout(gpu_creation_check_timeout_seconds)
    root_login_key = _resolve_root_login_ssh_public_key(
        spec, explicit_file=root_login_ssh_public_key_file
    )
    spec = replace(
        spec,
        root_login_ssh_public_key=root_login_key.value,
        ssh_public_keys=[],
    )
    spec.validate()
    _log(
        on_status,
        "login-node root SSH access enabled: "
        f"source={root_login_key.source}; fingerprint={root_login_key.fingerprint}",
    )
    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(
        terraform_dir, work_root, solutions_library_ref
    )
    _assert_solutions_library_contract(recipe_dir, ref=solutions_library_ref)
    _log(on_status, f"verified solutions-library contract {solutions_library_ref[:12]}")
    region, tenant_id, project_id, persisted_subnet_id, _saved = (
        _resolve_deploy_environment(spec, recipe_dir, project=project)
    )
    if source_preflight_only:
        result = {
            "name": spec.name,
            "region": region,
            "install_dir": str(recipe_dir / "installations" / spec.name),
            "kube_context": f"nebius-{spec.name}-slurm",
            "worker_pools": [pool.name for pool in spec.workers],
            "docker_cache_pools": [
                pool.name for pool in spec.workers if pool.docker_cache
            ],
            "control_plane": {
                "system_min_size": spec.system_min_size,
                "system_max_size": spec.effective_system_max_size(),
            },
            "workers": worker_capacity_summary(spec),
            "solutions_library_ref": solutions_library_ref,
            "status": "source-preflight-passed",
            "provider_mutation": False,
        }
        _log(
            on_status,
            f"deploy source preflight passed: {spec.name}; provider mutation disabled",
        )
        return result

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    install_dir = recipe_dir / "installations" / spec.name
    spec, capacity_workers = _resolve_reserved_worker_capacity(
        spec,
        install_dir=install_dir,
        nebius_bin=nebius_bin,
        tenant_id=tenant_id,
        project_id=project_id,
        region=region,
        env=_nebius_cli_env(profile),
        on_status=on_status,
    )
    install_dir = _prepare_installation(recipe_dir, spec, region)
    _log(on_status, f"Installation dir: {install_dir}")

    subnet_id = persisted_subnet_id or _resolve_subnet(
        nebius_bin, project_id, _nebius_cli_env(profile)
    )
    env = _soperator_tf_env(
        nebius_bin,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
        profile=profile,
    )
    sidecar_persisted = False

    def persist_identity_before_apply() -> None:
        """Checkpoint the guarded identity immediately before provider mutation."""

        nonlocal sidecar_persisted
        if sidecar_persisted:
            return
        _write_env_sidecar(
            install_dir,
            region=region,
            tenant_id=tenant_id,
            project_id=project_id,
            subnet_id=subnet_id,
            o11y_profile=env["TF_VAR_o11y_profile"],
            auth_profile=profile,
            cluster_id=str((_saved or {}).get("cluster_id") or ""),
            cluster_name=spec.name,
            provider_cluster_name=str(
                (_saved or {}).get("provider_cluster_name") or ""
            ),
            owned_filesystem_ids=list((_saved or {}).get("owned_filesystem_ids") or []),
            owned_allocation_ids=list((_saved or {}).get("owned_allocation_ids") or []),
            owned_auxiliary_resources=list(
                (_saved or {}).get("owned_auxiliary_resources") or []
            ),
        )
        sidecar_persisted = True

    context = f"nebius-{spec.name}-slurm"
    replacement_plans_checked = 0
    safe_local_replacements_applied = 0
    _log(on_status, "terraform init")
    _run_terraform_command(
        [terraform_bin, "init"],
        cwd=install_dir,
        env=env,
        timeout=900,
        stream_output=stream_terraform_output,
    )

    if apply_fixes:
        # Two-phase apply: the soperator operator HelmRelease is reconciled inside
        # the full apply and blocks on the prometheus ServiceMonitor CRD. With
        # telemetry off, that CRD is not installed by the recipe, so a single
        # apply times out waiting for the operator. Phase 1 brings up the mk8s
        # cluster + node groups (and writes the kube context); we then refresh
        # admin credentials and install the monitoring CRDs so the operator can
        # install cleanly in phase 2.
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        _log(on_status, "terraform apply (phase 1: k8s cluster + node groups)")
        with _terraform_plan_without_unsafe_replacements(
            terraform_bin,
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
            targets=("module.k8s",),
            on_status=on_status,
        ) as guarded_plan:
            replacement_plans_checked += 1
            safe_local_replacements_applied += len(guarded_plan.safe_local_replacements)
            persist_identity_before_apply()
            _run_terraform_command(
                [terraform_bin, "apply", str(guarded_plan.path)],
                cwd=install_dir,
                env=env,
                timeout=timeout_minutes * 60,
                stream_output=stream_terraform_output,
            )
        cluster_id = _terraform_cluster_id(terraform_bin, install_dir, env)
        if cluster_id:
            provider_cluster_name = _terraform_cluster_name(
                terraform_bin, install_dir, env
            )
            if not provider_cluster_name:
                raise RuntimeError(
                    "phase-1 Terraform state has no exact Managed Kubernetes "
                    "provider name; durable ownership was not promoted"
                )
            _write_env_sidecar(
                install_dir,
                region=region,
                tenant_id=tenant_id,
                project_id=project_id,
                subnet_id=subnet_id,
                o11y_profile=env["TF_VAR_o11y_profile"],
                auth_profile=profile,
                cluster_id=cluster_id,
                cluster_name=spec.name,
                provider_cluster_name=provider_cluster_name,
                owned_filesystem_ids=list(
                    (_saved or {}).get("owned_filesystem_ids") or []
                ),
                owned_allocation_ids=list(
                    (_saved or {}).get("owned_allocation_ids") or []
                ),
                owned_auxiliary_resources=list(
                    (_saved or {}).get("owned_auxiliary_resources") or []
                ),
            )
            _log(on_status, "refreshing kube admin credentials")
            _refresh_kube_credentials(nebius_bin, cluster_id, context, env)
        _log(on_status, "installing monitoring CRDs (before operator reconcile)")
        _install_monitoring_crds(kubectl_bin, context, on_status=on_status)
        _log(
            on_status,
            f"terraform apply (phase 2: operator + Slurm; {len(spec.workers)} worker pool(s))",
        )
        # The SlurmCluster CRD is created by the operator *during* phase 2, and the
        # slurm-cluster HelmRelease then blocks on it accepting
        # plugStackConfig.ncclInspectorPreConf (a chart/CRD skew); the nodesets
        # chart likewise mounts a cluster-name-prefixed slurm-scripts configmap.
        # Both must be fixed mid-apply, so run a concurrent fixer while phase 2
        # blocks on wait_for_slurm_cluster_hr.
        stop = threading.Event()
        fixer = threading.Thread(
            target=_mid_apply_fix_loop,
            args=(kubectl_bin, context, spec.name),
            kwargs={"stop": stop, "on_status": on_status},
            daemon=True,
        )
        with _terraform_plan_without_unsafe_replacements(
            terraform_bin,
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
            on_status=on_status,
        ) as guarded_plan:
            replacement_plans_checked += 1
            safe_local_replacements_applied += len(guarded_plan.safe_local_replacements)
            fixer.start()
            try:
                _run_terraform_command(
                    [terraform_bin, "apply", str(guarded_plan.path)],
                    cwd=install_dir,
                    env=env,
                    timeout=timeout_minutes * 60,
                    stream_output=stream_terraform_output,
                )
            finally:
                stop.set()
                fixer.join(timeout=10)
    else:
        _log(on_status, f"terraform apply ({len(spec.workers)} worker pool(s))")
        with _terraform_plan_without_unsafe_replacements(
            terraform_bin,
            cwd=install_dir,
            env=env,
            timeout=timeout_minutes * 60,
            on_status=on_status,
        ) as guarded_plan:
            replacement_plans_checked += 1
            safe_local_replacements_applied += len(guarded_plan.safe_local_replacements)
            persist_identity_before_apply()
            _run_terraform_command(
                [terraform_bin, "apply", str(guarded_plan.path)],
                cwd=install_dir,
                env=env,
                timeout=timeout_minutes * 60,
                stream_output=stream_terraform_output,
            )

    # The applied Terraform state is the ownership authority. Snapshot every
    # exact auxiliary ID and atomically promote the sidecar before any success
    # result or post-deploy validation can be returned.
    try:
        cluster_id = _require_applied_cluster_id(terraform_bin, install_dir, env)
        provider_cluster_name = _terraform_cluster_name(terraform_bin, install_dir, env)
        if not provider_cluster_name:
            raise RuntimeError(
                "applied Terraform state has no exact Managed Kubernetes provider "
                "name; refusing to overwrite durable ownership or report success"
            )
        owned_auxiliary_resources = _terraform_owned_auxiliary_resources(
            terraform_bin, install_dir, env
        )
    except RuntimeError as exc:
        raise SoperatorStateCaptureError(
            {
                "name": spec.name,
                "region": region,
                "project_id": project_id,
                "install_dir": str(install_dir),
                "kube_context": context,
                "worker_pools": [pool.name for pool in spec.workers],
            },
            str(exc),
        ) from exc
    owned_filesystem_ids = sorted(
        item["id"] for item in owned_auxiliary_resources if item["kind"] == "filesystem"
    )
    owned_allocation_ids = sorted(
        item["id"] for item in owned_auxiliary_resources if item["kind"] == "allocation"
    )
    _write_env_sidecar(
        install_dir,
        region=region,
        tenant_id=tenant_id,
        project_id=project_id,
        subnet_id=subnet_id,
        o11y_profile=env["TF_VAR_o11y_profile"],
        auth_profile=profile,
        cluster_id=cluster_id,
        cluster_name=spec.name,
        provider_cluster_name=provider_cluster_name,
        owned_filesystem_ids=owned_filesystem_ids,
        owned_allocation_ids=owned_allocation_ids,
        owned_auxiliary_resources=owned_auxiliary_resources,
    )

    result: dict[str, Any] = {
        "name": spec.name,
        "region": region,
        "project_id": project_id,
        "install_dir": str(install_dir),
        "kube_context": context,
        "worker_pools": [p.name for p in spec.workers],
        "docker_cache_pools": [p.name for p in spec.workers if p.docker_cache],
        "control_plane": {
            "system_min_size": spec.system_min_size,
            "system_max_size": spec.effective_system_max_size(),
        },
        "workers": capacity_workers,
        "replacement_guard": {
            "status": "passed",
            "plans_checked": replacement_plans_checked,
            "replacement_count": 0,
            "safe_local_replacements_applied": safe_local_replacements_applied,
        },
    }

    if apply_fixes:
        kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        warnings = apply_post_deploy_fixes(context, kubectl_bin, on_status=on_status)
        result["post_deploy_fixes"] = "applied"
        result["post_deploy_fix_warnings"] = warnings

    # GPU validation is a deploy contract, not an optional repair. Keep it
    # active even when an operator deliberately selects --skip-fixes.
    if any(pool.is_gpu() for pool in spec.workers):
        try:
            kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
            result["gpu_creation_checks"] = _run_gpu_creation_checks(
                spec,
                context,
                kubectl_bin,
                timeout_seconds=gpu_creation_check_timeout_seconds,
                on_status=on_status,
            )
        except GPUCreationCheckError as exc:
            result["gpu_creation_checks"] = exc.completed_checks
            failure = DeploymentValidationFailure(
                code="gpu_creation_check_failed",
                message=str(exc),
                check="gpu-creation",
                pool=exc.pool,
                phase=exc.phase,
                cleanup_confirmed=exc.cleanup_confirmed,
            )
            raise SoperatorDeploymentValidationError(result, failure) from exc
        except Exception as exc:
            result["gpu_creation_checks"] = []
            failure = DeploymentValidationFailure(
                code="gpu_creation_check_failed",
                message=f"GPU creation-check preflight failed: {exc}",
                check="gpu-creation",
                phase="preflight",
            )
            raise SoperatorDeploymentValidationError(result, failure) from exc
    else:
        result["gpu_creation_checks"] = []

    result["status"] = "ready"
    result["deployment_status"] = "applied"
    result["validation"] = {
        "status": "passed",
        "check": "gpu-creation",
    }

    return result


def destroy_cluster(
    name: str,
    *,
    terraform_dir: Path | None = None,
    work_root: Path | None = None,
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF,
    project: str | None = None,
    timeout_minutes: int = 90,
    source_preflight_only: bool = False,
    on_status: Callable[[str], None] | None = None,
    profile: str = "",
) -> dict[str, Any] | None:
    """Destroy an npa-managed soperator cluster by name."""

    work_root = (work_root or Path.home() / ".npa" / "soperator").expanduser()
    recipe_dir = _resolve_solutions_library(
        terraform_dir, work_root, solutions_library_ref
    )
    _assert_solutions_library_contract(recipe_dir, ref=solutions_library_ref)
    install_dir = recipe_dir / "installations" / name
    if not install_dir.exists():
        raise ValueError(f"no installation found for cluster {name!r} at {install_dir}")
    if source_preflight_only:
        result = {
            "name": name,
            "status": "source-preflight-passed",
            "install_dir": str(install_dir),
            "solutions_library_ref": solutions_library_ref,
            "provider_mutation": False,
        }
        _log(
            on_status,
            f"destroy source preflight passed: {name}; provider mutation disabled",
        )
        return result

    if timeout_minutes <= 0:
        raise ValueError("soperator destroy timeout must be positive")
    destroy_deadline = time.monotonic() + timeout_minutes * 60

    def remaining(phase: str, maximum: int | None = None) -> int:
        seconds = int(destroy_deadline - time.monotonic())
        if seconds < 1:
            raise RuntimeError(
                f"Soperator destroy deadline exhausted during {phase}; exact "
                "ownership state was retained for retry"
            )
        return min(seconds, maximum) if maximum is not None else seconds

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")

    # ``terraform destroy`` still parses the config, so the region/tenant/project/
    # subnet/o11y variables (passed as env at apply time, never written to
    # terraform.tfvars) must be set or destroy fails on "No value for required
    # variable". A complete sidecar written at deploy time is authoritative.
    # A missing sidecar fails closed; an older partial sidecar takes the narrow
    # compatibility path below and re-resolves the omitted environment fields.
    saved = _load_env_sidecar(install_dir)
    if not saved:
        raise ValueError(
            f"persisted exact Soperator identity is missing at {install_dir / _ENV_SIDECAR}; "
            "refusing destroy by name"
        )
    if (
        saved
        and saved.get("region")
        and saved.get("tenant_id")
        and saved.get("project_id")
    ):
        env = _soperator_tf_env(
            nebius_bin,
            region=str(saved["region"]),
            tenant_id=str(saved["tenant_id"]),
            project_id=str(saved["project_id"]),
            subnet_id=str(saved.get("subnet_id") or ""),
            profile=str(saved.get("auth_profile") or profile),
            timeout=remaining("Nebius IAM token exchange", 120),
        )
        if saved.get("o11y_profile"):
            env["TF_VAR_o11y_profile"] = str(saved["o11y_profile"])
    else:
        envcfg = resolve_environment(project=project)
        region = envcfg.region
        tenant_id = envcfg.tenant_id
        project_id = envcfg.project_id
        if not (region and tenant_id and project_id):
            raise ValueError(
                "cannot resolve region/tenant/project to destroy "
                f"{name!r}: no env sidecar at {install_dir / _ENV_SIDECAR} and "
                "~/.npa config is incomplete (pass --project)"
            )
        subnet_id = _resolve_subnet(
            nebius_bin,
            project_id,
            _nebius_cli_env(profile),
            timeout=remaining("subnet resolution", 120),
        )
        env = _soperator_tf_env(
            nebius_bin,
            region=region,
            tenant_id=tenant_id,
            project_id=project_id,
            subnet_id=subnet_id,
            profile=profile,
            timeout=remaining("Nebius IAM token exchange", 120),
        )
    _log(on_status, f"terraform destroy: {name}")
    _run_stream(
        [terraform_bin, "init"],
        cwd=install_dir,
        env=env,
        timeout=remaining("Terraform initialization", 900),
    )
    terraform_cluster_id = _terraform_cluster_id(
        terraform_bin,
        install_dir,
        env,
        timeout=remaining("Terraform cluster-ID state read", 120),
    )
    terraform_provider_cluster_name = _terraform_cluster_name(
        terraform_bin,
        install_dir,
        env,
        timeout=remaining("Terraform cluster-name state read", 120),
    )
    persisted_cluster_id = str((saved or {}).get("cluster_id") or "")
    persisted_provider_cluster_name = str(
        (saved or {}).get("provider_cluster_name") or ""
    )
    if (
        terraform_cluster_id
        and persisted_cluster_id
        and terraform_cluster_id != persisted_cluster_id
    ):
        raise ValueError(
            "persisted Soperator cluster identity conflicts with Terraform state; refusing destroy"
        )
    cluster_id = persisted_cluster_id or terraform_cluster_id
    if (
        terraform_provider_cluster_name
        and persisted_provider_cluster_name
        and terraform_provider_cluster_name != persisted_provider_cluster_name
    ):
        raise ValueError(
            "persisted Soperator provider cluster name conflicts with Terraform "
            "state; refusing destroy"
        )
    provider_cluster_name = (
        persisted_provider_cluster_name or terraform_provider_cluster_name
    )
    project_id = str(
        (saved or {}).get("project_id") or env.get("TF_VAR_iam_project_id") or ""
    )
    if not cluster_id:
        raise ValueError(
            "persisted Soperator/Terraform state has no exact cluster ID; "
            "refusing destructive name inference"
        )

    # Reclaim CSI-provisioned PVC disks (NFS + any dynamic volumes) BEFORE the
    # cluster is torn down. Deleting the mk8s cluster does NOT cascade-delete the
    # NETWORK_SSD_IO_M3 disks backing PVCs, so they leak against the (small) IO_M3
    # quota across deploy/destroy cycles. Delete the PVCs while the cluster is
    # still reachable so the CSI provisioner releases their backing disks.
    if cluster_id:
        context = f"nebius-{name}-slurm"
        _refresh_kube_credentials(
            nebius_bin,
            cluster_id,
            context,
            env,
            timeout=remaining("kubeconfig refresh", 120),
        )
        kubectl_bin = shutil.which(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
        if kubectl_bin:
            _log(on_status, "reclaiming CSI PVC disks before teardown")
            _run_capture(
                [
                    kubectl_bin,
                    "--context",
                    context,
                    "delete",
                    "pvc",
                    "--all",
                    "--all-namespaces",
                    "--wait=false",
                    "--timeout=60s",
                ],
                env=env,
                check=False,
                timeout=remaining("PVC reclamation", 120),
            )
            # Give CSI a bounded share of the one destroy deadline.
            time.sleep(min(20, remaining("PVC reclamation convergence")))

    # Best-effort terraform destroy. The recipe's disk_cleanup local-exec and
    # occasional node-group deletion races can fail even when the cluster itself
    # is removable, so don't hard-fail here -- fall through to a direct delete +
    # state reset so the install dir is reusable and quota is freed.
    destroy = _run_capture(
        [terraform_bin, "destroy", "-auto-approve"],
        cwd=install_dir,
        env=env,
        timeout=remaining("Terraform destroy"),
        check=False,
    )
    if destroy.returncode != 0:
        _log(
            on_status,
            "terraform destroy reported errors; falling back to direct cleanup",
        )

    # Ensure the mk8s cluster is actually gone (cascades node groups + instances).
    if cluster_id:
        still_present, presence_error = _resolve_exact_cluster_presence(
            nebius_bin=nebius_bin,
            cluster_id=cluster_id,
            cluster_name=provider_cluster_name,
            project_id=project_id,
            env=env,
            deadline=destroy_deadline,
        )
        if still_present is None:
            return {
                "name": name,
                "status": "destroy-incomplete",
                "install_dir": str(install_dir),
                "errors": [presence_error],
            }
        if still_present:
            _log(on_status, f"deleting mk8s cluster {cluster_id} directly")
            deleted = _run_capture(
                [nebius_bin, "mk8s", "cluster", "delete", "--id", cluster_id],
                env=env,
                check=False,
                timeout=remaining("exact Managed Kubernetes delete"),
            )
            if deleted.returncode != 0 and not _is_not_found_result(deleted):
                return {
                    "name": name,
                    "status": "destroy-incomplete",
                    "install_dir": str(install_dir),
                    "errors": ["exact Managed Kubernetes delete failed"],
                }
            # Wait for the cluster to actually disappear before cleaning up VPC
            # allocations below. The delete call can return while the cluster (and
            # its cloud-controller-manager) still exists; if we delete the static-IP
            # allocation while the CCM is alive it will re-create a same-named orphan
            # that isn't in terraform state, and the next deploy fails with
            # "Allocation ... already exists" (AlreadyExists). Poll get until gone.
            confirmed_absent = False
            last_absence_error = ""
            while time.monotonic() < destroy_deadline:
                try:
                    gone = _run_capture(
                        [
                            nebius_bin,
                            "mk8s",
                            "cluster",
                            "get",
                            "--id",
                            cluster_id,
                            "--format",
                            "json",
                        ],
                        env=env,
                        check=False,
                        timeout=remaining("Managed Kubernetes absence check", 120),
                    )
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    last_absence_error = type(exc).__name__
                    time.sleep(min(2, max(0, destroy_deadline - time.monotonic())))
                    continue
                if _is_not_found_result(gone):
                    confirmed_absent = True
                    break
                if gone.returncode != 0 or not gone.stdout.strip():
                    if _provider_read_terminal(gone):
                        return {
                            "name": name,
                            "status": "destroy-incomplete",
                            "install_dir": str(install_dir),
                            "errors": [
                                "exact Managed Kubernetes absence check failed terminally"
                            ],
                        }
                    last_absence_error = "transient or unreadable provider response"
                    time.sleep(min(2, max(0, destroy_deadline - time.monotonic())))
                    continue
                last_absence_error = "cluster is still present"
                time.sleep(min(2, max(0, destroy_deadline - time.monotonic())))
            if not confirmed_absent:
                return {
                    "name": name,
                    "status": "destroy-incomplete",
                    "install_dir": str(install_dir),
                    "errors": [
                        "exact Managed Kubernetes absence was not confirmed before "
                        f"the configured deadline: {last_absence_error}"
                    ],
                }

    # A direct exact-ID cluster delete cannot prove that Terraform-owned
    # filesystems, allocations, node groups, and cache disks were reclaimed.
    # Retain the complete installation/state whenever Terraform itself failed;
    # a later retry can reconcile those exact resource IDs. Never replace that
    # ownership proof with prefix/name sweeps.
    if destroy.returncode != 0:
        return {
            "name": name,
            "status": "destroy-incomplete",
            "install_dir": str(install_dir),
            "errors": [
                "Terraform teardown failed; exact auxiliary-resource state retained"
            ],
        }

    # Clean up only exact IDs captured from the applied Terraform state. Provider
    # inventory, delete, and post-delete absence evidence are all mandatory;
    # ambiguity retains the sidecar and Terraform state for a safe retry.
    owned_filesystem_ids = {
        str(value) for value in (saved or {}).get("owned_filesystem_ids", []) if value
    }
    owned_allocation_ids = {
        str(value) for value in (saved or {}).get("owned_allocation_ids", []) if value
    }
    auxiliary_errors: list[str] = []
    auxiliary_deadline = destroy_deadline
    if not project_id and (owned_filesystem_ids or owned_allocation_ids):
        auxiliary_errors.append(
            "persisted project identity is missing; exact auxiliary absence cannot be verified"
        )
    elif project_id:
        auxiliary_errors.extend(
            _cleanup_owned_provider_ids(
                nebius_bin=nebius_bin,
                project_id=project_id,
                env=env,
                owned_ids=owned_filesystem_ids,
                service=("compute", "filesystem"),
                description="filesystem",
                on_status=on_status,
                deadline=auxiliary_deadline,
            )
        )
        auxiliary_errors.extend(
            _cleanup_owned_provider_ids(
                nebius_bin=nebius_bin,
                project_id=project_id,
                env=env,
                owned_ids=owned_allocation_ids,
                service=("vpc", "allocation"),
                description="VPC allocation",
                on_status=on_status,
                deadline=auxiliary_deadline,
            )
        )
        raw_records = (saved or {}).get("owned_auxiliary_resources") or []
        if raw_records and not all(isinstance(item, dict) for item in raw_records):
            auxiliary_errors.append("persisted typed auxiliary ownership is malformed")
        else:
            auxiliary_errors.extend(
                _reconcile_recreated_auxiliary_resources(
                    nebius_bin=nebius_bin,
                    project_id=project_id,
                    cluster_name=str((saved or {}).get("cluster_name") or name),
                    env=env,
                    records=[dict(item) for item in raw_records],
                    deadline=auxiliary_deadline,
                    on_status=on_status,
                )
            )
    if auxiliary_errors:
        return {
            "name": name,
            "status": "destroy-incomplete",
            "install_dir": str(install_dir),
            "errors": auxiliary_errors,
        }

    # Reset local terraform state so the install dir is clean for a redeploy.
    for stale in install_dir.glob("terraform.tfstate*"):
        try:
            stale.unlink()
        except OSError:
            pass
    _log(on_status, f"destroy complete: {name}")
    return {"name": name, "status": "destroyed", "install_dir": str(install_dir)}


def apply_post_deploy_fixes(
    context: str,
    kubectl_bin: str,
    *,
    namespace: str = "soperator",
    on_status: Callable[[str], None] | None = None,
    timeout_minutes: int = 20,
) -> list[str]:
    """Apply idempotent repairs after a successful Terraform reconciliation.

    Monitoring namespace/CRD/dashboard repair is best-effort here: RBAC or a
    transient Flux error is retained in the returned diagnostics but cannot turn
    an already healthy Terraform apply into a reported deploy failure. A
    superseded ActiveChecks wait hook is reset only when its attempted generation
    is older than the desired generation. The CRD and scripts compatibility
    fixes remain polled/best-effort, followed by worker address/RESUME recovery.
    Returns non-secret warning strings.
    """

    warnings: list[str] = []
    try:
        _install_monitoring_crds(kubectl_bin, context, on_status=on_status)
    except RuntimeError as exc:
        warning = f"monitoring repair skipped after successful apply: {exc}"
        warnings.append(warning)
        _log(on_status, f"post-deploy warning: {warning}")

    reset_activechecks = _abort_superseded_activechecks_upgrade(
        kubectl_bin,
        context,
        namespace=namespace,
    )
    if reset_activechecks:
        _log(
            on_status,
            "post-deploy: aborted a superseded ActiveChecks Helm action so "
            "the newer generation can reconcile",
        )

    _log(
        on_status, "post-deploy: patching SlurmCluster CRD + ensuring scripts configmap"
    )
    deadline = time.monotonic() + timeout_minutes * 60
    crd_done = False
    cm_done = False
    while time.monotonic() < deadline and not (crd_done and cm_done):
        crd_done = crd_done or _patch_slurmcluster_crd(kubectl_bin, context)
        cm_done = cm_done or _ensure_scripts_configmap(kubectl_bin, context, namespace)
        if crd_done and cm_done:
            break
        time.sleep(15)
    if not crd_done:
        _log(
            on_status,
            "post-deploy: SlurmCluster CRD not present yet; skipped CRD patch",
        )
    if not cm_done:
        _log(on_status, "post-deploy: slurm-scripts configmap not present yet; skipped")

    _register_slurm_workers(kubectl_bin, context, namespace, on_status=on_status)
    _log(
        on_status, "post-deploy: fixes applied" + (" with warnings" if warnings else "")
    )
    return warnings


def _register_slurm_workers(
    kubectl_bin: str,
    context: str,
    namespace: str,
    *,
    on_status: Callable[[str], None] | None = None,
    wait_minutes: int = 10,
) -> None:
    """Best-effort: bring DOWN worker nodes to IDLE after registration races.

    Dynamic-node registration can race worker readiness and leave slurmctld
    resolving a bare short name. Set
    the FQDN NodeAddr and RESUME any node that is down. Idempotent and non-fatal.
    """

    kube_env = _nebius_cli_env()
    ctl = ["exec", "-n", namespace, "controller-0", "-c", "slurmctld", "--"]

    def slurmctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _run_capture(
            [kubectl_bin, "--context", context, *ctl, *args], env=kube_env, check=False
        )

    deadline = time.monotonic() + wait_minutes * 60
    while time.monotonic() < deadline:
        info = slurmctl(["sinfo", "-h", "-N", "-o", "%N %t"])
        if info.returncode != 0 or not info.stdout.strip():
            time.sleep(15)
            continue
        down = [
            line.split()[0]
            for line in info.stdout.splitlines()
            if line.split()
            and (line.split()[1].endswith("*") or "down" in line.split()[1].lower())
        ]
        if not down:
            _log(on_status, "post-deploy: all Slurm worker nodes are responding")
            return
        for node in sorted(set(down)):
            fqdn = f"{node}.soperator-nodeset-svc.{namespace}.svc.cluster.local"
            slurmctl(["scontrol", "update", f"NodeName={node}", f"NodeAddr={fqdn}"])
            slurmctl(["scontrol", "update", f"NodeName={node}", "State=RESUME"])
        _log(
            on_status,
            f"post-deploy: registered worker node(s): {', '.join(sorted(set(down)))}",
        )
        time.sleep(15)


def _gpu_count_from_preset(preset: str) -> int:
    """Return the leading GPU count from an upstream GPU preset name."""

    match = re.match(r"^([1-9][0-9]*)gpu-", preset)
    if match is None:
        raise RuntimeError(
            f"GPU creation check cannot derive a GPU count from preset {preset!r}"
        )
    return int(match.group(1))


def _validate_gpu_creation_check_timeout(timeout_seconds: int) -> int:
    """Validate the independent end-to-end GPU creation-check budget."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError(
            "gpu creation-check timeout must be an integer of at least 1 second"
        )
    return timeout_seconds


def _tail_diagnostic(*parts: str | bytes | None) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, bytes):
            text_parts.append(part.decode(errors="replace"))
        elif part:
            text_parts.append(part)
    return "\n".join("\n".join(text_parts).strip().splitlines()[-20:])


def _slurm_time_limit(seconds: int) -> str:
    """Render a Slurm wall-time value without dropping a sub-minute remainder."""

    days, remainder = divmod(max(1, seconds), 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{clock}" if days else clock


def _slurm_exec_command(
    kubectl_bin: str,
    context: str,
    namespace: str,
    pod: str,
    container: str | None,
    args: list[str],
    *,
    jailed: bool = False,
) -> list[str]:
    command = [
        kubectl_bin,
        "--context",
        context,
        "exec",
        "-n",
        namespace,
        pod,
    ]
    if container:
        command += ["-c", container]
    command += ["--"]
    if jailed:
        command += ["chroot", "/mnt/jail"]
    return [*command, *args]


def _discover_slurm_gpu_nodes(
    spec: SoperatorSpec,
    context: str,
    kubectl_bin: str,
    namespace: str,
    *,
    timeout_seconds: int,
    kube_env: dict[str, str],
) -> dict[str, list[str]]:
    """Discover node names from Slurm and validate their spec-pool mapping."""

    command = _slurm_exec_command(
        kubectl_bin,
        context,
        namespace,
        "controller-0",
        "slurmctld",
        ["sinfo", "-h", "-N", "-o", "%N"],
    )
    try:
        completed = _run_capture(
            command,
            env=kube_env,
            check=False,
            timeout=timeout_seconds,
        )
    except (BackendCommandError, subprocess.TimeoutExpired) as exc:
        diagnostic = _tail_diagnostic(exc.stderr, exc.stdout)
        raise GPUCreationCheckError(
            "GPU creation check timed out while discovering live Slurm nodes"
            + (f":\n{diagnostic}" if diagnostic else ""),
            pool=None,
            phase="node-discovery",
        ) from exc
    if completed.returncode != 0:
        diagnostic = _tail_diagnostic(completed.stderr, completed.stdout)
        raise GPUCreationCheckError(
            "GPU creation check could not query live Slurm nodes"
            + (f":\n{diagnostic}" if diagnostic else ""),
            pool=None,
            phase="node-discovery",
        )

    discovered = sorted(
        {
            line.strip().split()[0]
            for line in completed.stdout.splitlines()
            if line.strip()
        }
    )
    if not discovered:
        raise GPUCreationCheckError(
            "GPU creation check found no nodes in live Slurm state",
            pool=None,
            phase="node-discovery",
        )

    mapping: dict[str, list[str]] = {}
    for pool in (worker for worker in spec.workers if worker.is_gpu()):
        pattern = re.compile(rf"^{re.escape(pool.name)}-[0-9]+$")
        nodes = [node for node in discovered if pattern.fullmatch(node)]
        if len(nodes) != pool.size:
            shown = ", ".join(discovered)
            raise GPUCreationCheckError(
                f"GPU creation check cannot map pool {pool.name!r}: expected "
                f"{pool.size} live Slurm node(s) named {pool.name}-<index>, found "
                f"{len(nodes)}; Slurm reported: {shown}. Verify the upstream "
                "nodeset naming contract before retrying",
                pool=pool.name,
                phase="node-mapping",
            )
        mapping[pool.name] = nodes
    return mapping


def _cancel_and_verify_gpu_check_job(
    context: str,
    kubectl_bin: str,
    namespace: str,
    job_name: str,
    *,
    kube_env: dict[str, str],
) -> tuple[bool, str]:
    """Cancel one uniquely named gate job and prove it left Slurm's queue."""

    scancel = _slurm_exec_command(
        kubectl_bin,
        context,
        namespace,
        "login-0",
        None,
        ["scancel", "--name", job_name],
        jailed=True,
    )
    diagnostics: list[str] = []
    try:
        cancelled = _run_capture(
            scancel,
            env=kube_env,
            check=False,
            timeout=_GPU_CHECK_CLEANUP_TIMEOUT_SECONDS,
        )
        if cancelled.returncode != 0:
            detail = _tail_diagnostic(cancelled.stderr, cancelled.stdout)
            if detail:
                diagnostics.append(f"scancel: {detail}")
    except (BackendCommandError, subprocess.TimeoutExpired):
        diagnostics.append("scancel timed out")

    squeue = _slurm_exec_command(
        kubectl_bin,
        context,
        namespace,
        "login-0",
        None,
        ["squeue", "--noheader", "--name", job_name, "--format=%A"],
        jailed=True,
    )
    deadline = time.monotonic() + _GPU_CHECK_CLEANUP_TIMEOUT_SECONDS
    while True:
        remaining = max(1, int(deadline - time.monotonic()))
        try:
            queued = _run_capture(
                squeue,
                env=kube_env,
                check=False,
                timeout=remaining,
            )
        except (BackendCommandError, subprocess.TimeoutExpired):
            diagnostics.append("squeue cleanup verification timed out")
            return False, "; ".join(diagnostics)
        if queued.returncode == 0 and not queued.stdout.strip():
            return True, "; ".join(diagnostics)
        if queued.returncode != 0:
            detail = _tail_diagnostic(queued.stderr, queued.stdout)
            diagnostics.append(
                "could not verify Slurm queue" + (f": {detail}" if detail else "")
            )
            return False, "; ".join(diagnostics)
        if time.monotonic() >= deadline:
            diagnostics.append(f"job {job_name} remains in Slurm after cancellation")
            return False, "; ".join(diagnostics)
        time.sleep(1)


def _verify_gpu_check_job_absent(
    context: str,
    kubectl_bin: str,
    namespace: str,
    job_name: str,
    *,
    kube_env: dict[str, str],
) -> bool:
    """Allow Slurm's completion accounting to settle, then prove queue absence."""

    command = _slurm_exec_command(
        kubectl_bin,
        context,
        namespace,
        "login-0",
        None,
        ["squeue", "--noheader", "--name", job_name, "--format=%A"],
        jailed=True,
    )
    deadline = time.monotonic() + _GPU_CHECK_CLEANUP_TIMEOUT_SECONDS
    while True:
        remaining = max(1, int(deadline - time.monotonic()))
        try:
            queued = _run_capture(
                command,
                env=kube_env,
                check=False,
                timeout=remaining,
            )
        except (BackendCommandError, subprocess.TimeoutExpired):
            return False
        if queued.returncode != 0:
            return False
        if not queued.stdout.strip():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def _run_gpu_creation_checks(
    spec: SoperatorSpec,
    context: str,
    kubectl_bin: str,
    *,
    namespace: str = "soperator",
    timeout_seconds: int = DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS,
    on_status: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run real CUDA samples on every GPU worker through the login jail.

    The pinned 4.1.6 Terraform surface exposes REST separately from accounting,
    but the exact operator implementation skips REST reconciliation when
    accounting is disabled. Its REST-backed ActiveCheck controller therefore
    cannot provide creation-time GPU validation for that combination. This
    direct Slurm check is the safe runtime contract: one exclusive task per
    worker receives every GPU on that worker and requires deviceQuery,
    vectorAdd, simpleMultiGPU, and p2pBandwidthLatencyTest to all report PASS.
    Any failed/missing node or CUDA result fails the deploy.
    """

    timeout_seconds = _validate_gpu_creation_check_timeout(timeout_seconds)
    checks: list[dict[str, Any]] = []
    kube_env = _nebius_cli_env()
    gpu_pools = [worker for worker in spec.workers if worker.is_gpu()]
    if not gpu_pools:
        return checks
    deadline = time.monotonic() + timeout_seconds
    node_mapping = _discover_slurm_gpu_nodes(
        spec,
        context,
        kubectl_bin,
        namespace,
        timeout_seconds=timeout_seconds,
        kube_env=kube_env,
    )
    for pool in gpu_pools:
        remaining = int(deadline - time.monotonic())
        if remaining < 1:
            raise GPUCreationCheckError(
                f"GPU creation check exceeded its {timeout_seconds}-second end-to-end "
                f"timeout before pool {pool.name!r} could start",
                pool=pool.name,
                phase="queue",
                completed_checks=checks,
            )
        gpu_count = _gpu_count_from_preset(pool.preset)
        nodes = node_mapping[pool.name]
        node_list = ",".join(nodes)
        task_script = f"""
set -uo pipefail
if ! gpu_inventory=$(nvidia-smi -L); then
  echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL phase=nvidia-smi-inventory"
  exit 1
fi
gpu_count=$(printf '%s\n' "$gpu_inventory" | wc -l)
if [ "$gpu_count" -ne {gpu_count} ]; then
  echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL expected_gpus={gpu_count} actual_gpus=$gpu_count"
  exit 1
fi
if ! gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u); then
  echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL phase=nvidia-smi-name"
  exit 1
fi
case "$gpu_name" in
  "NVIDIA H100") platform=8xH100 ;;
  "NVIDIA H200") platform=8xH200 ;;
  "NVIDIA B200") platform=8xB200 ;;
  "NVIDIA B300") platform=8xB300 ;;
  "NVIDIA GB300") platform=4xGB300 ;;
  *) echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL unsupported_gpu=$gpu_name"; exit 1 ;;
esac
echo "NPA_GPU_CREATION_CHECK_START host=$(hostname) gpu_count=$gpu_count platform=$platform"
command_rc=0
out=$(health-checker run -e soperator -p "$platform" \
  -n deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest \
  -f json-partial \
  --tests-stdout-path /opt/soperator-outputs/health_checker_cmd_stdout) || command_rc=$?
status=$(printf '%s\n' "$out" | awk '/^[[:space:]]*{{/,/^[[:space:]]*}}/' | jq -r '.status // empty') || {{
  echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=FAIL command_rc=$command_rc phase=result-parse"
  exit 1
}}
echo "NPA_GPU_CREATION_CHECK_RESULT host=$(hostname) status=$status command_rc=$command_rc"
if [ "$command_rc" -ne 0 ] || [ "$status" != PASS ]; then
  exit 1
fi
""".strip()
        job_name = f"npa-gpu-check-{uuid.uuid4().hex[:12]}"
        command = [
            *_slurm_exec_command(
                kubectl_bin,
                context,
                namespace,
                "login-0",
                None,
                [],
                jailed=True,
            ),
            "srun",
            "--label",
            f"--job-name={job_name}",
            f"--nodes={pool.size}",
            f"--ntasks={pool.size}",
            "--ntasks-per-node=1",
            f"--gpus-per-node={gpu_count}",
            "--exclusive",
            "--kill-on-bad-exit=1",
            "--wait=10",
            f"--immediate={remaining}",
            f"--time={_slurm_time_limit(remaining)}",
            f"--nodelist={node_list}",
            "bash",
            "-lc",
            task_script,
        ]
        _log(
            on_status,
            "GPU creation check: "
            f"pool={pool.name}; nodes={pool.size}; GPUs/node={gpu_count}; "
            f"timeout={remaining}s; "
            "tests=deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest",
        )
        try:
            completed = _run_capture(
                command,
                env=kube_env,
                check=False,
                timeout=remaining,
            )
        except (BackendCommandError, subprocess.TimeoutExpired) as exc:
            cleanup_confirmed, cleanup_detail = _cancel_and_verify_gpu_check_job(
                context,
                kubectl_bin,
                namespace,
                job_name,
                kube_env=kube_env,
            )
            diagnostic = _tail_diagnostic(exc.stderr, exc.stdout)
            detail_parts = [part for part in (diagnostic, cleanup_detail) if part]
            detail = "\n".join(detail_parts)
            raise GPUCreationCheckError(
                f"GPU creation check timed out after {remaining} seconds for pool "
                f"{pool.name!r}; the queued/running Slurm job was cancelled"
                + (f":\n{detail}" if detail else ""),
                pool=pool.name,
                phase="process-timeout",
                cleanup_confirmed=cleanup_confirmed,
                completed_checks=checks,
            ) from exc
        passes = completed.stdout.count("NPA_GPU_CREATION_CHECK_RESULT")
        passes_with_status = completed.stdout.count("status=PASS")
        if (
            completed.returncode != 0
            or passes != pool.size
            or passes_with_status != pool.size
        ):
            cleanup_confirmed, cleanup_detail = _cancel_and_verify_gpu_check_job(
                context,
                kubectl_bin,
                namespace,
                job_name,
                kube_env=kube_env,
            )
            diagnostic = _tail_diagnostic(completed.stderr, completed.stdout)
            detail_parts = [part for part in (diagnostic, cleanup_detail) if part]
            detail = "\n".join(detail_parts)
            raise GPUCreationCheckError(
                f"GPU creation check failed for pool {pool.name!r} "
                f"({passes_with_status}/{pool.size} workers reported PASS)"
                + (f":\n{detail}" if detail else ""),
                pool=pool.name,
                phase="slurm-step",
                cleanup_confirmed=cleanup_confirmed,
                completed_checks=checks,
            )
        if not _verify_gpu_check_job_absent(
            context,
            kubectl_bin,
            namespace,
            job_name,
            kube_env=kube_env,
        ):
            cleanup_confirmed, cleanup_detail = _cancel_and_verify_gpu_check_job(
                context,
                kubectl_bin,
                namespace,
                job_name,
                kube_env=kube_env,
            )
            raise GPUCreationCheckError(
                f"GPU creation check tasks passed for pool {pool.name!r}, but Slurm "
                "still reported the gate job; cancellation was requested"
                + (f": {cleanup_detail}" if cleanup_detail else ""),
                pool=pool.name,
                phase="cleanup-verification",
                cleanup_confirmed=cleanup_confirmed,
                completed_checks=checks,
            )
        checks.append(
            {
                "pool": pool.name,
                "nodes": pool.size,
                "gpus_per_node": gpu_count,
                "tests": [
                    "deviceQuery",
                    "vectorAdd",
                    "simpleMultiGPU",
                    "p2pBandwidthLatencyTest",
                ],
                "status": "PASS",
            }
        )
        _log(
            on_status,
            f"GPU creation check passed: pool={pool.name}; workers={pool.size}",
        )
    return checks
