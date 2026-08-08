"""Immutable whole-path capacity plans shared by every provisioning entrypoint.

The important property of this module is not a particular CLI spelling.  It is
that validation and mutation consume the *same resolved object*.  A workflow
submit, ``provision-if-absent`` and ``cluster up`` must not each reinterpret
defaults or ask different quota questions after one of them has already begun
creating infrastructure.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from npa.cli.cluster.capacity import gpu_quota_name


INSTANCE_QUOTA = "compute.instance.count"
DISK_QUOTA = "compute.disk.count"
NETWORK_SSD_BYTES_QUOTA = "compute.disk.size.network-ssd"
PUBLIC_IP_QUOTA = "vpc.ipv4-address.public.count"

GIB = 1024**3
DEFAULT_AGENT_ROOT_DISK_GIB = 100
DEFAULT_CPU_DISK_GIB = 128
DEFAULT_GPU_DISK_GIB = 1023

DEFAULT_CPU_NODES = 1
DEFAULT_CPU_PLATFORM = "cpu-d3"
DEFAULT_CPU_PRESET = "8vcpu-32gb"
DEFAULT_GPU_NODES = 1
DEFAULT_GPU_PLATFORM = "gpu-rtx6000"
DEFAULT_GPU_PRESET = "1gpu-24vcpu-218gb"

_UNBOUNDED = frozenset({"-1", "inf", "infinite", "unbounded", "unlimited"})


class PreflightBlockedError(RuntimeError):
    """A mutation was refused because the complete plan was not green."""


@dataclass(frozen=True)
class ResolvedTopology:
    """The exact additive topology used for both quota arithmetic and execution."""

    cluster_name: str = "npa-cluster"
    agent_requested: bool = False
    agent_exists: bool = False
    control_plane_instances: int = 0
    control_plane_disks: int = 0
    control_plane_disk_gib: int = 0
    agent_root_disk_gib: int = DEFAULT_AGENT_ROOT_DISK_GIB
    cpu_nodes: int = DEFAULT_CPU_NODES
    existing_cpu_nodes: int = 0
    cpu_platform: str = DEFAULT_CPU_PLATFORM
    cpu_preset: str = DEFAULT_CPU_PRESET
    cpu_disk_gib: int = DEFAULT_CPU_DISK_GIB
    gpu_nodes: int = DEFAULT_GPU_NODES
    existing_gpu_nodes: int = 0
    gpu_platform: str = DEFAULT_GPU_PLATFORM
    gpu_preset: str = DEFAULT_GPU_PRESET
    gpu_disk_gib: int = DEFAULT_GPU_DISK_GIB
    gpu_preemptible: bool = False
    public_node_ips: bool = False
    accelerator: str = "RTXPRO6000:1"

    @property
    def new_agent_instances(self) -> int:
        return int(self.agent_requested and not self.agent_exists)

    @property
    def new_cpu_nodes(self) -> int:
        return max(0, self.cpu_nodes - self.existing_cpu_nodes)

    @property
    def new_gpu_nodes(self) -> int:
        return max(0, self.gpu_nodes - self.existing_gpu_nodes)

    @property
    def required_instances(self) -> int:
        return (
            self.new_agent_instances
            + max(0, self.control_plane_instances)
            + self.new_cpu_nodes
            + self.new_gpu_nodes
        )

    @property
    def required_disks(self) -> int:
        return (
            self.new_agent_instances
            + max(0, self.control_plane_disks)
            + self.new_cpu_nodes
            + self.new_gpu_nodes
        )

    @property
    def required_network_ssd_bytes(self) -> int:
        """Incremental NETWORK_SSD bytes for the exact resources still missing."""

        return GIB * (
            self.new_agent_instances * self.agent_root_disk_gib
            + max(0, self.control_plane_disks) * self.control_plane_disk_gib
            + self.new_cpu_nodes * self.cpu_disk_gib
            + self.new_gpu_nodes * self.gpu_disk_gib
        )

    @property
    def required_public_ips(self) -> int:
        node_ips = (
            self.new_cpu_nodes + self.new_gpu_nodes if self.public_node_ips else 0
        )
        return self.new_agent_instances + node_ips

    @property
    def required_gpus(self) -> int:
        return self.new_gpu_nodes * _gpus_per_node(self.gpu_preset)

    def quota_requirements(self) -> dict[str, int]:
        requirements = {
            INSTANCE_QUOTA: self.required_instances,
            DISK_QUOTA: self.required_disks,
            NETWORK_SSD_BYTES_QUOTA: self.required_network_ssd_bytes,
            PUBLIC_IP_QUOTA: self.required_public_ips,
        }
        quota_name = gpu_quota_name(self.gpu_platform)
        # Preemptible capacity can change the GPU capacity pool.  It never
        # changes the hard instance/disk/IP arithmetic above.
        if quota_name and self.required_gpus and not self.gpu_preemptible:
            requirements[quota_name] = self.required_gpus
        return requirements

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "new_agent_instances": self.new_agent_instances,
                "new_cpu_nodes": self.new_cpu_nodes,
                "new_gpu_nodes": self.new_gpu_nodes,
                "required_instances": self.required_instances,
                "required_disks": self.required_disks,
                "required_network_ssd_bytes": self.required_network_ssd_bytes,
                "required_network_ssd_gib": _bytes_to_gib_text(
                    self.required_network_ssd_bytes
                ),
                "required_public_ips": self.required_public_ips,
                "required_gpus": self.required_gpus,
            }
        )
        return payload


@dataclass(frozen=True)
class QuotaObservation:
    """One provider allowance before it is combined with a request delta."""

    name: str
    used: int | None = None
    limit: int | None = None
    state: str = "unknown"  # known | unbounded | unsupported | unknown
    reason: str = ""


@dataclass(frozen=True)
class QuotaDecision:
    """Arithmetic and decision for one exact quota name."""

    name: str
    required: int
    used: int | None
    limit: int | None
    available: int | None
    shortfall: int | None
    status: str  # ready | blocked | unknown | unbounded
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.name == NETWORK_SSD_BYTES_QUOTA:
            payload.update(
                {
                    "unit": "bytes",
                    "required_gib": _bytes_to_gib_text(self.required),
                    "used_gib": _bytes_to_gib_text(self.used),
                    "limit_gib": _bytes_to_gib_text(self.limit),
                    "available_gib": _bytes_to_gib_text(self.available),
                    "shortfall_gib": _bytes_to_gib_text(self.shortfall),
                }
            )
        else:
            payload["unit"] = "count"
        return payload


@dataclass(frozen=True)
class PreflightCheck:
    """A non-quota whole-path prerequisite without secret-bearing detail."""

    name: str
    status: str  # ready | blocked | unknown
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExistingCapacity:
    """Matching provider resources that an interrupted operation can reuse."""

    cpu_nodes: int = 0
    gpu_nodes: int = 0
    check: PreflightCheck = PreflightCheck(
        name="existing_cluster_resources", status="ready"
    )


@dataclass(frozen=True)
class WholePathPreflightPlan:
    """Frozen plan whose structured form is safe for logs and receipts."""

    project_alias: str
    project_id: str
    tenant_id: str
    region: str
    topology: ResolvedTopology
    quotas: tuple[QuotaDecision, ...] = ()
    checks: tuple[PreflightCheck, ...] = ()
    decision: str = "unknown"  # ready | blocked | unknown
    reasons: tuple[str, ...] = ()
    source_action: str = "not-required"
    input_action: str = "not-required"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_alias": self.project_alias,
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "topology": self.topology.to_dict(),
            "quotas": [item.to_dict() for item in self.quotas],
            "checks": [item.to_dict() for item in self.checks],
            "decision": self.decision,
            "reasons": list(self.reasons),
            "source_action": self.source_action,
            "input_action": self.input_action,
        }

    def assert_mutation_ready(self) -> None:
        if self.decision == "ready":
            return
        detail = "; ".join(self.reasons) or "preflight result is not ready"
        raise PreflightBlockedError(
            f"Whole-path preflight {self.decision}; no resources were created: {detail}"
        )


_RESOLVED_PLAN: ContextVar[WholePathPreflightPlan | None] = ContextVar(
    "npa_resolved_provisioning_plan", default=None
)


@contextmanager
def resolved_plan_context(plan: WholePathPreflightPlan) -> Iterator[None]:
    """Make one immutable outer plan authoritative for nested provisioning."""

    token = _RESOLVED_PLAN.set(plan)
    try:
        yield
    finally:
        _RESOLVED_PLAN.reset(token)


def current_resolved_plan() -> WholePathPreflightPlan | None:
    """Return the plan inherited by the current transactional call chain."""

    return _RESOLVED_PLAN.get()


QuotaReader = Callable[[str, str, Sequence[str]], Mapping[str, QuotaObservation]]


def resolve_topology(
    *,
    cluster_name: str = "npa-cluster",
    accelerator: str = "",
    agent_requested: bool = False,
    agent_exists: bool = False,
    cpu_nodes: int = -1,
    existing_cpu_nodes: int = 0,
    cpu_platform: str = "",
    cpu_preset: str = "",
    gpu_nodes: int = -1,
    existing_gpu_nodes: int = 0,
    gpu_platform: str = "",
    gpu_preset: str = "",
    preemptible: bool | None = None,
    public_node_ips: bool = False,
    control_plane_instances: int = 0,
    control_plane_disks: int = 0,
    control_plane_disk_gib: int = 0,
    agent_root_disk_gib: int = DEFAULT_AGENT_ROOT_DISK_GIB,
    cpu_disk_gib: int = DEFAULT_CPU_DISK_GIB,
    gpu_disk_gib: int = DEFAULT_GPU_DISK_GIB,
) -> ResolvedTopology:
    """Resolve every default once, including the canonical PAIDF shape."""

    requested_accelerator = str(accelerator or "").strip()
    inferred_platform, inferred_preset = _shape_for_accelerator(requested_accelerator)
    return ResolvedTopology(
        cluster_name=str(cluster_name or "npa-cluster").strip() or "npa-cluster",
        agent_requested=bool(agent_requested),
        agent_exists=bool(agent_exists),
        control_plane_instances=max(0, int(control_plane_instances)),
        control_plane_disks=max(0, int(control_plane_disks)),
        control_plane_disk_gib=max(0, int(control_plane_disk_gib)),
        agent_root_disk_gib=_positive_gib(
            agent_root_disk_gib, "agent_root_disk_gib"
        ),
        cpu_nodes=DEFAULT_CPU_NODES
        if cpu_nodes is None or cpu_nodes < 0
        else int(cpu_nodes),
        existing_cpu_nodes=max(0, int(existing_cpu_nodes)),
        cpu_platform=str(cpu_platform or DEFAULT_CPU_PLATFORM).strip(),
        cpu_preset=str(cpu_preset or DEFAULT_CPU_PRESET).strip(),
        cpu_disk_gib=_positive_gib(cpu_disk_gib, "cpu_disk_gib"),
        gpu_nodes=DEFAULT_GPU_NODES
        if gpu_nodes is None or gpu_nodes < 0
        else int(gpu_nodes),
        existing_gpu_nodes=max(0, int(existing_gpu_nodes)),
        gpu_platform=str(
            gpu_platform or inferred_platform or DEFAULT_GPU_PLATFORM
        ).strip(),
        gpu_preset=str(gpu_preset or inferred_preset or DEFAULT_GPU_PRESET).strip(),
        gpu_disk_gib=_positive_gib(gpu_disk_gib, "gpu_disk_gib"),
        gpu_preemptible=bool(preemptible) if preemptible is not None else False,
        public_node_ips=bool(public_node_ips),
        accelerator=requested_accelerator or "RTXPRO6000:1",
    )


def build_whole_path_plan(
    *,
    project_alias: str,
    project_id: str,
    tenant_id: str,
    region: str,
    topology: ResolvedTopology,
    quota_reader: QuotaReader | None = None,
    checks: Sequence[PreflightCheck] = (),
    mutation: bool,
    source_action: str = "not-required",
    input_action: str = "not-required",
) -> WholePathPreflightPlan:
    """Read all cumulative allowances and return one aggregate decision.

    Unknown values are intentionally represented in a read-only plan.  The same
    plan is fail-closed when *mutation* is true.
    """

    identity_checks = list(checks)
    for name, value in (
        ("project_id", project_id),
        ("tenant_id", tenant_id),
        ("region", region),
    ):
        if not str(value or "").strip():
            identity_checks.append(
                PreflightCheck(
                    name=name, status="blocked", reason=f"{name} is required"
                )
            )

    requirements = topology.quota_requirements()
    reader = quota_reader or read_provider_quotas
    try:
        observations = reader(tenant_id, region, tuple(requirements))
    except Exception as exc:  # noqa: BLE001 - normalized without provider secrets
        observations = {
            name: QuotaObservation(
                name=name,
                state="unknown",
                reason=f"provider/RBAC query failed: {type(exc).__name__}: {exc}",
            )
            for name in requirements
        }
    quotas = tuple(
        assess_quota(
            observations.get(name, QuotaObservation(name=name)),
            required=required,
        )
        for name, required in requirements.items()
    )
    blocked = [
        *(
            f"{item.name}: {item.reason}"
            for item in identity_checks
            if item.status == "blocked"
        ),
        *(f"{item.name}: {item.reason}" for item in quotas if item.status == "blocked"),
    ]
    unknown = [
        *(
            f"{item.name}: {item.reason}"
            for item in identity_checks
            if item.status == "unknown"
        ),
        *(f"{item.name}: {item.reason}" for item in quotas if item.status == "unknown"),
    ]
    if blocked:
        decision = "blocked"
        reasons = tuple(blocked)
    elif unknown:
        decision = "blocked" if mutation else "unknown"
        prefix = "unverified mutation prerequisite" if mutation else "unknown"
        reasons = tuple(f"{prefix}: {item}" for item in unknown)
    else:
        decision = "ready"
        reasons = ()
    return WholePathPreflightPlan(
        project_alias=str(project_alias or ""),
        project_id=str(project_id or ""),
        tenant_id=str(tenant_id or ""),
        region=str(region or ""),
        topology=topology,
        quotas=quotas,
        checks=tuple(identity_checks),
        decision=decision,
        reasons=reasons,
        source_action=source_action,
        input_action=input_action,
    )


def assess_quota(observation: QuotaObservation, *, required: int) -> QuotaDecision:
    """Apply exact integer arithmetic to one provider observation."""

    need = max(0, int(required))
    if need == 0:
        return QuotaDecision(
            name=observation.name,
            required=0,
            used=observation.used,
            limit=observation.limit,
            available=(
                max(0, observation.limit - (observation.used or 0))
                if observation.limit is not None
                else None
            ),
            shortfall=0,
            status="ready",
            reason="no new quota-backed resources are required",
        )
    if observation.state == "unbounded":
        return QuotaDecision(
            name=observation.name,
            required=need,
            used=observation.used,
            limit=None,
            available=None,
            shortfall=0,
            status="unbounded",
            reason=observation.reason or "provider reports this allowance as unbounded",
        )
    if (
        observation.state != "known"
        or observation.limit is None
        or observation.used is None
    ):
        return QuotaDecision(
            name=observation.name,
            required=need,
            used=observation.used,
            limit=observation.limit,
            available=None,
            shortfall=None,
            status="unknown",
            reason=observation.reason or "quota allowance is missing or unreadable",
        )
    available = max(0, observation.limit - observation.used)
    shortfall = max(0, need - available)
    status = "blocked" if shortfall else "ready"
    required_limit = observation.used + need
    reason = (
        f"required={need}, used={observation.used}, limit={observation.limit}, "
        f"available={available}, shortfall={shortfall}; required new limit={required_limit}"
    )
    if observation.name == NETWORK_SSD_BYTES_QUOTA:
        reason += (
            f" bytes (required={_bytes_to_gib_text(need)} GiB, "
            f"available={_bytes_to_gib_text(available)} GiB, "
            f"shortfall={_bytes_to_gib_text(shortfall)} GiB); request quota "
            f"{NETWORK_SSD_BYTES_QUOTA} >= {required_limit} bytes"
        )
    return QuotaDecision(
        name=observation.name,
        required=need,
        used=observation.used,
        limit=observation.limit,
        available=available,
        shortfall=shortfall,
        status=status,
        reason=reason,
    )


def parse_quota_allowances(
    payload: Mapping[str, Any], *, region: str, names: Sequence[str]
) -> dict[str, QuotaObservation]:
    """Parse one provider list response with explicit malformed/missing states."""

    items = payload.get("items")
    if not isinstance(items, list):
        return {
            name: QuotaObservation(
                name=name,
                state="unknown",
                reason="malformed quota response: items is not a list",
            )
            for name in names
        }
    candidates: dict[tuple[str, str], list[QuotaObservation]] = {}
    wanted = set(names)
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        metadata = raw.get("metadata")
        spec = raw.get("spec")
        status = raw.get("status")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        spec = spec if isinstance(spec, Mapping) else {}
        status = status if isinstance(status, Mapping) else {}
        name = str(metadata.get("name") or "")
        if name not in wanted:
            continue
        raw_limit = spec.get("limit")
        raw_usage = status.get("usage", 0)
        raw_available = status.get("available")
        item_region = str(spec.get("region") or "").strip()
        if str(raw_limit or "").strip().lower() in _UNBOUNDED:
            observation = QuotaObservation(name=name, state="unbounded")
        else:
            limit = _int_or_none(raw_limit)
            usage = _int_or_none(raw_usage)
            if limit is None or usage is None:
                observation = QuotaObservation(
                    name=name,
                    state="unknown",
                    reason="malformed quota allowance: limit/usage is not an integer",
                )
            else:
                limit = max(0, limit)
                usage = max(0, usage)
                available = (
                    _int_or_none(raw_available) if raw_available is not None else None
                )
                if usage > limit:
                    observation = QuotaObservation(
                        name=name,
                        state="unknown",
                        reason="contradictory quota allowance: usage exceeds limit",
                    )
                elif available is not None and available != limit - usage:
                    observation = QuotaObservation(
                        name=name,
                        state="unknown",
                        reason=(
                            "contradictory quota allowance: status.available does "
                            "not equal spec.limit - status.usage"
                        ),
                    )
                elif raw_available is not None and available is None:
                    observation = QuotaObservation(
                        name=name,
                        state="unknown",
                        reason="malformed quota allowance: available is not an integer",
                    )
                else:
                    observation = QuotaObservation(
                        name=name, used=usage, limit=limit, state="known"
                    )
        candidates.setdefault((name, item_region), []).append(observation)

    parsed: dict[str, QuotaObservation] = {}
    for name in names:
        selected = candidates.get((name, region)) or candidates.get((name, "")) or []
        if not selected:
            parsed[name] = QuotaObservation(
                name=name,
                state="unsupported",
                reason=f"quota {name} has no allowance for region {region}",
            )
            continue
        first = selected[0]
        if any(item != first for item in selected[1:]):
            parsed[name] = QuotaObservation(
                name=name,
                state="unknown",
                reason="contradictory duplicate quota allowances",
            )
        else:
            parsed[name] = first
    return parsed


def read_provider_quotas(
    tenant_id: str, region: str, names: Sequence[str]
) -> Mapping[str, QuotaObservation]:
    """Read one Nebius quota snapshot for all requested names."""

    if not tenant_id or not region:
        return {
            name: QuotaObservation(
                name=name,
                state="unknown",
                reason="tenant_id and region are required for the quota query",
            )
            for name in names
        }
    from npa.clients.nebius import list_quota_allowances

    return parse_quota_allowances(
        list_quota_allowances(tenant_id), region=region, names=names
    )


def discover_existing_capacity(
    *,
    project_id: str,
    cluster_name: str,
    cpu_platform: str,
    cpu_preset: str,
    gpu_platform: str,
    gpu_preset: str,
    client: Any | None = None,
) -> ExistingCapacity:
    """Count exact matching worker nodes without treating a control plane as ready.

    A provider ``RUNNING`` control plane with no node groups deliberately returns
    zero. A partial cluster with only its CPU group returns one CPU node, so a
    resume checks quota only for the missing GPU delta.
    """

    if not str(project_id or "").strip():
        return ExistingCapacity(
            check=PreflightCheck(
                name="existing_cluster_resources",
                status="unknown",
                reason="project_id is required to inspect existing cluster resources",
            )
        )
    try:
        from npa.cluster.api import MK8sClient
        from npa.cluster.exceptions import ClusterNotFoundError

        provider = client or MK8sClient(timeout=120, poll_interval=5.0)
        try:
            cluster = provider.get_cluster(cluster_name, project_id=project_id)
        except ClusterNotFoundError:
            return ExistingCapacity()
        groups = provider.list_node_groups(cluster.id)
    except Exception as exc:  # noqa: BLE001 - represented explicitly/fail-closed upstream
        return ExistingCapacity(
            check=PreflightCheck(
                name="existing_cluster_resources",
                status="unknown",
                reason=f"provider/RBAC inventory failed: {type(exc).__name__}: {exc}",
            )
        )

    cpu_nodes = 0
    gpu_nodes = 0
    for group in groups:
        count = max(0, int(getattr(group, "node_count", 0) or 0))
        platform = str(getattr(group, "platform", "") or "").strip()
        preset = str(getattr(group, "preset", "") or "").strip()
        if platform == cpu_platform and preset == cpu_preset:
            cpu_nodes += count
        elif platform == gpu_platform and preset == gpu_preset:
            gpu_nodes += count
    return ExistingCapacity(
        cpu_nodes=cpu_nodes,
        gpu_nodes=gpu_nodes,
        check=PreflightCheck(
            name="existing_cluster_resources",
            status="ready",
            reason=(
                f"matching existing nodes: cpu={cpu_nodes}, gpu={gpu_nodes}; "
                "only the missing delta is quota-checked"
            ),
        ),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _positive_gib(value: Any, name: str) -> int:
    parsed = _int_or_none(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer GiB value")
    return parsed


def _bytes_to_gib_text(value: int | None) -> str | None:
    if value is None:
        return None
    quotient, remainder = divmod(int(value), GIB)
    if remainder == 0:
        return str(quotient)
    return f"{int(value) / GIB:.3f}".rstrip("0").rstrip(".")


def _gpus_per_node(preset: str) -> int:
    value = str(preset or "").strip().lower()
    if "gpu" not in value:
        return 1
    try:
        return max(1, int(value.split("gpu", 1)[0]))
    except ValueError:
        return 1


def _shape_for_accelerator(accelerator: str) -> tuple[str, str]:
    value = accelerator.split(":", 1)[0].strip().lower().replace("-", "")
    if value in {"rtxpro6000", "rtx6000"}:
        return DEFAULT_GPU_PLATFORM, DEFAULT_GPU_PRESET
    if value == "h100":
        return "gpu-h100-sxm", "1gpu-16vcpu-200gb"
    if value == "h200":
        return "gpu-h200-sxm", "1gpu-16vcpu-200gb"
    if value == "l40s":
        return "gpu-l40s-d", "1gpu-16vcpu-96gb"
    return "", ""


__all__ = [
    "DEFAULT_CPU_NODES",
    "DEFAULT_AGENT_ROOT_DISK_GIB",
    "DEFAULT_CPU_DISK_GIB",
    "DEFAULT_CPU_PLATFORM",
    "DEFAULT_CPU_PRESET",
    "DEFAULT_GPU_NODES",
    "DEFAULT_GPU_DISK_GIB",
    "DEFAULT_GPU_PLATFORM",
    "DEFAULT_GPU_PRESET",
    "DISK_QUOTA",
    "GIB",
    "ExistingCapacity",
    "INSTANCE_QUOTA",
    "NETWORK_SSD_BYTES_QUOTA",
    "PUBLIC_IP_QUOTA",
    "PreflightBlockedError",
    "PreflightCheck",
    "QuotaDecision",
    "QuotaObservation",
    "ResolvedTopology",
    "WholePathPreflightPlan",
    "assess_quota",
    "build_whole_path_plan",
    "current_resolved_plan",
    "discover_existing_capacity",
    "parse_quota_allowances",
    "read_provider_quotas",
    "resolved_plan_context",
    "resolve_topology",
]
