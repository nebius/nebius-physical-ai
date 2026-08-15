"""Tenant quota preflight for fleet deploys.

Managed Kubernetes accepts a node group whose instances it cannot actually
create: the node group is stored, then the compute API rejects each instance
with a ``QuotaFailure`` and mk8s keeps retrying. Terraform sees only
``Still creating...`` and blocks until the per-cluster timeout, so a quota wall
looks like a hang rather than a rejection.

This module derives what a fleet's clusters need, validates explicitly bound
GPU capacity block groups, compares the remaining requirements against the
*tenant's* quota allowances for the target region, and lets the caller fail fast
with the shortfall. Reservation-backed GPUs do not consume the ordinary GPU
quota, but node/disk/GPU-cluster/storage quotas still apply. Project allowances
only subdivide the tenant allowance, so the tenant is the meaningful container
to check.

Only read APIs are used (``nebius capacity capacity-block-group list`` and
``nebius quotas quota-allowance list``). Quota evidence is fail-closed: an
unreadable, incomplete, malformed, or unpaged allowance response is not proof
that a deployment fits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Callable, Iterable

from npa.fleet.spec import ClusterSpec

_GIB = 1024**3
_CPU_DISK_GIB = 128
_FILESYSTEM_SIZE_QUOTA = "compute.filesystem.size.network-ssd"
_FILESYSTEM_SIZE_UNIT = "byte"
_BYTE_QUOTAS = {_FILESYSTEM_SIZE_QUOTA, "compute.disk.size.network-ssd"}
_VCPU_QUOTAS = {"compute.instance.non-gpu.vcpu"}
_OPTIONAL_UNADVERTISED_QUOTAS = frozenset(
    {
        "vpc.network.count",
        "vpc.subnet.count",
        "vpc.pool.count",
        "vpc.routetable.count",
        "vpc.route.count",
    }
)

# GPU quotas are keyed by accelerator family, not by the platform name:
# platform "gpu-h200-sxm" consumes "compute.instance.gpu.h200".
_GPU_PLATFORM_RE = re.compile(r"^gpu-([a-z0-9]+)")
_GPU_COUNT_RE = re.compile(r"(\d+)gpu")
_VCPU_COUNT_RE = re.compile(r"(\d+)vcpu")


def gpu_family(platform: str) -> str:
    """Return the quota family for a GPU platform (``gpu-h200-sxm`` -> ``h200``)."""

    match = _GPU_PLATFORM_RE.match(platform.strip().lower())
    return match.group(1) if match else ""


def _preset_gpus(preset: str) -> int:
    match = _GPU_COUNT_RE.search(preset.lower())
    return int(match.group(1)) if match else 0


def _preset_vcpus(preset: str) -> int:
    match = _VCPU_COUNT_RE.search(preset.lower())
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class QuotaShortfall:
    """One quota whose tenant limit cannot cover what the fleet requires."""

    name: str
    region: str
    required: int
    limit: int
    available: int | None = None
    unit: str = ""

    def describe(self) -> str:
        available = self.limit if self.available is None else self.available
        return (
            f"{self.name} [{self.region}]: needs {self.required}{f' {self.unit}' if self.unit else ''}, "
            f"{available} available from tenant limit {self.limit}"
        )


@dataclass(frozen=True)
class ReservationRequirement:
    """GPU capacity required from one explicitly bound capacity block group."""

    reservation_id: str
    region: str
    platform: str
    fabric: str
    required_gpus: int


@dataclass(frozen=True)
class ReservationShortfall:
    """A reservation validation failure that must block a STRICT deployment."""

    reservation_id: str
    reason: str

    def describe(self) -> str:
        return f"{self.reservation_id}: {self.reason}"


def required_reservations(
    clusters: Iterable[ClusterSpec], region: str
) -> dict[str, ReservationRequirement]:
    """Aggregate STRICT GPU requirements by explicit capacity block group ID."""

    requirements: dict[str, ReservationRequirement] = {}
    for cluster in clusters:
        gpu = cluster.gpu_nodes
        if not gpu or gpu.count <= 0 or not gpu.capacity_block_group:
            continue
        reservation_id = gpu.capacity_block_group
        required_gpus = gpu.count * _preset_gpus(gpu.preset)
        fabric = (
            cluster.infiniband_fabric if cluster.resolved_enable_gpu_cluster() else ""
        )
        previous = requirements.get(reservation_id)
        if previous is not None:
            if previous.platform != gpu.platform or previous.fabric != fabric:
                raise ValueError(
                    "one capacity_block_group cannot back incompatible GPU pools: "
                    f"{reservation_id}"
                )
            requirements[reservation_id] = ReservationRequirement(
                reservation_id=reservation_id,
                region=region,
                platform=gpu.platform,
                fabric=fabric,
                required_gpus=previous.required_gpus + required_gpus,
            )
            continue
        requirements[reservation_id] = ReservationRequirement(
            reservation_id=reservation_id,
            region=region,
            platform=gpu.platform,
            fabric=fabric,
            required_gpus=required_gpus,
        )
    return requirements


def _available_reserved_gpus(status: dict[str, Any]) -> int | None:
    """Return a conservative free-GPU count from capacity block group status."""

    raw_limit = status.get("current_limit")
    if raw_limit is None:
        return None
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return None
    raw_usage_count = status.get("usage")
    if raw_usage_count not in (None, ""):
        try:
            used_count = int(raw_usage_count)
        except (TypeError, ValueError):
            return None
        if used_count < 0:
            return None
        return max(0, limit - used_count)
    raw_usage = status.get("usage_percentage")
    if raw_usage in (None, ""):
        if status.get("usage_state") == "USAGE_STATE_NOT_USED":
            return limit
        return None
    try:
        # Prefer the exact usage counter above. The live API serializes this
        # nominally "percentage" fallback as a fraction: 0.17 means roughly
        # 17% consumed. Accept an older 0..100 representation as well, but treat
        # 1 as fully consumed to match the authoritative current wire format.
        used = Decimal(str(raw_usage))
        if used < 0:
            return None
        fraction = used if used <= 1 else used / Decimal(100)
        remaining = Decimal(limit) * (Decimal(1) - fraction)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return max(0, int(remaining.to_integral_value(rounding=ROUND_FLOOR)))


def parse_capacity_blocks(payload: str) -> dict[str, dict[str, Any]]:
    """Index capacity block group list JSON by ID, retaining validation fields."""

    try:
        items = json.loads(payload or "{}").get("items", [])
    except json.JSONDecodeError:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        metadata = item.get("metadata", {}) or {}
        status = item.get("status", {}) or {}
        affinity = (status.get("resource_affinity", {}) or {}).get(
            "compute_v1", {}
        ) or {}
        reservation_id = str(metadata.get("id") or "")
        if not reservation_id:
            continue
        indexed[reservation_id] = {
            "parent_id": str(metadata.get("parent_id") or ""),
            "region": str(status.get("region") or ""),
            "platform": str(affinity.get("platform") or ""),
            "fabric": str(affinity.get("fabric") or ""),
            "state": str(status.get("state") or ""),
            "available_gpus": _available_reserved_gpus(status),
        }
    return indexed


def find_reservation_shortfalls(
    requirements: dict[str, ReservationRequirement],
    blocks: dict[str, dict[str, Any]],
    tenant_id: str,
) -> list[ReservationShortfall]:
    """Validate explicit capacity block groups and their remaining GPU capacity."""

    shortfalls: list[ReservationShortfall] = []
    for reservation_id, requirement in requirements.items():
        block = blocks.get(reservation_id)
        if block is None:
            shortfalls.append(
                ReservationShortfall(reservation_id, "not found in tenant")
            )
            continue
        if block["parent_id"] != tenant_id:
            shortfalls.append(
                ReservationShortfall(reservation_id, "belongs to another tenant")
            )
        elif block["state"] != "STATE_ACTIVE":
            shortfalls.append(
                ReservationShortfall(
                    reservation_id, f"is not active ({block['state'] or 'unknown'})"
                )
            )
        elif block["region"] != requirement.region:
            shortfalls.append(
                ReservationShortfall(
                    reservation_id,
                    f"region {block['region']!r} does not match {requirement.region!r}",
                )
            )
        elif block["platform"] != requirement.platform:
            shortfalls.append(
                ReservationShortfall(
                    reservation_id,
                    f"platform {block['platform']!r} does not match {requirement.platform!r}",
                )
            )
        # A fabric is a workload requirement only for an explicitly clustered
        # 8-GPU pool. Single-GPU STRICT reservations are still fabric-scoped by
        # the provider, but the capacity-block ID itself supplies that placement;
        # requiring the non-clustered node pool's deliberately empty fabric to
        # equal the block's concrete fabric rejects every such reservation.
        elif requirement.fabric and block["fabric"] != requirement.fabric:
            shortfalls.append(
                ReservationShortfall(
                    reservation_id,
                    f"fabric {block['fabric']!r} does not match {requirement.fabric!r}",
                )
            )
        elif block["available_gpus"] is None:
            shortfalls.append(
                ReservationShortfall(
                    reservation_id, "remaining GPU capacity is unavailable"
                )
            )
        elif block["available_gpus"] < requirement.required_gpus:
            shortfalls.append(
                ReservationShortfall(
                    reservation_id,
                    f"needs {requirement.required_gpus} GPUs, only "
                    f"{block['available_gpus']} reserved GPUs remain",
                )
            )
    return shortfalls


def reservation_shortfall_message(shortfalls: list[ReservationShortfall]) -> str:
    """Format reservation failures without suggesting an on-demand bypass."""

    return "\n".join(
        [
            "reserved capacity is insufficient or incompatible for this fleet:",
            *(f"  - {shortfall.describe()}" for shortfall in shortfalls),
            "STRICT reservation-backed node groups will not fall back to on-demand capacity.",
        ]
    )


def required_quotas(
    clusters: Iterable[ClusterSpec], *, new_projects: int = 0
) -> dict[str, int]:
    """Aggregate the tenant quota amounts *clusters* need in one region.

    GPU-node vCPUs are deliberately not added to ``compute.instance.non-gpu.vcpu``
    -- a GPU instance is accounted against its GPU family quota instead.
    """

    needed: dict[str, int] = {}

    def add(name: str, amount: int) -> None:
        if amount > 0:
            needed[name] = needed.get(name, 0) + amount

    for cluster in clusters:
        cpu, gpu = cluster.cpu_nodes, cluster.gpu_nodes
        nodes = cluster.cpu_count() + cluster.gpu_count()
        add("mk8s.cluster.count", 1)
        # Every private worker consumes two tenant VPC allocations: its /32
        # address plus the delegated pod alias range. The managed control plane
        # endpoint is service-owned and consumes neither a tenant allocation nor
        # public-address quota; worker public IPs are disabled by the recipe.
        add("vpc.allocation.count", 2 * nodes)
        # Managed control-plane etcd is service-owned: it consumes control-plane
        # IP allocations, but not the tenant's Compute VM or disk quotas. Only
        # node-group VMs and their explicitly rendered boot disks count here.
        add("compute.instance.count", nodes)
        add("compute.disk.count", nodes)  # one boot disk per node
        if cpu and cpu.count > 0:
            cpu_disk_gib = cpu.disk_size_gib if cpu.disk_size_gib > 0 else _CPU_DISK_GIB
            add("compute.disk.size.network-ssd", cpu.count * cpu_disk_gib * _GIB)
        if gpu and gpu.count > 0:
            gpu_disk_gib = (
                gpu.disk_size_gib
                if gpu.disk_size_gib > 0
                else cluster.gpu_disk_size_gib
            )
            add("compute.disk.size.network-ssd", gpu.count * gpu_disk_gib * _GIB)
        if cpu and cpu.count > 0:
            add("compute.instance.non-gpu.vcpu", cpu.count * _preset_vcpus(cpu.preset))
        if gpu and gpu.count > 0 and not gpu.capacity_block_group:
            family = gpu_family(gpu.platform)
            if family:
                add(
                    f"compute.instance.gpu.{family}",
                    gpu.count * _preset_gpus(gpu.preset),
                )
        if cluster.resolved_enable_gpu_cluster():
            add("compute.gpucluster.count", 1)
        if cluster.enable_filestore and not cluster.existing_filestore:
            add("compute.filesystem.count", 1)
            add(_FILESYSTEM_SIZE_QUOTA, cluster.filestore_disk_size_gibibytes * _GIB)
    # A create-on-demand project needs one default/owned topology before the
    # recipe can run. Live inventory contains private and public project pools,
    # plus the network's default route table and egress route.
    add("vpc.network.count", new_projects)
    add("vpc.subnet.count", new_projects)
    add("vpc.pool.count", 2 * new_projects)
    add("vpc.routetable.count", new_projects)
    add("vpc.route.count", new_projects)
    return needed


def _quota_units(name: str) -> frozenset[str]:
    """Return explicitly supported wire units for one required quota."""

    if name in _BYTE_QUOTAS:
        return frozenset({_FILESYSTEM_SIZE_UNIT})
    if name in _VCPU_QUOTAS:
        # Current tenant payloads use count; older/catalog-specific payloads use
        # the semantically equivalent vcpu label.
        return frozenset({"count", "vcpu"})
    if name.startswith("compute.instance.gpu."):
        return frozenset({"count", "gpu"})
    return frozenset({"count"})


def _parse_uint(value: Any, *, field: str) -> int:
    """Parse a protobuf uint64 JSON value without truncating floats/bools."""

    if isinstance(value, bool):
        raise ValueError(f"{field} is not an unsigned integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"{field} is not an unsigned integer")
    if parsed < 0:
        raise ValueError(f"{field} is negative")
    return parsed


def _finite_allowance(name: str, item: dict[str, Any]) -> dict[str, Any]:
    """Interpret one finite allowance, raising when capacity is not provable."""

    spec = item["spec"]
    status = item.get("status")
    try:
        parsed_limit = _parse_uint(spec["limit"], field="limit")
        if not isinstance(status, dict):
            raise ValueError("finite allowance has no status mapping")
        state = str(status.get("state") or "")
        if state and state != "STATE_ACTIVE":
            raise ValueError(f"allowance state is not active: {state}")
        unit = str(status.get("unit") or "").strip().lower()
        compatible_units = _quota_units(name)
        if unit not in compatible_units:
            expected = ", ".join(sorted(compatible_units))
            raise ValueError(
                f"quota {name!r} reported incompatible unit {unit!r}; "
                f"expected one of: {expected}"
            )

        raw_usage = status.get("usage")
        if raw_usage not in (None, ""):
            usage = _parse_uint(raw_usage, field="usage")
            available = max(0, parsed_limit - usage)
            usage_source = "exact"
        elif status.get("usage_percentage") not in (None, ""):
            fraction = Decimal(str(status["usage_percentage"]))
            # The API defines this as a 0..1 fraction, but explicitly permits
            # values above 1 when consumption exceeds the allowance.
            if not fraction.is_finite() or fraction < 0:
                raise ValueError("usage_percentage is not a non-negative fraction")
            available = max(
                0,
                int(
                    (
                        Decimal(parsed_limit) * (Decimal(1) - fraction)
                    ).to_integral_value(rounding=ROUND_FLOOR)
                ),
            )
            usage = max(0, parsed_limit - available)
            usage_source = "fraction"
        elif status.get("usage_state") == "USAGE_STATE_NOT_USED":
            usage = 0
            available = parsed_limit
            usage_source = "not-used"
        else:
            raise ValueError(
                "finite allowance has no usage, usage_percentage, or not-used state"
            )
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "incompatible unit" in str(exc):
            raise
        raise ValueError(f"quota {name!r} has unusable capacity evidence") from exc
    return {
        "limit": parsed_limit,
        "available": available,
        "unit": unit,
        "unlimited": False,
        "usage": usage,
        "usage_source": usage_source,
    }


def _unlimited_allowance(name: str, item: dict[str, Any]) -> dict[str, Any]:
    """Interpret an allowance whose optional API ``limit`` field is omitted."""

    status = item.get("status")
    if status is not None and not isinstance(status, dict):
        raise ValueError(f"quota {name!r} has a non-mapping status")
    state = str((status or {}).get("state") or "")
    if state and state != "STATE_ACTIVE":
        raise ValueError(f"quota {name!r} allowance state is not active: {state}")
    unit = str((status or {}).get("unit") or "").strip().lower()
    if unit and unit not in _quota_units(name):
        expected = ", ".join(sorted(_quota_units(name)))
        raise ValueError(
            f"quota {name!r} reported incompatible unit {unit!r}; "
            f"expected one of: {expected}"
        )
    return {
        "limit": None,
        "available": None,
        "unit": unit,
        "unlimited": True,
        "usage_source": "unlimited",
    }


def _select_allowance(
    name: str,
    candidates: list[tuple[str, dict[str, Any]]],
    *,
    container_id: str,
) -> dict[str, Any] | None:
    """Select one deterministic allowance for the requested container scope."""

    if container_id:
        authoritative = [item for parent, item in candidates if parent == container_id]
        if authoritative:
            selected = authoritative
            # A finite legacy/unscoped candidate alongside an authoritative
            # unlimited record is contradictory rather than proof of infinity.
            if all(item["spec"].get("limit") is None for item in selected) and any(
                not parent and item["spec"].get("limit") is not None
                for parent, item in candidates
            ):
                raise ValueError(
                    f"quota {name!r} has ambiguous scoped/unscoped candidates"
                )
        else:
            # Older fixtures/clients may omit parent metadata. They are usable
            # only when no explicitly scoped target allowance was returned.
            selected = [item for parent, item in candidates if not parent]
    else:
        selected = [item for _parent, item in candidates]
    if not selected:
        return None

    finite = [item for item in selected if item["spec"].get("limit") is not None]
    if finite:
        parsed = [_finite_allowance(name, item) for item in finite]
        first = parsed[0]
        if any(candidate != first for candidate in parsed[1:]):
            raise ValueError(f"quota {name!r} has ambiguous finite candidates")
        return first
    parsed_unlimited = [_unlimited_allowance(name, item) for item in selected]
    units = {candidate["unit"] for candidate in parsed_unlimited if candidate["unit"]}
    if len(units) > 1:
        raise ValueError(f"quota {name!r} has ambiguous unlimited candidates")
    result = parsed_unlimited[0]
    result["unit"] = next(iter(units), "")
    return result


def parse_allowances(
    payload: str,
    region: str,
    required: Iterable[str] | None = None,
    *,
    container_id: str = "",
) -> dict[str, dict[str, Any]]:
    """Select and index allowances for *region* and the requested container.

    ``QuotaAllowanceSpec.limit`` is optional. An omitted limit is treated as
    unlimited only after scope selection proves that the record belongs to the
    requested tenant (or is the sole legacy unscoped evidence). A concrete limit
    always beats an unscoped unlimited collision, independent of response order.
    """

    required_names = set(required or ())
    try:
        document = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("could not parse quota allowance JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ValueError("quota allowance JSON has a non-list 'items' field")
    if document.get("next_page_token"):
        raise ValueError("quota allowance JSON is incomplete after paginated read")

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for item in document["items"]:
        if not isinstance(item, dict):
            raise ValueError("quota allowance JSON contains a non-mapping item")
        meta = item.get("metadata")
        spec = item.get("spec")
        if not isinstance(meta, dict) or not isinstance(spec, dict):
            raise ValueError("quota allowance has non-mapping metadata or spec")
        name = str(meta.get("name") or "")
        if not name or str(spec.get("region") or "") != region:
            continue
        if required_names and name not in required_names:
            continue
        scope_values = {
            str(meta.get(field) or "")
            for field in ("parent_id", "container_id")
            if meta.get(field)
        }
        if len(scope_values) > 1:
            raise ValueError(f"quota {name!r} has conflicting container metadata")
        parent_id = next(iter(scope_values), "")
        grouped.setdefault(name, []).append((parent_id, item))

    indexed: dict[str, dict[str, Any]] = {}
    for name in sorted(grouped):
        selected = _select_allowance(
            name, grouped[name], container_id=container_id
        )
        if selected is not None:
            indexed[name] = selected
    return indexed


def find_shortfalls(
    needed: dict[str, int], allowances: dict[str, dict[str, Any]], region: str
) -> list[QuotaShortfall]:
    """Return the quotas whose tenant limit is below what the fleet requires.

    Current consumption is subtracted using the exact usage count when present,
    or the authoritative live 0..1 fractional ``usage_percentage`` wire field.
    """

    shortfalls: list[QuotaShortfall] = []
    for name, required in sorted(needed.items()):
        allowance = allowances.get(name)
        if allowance is None:
            continue
        if allowance.get("unlimited"):
            continue
        available = allowance["available"]
        if required > available:
            shortfalls.append(
                QuotaShortfall(
                    name=name,
                    region=region,
                    required=required,
                    limit=allowance["limit"],
                    available=available,
                    unit=allowance["unit"],
                )
            )
    return shortfalls


def shortfall_message(shortfalls: list[QuotaShortfall], tenant_id: str) -> str:
    lines = [
        f"tenant {tenant_id} quota is too low for this fleet:",
        *(f"  - {s.describe()}" for s in shortfalls),
        "Project-level allowances only subdivide the tenant allowance, so raising "
        "these is a tenant (root) operation -- ask the Nebius account team. Deploy "
        "with --no-preflight to attempt it anyway (node groups will stay "
        "PROVISIONING while the compute API rejects each instance).",
    ]
    return "\n".join(lines)


def preflight_region(
    *,
    nebius_bin: str,
    tenant_id: str,
    region: str,
    clusters: Iterable[ClusterSpec],
    env: dict[str, str],
    new_projects: int = 0,
    profile: str = "",
    run_capture: Callable[..., Any],
    nebius_argv: Callable[[str, str], list[str]],
    on_status: Callable[[str], None] | None = None,
) -> list[QuotaShortfall]:
    """Check one region's tenant allowances against *clusters*' requirements.

    Returns quota shortfalls (empty when the fleet fits). Unreadable, incomplete,
    or malformed capacity/quota evidence raises before project creation. A
    preflight that cannot prove remaining capacity must not authorize mutation.
    """

    clusters = list(clusters)
    reservations = required_reservations(clusters, region)
    if reservations:
        reservation_result = run_capture(
            [
                *nebius_argv(nebius_bin, profile),
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
            check=False,
            timeout=120,
        )
        reservation_payload = getattr(reservation_result, "stdout", "") or ""
        if (
            getattr(reservation_result, "returncode", 1) != 0
            or not reservation_payload.strip()
        ):
            raise ValueError(
                "could not read capacity block groups; refusing to bypass STRICT "
                "reservation preflight"
            )
        blocks = parse_capacity_blocks(reservation_payload)
        reservation_shortfalls = find_reservation_shortfalls(
            reservations, blocks, tenant_id
        )
        if reservation_shortfalls:
            raise ValueError(reservation_shortfall_message(reservation_shortfalls))
        if on_status is not None:
            reserved = sum(item.required_gpus for item in reservations.values())
            on_status(
                f"validated {reserved} GPU(s) against {len(reservations)} active "
                "capacity block group(s); ordinary GPU quota excluded"
            )

    needed = required_quotas(clusters, new_projects=new_projects)
    if not needed:
        return []
    result = run_capture(
        [
            *nebius_argv(nebius_bin, profile),
            "quotas",
            "quota-allowance",
            "list",
            "--parent-id",
            tenant_id,
            "--all",
            "--format",
            "json",
        ],
        env=env,
        check=False,
        timeout=120,
    )
    if (
        getattr(result, "returncode", 1) != 0
        or not (getattr(result, "stdout", "") or "").strip()
    ):
        raise ValueError(
            f"could not read tenant quota allowances for {region}; refusing to "
            "create projects or resources without capacity evidence"
        )
    allowances = parse_allowances(
        result.stdout, region, required=needed, container_id=tenant_id
    )
    missing = sorted(set(needed) - set(allowances))
    required_missing = [
        name for name in missing if name not in _OPTIONAL_UNADVERTISED_QUOTAS
    ]
    if required_missing:
        raise ValueError(
            f"tenant quota response for {region} omitted required allowances: "
            f"{', '.join(required_missing)}"
        )
    if on_status is not None:
        for name in missing:
            on_status(
                f"quota {name} [{region}]: unadvertised optional quota in completed "
                "tenant catalog; treating it as not quota-controlled in this region"
            )
        for name, required_amount in sorted(needed.items()):
            if name not in allowances:
                continue
            allowance = allowances[name]
            unit = f" {allowance['unit']}" if allowance["unit"] else ""
            if allowance.get("unlimited"):
                on_status(
                    f"quota {name} [{region}]: needs {required_amount}{unit}; "
                    "tenant allowance is unlimited"
                )
            else:
                on_status(
                    f"quota {name} [{region}]: needs {required_amount}{unit}; "
                    f"{allowance['available']} available from limit "
                    f"{allowance['limit']} (usage evidence: "
                    f"{allowance['usage_source']})"
                )
    return find_shortfalls(needed, allowances, region)
