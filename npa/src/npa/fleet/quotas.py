"""Tenant quota preflight for fleet deploys.

Managed Kubernetes accepts a node group whose instances it cannot actually
create: the node group is stored, then the compute API rejects each instance
with a ``QuotaFailure`` and mk8s keeps retrying. Terraform sees only
``Still creating...`` and blocks until the per-cluster timeout, so a quota wall
looks like a hang rather than a rejection.

This module derives what a fleet's clusters need, compares it against the
*tenant's* quota allowances for the target region, and lets the caller fail fast
with the shortfall. Project allowances only subdivide the tenant allowance, so
the tenant is the meaningful container to check; raising a tenant allowance is a
``root-g00root`` operation that a tenant-scoped service account cannot perform,
which makes an early, explicit error much more useful than a timeout.

Only read APIs are used (``nebius quotas quota-allowance list``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from npa.fleet.spec import ClusterSpec

_GIB = 1024**3

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
    unit: str = ""

    def describe(self) -> str:
        return (
            f"{self.name} [{self.region}]: needs {self.required}{f' {self.unit}' if self.unit else ''}, "
            f"tenant limit {self.limit}"
        )


def required_quotas(clusters: Iterable[ClusterSpec]) -> dict[str, int]:
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
        add("compute.instance.count", nodes)
        add("compute.disk.count", nodes)  # one boot disk per node
        if cpu and cpu.count > 0:
            add("compute.instance.non-gpu.vcpu", cpu.count * _preset_vcpus(cpu.preset))
        if gpu and gpu.count > 0:
            family = gpu_family(gpu.platform)
            if family:
                add(f"compute.instance.gpu.{family}", gpu.count * _preset_gpus(gpu.preset))
        if cluster.resolved_enable_gpu_cluster():
            add("compute.gpucluster.count", 1)
        if cluster.enable_filestore and not cluster.existing_filestore:
            add("compute.filesystem.count", 1)
            add(
                "compute.filesystem.size.network-ssd",
                cluster.filestore_disk_size_gibibytes * _GIB,
            )
    return needed


def parse_allowances(payload: str, region: str) -> dict[str, dict[str, Any]]:
    """Index ``quota-allowance list`` JSON by quota name for *region*.

    An allowance with an unset ``limit`` means "no limit at this container"
    (project allowances are usually unset and draw on the tenant pool), so those
    entries are skipped rather than read as a limit of zero.
    """

    try:
        items = json.loads(payload or "{}").get("items", [])
    except json.JSONDecodeError:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in items if isinstance(items, list) else []:
        meta = item.get("metadata", {}) or {}
        spec = item.get("spec", {}) or {}
        name = str(meta.get("name") or "")
        if not name or str(spec.get("region") or "") != region:
            continue
        limit = spec.get("limit")
        if limit is None:
            continue
        try:
            indexed[name] = {
                "limit": int(limit),
                "unit": str((item.get("status", {}) or {}).get("unit") or ""),
            }
        except (TypeError, ValueError):
            continue
    return indexed


def find_shortfalls(needed: dict[str, int], allowances: dict[str, dict[str, Any]], region: str) -> list[QuotaShortfall]:
    """Return the quotas whose tenant limit is below what the fleet requires.

    Current consumption is not subtracted: the allowance API reports usage only
    as a percentage, and treating that as an exact figure could reject a
    deployable fleet. This therefore flags definite walls (most importantly a
    limit of 0) and stays quiet when the limit merely looks tight.
    """

    shortfalls: list[QuotaShortfall] = []
    for name, required in sorted(needed.items()):
        allowance = allowances.get(name)
        if allowance is None:  # not advertised for this region -> nothing to assert
            continue
        if required > allowance["limit"]:
            shortfalls.append(
                QuotaShortfall(
                    name=name,
                    region=region,
                    required=required,
                    limit=allowance["limit"],
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
    profile: str = "",
    run_capture: Callable[..., Any],
    nebius_argv: Callable[[str, str], list[str]],
    on_status: Callable[[str], None] | None = None,
) -> list[QuotaShortfall]:
    """Check one region's tenant allowances against *clusters*' requirements.

    Returns the shortfalls (empty when the fleet fits). A quota API that cannot
    be read is reported and treated as "no shortfall": losing the preflight must
    not block a deploy that would otherwise succeed.
    """

    needed = required_quotas(clusters)
    if not needed:
        return []
    result = run_capture(
        [
            *nebius_argv(nebius_bin, profile),
            "quotas", "quota-allowance", "list", "--parent-id", tenant_id, "--format", "json",
        ],
        env=env,
        check=False,
        timeout=120,
    )
    if getattr(result, "returncode", 1) != 0 or not (getattr(result, "stdout", "") or "").strip():
        if on_status is not None:
            on_status(
                f"WARNING: could not read tenant quota allowances for {region}; "
                "skipping quota preflight"
            )
        return []
    allowances = parse_allowances(result.stdout, region)
    if not allowances:
        if on_status is not None:
            on_status(
                f"WARNING: no quota allowances reported for region {region}; "
                "skipping quota preflight"
            )
        return []
    return find_shortfalls(needed, allowances, region)
