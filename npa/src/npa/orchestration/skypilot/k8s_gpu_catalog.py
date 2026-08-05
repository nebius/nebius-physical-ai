"""Managed-Kubernetes GPU catalog discovery for SkyPilot-backed submits.

Kubernetes clusters do not advertise the short marketing GPU names that workflow
specs are written against. SkyPilot derives its accelerator name from the node's
GPU labels, so the same physical card can surface as ``RTX6000`` (Nebius label),
``RTXPRO-6000-BLACKWELL-SERVER-EDITION`` (NVIDIA GPU-feature-discovery label), or
nothing at all while the GPU operator is still labelling nodes. A spec pinned to
``RTXPRO6000`` then fails prechecks with no hint about the name to use instead.

This module reads what the cluster actually advertises (``sky show-gpus``) and
maps a requested accelerator onto it, including the per-node quantity limit that
makes ``NAME:2`` unschedulable on a fleet of single-GPU nodes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
import subprocess
import time

from npa.orchestration.skypilot._bin import SkyBin, resolve_sky_bin
from npa.orchestration.skypilot.gpu_catalog import (
    AcceleratorRequest,
    parse_accelerator_request,
)

DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 180
DEFAULT_READINESS_TIMEOUT_SECONDS = 600
DEFAULT_READINESS_POLL_SECONDS = 10.0


class KubernetesGpuCatalogError(RuntimeError):
    """Raised when the live Kubernetes GPU catalog cannot be discovered."""


class UnsatisfiableAcceleratorError(ValueError):
    """Raised when a requested accelerator cannot be scheduled on the cluster."""


class PermanentlyUnsatisfiableAcceleratorError(UnsatisfiableAcceleratorError):
    """Raised when more discovery time cannot make the request schedulable."""


@dataclass(frozen=True)
class KubernetesGpuCatalog:
    """Accelerators a Kubernetes cluster advertises, with per-node quantities."""

    quantities_by_accelerator: dict[str, frozenset[int]]
    context: str = ""
    raw_output: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.quantities_by_accelerator

    def format_available(self) -> str:
        entries = []
        for name in sorted(self.quantities_by_accelerator, key=str.casefold):
            quantities = ", ".join(
                str(quantity)
                for quantity in sorted(self.quantities_by_accelerator[name])
            )
            entries.append(f"{name}:{{{quantities}}}")
        return "; ".join(entries) if entries else "none"

    def max_per_node(self, name: str) -> int:
        quantities = self.quantities_by_accelerator.get(name) or frozenset()
        return max(quantities) if quantities else 0


@dataclass(frozen=True)
class AcceleratorResolution:
    """The accelerator spec to submit with, and why it differs from the request."""

    requested: str
    resolved: str
    remapped: bool
    catalog: KubernetesGpuCatalog

    def describe(self) -> str:
        if not self.remapped:
            return f"accelerator {self.resolved} matches this cluster"
        return (
            f"accelerator {self.requested} is not advertised by this cluster; "
            f"using {self.resolved} instead"
        )


def _normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(name or "").casefold())


def _is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


def parse_kubernetes_gpu_catalog(
    output: str, *, context: str = ""
) -> KubernetesGpuCatalog:
    """Parse ``sky show-gpus --infra k8s`` output into per-node quantities.

    Only the per-context tables are read: they are the ones that carry
    ``REQUESTABLE_QTY_PER_NODE``, which is what decides whether ``NAME:2`` can
    ever be scheduled. The leading cluster-wide summary table and the trailing
    per-node availability table are skipped.
    """

    wanted = str(context or "").strip()
    quantities_by_accelerator: dict[str, frozenset[int]] = {}
    current_context = ""
    in_table = False
    for raw_line in str(output or "").splitlines():
        line = raw_line.strip()
        if not line:
            in_table = False
            continue
        if line.lower().startswith("context:"):
            current_context = line.split(":", 1)[1].strip()
            in_table = False
            continue
        if line.lower().startswith("kubernetes per-node gpu availability"):
            break
        if "REQUESTABLE_QTY_PER_NODE" in line:
            in_table = True
            continue
        if line.startswith("GPU") or line.startswith(("WARNING:", "Hint:", "The --")):
            in_table = False
            continue
        if not in_table:
            continue
        if wanted and current_context and current_context != wanted:
            continue
        columns = re.split(r"\s{2,}", line)
        if len(columns) < 2:
            continue
        name = columns[0].strip()
        quantities = frozenset(int(value) for value in re.findall(r"\d+", columns[1]))
        if not name or not quantities:
            continue
        merged = quantities_by_accelerator.get(name, frozenset()) | quantities
        quantities_by_accelerator[name] = merged
    return KubernetesGpuCatalog(
        quantities_by_accelerator=quantities_by_accelerator,
        context=wanted or current_context,
        raw_output=output,
    )


def discover_kubernetes_gpu_catalog(
    *,
    context: str = "",
    sky_bin: SkyBin = None,
    timeout: int = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> KubernetesGpuCatalog:
    """Ask SkyPilot which accelerators the target Kubernetes cluster advertises."""

    sky_executable = str(resolve_sky_bin(sky_bin))
    infra = f"k8s/{context}" if context else "k8s"
    cmd = [sky_executable, "show-gpus", "--infra", infra]
    execute = runner or subprocess.run
    try:
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise KubernetesGpuCatalogError(
            f"Unable to run `{' '.join(cmd)}`: {exc}"
        ) from exc
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        raise KubernetesGpuCatalogError(
            f"`{' '.join(cmd)}` failed: {detail}"
        )
    return parse_kubernetes_gpu_catalog(output, context=context)


def kubernetes_allocatable_gpu_count(
    *,
    context: str = "",
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> int | None:
    """Return Kubernetes' allocatable GPU total, or ``None`` when unavailable."""

    import json

    cmd = ["kubectl"]
    if context:
        cmd.extend(["--context", context])
    cmd.extend(["get", "nodes", "-o", "json"])
    execute = runner or subprocess.run
    try:
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
        return sum(
            int(((item.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0))
            for item in payload.get("items", [])
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wait_for_kubernetes_accelerators(
    accelerators: list[str],
    *,
    context: str = "",
    sky_bin: SkyBin = None,
    timeout: float = DEFAULT_READINESS_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_READINESS_POLL_SECONDS,
    discover: Callable[[], KubernetesGpuCatalog] | None = None,
    allocatable: Callable[[], int | None] | None = None,
    on_status: Callable[[str], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, AcceleratorResolution]:
    """Wait until both Kubernetes and SkyPilot see every requested accelerator.

    Kubernetes allocatable capacity is reported independently because it can be
    healthy minutes before SkyPilot's catalog has consumed the node labels.  A
    timeout is observational only: it never deletes or scales the cluster.
    """

    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("GPU readiness timeout and poll interval must be positive")
    requested = [item for item in accelerators if str(item).strip()]
    if not requested:
        return {}
    get_catalog = discover or (
        lambda: discover_kubernetes_gpu_catalog(
            context=context,
            sky_bin=sky_bin,
            timeout=max(1, min(int(timeout), DEFAULT_DISCOVERY_TIMEOUT_SECONDS)),
        )
    )
    get_allocatable = allocatable or (
        lambda: kubernetes_allocatable_gpu_count(context=context)
    )
    deadline = monotonic() + timeout
    last_failure = "SkyPilot accelerator catalog has not been queried"
    attempt = 0
    while True:
        attempt += 1
        count = get_allocatable()
        if on_status:
            shown = "unknown" if count is None else str(count)
            on_status(
                f"GPU readiness attempt {attempt}: Kubernetes allocatable={shown}; "
                "SkyPilot discovery=pending"
            )
        try:
            catalog = get_catalog()
            resolved = {
                accelerator: resolve_kubernetes_accelerator(accelerator, catalog=catalog)
                for accelerator in requested
            }
        except PermanentlyUnsatisfiableAcceleratorError:
            # The catalog is already populated and proves that one node can
            # never satisfy the quantity (or that the name is ambiguous).
            # Waiting for the same discovery result only burns the readiness
            # timeout and hides the actionable error.
            raise
        except (KubernetesGpuCatalogError, UnsatisfiableAcceleratorError) as exc:
            last_failure = str(exc)
        else:
            if on_status:
                on_status(
                    "GPU readiness: Kubernetes allocatable="
                    + ("unknown" if count is None else str(count))
                    + "; SkyPilot discovery=ready ("
                    + ", ".join(item.resolved for item in resolved.values())
                    + ")"
                )
            return resolved
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise KubernetesGpuCatalogError(
                "Timed out after "
                f"{timeout:g}s waiting for SkyPilot to discover a compatible GPU in "
                f"context {context or '<current>'}. Kubernetes allocatable="
                f"{'unknown' if count is None else count}; last SkyPilot result: "
                f"{last_failure}. Capacity was left running; retry the same submit or run "
                "`npa provision-if-absent --gpu-readiness-timeout <seconds>`."
            )
        sleeper(min(poll_interval, remaining))


def spec_accelerators(resources: object) -> list[str]:
    """Return the distinct Kubernetes accelerator specs declared by a spec's profiles.

    Only ``cloud: kubernetes`` profiles are considered; Nebius VM profiles are
    validated against the VM catalog instead.
    """

    found: list[str] = []
    if not isinstance(resources, dict):
        return found
    for profile in resources.values():
        if not isinstance(profile, dict):
            continue
        cloud = str(profile.get("cloud") or "").strip().casefold()
        if cloud not in {"kubernetes", "k8s"}:
            continue
        accelerator = str(profile.get("accelerators") or "").strip()
        if accelerator and accelerator not in found:
            found.append(accelerator)
    return found


def context_from_infra(infra: str) -> str:
    """Extract the Kubernetes context from a SkyPilot ``--infra`` value."""

    value = str(infra or "").strip()
    for prefix in ("k8s/", "kubernetes/"):
        if value.startswith(prefix):
            return value[len(prefix):].strip()
    return ""


def _candidate_names(request: AcceleratorRequest, catalog: KubernetesGpuCatalog) -> list[str]:
    """Return catalog names matching ``request``, strongest match tier first."""

    names = list(catalog.quantities_by_accelerator)
    exact = [name for name in names if name.casefold() == request.name.casefold()]
    if exact:
        return exact
    wanted = _normalize(request.name)
    if not wanted:
        return []
    prefixed = [name for name in names if _normalize(name).startswith(wanted)]
    if prefixed:
        return prefixed
    # `RTX6000` (Nebius node label) must still reach
    # `RTXPRO-6000-BLACKWELL-SERVER-EDITION` (GPU-feature-discovery label).
    return [name for name in names if _is_subsequence(wanted, _normalize(name))]


def resolve_kubernetes_accelerator(
    accelerator: str,
    *,
    catalog: KubernetesGpuCatalog,
) -> AcceleratorResolution:
    """Map a requested accelerator spec onto what the cluster advertises.

    Raises ``UnsatisfiableAcceleratorError`` when the name cannot be matched, is
    ambiguous, or the requested per-task quantity exceeds what a single node can
    provide (SkyPilot places all GPUs of one task on one node).
    """

    request = parse_accelerator_request(accelerator)
    if catalog.is_empty:
        raise UnsatisfiableAcceleratorError(
            "This cluster advertises no GPUs to SkyPilot. "
            "Suggested action: wait for the NVIDIA GPU operator to finish labelling "
            "nodes, then rerun; `kubectl get nodes -L nvidia.com/gpu.product` shows progress."
        )
    matches = _candidate_names(request, catalog)
    if not matches:
        raise UnsatisfiableAcceleratorError(
            f"Accelerator {request.name!r} is not advertised by this cluster. "
            f"Available: {catalog.format_available()}. "
            f"Suggested action: export NPA_WORKFLOW_GPU_ACCELERATOR=<name>:<qty> "
            "using one of the names above."
        )
    if len(matches) > 1:
        options = ", ".join(sorted(matches))
        raise PermanentlyUnsatisfiableAcceleratorError(
            f"Accelerator {request.name!r} matches more than one advertised accelerator "
            f"({options}). Suggested action: export NPA_WORKFLOW_GPU_ACCELERATOR=<name>:<qty> "
            "with the exact name you want."
        )
    resolved_name = matches[0]
    allowed = catalog.quantities_by_accelerator[resolved_name]
    if request.quantity not in allowed:
        max_per_node = catalog.max_per_node(resolved_name)
        if request.quantity > max_per_node:
            raise PermanentlyUnsatisfiableAcceleratorError(
                f"{resolved_name}:{request.quantity} cannot be scheduled: this cluster's "
                f"nodes offer at most {max_per_node} of that GPU each, and SkyPilot places "
                "all GPUs of one task on a single node. Adding nodes does not help. "
                f"Suggested action: export NPA_WORKFLOW_GPU_ACCELERATOR={resolved_name}:{max_per_node} "
                "and let the workflow fan out across steps instead of across GPUs."
            )
        offered = ", ".join(str(value) for value in sorted(allowed))
        raise PermanentlyUnsatisfiableAcceleratorError(
            f"{resolved_name}:{request.quantity} is not a requestable quantity on this "
            f"cluster (it offers {offered} per node). Suggested action: export "
            f"NPA_WORKFLOW_GPU_ACCELERATOR={resolved_name}:<one of {offered}>."
        )
    resolved = f"{resolved_name}:{request.quantity}"
    return AcceleratorResolution(
        requested=request.spec,
        resolved=resolved,
        remapped=resolved != request.spec,
        catalog=catalog,
    )
