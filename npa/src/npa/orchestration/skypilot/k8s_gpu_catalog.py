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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
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


Kubeconfig = str | os.PathLike[str] | None


def exact_kubernetes_context_config(context: str) -> str:
    """Return SkyPilot's exact single-context configuration override."""

    exact_context = str(context or "").strip()
    if not exact_context:
        return ""
    allowed_contexts = json.dumps([exact_context], separators=(",", ":"))
    return f"kubernetes.allowed_contexts={allowed_contexts}"


def _kubeconfig_env(kubeconfig: Kubeconfig) -> dict[str, str] | None:
    if kubeconfig is None or not os.fspath(kubeconfig).strip():
        return None
    env = os.environ.copy()
    env["KUBECONFIG"] = str(Path(kubeconfig).expanduser())
    return env


def _kubectl_failure(*, action: str, returncode: int, output: str) -> str:
    """Classify kubectl failures without echoing raw private resource details."""

    lowered = str(output or "").casefold()
    if "forbidden" in lowered or "cannot list resource" in lowered:
        return (
            f"Kubernetes RBAC denied {action} in the exact context; grant the "
            "operator get/list access to nodes and patch/update access when label "
            "repair is requested"
        )
    if "unauthorized" in lowered or "authentication" in lowered:
        return (
            f"Kubernetes authentication failed while {action} in the exact context; "
            "refresh the selected kubeconfig credentials"
        )
    return (
        f"kubectl failed while {action} in the exact context (exit {returncode}); "
        "verify the selected kubeconfig, context, API reachability, and node RBAC"
    )


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
class KubernetesGpuNode:
    """Per-node schedulable/free GPU evidence used by gang preflight."""

    name: str
    ready: bool
    schedulable: bool
    products: tuple[str, ...]
    capacity: int
    allocatable: int
    committed: int
    free: int
    exclusion: str = ""
    allocatable_cpu_millis: int = 0
    committed_cpu_millis: int = 0
    free_cpu_millis: int = 0
    allocatable_memory_bytes: int = 0
    committed_memory_bytes: int = 0
    free_memory_bytes: int = 0
    allocatable_pods: int = 0
    committed_pods: int = 0
    free_pod_slots: int = 0
    labels: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ready": self.ready,
            "schedulable": self.schedulable,
            "products": list(self.products),
            "capacity": self.capacity,
            "allocatable": self.allocatable,
            "committed": self.committed,
            "free": self.free,
            "exclusion": self.exclusion,
            "allocatable_cpu_millis": self.allocatable_cpu_millis,
            "committed_cpu_millis": self.committed_cpu_millis,
            "free_cpu_millis": self.free_cpu_millis,
            "allocatable_memory_bytes": self.allocatable_memory_bytes,
            "committed_memory_bytes": self.committed_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "allocatable_pods": self.allocatable_pods,
            "committed_pods": self.committed_pods,
            "free_pod_slots": self.free_pod_slots,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True)
class KubernetesGpuInventory:
    """Provider-independent Kubernetes GPU readiness evidence."""

    context: str
    ready_nodes: int
    eligible_gpu_nodes: int
    capacity: int
    allocatable: int
    products: tuple[str, ...]
    node_labels: dict[str, dict[str, str]]
    error: str = ""
    nodes: tuple[KubernetesGpuNode, ...] = ()
    unbound_pending_gpu_pods: int = 0
    unbound_pending_gpu_requests: int = 0

    def to_dict(self) -> dict[str, object]:
        product = (
            self.products[0]
            if len(self.products) == 1
            else "multiple"
            if self.products
            else "unknown/unlabeled"
        )
        return {
            "context": self.context,
            "ready_nodes": self.ready_nodes,
            "eligible_gpu_nodes": self.eligible_gpu_nodes,
            "capacity": self.capacity,
            "allocatable": self.allocatable,
            "allocatable_gpus": self.allocatable,
            "products": list(self.products),
            "accelerator_product": product,
            "label_readiness": "ready"
            if self.products or self.allocatable == 0
            else "blocked_missing_product_label",
            "node_labels": self.node_labels,
            "error": self.error,
            "nodes": [node.to_dict() for node in self.nodes],
            "unbound_pending_gpu_pods": self.unbound_pending_gpu_pods,
            "unbound_pending_gpu_requests": self.unbound_pending_gpu_requests,
        }


def _gpu_quantity(container: object) -> int:
    if not isinstance(container, dict):
        return 0
    resources = container.get("resources") or {}
    if not isinstance(resources, dict):
        return 0
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    raw = (
        requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0))
        if isinstance(requests, dict) and isinstance(limits, dict)
        else 0
    )
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        raise KubernetesGpuCatalogError(f"invalid pod nvidia.com/gpu request {raw!r}")


def _cpu_millis(raw: object) -> int:
    value = str(raw or "0").strip()
    try:
        if value.endswith("m"):
            return max(0, int(Decimal(value[:-1])))
        if value.endswith("u"):
            return max(0, int(Decimal(value[:-1]) / 1000))
        if value.endswith("n"):
            return max(0, int(Decimal(value[:-1]) / 1_000_000))
        return max(0, int(Decimal(value) * 1000))
    except (InvalidOperation, ValueError) as exc:
        raise KubernetesGpuCatalogError(
            f"invalid Kubernetes CPU quantity {raw!r}"
        ) from exc


_MEMORY_FACTORS = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
    "K": 1000,
    "m": Decimal("0.001"),
}


def _memory_bytes(raw: object) -> int:
    value = str(raw or "0").strip()
    suffix = next((item for item in _MEMORY_FACTORS if value.endswith(item)), "")
    number = value[: -len(suffix)] if suffix else value
    try:
        return max(0, int(Decimal(number) * _MEMORY_FACTORS.get(suffix, 1)))
    except (InvalidOperation, ValueError) as exc:
        raise KubernetesGpuCatalogError(
            f"invalid Kubernetes memory quantity {raw!r}"
        ) from exc


def _container_request(container: object, resource: str) -> int:
    if not isinstance(container, dict):
        return 0
    resources = container.get("resources") or {}
    if not isinstance(resources, dict):
        return 0
    requests = resources.get("requests") or {}
    if not isinstance(requests, dict):
        return 0
    raw = requests.get(resource, 0)
    return _cpu_millis(raw) if resource == "cpu" else _memory_bytes(raw)


def _pod_commitment(pod: object) -> tuple[str, int, int, int, int]:
    if not isinstance(pod, dict):
        return "", 0, 0, 0, 0
    spec = pod.get("spec") or {}
    status = pod.get("status") or {}
    if not isinstance(spec, dict) or not isinstance(status, dict):
        return "", 0, 0, 0, 0
    if str(status.get("phase") or "") in {"Succeeded", "Failed"}:
        return "", 0, 0, 0, 0
    node_name = str(spec.get("nodeName") or "").strip()
    containers = spec.get("containers") or []
    init_containers = spec.get("initContainers") or []
    regular_gpu = sum(_gpu_quantity(item) for item in containers)
    init_gpu = max(
        (_gpu_quantity(item) for item in spec.get("initContainers") or []),
        default=0,
    )
    regular_cpu = sum(_container_request(item, "cpu") for item in containers)
    init_cpu = max(
        (_container_request(item, "cpu") for item in init_containers), default=0
    )
    regular_memory = sum(_container_request(item, "memory") for item in containers)
    init_memory = max(
        (_container_request(item, "memory") for item in init_containers), default=0
    )
    overhead = spec.get("overhead") or {}
    overhead_cpu = (
        _cpu_millis(overhead.get("cpu", 0)) if isinstance(overhead, dict) else 0
    )
    overhead_memory = (
        _memory_bytes(overhead.get("memory", 0)) if isinstance(overhead, dict) else 0
    )
    return (
        node_name,
        max(regular_gpu, init_gpu),
        max(regular_cpu, init_cpu) + overhead_cpu,
        max(regular_memory, init_memory) + overhead_memory,
        1,
    )


def discover_kubernetes_gpu_inventory(
    *,
    context: str = "",
    kubeconfig: Kubeconfig = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> KubernetesGpuInventory:
    """Read Ready/schedulable nodes, GPU quantities, and raw product labels."""

    cmd = ["kubectl"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        cmd.extend(["--kubeconfig", str(Path(kubeconfig).expanduser())])
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
            env=_kubeconfig_env(kubeconfig),
        )
        if result.returncode != 0:
            return KubernetesGpuInventory(
                context,
                0,
                0,
                0,
                0,
                (),
                {},
                _kubectl_failure(
                    action="reading GPU node inventory",
                    returncode=result.returncode,
                    output=result.stderr or result.stdout,
                ),
            )
        payload = json.loads(result.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return KubernetesGpuInventory(
            context, 0, 0, 0, 0, (), {}, "kubectl node inventory unavailable"
        )
    pod_cmd = ["kubectl"]
    if kubeconfig is not None and os.fspath(kubeconfig).strip():
        pod_cmd.extend(["--kubeconfig", str(Path(kubeconfig).expanduser())])
    if context:
        pod_cmd.extend(["--context", context])
    pod_cmd.extend(["get", "pods", "--all-namespaces", "-o", "json"])
    try:
        pod_result = execute(
            pod_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=_kubeconfig_env(kubeconfig),
        )
        if pod_result.returncode != 0:
            return KubernetesGpuInventory(
                context,
                0,
                0,
                0,
                0,
                (),
                {},
                "kubectl pod inventory failed; free shared GPU capacity is unknown",
            )
        pod_payload = json.loads(pod_result.stdout or "{}")
        committed_by_node: dict[str, tuple[int, int, int, int]] = {}
        unbound_pending_gpu_pods = 0
        unbound_pending_gpu_requests = 0
        for pod in pod_payload.get("items", []):
            node_name, gpu, cpu, memory, pod_slots = _pod_commitment(pod)
            if node_name:
                prior = committed_by_node.get(node_name, (0, 0, 0, 0))
                committed_by_node[node_name] = (
                    prior[0] + gpu,
                    prior[1] + cpu,
                    prior[2] + memory,
                    prior[3] + pod_slots,
                )
            elif gpu > 0:
                # An unbound active GPU pod has already made a claim on shared
                # capacity, but Kubernetes has not authoritatively chosen which
                # compatible node will satisfy it. Do not pretend every node is
                # still free for a new gang.
                unbound_pending_gpu_pods += 1
                unbound_pending_gpu_requests += gpu
    except (OSError, ValueError, subprocess.SubprocessError, KubernetesGpuCatalogError):
        return KubernetesGpuInventory(
            context,
            0,
            0,
            0,
            0,
            (),
            {},
            "kubectl pod inventory unavailable; free shared GPU capacity is unknown",
        )

    ready_nodes = 0
    eligible_nodes = 0
    capacity = 0
    allocatable = 0
    products: set[str] = set()
    labels_by_node: dict[str, dict[str, str]] = {}
    node_records: list[KubernetesGpuNode] = []
    for item in payload.get("items", []):
        metadata = item.get("metadata") or {}
        spec = item.get("spec") or {}
        status = item.get("status") or {}
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions") or []
            if isinstance(condition, dict)
        )
        if ready:
            ready_nodes += 1
        disallowed_taint = any(
            str(taint.get("effect") or "") == "NoExecute"
            or (
                str(taint.get("effect") or "") == "NoSchedule"
                and str(taint.get("key") or "") != "nvidia.com/gpu"
            )
            for taint in spec.get("taints") or []
            if isinstance(taint, dict)
        )
        node_allocatable = int(
            (status.get("allocatable") or {}).get("nvidia.com/gpu", 0)
        )
        node_capacity = int((status.get("capacity") or {}).get("nvidia.com/gpu", 0))
        node_cpu = _cpu_millis((status.get("allocatable") or {}).get("cpu", 0))
        node_memory = _memory_bytes((status.get("allocatable") or {}).get("memory", 0))
        try:
            node_pods = max(0, int((status.get("allocatable") or {}).get("pods", 0)))
        except (TypeError, ValueError):
            node_pods = 0
        all_labels = {
            str(key): str(value)
            for key, value in (metadata.get("labels") or {}).items()
        }
        raw_labels = {
            str(key): str(value)
            for key, value in all_labels.items()
            if "gpu" in str(key).casefold() or "accelerator" in str(key).casefold()
        }
        name = str(metadata.get("name") or "")
        if name:
            labels_by_node[name] = raw_labels
        blocked = bool(spec.get("unschedulable")) or disallowed_taint
        node_products: set[str] = set()
        for key, value in raw_labels.items():
            if (
                key
                in {
                    "nvidia.com/gpu.product",
                    "nebius.com/gpu",
                    "nebius.com/gpu-name",
                    "node.kubernetes.io/instance-type",
                    "skypilot.co/accelerator",
                }
                or "product" in key.casefold()
            ):
                if value:
                    node_products.add(value)
        committed, committed_cpu, committed_memory, committed_pods = (
            committed_by_node.get(name, (0, 0, 0, 0))
        )
        free = max(0, node_allocatable - committed)
        free_cpu = max(0, node_cpu - committed_cpu)
        free_memory = max(0, node_memory - committed_memory)
        free_pods = max(0, node_pods - committed_pods)
        exclusion = (
            "not-ready"
            if not ready
            else "cordoned-or-unsupported-taint"
            if blocked
            else "no-allocatable-gpu"
            if node_allocatable <= 0
            else ""
        )
        if name:
            node_records.append(
                KubernetesGpuNode(
                    name=name,
                    ready=ready,
                    schedulable=not blocked,
                    products=tuple(sorted(node_products)),
                    capacity=node_capacity,
                    allocatable=node_allocatable,
                    committed=committed,
                    free=free,
                    exclusion=exclusion,
                    allocatable_cpu_millis=node_cpu,
                    committed_cpu_millis=committed_cpu,
                    free_cpu_millis=free_cpu,
                    allocatable_memory_bytes=node_memory,
                    committed_memory_bytes=committed_memory,
                    free_memory_bytes=free_memory,
                    allocatable_pods=node_pods,
                    committed_pods=committed_pods,
                    free_pod_slots=free_pods,
                    labels=tuple(sorted(all_labels.items())),
                )
            )
        if ready and not blocked and node_allocatable > 0:
            eligible_nodes += 1
            allocatable += node_allocatable
            capacity += node_capacity
            # Report one canonical product per node. Nebius and GPU Feature
            # Discovery can advertise two aliases for the same physical card;
            # treating both as separate products makes a homogeneous pool look
            # heterogeneous in CLI readiness output.
            product = next(
                (
                    raw_labels.get(key, "")
                    for key in ("nvidia.com/gpu.product", "nebius.com/gpu-name")
                    if raw_labels.get(key)
                ),
                "",
            )
            if not product:
                product = next(
                    (
                        value
                        for key, value in sorted(raw_labels.items())
                        if "product" in key.casefold() and value
                    ),
                    "",
                )
            if product:
                products.add(product)
    return KubernetesGpuInventory(
        context=context,
        ready_nodes=ready_nodes,
        eligible_gpu_nodes=eligible_nodes,
        capacity=capacity,
        allocatable=allocatable,
        products=tuple(sorted(products)),
        node_labels=labels_by_node,
        nodes=tuple(sorted(node_records, key=lambda item: item.name)),
        unbound_pending_gpu_pods=unbound_pending_gpu_pods,
        unbound_pending_gpu_requests=unbound_pending_gpu_requests,
    )


def _matches_node_selector_requirement(
    *, name: str, labels: Mapping[str, str], requirement: Mapping[str, object]
) -> bool:
    key = str(requirement.get("key") or "").strip()
    operator = str(requirement.get("operator") or "").strip()
    raw_values = requirement.get("values") or []
    values = [str(item) for item in raw_values] if isinstance(raw_values, list) else []
    if not key or operator not in {"In", "NotIn", "Exists", "DoesNotExist", "Gt", "Lt"}:
        raise KubernetesGpuCatalogError(
            "resource-profile required nodeAffinity has an unsupported selector"
        )
    actual = name if key == "metadata.name" else labels.get(key)
    if operator == "In":
        return actual is not None and actual in values
    if operator == "NotIn":
        return actual is not None and actual not in values
    if operator == "Exists":
        return actual is not None
    if operator == "DoesNotExist":
        return actual is None
    if len(values) != 1 or actual is None:
        raise KubernetesGpuCatalogError(
            "resource-profile Gt/Lt nodeAffinity requires one numeric value"
        )
    try:
        left, right = int(actual), int(values[0])
    except ValueError as exc:
        raise KubernetesGpuCatalogError(
            "resource-profile Gt/Lt nodeAffinity values must be integers"
        ) from exc
    return left > right if operator == "Gt" else left < right


def _node_matches_pod_spec(
    node: KubernetesGpuNode, pod_spec: Mapping[str, object]
) -> bool:
    labels = dict(node.labels)
    node_name = str(pod_spec.get("nodeName") or "").strip()
    if node_name and node.name != node_name:
        return False
    selector = pod_spec.get("nodeSelector") or {}
    if not isinstance(selector, Mapping):
        raise KubernetesGpuCatalogError(
            "resource-profile kubernetes.pod_config.spec.nodeSelector must be a mapping"
        )
    if any(labels.get(str(key)) != str(value) for key, value in selector.items()):
        return False
    affinity = pod_spec.get("affinity") or {}
    if not isinstance(affinity, Mapping):
        raise KubernetesGpuCatalogError(
            "resource-profile kubernetes pod affinity must be a mapping"
        )
    if affinity.get("podAffinity") or affinity.get("podAntiAffinity"):
        raise KubernetesGpuCatalogError(
            "existing-capacity preflight cannot authoritatively evaluate required "
            "pod affinity/anti-affinity; remove it or preflight with provider evidence"
        )
    node_affinity = affinity.get("nodeAffinity") or {}
    if not isinstance(node_affinity, Mapping):
        raise KubernetesGpuCatalogError(
            "resource-profile nodeAffinity must be a mapping"
        )
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution") or {}
    if not required:
        return True
    if not isinstance(required, Mapping):
        raise KubernetesGpuCatalogError(
            "resource-profile required nodeAffinity must be a mapping"
        )
    raw_terms = required.get("nodeSelectorTerms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise KubernetesGpuCatalogError(
            "resource-profile required nodeAffinity must contain nodeSelectorTerms"
        )
    for term in raw_terms:
        if not isinstance(term, Mapping):
            raise KubernetesGpuCatalogError(
                "resource-profile nodeSelectorTerm must be a mapping"
            )
        expressions = term.get("matchExpressions") or []
        fields = term.get("matchFields") or []
        if not isinstance(expressions, list) or not isinstance(fields, list):
            raise KubernetesGpuCatalogError(
                "resource-profile node affinity requirements must be lists"
            )
        requirements = [*expressions, *fields]
        if all(
            isinstance(item, Mapping)
            and _matches_node_selector_requirement(
                name=node.name, labels=labels, requirement=item
            )
            for item in requirements
        ):
            return True
    return False


def preflight_kubernetes_gpu_gang(
    inventory: KubernetesGpuInventory,
    *,
    accelerator: str,
    node_count: int,
    cpus: object = 0,
    memory: object = 0,
    allowed_nodes: tuple[str, ...] | list[str] | None = None,
    pod_spec: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require N distinct compatible nodes with enough currently free GPUs."""

    if inventory.error:
        raise KubernetesGpuCatalogError(inventory.error)
    if inventory.unbound_pending_gpu_pods:
        raise KubernetesGpuCatalogError(
            "free shared GPU capacity is indeterminate: Kubernetes has "
            f"{inventory.unbound_pending_gpu_pods} active unbound GPU pod(s) "
            f"requesting {inventory.unbound_pending_gpu_requests} GPU(s); wait for "
            "authoritative placement or remove only the owned pending workload"
        )
    request = parse_accelerator_request(accelerator)
    expected = int(node_count)
    if expected < 1:
        raise ValueError("node_count must be positive")
    wanted = _normalize(request.name)
    requested_cpu = _cpu_millis(cpus)
    requested_memory = _memory_bytes(memory)
    allowed = {str(name).strip() for name in (allowed_nodes or ()) if str(name).strip()}
    selected_pod_spec = pod_spec or {}
    if not isinstance(selected_pod_spec, Mapping):
        raise KubernetesGpuCatalogError("resource-profile pod spec must be a mapping")
    if selected_pod_spec.get("topologySpreadConstraints"):
        raise KubernetesGpuCatalogError(
            "existing-capacity preflight cannot authoritatively evaluate topology "
            "spread constraints"
        )
    alias_group = next(
        (group for group in _EXPLICIT_ACCELERATOR_ALIASES if wanted in group),
        frozenset({wanted}),
    )

    def compatible(node: KubernetesGpuNode) -> bool:
        return any(_normalize(product) in alias_group for product in node.products)

    candidates = [
        node
        for node in inventory.nodes
        if node.ready
        and node.schedulable
        and (not allowed or node.name in allowed)
        and _node_matches_pod_spec(node, selected_pod_spec)
        and compatible(node)
        and node.free >= request.quantity
        and node.free_cpu_millis >= requested_cpu
        and node.free_memory_bytes >= requested_memory
        and node.free_pod_slots >= 1
    ]
    if len(candidates) < expected:
        raise UnsatisfiableAcceleratorError(
            f"Kubernetes context {inventory.context or '<current>'} has "
            f"{len(candidates)} distinct compatible schedulable node(s) with at "
            f"least {request.quantity} free {request.name} GPU(s), but the gang "
            f"requires {expected}. Active pod GPU commitments are subtracted; "
            f"each rank also requires {requested_cpu / 1000:g} CPU and "
            f"{requested_memory} memory bytes. Active pod GPU/CPU/memory requests "
            "and allocatable pod slots are checked. SkyPilot allowed_nodes affinity "
            f"is applied ({sorted(allowed) if allowed else 'unrestricted'}); "
            "aggregate capacity on one node cannot satisfy "
            "multiple gang ranks."
        )
    return {
        "context": inventory.context,
        "accelerator": request.spec,
        "node_count": expected,
        "compatible_free_nodes": len(candidates),
        "selected_nodes": [node.name for node in candidates[:expected]],
        "cpus_per_node": requested_cpu / 1000,
        "memory_bytes_per_node": requested_memory,
        "allowed_nodes": sorted(allowed),
    }


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


_EXPLICIT_ACCELERATOR_ALIASES = (
    frozenset(
        {
            "rtx6000",
            "rtxpro6000",
            "rtxpro6000blackwellserveredition",
            "nvidiartxpro6000blackwellserveredition",
        }
    ),
)


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
    kubeconfig: Kubeconfig = None,
    sky_bin: SkyBin = None,
    timeout: int = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> KubernetesGpuCatalog:
    """Ask SkyPilot which accelerators the target Kubernetes cluster advertises."""

    sky_executable = str(resolve_sky_bin(sky_bin))
    infra = f"k8s/{context}" if context else "k8s"
    cmd = [sky_executable, "show-gpus", "--infra", infra]
    config_override = exact_kubernetes_context_config(context)
    if config_override:
        cmd[2:2] = ["--config", config_override]
    execute = runner or subprocess.run
    try:
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=_kubeconfig_env(kubeconfig),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise KubernetesGpuCatalogError(
            f"Unable to run `{' '.join(cmd)}`: {exc}"
        ) from exc
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0 and "kubernetes is not enabled" in output.casefold():
        check_cmd = [sky_executable, "check"]
        if config_override:
            check_cmd.extend(["--config", config_override])
        check_cmd.append("kubernetes")
        checked = execute(
            check_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=_kubeconfig_env(kubeconfig),
        )
        checked_output = "\n".join(
            part for part in (checked.stdout, checked.stderr) if part
        )
        if (
            checked.returncode != 0
            or "kubernetes: disabled" in checked_output.casefold()
        ):
            detail = (
                checked.stderr or checked.stdout or f"exit {checked.returncode}"
            ).strip()
            raise KubernetesGpuCatalogError(
                "SkyPilot Kubernetes discovery was disabled after API-server "
                f"restart and `sky check kubernetes` failed: {detail}"
            )
        result = execute(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=_kubeconfig_env(kubeconfig),
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        raise KubernetesGpuCatalogError(f"`{' '.join(cmd)}` failed: {detail}")
    return parse_kubernetes_gpu_catalog(output, context=context)


_KNOWN_SKYPILOT_LABELS = {
    "b200": "B200",
    "nvidiab200": "B200",
    "rtx6000": "rtxpro6000",
    "rtxpro6000": "rtxpro6000",
    "rtxpro6000blackwellserveredition": "rtxpro6000",
    "nvidiartxpro6000blackwellserveredition": "rtxpro6000",
}


def _known_skypilot_label(labels: dict[str, str]) -> str:
    for key in ("nvidia.com/gpu.product", "nebius.com/gpu-name"):
        normalized = _normalize(labels.get(key, ""))
        if normalized in _KNOWN_SKYPILOT_LABELS:
            return _KNOWN_SKYPILOT_LABELS[normalized]
    return ""


def label_known_kubernetes_gpus_for_skypilot(
    *,
    context: str,
    kubeconfig: Kubeconfig = None,
    inventory: KubernetesGpuInventory | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> int:
    """Add SkyPilot labels only for exact, reviewed GFD product aliases.

    SkyPilot 0.12.2 treats any GFD label as "already labelled", but its GPU name
    catalog does not yet recognize the B200 or RTX PRO 6000 Blackwell product
    strings exposed by managed Kubernetes.
    Its own ``sky gpus label`` therefore performs no mutation while
    ``sky gpus list`` remains empty.  NPA bridges only explicit equivalences;
    unknown or adjacent products remain untouched and fail closed.
    """

    exact_context = str(context or "").strip()
    if not exact_context:
        raise KubernetesGpuCatalogError(
            "Refusing to label GPUs without an exact Kubernetes context"
        )
    observed = inventory or discover_kubernetes_gpu_inventory(
        context=exact_context, kubeconfig=kubeconfig
    )
    if observed.error:
        raise KubernetesGpuCatalogError(
            f"Cannot label GPUs because Kubernetes inventory failed: {observed.error}"
        )
    execute = runner or subprocess.run
    labelled = 0
    for node, labels in observed.node_labels.items():
        if labels.get("skypilot.co/accelerator"):
            continue
        accelerator = _known_skypilot_label(labels)
        if not accelerator:
            continue
        cmd = [
            "kubectl",
        ]
        if kubeconfig is not None and os.fspath(kubeconfig).strip():
            cmd.extend(["--kubeconfig", str(Path(kubeconfig).expanduser())])
        cmd.extend(
            [
                "--context",
                exact_context,
                "label",
                "node",
                node,
                f"skypilot.co/accelerator={accelerator}",
            ]
        )
        try:
            result = execute(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
                env=_kubeconfig_env(kubeconfig),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise KubernetesGpuCatalogError(
                "Could not run kubectl for the explicitly requested GPU label repair; "
                "verify kubectl installation and the exact kubeconfig/context"
            ) from exc
        if result.returncode != 0:
            raise KubernetesGpuCatalogError(
                _kubectl_failure(
                    action="repairing the reviewed SkyPilot GPU node label",
                    returncode=result.returncode,
                    output=result.stderr or result.stdout,
                )
                + "; existing node labels were preserved"
            )
        labelled += 1
    return labelled


def kubernetes_allocatable_gpu_count(
    *,
    context: str = "",
    kubeconfig: Kubeconfig = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> int | None:
    """Return Kubernetes' allocatable GPU total, or ``None`` when unavailable."""

    inventory = discover_kubernetes_gpu_inventory(
        context=context, kubeconfig=kubeconfig, runner=runner
    )
    return None if inventory.error else inventory.allocatable


def wait_for_kubernetes_accelerators(
    accelerators: list[str],
    *,
    context: str = "",
    kubeconfig: Kubeconfig = None,
    sky_bin: SkyBin = None,
    label_known_gpus: bool = False,
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
    healthy minutes before SkyPilot's catalog has consumed the node labels.
    Discovery is read-only by default. Setup callers may explicitly opt into the
    narrow, idempotent known-product label repair with ``label_known_gpus=True``;
    a timeout never deletes or scales the cluster.
    """

    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("GPU readiness timeout and poll interval must be positive")
    requested = [str(item).strip() for item in accelerators if str(item).strip()]
    initial_inventory: KubernetesGpuInventory | None = None
    if label_known_gpus:
        if not str(context or "").strip():
            raise KubernetesGpuCatalogError(
                "Refusing explicit GPU label repair without an exact Kubernetes context"
            )
        initial_inventory = discover_kubernetes_gpu_inventory(
            context=context, kubeconfig=kubeconfig
        )
        if on_status:
            on_status(
                "GPU label mutation requested: exact-context known-product repair "
                "is enabled"
            )
        labelled = label_known_kubernetes_gpus_for_skypilot(
            context=context,
            kubeconfig=kubeconfig,
            inventory=initial_inventory,
        )
        if on_status:
            on_status(
                f"GPU label mutation: added {labelled} reviewed "
                "skypilot.co/accelerator label(s)"
                if labelled
                else "GPU label mutation: no label changes were required"
            )
        if not requested:
            known = sorted(
                {
                    labels.get("skypilot.co/accelerator")
                    or _known_skypilot_label(labels)
                    for labels in initial_inventory.node_labels.values()
                }
                - {""}
            )
            if not known:
                raise KubernetesGpuCatalogError(
                    "SkyPilot smoke auto-detection found no reviewed GPU product "
                    "mapping in the exact Kubernetes context; pass an explicit "
                    "accelerator after confirming the node product"
                )
            requested = [f"{known[0]}:1"]
    if not requested:
        return {}
    get_catalog = discover or (
        lambda: discover_kubernetes_gpu_catalog(
            context=context,
            kubeconfig=kubeconfig,
            sky_bin=sky_bin,
            timeout=max(1, min(int(timeout), DEFAULT_DISCOVERY_TIMEOUT_SECONDS)),
        )
    )
    get_allocatable = allocatable or (
        lambda: kubernetes_allocatable_gpu_count(context=context, kubeconfig=kubeconfig)
    )
    deadline = monotonic() + timeout
    last_failure = "SkyPilot accelerator catalog has not been queried"
    attempt = 0
    while True:
        attempt += 1
        inventory = initial_inventory
        initial_inventory = None
        if inventory is None and allocatable is None:
            inventory = discover_kubernetes_gpu_inventory(
                context=context, kubeconfig=kubeconfig
            )
        count = (
            inventory.allocatable
            if inventory is not None and not inventory.error
            else get_allocatable()
        )
        if on_status:
            shown = "unknown" if count is None else str(count)
            on_status(
                f"GPU readiness attempt {attempt}: Kubernetes allocatable={shown}; "
                "SkyPilot discovery=pending"
            )
        try:
            required = max(
                parse_accelerator_request(accelerator).quantity
                for accelerator in requested
            )
            if count is not None and count < required:
                raise UnsatisfiableAcceleratorError(
                    f"Kubernetes has {count} eligible allocatable GPU(s), but the "
                    f"request needs at least {required}."
                )
            if (
                inventory is not None
                and inventory.allocatable > 0
                and not inventory.products
            ):
                raise UnsatisfiableAcceleratorError(
                    "Kubernetes GPU capacity is Ready and allocatable, but accelerator product "
                    "labels are missing; wait for GPU Feature Discovery/NFD before SkyPilot use."
                )
            catalog = get_catalog()
            resolved = {
                accelerator: resolve_kubernetes_accelerator(
                    accelerator, catalog=catalog
                )
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
            return value[len(prefix) :].strip()
    return ""


def _candidate_names(
    request: AcceleratorRequest, catalog: KubernetesGpuCatalog
) -> list[str]:
    """Return exact-normalized or explicitly registered catalog aliases."""

    names = list(catalog.quantities_by_accelerator)
    wanted = _normalize(request.name)
    if not wanted:
        return []
    exact = [name for name in names if _normalize(name) == wanted]
    if exact:
        return exact
    alias_group = next(
        (group for group in _EXPLICIT_ACCELERATOR_ALIASES if wanted in group), None
    )
    if alias_group is None:
        return []
    return [name for name in names if _normalize(name) in alias_group]


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
            "NPA does not auto-select prefix or fuzzy candidates because a nearby "
            "product can have materially different capacity and cost. "
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
