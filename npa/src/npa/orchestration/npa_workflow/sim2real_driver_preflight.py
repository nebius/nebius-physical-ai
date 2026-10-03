"""Bind the Sim2Real managed-driver check to actual Isaac pod placement."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.kubernetes_prerequisites import (
    cpu_millicores,
    integer_resource,
    memory_bytes,
)
from npa.orchestration.npa_workflow.skypilot_render import normalize_resources
from npa.orchestration.npa_workflow.spec import resolve_resource_profile
from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuInventory,
    KubernetesGpuNode,
    UnsatisfiableAcceleratorError,
    preflight_kubernetes_gpu_gang,
)
from npa.orchestration.skypilot.registry_preflight import (
    merge_kubernetes_pull_placement,
)
from npa.orchestration.skypilot.resource_quantities import kubernetes_gpu_quantities

_ISAAC_STATES = ("stage-07-rollouts", "stage-09-ppo", "stage-10-gold")
_DEFAULT_POD = {
    "nodeSelector": {"kubernetes.io/os": "linux"},
    "tolerations": [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    ],
}


def _pod_spec(kubernetes):
    if not isinstance(kubernetes, Mapping):
        raise ValueError("Isaac Kubernetes configuration must be a mapping")
    pod = kubernetes.get("pod_config") or {}
    if not isinstance(pod, Mapping) or not isinstance(pod.get("spec", {}), Mapping):
        raise ValueError("Isaac pod configuration must contain a spec mapping")
    return pod.get("spec", {})


def _merged_pod(base, override):
    return json.loads(
        merge_kubernetes_pull_placement(json.dumps(base), json.dumps(override))
    )


def _configured_pod(context, global_config_path):
    document = {}
    if global_config_path is not None:
        document = yaml.safe_load(Path(global_config_path).read_text())
        if document is None:
            document = {}
    if not isinstance(document, Mapping):
        raise ValueError("selected SkyPilot configuration must be a mapping")
    kubernetes = document.get("kubernetes", {})
    base = _merged_pod(_DEFAULT_POD, _pod_spec(kubernetes))
    contexts = kubernetes.get("context_configs", {})
    if not isinstance(contexts, Mapping):
        raise ValueError("SkyPilot context_configs must be a mapping")
    return _merged_pod(base, _pod_spec(contexts.get(context) or {}))


def isaac_render_placements(
    spec, *, context, global_config_path=None, allowed_nodes=()
):
    """Resolve canonical Isaac stage placement using the renderer's contracts.

    Args:
        spec: Resolved canonical Sim2Real workflow specification.
        context: Exact selected Kubernetes context.
        global_config_path: Selected SkyPilot configuration file, if any.
        allowed_nodes: Node allowlist resolved by the submit path.
    Returns:
        Effective accelerator, capacity and pod constraints for each Isaac state.
    Raises:
        ValueError: Required states or placement configuration are invalid.
        OSError: The selected configuration cannot be read.
    """
    base = _configured_pod(context, global_config_path)
    placements = []
    for name in _ISAAC_STATES:
        state = spec.states.get(name)
        if state is None or state.resources not in spec.resources:
            raise ValueError("canonical Sim2Real Isaac resource profile is missing")
        resolved = resolve_resource_profile(
            state.resources,
            spec.resources[state.resources],
            config=spec.config,
            run={"id": "driver-preflight"},
        )
        placements.append(_resource_placement(resolved, base, allowed_nodes))
    return placements


def _resource_placement(resolved, base, allowed_nodes):
    effective = normalize_resources(resolved)
    accelerator = str(effective.get("accelerators") or "")
    cpus, memory = kubernetes_gpu_quantities(effective, accelerator=accelerator)
    return {
        "accelerator": accelerator,
        "cpus": cpus,
        "memory": memory,
        "allowed_nodes": allowed_nodes,
        "pod_spec": _merged_pod(base, _pod_spec(resolved.get("kubernetes") or {})),
    }


def _tolerates(taint, tolerations):
    for tolerance in tolerations:
        if not isinstance(tolerance, Mapping):
            raise ValueError("Isaac pod tolerations must contain mappings")
        if tolerance.get("effect") not in (None, "", taint.get("effect")):
            continue
        operator = tolerance.get("operator", "Equal")
        if operator == "Exists" and tolerance.get("key", "") in ("", taint.get("key")):
            return True
        if (
            operator == "Equal"
            and tolerance.get("key") == taint.get("key")
            and tolerance.get("value", "") == taint.get("value", "")
        ):
            return True
    return False


def _node_available(node, pod_spec):
    spec, status = node.get("spec") or {}, node.get("status") or {}
    if not isinstance(spec, Mapping) or not isinstance(status, Mapping):
        raise ValueError("Isaac node scheduling evidence is invalid")
    if spec.get("unschedulable"):
        return False
    conditions = status.get("conditions")
    if (
        not isinstance(conditions, list)
        or not all(isinstance(item, Mapping) for item in conditions)
        or not any(item.get("type") == "Ready" for item in conditions)
    ):
        raise ValueError("Isaac node readiness evidence is missing")
    if not any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in conditions
    ):
        return False
    tolerations = pod_spec.get("tolerations", [])
    if not isinstance(tolerations, list):
        raise ValueError("Isaac pod tolerations must be a list")
    for taint in spec.get("taints") or []:
        if not isinstance(taint, Mapping):
            raise ValueError("Isaac node taint evidence is invalid")
        if taint.get("effect") in {"NoSchedule", "NoExecute"} and not _tolerates(
            taint, tolerations
        ):
            return False
    return True


def _node_products(labels):
    return tuple(
        str(labels[key])
        for key in (
            "nvidia.com/gpu.product",
            "nebius.com/gpu-name",
            "skypilot.co/accelerator",
        )
        if labels.get(key)
    )


def _placement_node(node):
    metadata, status = node.get("metadata"), node.get("status")
    if not isinstance(metadata, Mapping) or not metadata.get("name"):
        raise ValueError("Isaac node metadata evidence is invalid")
    if not isinstance(status, Mapping):
        raise ValueError("Isaac node status evidence is invalid")
    labels, allocatable = metadata.get("labels", {}), status.get("allocatable") or {}
    if not isinstance(labels, Mapping):
        raise ValueError("Isaac node label evidence is invalid")
    if not isinstance(allocatable, Mapping) or any(
        key not in allocatable for key in ("cpu", "memory", "nvidia.com/gpu")
    ):
        raise ValueError("Isaac node allocatable resource evidence is missing")
    gpu = integer_resource(allocatable["nvidia.com/gpu"])
    cpu, memory = (
        cpu_millicores(allocatable["cpu"]),
        memory_bytes(allocatable["memory"]),
    )
    return KubernetesGpuNode(
        name=str(metadata.get("name") or ""),
        ready=True,
        schedulable=True,
        products=_node_products(labels),
        labels=tuple(labels.items()),
        capacity=gpu,
        allocatable=gpu,
        committed=0,
        free=gpu,
        exclusion="",
        allocatable_cpu_millis=cpu,
        free_cpu_millis=cpu,
        allocatable_memory_bytes=memory,
        free_memory_bytes=memory,
        allocatable_pods=1,
        free_pod_slots=1,
    )


def node_can_host_isaac(node, placements=None):
    """Check allocatable placement without claiming or reserving free capacity.

    Args:
        node: Authoritative Kubernetes node record.
        placements: Effective Isaac stage placements; None uses canonical defaults.
    Returns:
        Whether any Isaac task can be scheduled on this node.
    Raises:
        ValueError: Node placement evidence is missing or malformed.
        RuntimeError: Placement constraints cannot be evaluated authoritatively.
    """
    if placements is None:
        placements = [{"accelerator": "RTXPRO6000:1", "pod_spec": _DEFAULT_POD}]
    for placement in placements:
        if not _node_available(node, placement["pod_spec"]):
            continue
        record = _placement_node(node)
        inventory = KubernetesGpuInventory(
            context="driver-preflight",
            ready_nodes=1,
            eligible_gpu_nodes=1,
            capacity=record.capacity,
            allocatable=record.allocatable,
            products=record.products,
            node_labels={},
            nodes=(record,),
        )
        try:
            preflight_kubernetes_gpu_gang(inventory, node_count=1, **placement)
        except UnsatisfiableAcceleratorError:
            continue
        return True
    return False
