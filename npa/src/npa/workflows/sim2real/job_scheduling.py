"""Kueue admission and Kubernetes-native failure policy for GPU siblings."""

from __future__ import annotations

import copy
import os
import re
from typing import Any

from npa.workflows.sim2real.k8s_client import QUEUE_LABEL
from npa.workflows.sim2real.models import Sim2RealLoopError


KUEUE_VERSION = "0.17.3"
KUEUE_API_VERSION = "kueue.x-k8s.io/v1beta2"
DEFAULT_RESOURCE_FLAVOR = "sim2real-rtx-pro-6000"
DEFAULT_CLUSTER_QUEUE = "sim2real-gpu-cluster"
DEFAULT_LOCAL_QUEUE = "sim2real-gpu"
DEFAULT_PRIORITY_CLASS = "sim2real-production"
ISAAC_CACHE_VOLUME = "isaac-runtime-cache"
ISAAC_CACHE_MOUNT = "/opt/isaac-cache"
_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?$")


def require_image_digest(image: str) -> str:
    """Return an immutable image reference or fail before Job creation."""

    normalized = str(image or "").removeprefix("docker:").strip()
    if "@sha256:" not in normalized:
        raise Sim2RealLoopError(
            f"Sim2Real production Jobs require image@sha256 provenance, got {image!r}"
        )
    algorithm, digest = normalized.rsplit("@", 1)[-1].split(":", 1)
    if algorithm != "sha256" or len(digest) != 64:
        raise Sim2RealLoopError(f"invalid immutable image digest: {image!r}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise Sim2RealLoopError(f"invalid immutable image digest: {image!r}") from exc
    return normalized


def _configure_isaac_runtime_cache(
    manifest: dict[str, Any], *, immutable_image: str, cache_pvc: str = ""
) -> dict[str, str]:
    """Mount the operator-fetched Isaac closure read-only on Isaac GPU Jobs.

    NVIDIA's proprietary Isaac wheels cannot be redistributed in the public
    workbench image. The operator therefore warms one content-addressed PVC on
    a CPU node after accepting the EULA. GPU Jobs must consume that closure
    offline and read-only; they never run pip or mutate dependency state after
    Kueue admission.
    """

    expected_image = (
        os.environ.get("NPA_SIM2REAL_ISAAC_IMAGE", "").strip()
        or os.environ.get("ISAAC_IMAGE", "").strip()
    )
    if not expected_image:
        return {}
    try:
        expected_immutable = require_image_digest(expected_image)
    except Sim2RealLoopError:
        # The selected Job image is still independently digest-validated. A
        # malformed non-selected alias must not make unrelated GPU Jobs Isaac.
        return {}
    if immutable_image != expected_immutable:
        return {}

    pvc = (
        cache_pvc.strip() or os.environ.get("NPA_SIM2REAL_ISAAC_CACHE_PVC", "").strip()
    )
    if not pvc:
        raise Sim2RealLoopError(
            "Isaac GPU Jobs require NPA_SIM2REAL_ISAAC_CACHE_PVC so pinned "
            "dependencies are warmed before GPU admission"
        )
    if len(pvc) > 253 or not _DNS_SUBDOMAIN_RE.fullmatch(pvc):
        raise Sim2RealLoopError(
            f"NPA_SIM2REAL_ISAAC_CACHE_PVC is not a DNS-safe PVC name: {pvc!r}"
        )

    pod_spec = manifest["spec"]["template"]["spec"]
    containers = list(pod_spec.get("containers") or [])
    if len(containers) != 1:
        raise Sim2RealLoopError(
            "Isaac runtime-cache contract requires exactly one application container"
        )
    container = containers[0]
    volumes = list(pod_spec.get("volumes") or [])
    if any(item.get("name") == ISAAC_CACHE_VOLUME for item in volumes):
        raise Sim2RealLoopError("Isaac runtime-cache volume name is already in use")
    mounts = list(container.get("volumeMounts") or [])
    if any(
        item.get("name") == ISAAC_CACHE_VOLUME
        or item.get("mountPath") == ISAAC_CACHE_MOUNT
        for item in mounts
    ):
        raise Sim2RealLoopError("Isaac runtime-cache mount collides with the Job")

    volumes.append(
        {
            "name": ISAAC_CACHE_VOLUME,
            "persistentVolumeClaim": {"claimName": pvc, "readOnly": True},
        }
    )
    mounts.append(
        {
            "name": ISAAC_CACHE_VOLUME,
            "mountPath": ISAAC_CACHE_MOUNT,
            "readOnly": True,
        }
    )
    container["volumeMounts"] = mounts
    env = list(container.get("env") or [])
    by_name = {str(item.get("name") or ""): item for item in env}
    required_env = {
        "NPA_SIM2REAL_ISAAC_CACHE_PVC": pvc,
        "NPA_ISAAC_CACHE_DIR": ISAAC_CACHE_MOUNT,
        "NPA_ISAAC_CACHE_READONLY": "1",
        "NPA_ISAAC_BOOTSTRAP_OFFLINE": "1",
    }
    for name, value in required_env.items():
        existing = by_name.get(name)
        if existing is not None and existing.get("value") != value:
            raise Sim2RealLoopError(
                f"Isaac runtime-cache env {name} conflicts with fail-closed value"
            )
        if existing is None:
            env.append({"name": name, "value": value})
    container["env"] = env
    pod_spec["containers"] = containers
    pod_spec["volumes"] = volumes
    annotations = manifest.setdefault("metadata", {}).setdefault("annotations", {})
    annotations["sim2real.npa.dev/isaac-cache-pvc"] = pvc
    annotations["sim2real.npa.dev/runtime-dependencies"] = (
        "content-addressed-readonly-pvc"
    )
    return {
        "pvc": pvc,
        "mount_path": ISAAC_CACHE_MOUNT,
        "mode": "offline-readonly",
    }


def configure_gpu_job(
    manifest: dict[str, Any],
    *,
    image: str,
    product: str,
    gpu_resource: str,
    gpu_count: int,
    queue_name: str = "",
    priority_class: str = "",
    isaac_cache_pvc: str = "",
) -> dict[str, Any]:
    """Apply queue admission and a fail-closed Pod failure policy."""

    immutable = require_image_digest(image)
    configured = copy.deepcopy(manifest)
    metadata = configured.setdefault("metadata", {})
    spec = configured.setdefault("spec", {})
    pod_spec = spec.setdefault("template", {}).setdefault("spec", {})
    containers = list(pod_spec.get("containers") or [])
    if not containers:
        raise Sim2RealLoopError("GPU Job manifest contains no containers")
    # The selected component image must be the exact digest being attested.
    containers[0]["image"] = immutable
    containers[0]["imagePullPolicy"] = "IfNotPresent"
    pod_spec["containers"] = containers
    _configure_isaac_runtime_cache(
        configured,
        immutable_image=immutable,
        cache_pvc=isaac_cache_pvc,
    )
    pod_spec["restartPolicy"] = "Never"
    pod_spec["nodeSelector"] = {"nvidia.com/gpu.product": product}
    priority = priority_class or os.environ.get(
        "NPA_SIM2REAL_KUEUE_PRIORITY_CLASS", DEFAULT_PRIORITY_CLASS
    )
    pod_spec["priorityClassName"] = priority

    retries = int(os.environ.get("NPA_SIM2REAL_K8S_INFRA_RETRIES", "3") or 3)
    if retries < 1:
        raise Sim2RealLoopError("NPA_SIM2REAL_K8S_INFRA_RETRIES must be >= 1")
    spec["backoffLimit"] = retries
    spec.pop("ttlSecondsAfterFinished", None)
    # `DisruptionTarget` is a structured Pod condition set for preemption,
    # eviction, and similar infrastructure termination. Ignore it for the
    # application retry counter. Any nonzero application-container exit fails
    # immediately; remaining pod failures consume the bounded native backoff.
    spec["podFailurePolicy"] = {
        "rules": [
            {
                "action": "Ignore",
                "onPodConditions": [{"type": "DisruptionTarget", "status": "True"}],
            },
            {
                "action": "FailJob",
                "onExitCodes": {"operator": "NotIn", "values": [0]},
            },
        ]
    }
    if int(os.environ.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "0") or 0) == 0:
        spec.pop("activeDeadlineSeconds", None)

    queue = queue_name or os.environ.get(
        "NPA_SIM2REAL_KUEUE_LOCAL_QUEUE", DEFAULT_LOCAL_QUEUE
    )
    labels = metadata.setdefault("labels", {})
    labels[QUEUE_LABEL] = queue
    template_labels = (
        spec.setdefault("template", {})
        .setdefault("metadata", {})
        .setdefault("labels", {})
    )
    template_labels["sim2real.npa.dev/gpu-product"] = product[:63]
    # Kueue admits batch Jobs bearing the queue label. Setting suspend makes
    # ownership explicit even if the webhook is temporarily unavailable.
    spec["suspend"] = True
    metadata.setdefault("annotations", {})["sim2real.npa.dev/failure-policy"] = (
        "podFailurePolicy:v1"
    )
    metadata["annotations"]["sim2real.npa.dev/gpu-request"] = (
        f"{gpu_resource}={gpu_count}"
    )
    return configured


def kueue_queue_manifests(
    *,
    namespace: str,
    gpu_product: str,
    gpu_resource: str = "nvidia.com/gpu",
    gpu_quota: int,
    cpu_quota: int | str,
    memory_quota: str,
    resource_flavor: str = DEFAULT_RESOURCE_FLAVOR,
    cluster_queue: str = DEFAULT_CLUSTER_QUEUE,
    local_queue: str = DEFAULT_LOCAL_QUEUE,
    priority_class: str = DEFAULT_PRIORITY_CLASS,
) -> list[dict[str, Any]]:
    """Return the exact queue/flavor/quota resources for the isolated cluster."""

    if gpu_quota < 1:
        raise ValueError("gpu_quota must be >= 1")
    if not str(cpu_quota).strip() or str(cpu_quota).strip() == "0":
        raise ValueError("cpu_quota must be a positive Kubernetes quantity")
    if not str(memory_quota).strip() or str(memory_quota).strip() == "0":
        raise ValueError("memory_quota must be a positive Kubernetes quantity")
    return [
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "ResourceFlavor",
            "metadata": {"name": resource_flavor},
            "spec": {"nodeLabels": {"nvidia.com/gpu.product": gpu_product}},
        },
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "ClusterQueue",
            "metadata": {"name": cluster_queue},
            "spec": {
                "namespaceSelector": {
                    "matchLabels": {"kubernetes.io/metadata.name": namespace}
                },
                "queueingStrategy": "BestEffortFIFO",
                "resourceGroups": [
                    {
                        # Kueue requires quota for every requested resource in a
                        # Workload, not only the accelerator that gates fan-out.
                        "coveredResources": [gpu_resource, "cpu", "memory"],
                        "flavors": [
                            {
                                "name": resource_flavor,
                                "resources": [
                                    {"name": gpu_resource, "nominalQuota": gpu_quota},
                                    {"name": "cpu", "nominalQuota": cpu_quota},
                                    {
                                        "name": "memory",
                                        "nominalQuota": memory_quota,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
        },
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "LocalQueue",
            "metadata": {"name": local_queue, "namespace": namespace},
            "spec": {"clusterQueue": cluster_queue},
        },
        {
            "apiVersion": "scheduling.k8s.io/v1",
            "kind": "PriorityClass",
            "metadata": {"name": priority_class},
            "value": 100000,
            "globalDefault": False,
            "preemptionPolicy": "Never",
            "description": "NPA Sim2Real production GPU siblings",
        },
    ]
