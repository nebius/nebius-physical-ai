"""Managed-jobs controller convention for NPA SkyPilot submissions.

The architectural default is a Kubernetes-hosted managed-jobs controller so
Workbench submissions keep the controller and task pods on MK8s.  The Nebius
CPU VM controller from the W9 bootstrap remains as an explicit fallback for
clusters that cannot schedule the controller pod.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from npa.cluster.config import DEFAULT_REGION

ControllerBackend = Literal["kubernetes", "nebius"]

DEFAULT_CONTROLLER_BACKEND: ControllerBackend = "kubernetes"
# Keep the shared controller inside the canonical 8-vCPU/32-GiB PAIDF CPU pool
# alongside one 4-vCPU/16-GiB stage. This matches SkyPilot's standalone
# controller shape and leaves Kubernetes reserve instead of silently requiring
# the old 16-vCPU node upgrade.
DEFAULT_K8S_CONTROLLER_CPUS = 2
DEFAULT_K8S_CONTROLLER_MEMORY_GB = 8
DEFAULT_CONTROLLER_INSTANCE_TYPE = "cpu-e2_2vcpu-8gb"
DEFAULT_CONTROLLER_CPUS = 2
DEFAULT_CONTROLLER_MEMORY_GB = 8
DEFAULT_CONTROLLER_DISK_SIZE_GB = 64
DEFAULT_JOBS_CONTROLLER_AUTOSTOP = False
# Kubernetes starts SkyPilot's long-lived sshd from the container working
# directory.  If that directory is an ephemeral setup path which SkyPilot later
# removes, every subsequent rsync receiver inherits a deleted cwd and fails
# before a managed workload can become observable.  /tmp is part of the base
# filesystem contract for both controller and workload images and is not one of
# SkyPilot's per-launch cleanup directories.
KUBERNETES_SKYPILOT_WORKING_DIR = "/tmp"


def controller_resources_kubernetes() -> dict[str, Any]:
    """Return the default Kubernetes controller resources for SkyPilot managed jobs."""

    return {
        "cloud": "kubernetes",
        "cpus": DEFAULT_K8S_CONTROLLER_CPUS,
        "memory": DEFAULT_K8S_CONTROLLER_MEMORY_GB,
        "autostop": DEFAULT_JOBS_CONTROLLER_AUTOSTOP,
    }


def controller_resources_nebius_vm() -> dict[str, Any]:
    """Return the fallback Nebius CPU VM resources for SkyPilot managed jobs."""

    return {
        "cloud": "nebius",
        "region": DEFAULT_REGION,
        "instance_type": DEFAULT_CONTROLLER_INSTANCE_TYPE,
        "cpus": DEFAULT_CONTROLLER_CPUS,
        "memory": DEFAULT_CONTROLLER_MEMORY_GB,
        "disk_size": DEFAULT_CONTROLLER_DISK_SIZE_GB,
        "autostop": DEFAULT_JOBS_CONTROLLER_AUTOSTOP,
    }


def default_controller_resources(
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
) -> dict[str, Any]:
    """Return NPA's default resources for the selected controller backend."""

    return _controller_resources_for_backend(controller_backend)


def apply_controller_override(
    yaml_dict: dict[str, Any],
    *,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    controller_region: str | None = None,
) -> dict[str, Any]:
    """Inject NPA's managed-jobs controller resources into a SkyPilot config.

    The function is idempotent and preserves an explicitly larger controller
    resource block. When ``controller_region`` is provided (for the Kubernetes
    backend this is the kube context, e.g. derived from a job's
    ``--infra k8s/<context>``), the controller is co-located there so it shares
    the region — and therefore object-storage reachability — of its jobs. The
    region is never hard-coded; callers pass it from the submission target.
    """

    updated = deepcopy(yaml_dict)
    jobs = updated.setdefault("jobs", {})
    controller = jobs.setdefault("controller", {})
    existing = controller.get("resources")
    default = _controller_resources_for_backend(controller_backend)

    if isinstance(existing, dict) and _is_at_least_default(existing, default):
        existing["autostop"] = DEFAULT_JOBS_CONTROLLER_AUTOSTOP
        _apply_controller_region(existing, controller_backend, controller_region)
        _apply_kubernetes_durable_working_directory(updated, controller_backend)
        return updated

    merged = deepcopy(default)
    if isinstance(existing, dict) and _compatible_controller_cloud(existing, default):
        merged.update(
            {
                key: value
                for key, value in existing.items()
                if key not in _unsupported_override_keys(controller_backend)
            }
        )
        merged["autostop"] = DEFAULT_JOBS_CONTROLLER_AUTOSTOP
        if not _is_at_least_default(merged, default):
            merged = default

    _apply_controller_region(merged, controller_backend, controller_region)
    controller["resources"] = merged
    _apply_kubernetes_durable_working_directory(updated, controller_backend)
    return updated


def _apply_kubernetes_durable_working_directory(
    config: dict[str, Any], controller_backend: ControllerBackend
) -> None:
    """Start SkyPilot's Kubernetes container processes from a durable cwd.

    SkyPilot applies the global Kubernetes pod config to the controller it
    creates before submission transport begins.  Kubernetes patch-merges the
    named ``ray-node`` container, so this preserves image pull secrets, mounts,
    environment, and other operator configuration while making the cwd used by
    the controller's sshd explicit.
    """

    if controller_backend != "kubernetes":
        return
    kubernetes = config.setdefault("kubernetes", {})
    if not isinstance(kubernetes, dict):
        raise ValueError("SkyPilot global config kubernetes section must be a mapping")
    pod_config = kubernetes.setdefault("pod_config", {})
    if not isinstance(pod_config, dict):
        raise ValueError("SkyPilot kubernetes.pod_config must be a mapping")
    spec = pod_config.setdefault("spec", {})
    if not isinstance(spec, dict):
        raise ValueError("SkyPilot kubernetes.pod_config.spec must be a mapping")
    containers = spec.setdefault("containers", [])
    if not isinstance(containers, list):
        raise ValueError(
            "SkyPilot kubernetes.pod_config.spec.containers must be a list"
        )
    ray_node = next(
        (
            item
            for item in containers
            if isinstance(item, dict) and item.get("name") == "ray-node"
        ),
        None,
    )
    if ray_node is None:
        ray_node = {"name": "ray-node"}
        containers.append(ray_node)
    ray_node["workingDir"] = KUBERNETES_SKYPILOT_WORKING_DIR


def _apply_controller_region(
    resources: dict[str, Any],
    controller_backend: ControllerBackend,
    controller_region: str | None,
) -> None:
    """Pin the controller to ``controller_region`` when the caller supplies one.

    For Kubernetes the SkyPilot ``region`` is the kube context; co-locating the
    controller with the target context keeps the controller and its jobs in the
    same region so bucket mounts resolve against the right object-storage
    endpoint. An explicit region already in the block is preserved.
    """

    region = (controller_region or "").strip()
    if not region:
        return
    resources.setdefault("region", region)


def _controller_resources_for_backend(
    controller_backend: ControllerBackend,
) -> dict[str, Any]:
    if controller_backend == "kubernetes":
        return controller_resources_kubernetes()
    if controller_backend == "nebius":
        return controller_resources_nebius_vm()
    raise ValueError("controller_backend must be 'kubernetes' or 'nebius'")


def _is_at_least_default(resources: dict[str, Any], default: dict[str, Any]) -> bool:
    if not _compatible_controller_cloud(resources, default):
        return False
    for key in _unsupported_override_keys(_backend_from_default(default)):
        if key in resources:
            return False
    for key in ("cpus", "memory", "disk_size"):
        if key not in default:
            continue
        actual = _number(resources.get(key))
        minimum = _number(default.get(key))
        if actual is None or minimum is None or actual < minimum:
            return False
    if "autostop" in default:
        if resources.get("autostop") is not DEFAULT_JOBS_CONTROLLER_AUTOSTOP:
            return False
    return True


def _backend_from_default(default: dict[str, Any]) -> ControllerBackend:
    if default.get("cloud") == "kubernetes":
        return "kubernetes"
    return "nebius"


def _compatible_controller_cloud(
    resources: dict[str, Any], default: dict[str, Any]
) -> bool:
    return resources.get("cloud") in {None, default.get("cloud")}


def _unsupported_override_keys(controller_backend: ControllerBackend) -> set[str]:
    if controller_backend == "kubernetes":
        return {"disk_size"}
    return set()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
