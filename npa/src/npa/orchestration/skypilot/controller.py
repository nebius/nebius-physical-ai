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
DEFAULT_K8S_CONTROLLER_CPUS = 4
DEFAULT_K8S_CONTROLLER_MEMORY_GB = 16
DEFAULT_CONTROLLER_INSTANCE_TYPE = "cpu-e2_2vcpu-8gb"
DEFAULT_CONTROLLER_CPUS = 2
DEFAULT_CONTROLLER_MEMORY_GB = 8
DEFAULT_CONTROLLER_DISK_SIZE_GB = 64
DEFAULT_CONTROLLER_AUTOSTOP_IDLE_MINUTES = 5


def controller_resources_kubernetes() -> dict[str, Any]:
    """Return the default Kubernetes controller resources for SkyPilot managed jobs."""

    return {
        "cloud": "kubernetes",
        "cpus": DEFAULT_K8S_CONTROLLER_CPUS,
        "memory": DEFAULT_K8S_CONTROLLER_MEMORY_GB,
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
        "autostop": {
            "idle_minutes": DEFAULT_CONTROLLER_AUTOSTOP_IDLE_MINUTES,
            "down": False,
        },
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
) -> dict[str, Any]:
    """Inject NPA's managed-jobs controller resources into a SkyPilot config.

    The function is idempotent and preserves an explicitly larger controller
    resource block.
    """

    updated = deepcopy(yaml_dict)
    jobs = updated.setdefault("jobs", {})
    controller = jobs.setdefault("controller", {})
    existing = controller.get("resources")
    default = _controller_resources_for_backend(controller_backend)

    if isinstance(existing, dict) and _is_at_least_default(existing, default):
        return updated

    merged = deepcopy(default)
    if isinstance(existing, dict) and _compatible_controller_cloud(existing, default):
        merged.update(
            {
                key: value
                for key, value in existing.items()
                if key not in _unsupported_override_keys(controller_backend)
                and not (controller_backend == "nebius" and key == "autostop")
            }
        )
        if controller_backend == "nebius":
            merged["autostop"] = _safe_autostop(existing.get("autostop"))
        if not _is_at_least_default(merged, default):
            merged = default

    controller["resources"] = merged
    return updated


def _controller_resources_for_backend(controller_backend: ControllerBackend) -> dict[str, Any]:
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
        autostop = resources.get("autostop")
        if isinstance(autostop, dict) and autostop.get("down") is True:
            return False
    return True


def _backend_from_default(default: dict[str, Any]) -> ControllerBackend:
    if default.get("cloud") == "kubernetes":
        return "kubernetes"
    return "nebius"


def _compatible_controller_cloud(resources: dict[str, Any], default: dict[str, Any]) -> bool:
    return resources.get("cloud") in {None, default.get("cloud")}


def _unsupported_override_keys(controller_backend: ControllerBackend) -> set[str]:
    if controller_backend == "kubernetes":
        return {"autostop", "disk_size"}
    return set()


def _safe_autostop(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return controller_resources_nebius_vm()["autostop"]
    idle_minutes = value.get("idle_minutes", DEFAULT_CONTROLLER_AUTOSTOP_IDLE_MINUTES)
    return {"idle_minutes": idle_minutes, "down": False}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
