"""Managed-jobs controller convention for NPA SkyPilot submissions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from npa.cluster.config import DEFAULT_REGION

DEFAULT_CONTROLLER_INSTANCE_TYPE = "cpu-e2_2vcpu-8gb"
DEFAULT_CONTROLLER_CPUS = 2
DEFAULT_CONTROLLER_MEMORY_GB = 8
DEFAULT_CONTROLLER_DISK_SIZE_GB = 64
DEFAULT_CONTROLLER_AUTOSTOP_IDLE_MINUTES = 5


def default_controller_resources() -> dict[str, Any]:
    """Return the default Nebius CPU VM resources for SkyPilot managed jobs."""

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


def apply_controller_override(yaml_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject NPA's managed-jobs controller resources into a SkyPilot config.

    The function is idempotent and preserves an explicitly larger controller
    resource block.
    """

    updated = deepcopy(yaml_dict)
    jobs = updated.setdefault("jobs", {})
    controller = jobs.setdefault("controller", {})
    existing = controller.get("resources")
    default = default_controller_resources()

    if isinstance(existing, dict) and _is_at_least_default(existing):
        return updated

    merged = deepcopy(default)
    if isinstance(existing, dict):
        merged.update({key: value for key, value in existing.items() if key not in {"autostop"}})
        merged["autostop"] = _safe_autostop(existing.get("autostop"))
        if not _is_at_least_default(merged):
            merged = default

    controller["resources"] = merged
    return updated


def _is_at_least_default(resources: dict[str, Any]) -> bool:
    if resources.get("cloud") not in {None, "nebius"}:
        return True
    cpus = _number(resources.get("cpus"))
    memory = _number(resources.get("memory"))
    disk_size = _number(resources.get("disk_size"))
    if cpus is None or memory is None or disk_size is None:
        return False
    autostop = resources.get("autostop")
    if isinstance(autostop, dict) and autostop.get("down") is True:
        return False
    return (
        cpus >= DEFAULT_CONTROLLER_CPUS
        and memory >= DEFAULT_CONTROLLER_MEMORY_GB
        and disk_size >= DEFAULT_CONTROLLER_DISK_SIZE_GB
    )


def _safe_autostop(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return default_controller_resources()["autostop"]
    idle_minutes = value.get("idle_minutes", DEFAULT_CONTROLLER_AUTOSTOP_IDLE_MINUTES)
    return {"idle_minutes": idle_minutes, "down": False}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
