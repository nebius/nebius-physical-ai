"""GPU node-group defaults and helpers."""

from __future__ import annotations

from npa.cluster.config import (
    DEFAULT_GPU_DRIVER_PRESET,
    GPU_TYPE_DEFAULTS,
    NodeGroupConfig,
    default_node_group_name,
    resolve_gpu_preset,
)

L40S_WARNING = (
    "Warning: L40S node groups are supported but experimental for this milestone; "
    "prefer H100 or H200 unless L40S is explicitly required."
)


def gpu_type_from_platform(platform: str) -> str:
    """Return the NPA GPU type for a Nebius platform name, when known."""

    for gpu_type, defaults in GPU_TYPE_DEFAULTS.items():
        if defaults["platform"] == platform:
            return gpu_type
    return ""


__all__ = [
    "DEFAULT_GPU_DRIVER_PRESET",
    "GPU_TYPE_DEFAULTS",
    "L40S_WARNING",
    "NodeGroupConfig",
    "default_node_group_name",
    "gpu_type_from_platform",
    "resolve_gpu_preset",
]
