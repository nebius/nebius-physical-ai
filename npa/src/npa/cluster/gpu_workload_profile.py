"""Explicit workload profiles for Managed Kubernetes GPU clusters."""

from __future__ import annotations

from dataclasses import dataclass


RTX_RENDERING_PROFILE = "rtx-rendering"
RTX_RENDERING_PLATFORM = "gpu-rtx6000"
RTX_RENDERING_PRESET = "1gpu-24vcpu-218gb"
RTX_RENDERING_8GPU_PRESET = "8gpu-192vcpu-1744gb"
RTX_RENDERING_PRESETS = frozenset(
    {RTX_RENDERING_PRESET, RTX_RENDERING_8GPU_PRESET}
)
SUPPORTED_GPU_WORKLOAD_PROFILES = frozenset({"", RTX_RENDERING_PROFILE})


class GpuWorkloadProfileError(ValueError):
    """Raised when a workload profile conflicts with explicit topology."""


@dataclass(frozen=True)
class GpuWorkloadSelection:
    profile: str
    gpu_nodes: int
    gpu_platform: str
    gpu_preset: str
    gpu_driver_mode: str
    graphics_smoke: bool


def resolve_gpu_workload_profile(
    *,
    profile: str,
    gpu_nodes: int,
    gpu_platform: str,
    gpu_preset: str,
    gpu_driver_mode: str,
) -> GpuWorkloadSelection:
    """Resolve one public profile without changing the unprofiled defaults."""

    normalized = str(profile or "").strip().lower()
    if normalized not in SUPPORTED_GPU_WORKLOAD_PROFILES:
        supported = ", ".join(
            sorted(item for item in SUPPORTED_GPU_WORKLOAD_PROFILES if item)
        )
        raise GpuWorkloadProfileError(
            f"unsupported GPU workload profile {profile!r}; expected one of: {supported}"
        )
    if not normalized:
        return GpuWorkloadSelection(
            profile="",
            gpu_nodes=gpu_nodes,
            gpu_platform=gpu_platform,
            gpu_preset=gpu_preset,
            gpu_driver_mode=gpu_driver_mode,
            graphics_smoke=False,
        )

    if gpu_nodes == 0:
        raise GpuWorkloadProfileError(
            "GPU workload profile 'rtx-rendering' requires at least one GPU node"
        )
    if gpu_platform and gpu_platform != RTX_RENDERING_PLATFORM:
        raise GpuWorkloadProfileError(
            "GPU workload profile 'rtx-rendering' requires platform "
            f"{RTX_RENDERING_PLATFORM!r}, got {gpu_platform!r}"
        )
    if gpu_preset and gpu_preset not in RTX_RENDERING_PRESETS:
        raise GpuWorkloadProfileError(
            "GPU workload profile 'rtx-rendering' requires an RTX PRO 6000 preset "
            f"({', '.join(sorted(RTX_RENDERING_PRESETS))}), got {gpu_preset!r}"
        )
    if gpu_driver_mode and gpu_driver_mode not in {"auto", "operator"}:
        raise GpuWorkloadProfileError(
            "GPU workload profile 'rtx-rendering' requires gpu_driver_mode='operator'"
        )
    return GpuWorkloadSelection(
        profile=RTX_RENDERING_PROFILE,
        gpu_nodes=1 if gpu_nodes < 0 else gpu_nodes,
        gpu_platform=RTX_RENDERING_PLATFORM,
        gpu_preset=gpu_preset or RTX_RENDERING_PRESET,
        gpu_driver_mode="operator",
        graphics_smoke=True,
    )
