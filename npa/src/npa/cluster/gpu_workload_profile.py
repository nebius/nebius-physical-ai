"""Explicit workload profiles for Managed Kubernetes GPU clusters."""

from __future__ import annotations

import re
from dataclasses import dataclass


RTX_RENDERING_PROFILE = "rtx-rendering"
RTX_RENDERING_PLATFORM = "gpu-rtx6000"
RTX_RENDERING_PLATFORM_PATTERN = re.compile(r"gpu-rtx6000(?:-[a-z])?")
RTX_RENDERING_PRESET = "1gpu-24vcpu-218gb"
RTX_RENDERING_8GPU_PRESET = "8gpu-192vcpu-1744gb"
RTX_RENDERING_PRESETS = frozenset(
    {RTX_RENDERING_PRESET, RTX_RENDERING_8GPU_PRESET}
)
SUPPORTED_GPU_WORKLOAD_PROFILES = frozenset({"", RTX_RENDERING_PROFILE})


def validate_driver_package_repositories(files: object, *, profile: str) -> None:
    """Validate ConfigMap file keys without interpreting operator package policy."""
    if not isinstance(files, dict):
        raise ValueError("gpu_driver_package_repositories must be a filename/content mapping")
    if files and str(profile or "").strip().lower() != RTX_RENDERING_PROFILE:
        raise ValueError("GPU driver package repositories require the RTX rendering profile")
    for name, content in files.items():
        if (
            not isinstance(name, str) or name in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9._-]+", name)
            or not isinstance(content, str) or not content.strip()
        ):
            raise ValueError("GPU driver package repositories require safe filenames and nonempty text")


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
    if gpu_platform and not RTX_RENDERING_PLATFORM_PATTERN.fullmatch(gpu_platform):
        raise GpuWorkloadProfileError(
            "GPU workload profile 'rtx-rendering' requires platform "
            f"{RTX_RENDERING_PLATFORM!r} or its zonal variant, got {gpu_platform!r}"
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
        gpu_platform=gpu_platform or RTX_RENDERING_PLATFORM,
        gpu_preset=gpu_preset or RTX_RENDERING_PRESET,
        gpu_driver_mode="operator",
        graphics_smoke=True,
    )
