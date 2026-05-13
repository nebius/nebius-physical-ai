"""GPU platform resolution for Workbench Serverless Jobs."""

from __future__ import annotations

GPU_PLATFORM_ALIASES = {
    "h200": "gpu-h200-sxm",
    "gpu-h200-sxm": "gpu-h200-sxm",
    "h100": "gpu-h100-sxm",
    "gpu-h100-sxm": "gpu-h100-sxm",
    "b300": "gpu-b300-sxm",
    "gpu-b300-sxm": "gpu-b300-sxm",
    "b200": "gpu-b200-sxm-a",
    "gpu-b200-sxm-a": "gpu-b200-sxm-a",
    "l40s": "gpu-l40s-a",
    "gpu-l40s-a": "gpu-l40s-a",
    "gpu-l40s-d": "gpu-l40s-d",
    "gpu-rtx-pro-6000": "gpu-rtx6000",
    "rtx-pro-6000": "gpu-rtx6000",
    "rtx6000": "gpu-rtx6000",
    "gpu-rtx6000": "gpu-rtx6000",
}

GPU_PLATFORM_PRESETS = {
    "gpu-h200-sxm": {
        1: "1gpu-16vcpu-200gb",
        8: "8gpu-128vcpu-1600gb",
    },
    "gpu-h100-sxm": {
        1: "1gpu-16vcpu-200gb",
        8: "8gpu-128vcpu-1600gb",
    },
    "gpu-b300-sxm": {
        1: "1gpu-24vcpu-346gb",
        8: "8gpu-192vcpu-2768gb",
    },
    "gpu-b200-sxm-a": {
        1: "1gpu-20vcpu-224gb",
        8: "8gpu-160vcpu-1792gb",
    },
    "gpu-l40s-a": {
        1: "1gpu-40vcpu-160gb",
    },
    "gpu-l40s-d": {
        1: "1gpu-48vcpu-288gb",
        2: "2gpu-96vcpu-576gb",
        4: "4gpu-192vcpu-1152gb",
    },
    "gpu-rtx6000": {
        1: "1gpu-24vcpu-218gb",
        8: "8gpu-192vcpu-1744gb",
    },
}


def resolve_gpu_platform(gpu_type: str, gpu_count: int = 1) -> tuple[str, str, int]:
    """Return (platform_id, preset, gpu_count) for a GPU alias or platform name."""

    normalized = (gpu_type or "").strip().lower()
    if not normalized:
        raise ValueError("gpu_type is required")
    platform_id = GPU_PLATFORM_ALIASES.get(normalized)
    if not platform_id:
        raise ValueError(f"Unknown GPU type: {gpu_type}")
    resolved_count = gpu_count or 1
    preset = GPU_PLATFORM_PRESETS.get(platform_id, {}).get(resolved_count)
    if not preset:
        preset = f"{resolved_count}gpu-16vcpu-200gb"
    return platform_id, preset, resolved_count
