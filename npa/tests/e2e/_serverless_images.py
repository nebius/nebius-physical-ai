"""Shared helpers for live serverless tool e2e image/platform resolution."""

from __future__ import annotations

import os


_PLACEHOLDER_REGISTRY = "cr.eu-north1.nebius.cloud/your-registry-id"

# Platforms available on the shared us-central1 rtxpro project (verified live).
# Default to RTX6000: tenant H200 quota is often exhausted (limit 2).
_DEFAULT_SERVERLESS_GPU = "gpu-rtx6000"
_RT_CORE_SERVERLESS_GPU = "gpu-rtx6000"


def resolve_registry() -> str:
    """Return the live Nebius registry prefix, or the placeholder when unset."""
    return (
        os.environ.get("NPA_E2E_REGISTRY", "").strip()
        or os.environ.get("NPA_REGISTRY", "").strip()
        or _PLACEHOLDER_REGISTRY
    )


def resolve_image(image_or_repo_tag: str) -> str:
    """Rewrite placeholder registry images to ``NPA_REGISTRY`` / ``NPA_E2E_REGISTRY``.

    Accepts either a full image reference or ``npa-<tool>:<tag>``.
    """
    value = str(image_or_repo_tag or "").strip()
    registry = resolve_registry().rstrip("/")
    if not value:
        return value
    if value.startswith(_PLACEHOLDER_REGISTRY):
        suffix = value[len(_PLACEHOLDER_REGISTRY) :].lstrip("/")
        return f"{registry}/{suffix}"
    if "your-registry-id" in value:
        return value.replace("your-registry-id", registry.rsplit("/", 1)[-1])
    if value.startswith("npa-") and ":" in value and "/" not in value:
        return f"{registry}/{value}"
    return value


def resolve_serverless_gpu_type(default: str = _DEFAULT_SERVERLESS_GPU) -> str:
    """GPU platform for serverless job creates on the live project.

    Prefer ``NPA_E2E_SERVERLESS_GPU_TYPE``; otherwise map legacy L40S aliases onto
    an RT-core platform that exists on the shared rtxpro project (``gpu-rtx6000``).
    """
    explicit = os.environ.get("NPA_E2E_SERVERLESS_GPU_TYPE", "").strip()
    if explicit:
        return explicit
    legacy = str(default or "").strip().lower()
    if legacy in {"gpu-l40s-d", "gpu-l40s-a", "l40s", "gpu-l40s", "gpu-rtx-pro-6000"}:
        return _RT_CORE_SERVERLESS_GPU
    if legacy in {"h200", "gpu-h200"}:
        return "gpu-h200-sxm"
    return default or _DEFAULT_SERVERLESS_GPU


_RTX6000_PRESETS = frozenset({"1gpu-24vcpu-218gb", "8gpu-192vcpu-1744gb"})


def resolve_serverless_gpu_preset(
    default: str = "1gpu-24vcpu-218gb",
    *,
    platform: str | None = None,
) -> str:
    """GPU preset for serverless job creates on the live project.

    Prefer ``NPA_E2E_SERVERLESS_PRESET``. When the resolved platform is
    ``gpu-rtx6000`` and the caller still passes an H100/H200/L40S preset,
    remap to the RTX6000 catalog preset ``1gpu-24vcpu-218gb``.
    """
    explicit = os.environ.get("NPA_E2E_SERVERLESS_PRESET", "").strip()
    if explicit:
        return explicit
    resolved_platform = (platform or "").strip() or resolve_serverless_gpu_type()
    preset = str(default or "").strip() or "1gpu-24vcpu-218gb"
    if resolved_platform == "gpu-rtx6000" and preset not in _RTX6000_PRESETS:
        return "1gpu-24vcpu-218gb"
    return preset
