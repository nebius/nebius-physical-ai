"""NPA resource conventions mapped to SkyPilot resource blocks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from npa.cluster.config import (
    DEFAULT_NODE_PLATFORM,
    DEFAULT_NODE_PRESET,
    DEFAULT_REGION,
    GPU_TYPE_DEFAULTS,
    SUPPORTED_NODE_PRESETS,
)

SKYPILOT_VERSION = "0.12.2"
DEFAULT_BACKEND = "nebius"
DEFAULT_AUTOSTOP_IDLE_MINUTES = 5


class SkyPilotResourceError(ValueError):
    """Base error for invalid NPA SkyPilot resource specifications."""


class InvalidResourceSpecError(SkyPilotResourceError):
    """Raised when an NPA resource spec cannot be mapped to SkyPilot."""


@dataclass(frozen=True)
class NPASpec:
    """Validated NPA resource request."""

    backend: str = DEFAULT_BACKEND
    gpu: str | None = None
    count: int = 1
    cpus: int = 2
    memory_gb: int = 8
    region: str = DEFAULT_REGION


# Source: `sky show-gpus --cloud nebius --all` captured at
# /tmp/w9skypilot-integration-bootstrap-20260516T011706Z/phase-a/
# sky-show-gpus-nebius.txt. SkyPilot 0.12.2 reports H100, H200, and L40S
# for Nebius. RTXPRO6000 is the global SkyPilot spelling for RTX PRO 6000,
# but it is not present in the 0.12.2 Nebius catalog; see novel-issues.md.
NPA_GPU_TO_SKYPILOT_ACCELERATOR = {
    "h100": "H100",
    "h200": "H200",
    "l40s": "L40S",
    "rtx6000": "RTXPRO6000",
}

GPU_INSTANCE_TYPES_BY_COUNT = {
    "h100": {
        1: "gpu-h100-sxm_1gpu-16vcpu-200gb",
        8: "gpu-h100-sxm_8gpu-128vcpu-1600gb",
    },
    "h200": {
        1: "gpu-h200-sxm_1gpu-16vcpu-200gb",
        8: "gpu-h200-sxm_8gpu-128vcpu-1600gb",
    },
    "l40s": {
        1: "gpu-l40s-d_1gpu-16vcpu-96gb",
        2: "gpu-l40s-d_2gpu-64vcpu-384gb",
        4: "gpu-l40s-d_4gpu-128vcpu-768gb",
    },
    "rtx6000": {
        1: "gpu-rtx6000_1gpu-24vcpu-218gb",
    },
}

CPU_INSTANCE_TYPES = {
    (platform, preset): f"{platform}_{preset}"
    for platform, presets in SUPPORTED_NODE_PRESETS.items()
    for preset in presets
}


def _shape_from_preset(preset: str) -> tuple[int, int] | None:
    cpu_token, memory_token = preset.split("-", 1)
    if not cpu_token.endswith("vcpu") or not memory_token.endswith("gb"):
        return None
    return int(cpu_token.removesuffix("vcpu")), int(memory_token.removesuffix("gb"))


CPU_PRESET_BY_SHAPE = {
    _shape_from_preset(preset): f"{platform}_{preset}"
    for platform, presets in SUPPORTED_NODE_PRESETS.items()
    for preset in presets
    if (_shape_from_preset(preset) is not None)
}


def validate_npa_spec(spec: Mapping[str, Any] | NPASpec) -> None:
    """Validate an NPA resource spec before converting it to SkyPilot."""

    normalized = _normalize_spec(spec)
    if normalized.backend not in {"nebius", "kubernetes"}:
        raise InvalidResourceSpecError("backend must be 'nebius' or 'kubernetes'")
    if normalized.region != DEFAULT_REGION:
        raise InvalidResourceSpecError(f"unsupported region '{normalized.region}'. Supported: {DEFAULT_REGION}")
    if normalized.count <= 0:
        raise InvalidResourceSpecError("count must be positive")
    if normalized.cpus <= 0:
        raise InvalidResourceSpecError("cpus must be positive")
    if normalized.memory_gb <= 0:
        raise InvalidResourceSpecError("memory_gb must be positive")
    if normalized.gpu is None:
        return
    if normalized.gpu not in NPA_GPU_TO_SKYPILOT_ACCELERATOR:
        allowed = ", ".join(sorted(NPA_GPU_TO_SKYPILOT_ACCELERATOR))
        raise InvalidResourceSpecError(f"unsupported GPU type '{normalized.gpu}'. Supported: {allowed}")
    if normalized.backend == "nebius" and normalized.count not in GPU_INSTANCE_TYPES_BY_COUNT[normalized.gpu]:
        allowed = ", ".join(str(count) for count in sorted(GPU_INSTANCE_TYPES_BY_COUNT[normalized.gpu]))
        raise InvalidResourceSpecError(
            f"unsupported count {normalized.count} for Nebius GPU '{normalized.gpu}'. Supported: {allowed}"
        )


def resources_for_npa_spec(spec: Mapping[str, Any] | NPASpec) -> dict[str, Any]:
    """Return a SkyPilot resources block for an NPA resource convention dict."""

    normalized = _normalize_spec(spec)
    validate_npa_spec(normalized)

    if normalized.backend == "kubernetes":
        return _kubernetes_resources(normalized)
    return _nebius_resources(normalized)


def _normalize_spec(spec: Mapping[str, Any] | NPASpec) -> NPASpec:
    if isinstance(spec, NPASpec):
        return spec
    if not isinstance(spec, Mapping):
        raise InvalidResourceSpecError("spec must be a mapping or NPASpec")

    gpu_raw = spec.get("gpu")
    gpu = str(gpu_raw).lower().strip() if gpu_raw not in {None, ""} else None
    return NPASpec(
        backend=str(spec.get("backend", DEFAULT_BACKEND)).lower().strip(),
        gpu=gpu,
        count=int(spec.get("count", 1)),
        cpus=int(spec.get("cpus", 2)),
        memory_gb=int(spec.get("memory_gb", 8)),
        region=str(spec.get("region", DEFAULT_REGION)).strip() or DEFAULT_REGION,
    )


def _nebius_resources(spec: NPASpec) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "cloud": "nebius",
        "region": spec.region,
        "autostop": {"idle_minutes": DEFAULT_AUTOSTOP_IDLE_MINUTES, "down": False},
    }
    if spec.gpu is None:
        resources["instance_type"] = _cpu_instance_type(spec.cpus, spec.memory_gb)
        return resources

    accelerator = NPA_GPU_TO_SKYPILOT_ACCELERATOR[spec.gpu]
    resources["instance_type"] = GPU_INSTANCE_TYPES_BY_COUNT[spec.gpu][spec.count]
    resources["accelerators"] = f"{accelerator}:{spec.count}"
    return resources


def _kubernetes_resources(spec: NPASpec) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "cloud": "kubernetes",
        "cpus": spec.cpus,
        "memory": spec.memory_gb,
    }
    if spec.gpu is not None:
        accelerator = NPA_GPU_TO_SKYPILOT_ACCELERATOR[spec.gpu]
        resources["accelerators"] = f"{accelerator}:{spec.count}"
    return resources


def _cpu_instance_type(cpus: int, memory_gb: int) -> str:
    if (cpus, memory_gb) in CPU_PRESET_BY_SHAPE:
        return CPU_PRESET_BY_SHAPE[(cpus, memory_gb)]
    default_key = (DEFAULT_NODE_PLATFORM, DEFAULT_NODE_PRESET)
    if (cpus, memory_gb) == _shape_from_preset(DEFAULT_NODE_PRESET):
        return CPU_INSTANCE_TYPES[default_key]
    raise InvalidResourceSpecError(
        f"unsupported Nebius CPU shape {cpus} vCPU / {memory_gb} GiB; use an NPA CPU preset shape"
    )


def gpu_defaults() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of NPA cluster GPU defaults for documentation/tests."""

    return {key: dict(value) for key, value in GPU_TYPE_DEFAULTS.items()}
