"""SONIC GPU-routing guardrails.

This module is the single source of truth for which GPU class each SONIC
pipeline stage may run on. The rules encode the physical constraints of the
pipeline so that a misroute fails fast with an actionable diagnostic instead of
wasting a live GPU launch:

* ``retarget`` is a CPU-only motion-library job.
* ``finetune`` / ``train`` / ``mujoco-eval`` are headless, state-based
  (proprioceptive) workloads. They run on datacenter-headless GPUs
  (H100/H200/A100/B200/B300) and also run on RT-core GPUs.
* ``isaac-render`` needs RT cores to rasterize frames, so it is restricted to
  RT-core GPUs (L40S, RTX PRO 6000, Blackwell ``sm_120``) and must never be
  routed to H100/H200/A100 or datacenter Blackwell (B200/B300) parts.

GPU classification itself is tool-neutral and lives in
:mod:`npa.workbench.gpu_classes`; this module holds only SONIC's policy over
those classes. Blackwell also requires a CUDA 12.8+ build; that build selection
is handled by the image manifest.

The helpers are pure and dependency-free so they can be reused across all three
tiers (raw YAML materialization, the SDK, and the CLI) without drift.
"""

from __future__ import annotations

from npa.workbench.gpu_classes import (
    CPU,
    DATACENTER_HEADLESS,
    RT_CORE,
    UNKNOWN,
    GpuRoutingError,
    classify_gpu_target,
    is_datacenter_headless_target,
    is_rt_core_target,
    require_rt_core_target,
)

# Workload identifiers used across the SONIC pipeline stages.
RETARGET = "retarget"
FINETUNE = "finetune"
TRAIN = "train"
MUJOCO_EVAL = "mujoco-eval"
ISAAC_RENDER = "isaac-render"

# Allowed GPU classes per workload.
_WORKLOAD_ALLOWED_CLASSES: dict[str, frozenset[str]] = {
    RETARGET: frozenset({CPU}),
    FINETUNE: frozenset({DATACENTER_HEADLESS, RT_CORE}),
    TRAIN: frozenset({DATACENTER_HEADLESS, RT_CORE}),
    MUJOCO_EVAL: frozenset({DATACENTER_HEADLESS, RT_CORE}),
    ISAAC_RENDER: frozenset({RT_CORE}),
}

_RENDER_HINT = (
    "Use an RT-core GPU such as --gpu-target l40s or "
    "--gpu-target gpu-rtx-pro-6000 (Blackwell sm_120). RT-core GPUs are the "
    "only ones that can rasterize Isaac-Lab render frames."
)


class SonicRoutingError(GpuRoutingError):
    """Raised when a SONIC workload is routed to an incompatible GPU class."""


def validate_render_gpu_target(gpu_target: str | None, *, what: str = "Isaac-Lab render") -> str:
    """Validate that a render workload targets an RT-core GPU."""

    return require_rt_core_target(
        gpu_target,
        what=what,
        hint=_RENDER_HINT,
        error_cls=SonicRoutingError,
    )


def validate_gpu_routing(*, workload: str, gpu_target: str | None) -> str:
    """Validate a workload/GPU pairing against the SONIC routing rules.

    Returns the resolved GPU class on success and raises
    :class:`SonicRoutingError` on a misroute. An empty/unknown target for a
    GPU workload is allowed so callers may rely on their own defaults; an
    explicitly-classified mismatch fails loud.
    """

    normalized_workload = (workload or "").strip().lower()
    allowed = _WORKLOAD_ALLOWED_CLASSES.get(normalized_workload)
    if allowed is None:
        choices = ", ".join(sorted(_WORKLOAD_ALLOWED_CLASSES))
        raise SonicRoutingError(
            f"unknown SONIC workload {workload!r}; choose one of: {choices}"
        )

    gpu_class = classify_gpu_target(gpu_target)

    if normalized_workload == RETARGET:
        if gpu_class in (CPU, UNKNOWN):
            return CPU
        raise SonicRoutingError(
            f"retarget is a CPU-only motion-library job; {gpu_target!r} requests "
            "a GPU. Run retarget without an accelerator."
        )

    if normalized_workload == ISAAC_RENDER:
        validate_render_gpu_target(gpu_target)
        return gpu_class if gpu_class != UNKNOWN else RT_CORE

    # Headless-capable workloads (finetune / train / mujoco-eval).
    if gpu_class == UNKNOWN:
        return UNKNOWN
    if gpu_class == CPU:
        raise SonicRoutingError(
            f"{normalized_workload} requires a GPU; {gpu_target!r} is CPU-only."
        )
    if gpu_class in allowed:
        return gpu_class
    raise SonicRoutingError(
        f"{normalized_workload} cannot run on GPU class {gpu_class!r} "
        f"({gpu_target!r}); allowed classes: {', '.join(sorted(allowed))}."
    )


__all__ = [
    "CPU",
    "DATACENTER_HEADLESS",
    "FINETUNE",
    "ISAAC_RENDER",
    "MUJOCO_EVAL",
    "RETARGET",
    "RT_CORE",
    "TRAIN",
    "UNKNOWN",
    "SonicRoutingError",
    "classify_gpu_target",
    "is_datacenter_headless_target",
    "is_rt_core_target",
    "validate_gpu_routing",
    "validate_render_gpu_target",
]
