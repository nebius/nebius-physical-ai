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

"Blackwell" is a marketing family, not one GPU class. The workstation part
(RTX PRO 6000, ``sm_120``) has RT cores; the datacenter parts (B200 ``sm_100``,
B300 ``sm_103``) do not, exactly like H100/H200. So a bare "blackwell" token is
only read as RT-core when no datacenter model number is present.

Blackwell also requires a CUDA 12.8+ build; that build selection is handled by
the image manifest, but the GPU class is decided here.

The helpers are pure and dependency-free so they can be reused across all three
tiers (raw YAML materialization, the SDK, and the CLI) without drift.
"""

from __future__ import annotations

CPU = "cpu"
RT_CORE = "rt-core"
DATACENTER_HEADLESS = "datacenter-headless"
UNKNOWN = "unknown"

# Workload identifiers used across the SONIC pipeline stages.
RETARGET = "retarget"
FINETUNE = "finetune"
TRAIN = "train"
MUJOCO_EVAL = "mujoco-eval"
ISAAC_RENDER = "isaac-render"

# Substring tokens (matched against a normalized gpu target) that classify a GPU.
# RT-core parts can rasterize frames; datacenter-headless parts cannot.
_RT_CORE_TOKENS = (
    "l40s",
    "rtx",
    "rtxpro",
    "rtx-pro",
    "rtx6000",
    "blackwell",
    "sm-120",
    "sm120",
)
_DATACENTER_HEADLESS_TOKENS = (
    "h100",
    "h200",
    "a100",
    "b200",
    "b300",
    "sm-100",
    "sm100",
    "sm-103",
    "sm103",
)
_CPU_TOKENS = (
    "cpu",
    "none",
    "host",
)

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


class SonicRoutingError(ValueError):
    """Raised when a SONIC workload is routed to an incompatible GPU class."""


def _normalize(gpu_target: str | None) -> str:
    return (gpu_target or "").strip().lower().replace("_", "-")


def classify_gpu_target(gpu_target: str | None) -> str:
    """Classify a GPU/provider target into a SONIC routing class.

    Returns one of :data:`CPU`, :data:`RT_CORE`, :data:`DATACENTER_HEADLESS`,
    or :data:`UNKNOWN`. An empty target is treated as :data:`UNKNOWN` so callers
    can decide whether to require an explicit selection.
    """

    normalized = _normalize(gpu_target)
    if not normalized:
        return UNKNOWN
    # Datacenter models are checked first: "b200"/"b300" are unambiguous, and a
    # name like "blackwell-b300" must not be read as RT-core just because it
    # carries the family name. Datacenter Blackwell has no RT cores.
    if any(token in normalized for token in _DATACENTER_HEADLESS_TOKENS):
        return DATACENTER_HEADLESS
    # RT-core tokens then win over the remaining datacenter set so an RTX or
    # workstation-Blackwell part is never misread as headless-only.
    if any(token in normalized for token in _RT_CORE_TOKENS):
        return RT_CORE
    if any(token == normalized or token in normalized for token in _CPU_TOKENS):
        return CPU
    return UNKNOWN


def is_rt_core_target(gpu_target: str | None) -> bool:
    """Return True when the target is an RT-core (render-capable) GPU."""

    return classify_gpu_target(gpu_target) == RT_CORE


def is_datacenter_headless_target(gpu_target: str | None) -> bool:
    """Return True when the target is a headless datacenter GPU (no RT cores)."""

    return classify_gpu_target(gpu_target) == DATACENTER_HEADLESS


def validate_render_gpu_target(gpu_target: str | None, *, what: str = "Isaac-Lab render") -> str:
    """Validate that a render workload targets an RT-core GPU.

    An empty target is allowed (the caller falls back to the RT-core default).
    A datacenter-headless or otherwise non-RT-core target fails loud.
    """

    normalized = _normalize(gpu_target)
    if not normalized:
        return ""
    gpu_class = classify_gpu_target(gpu_target)
    if gpu_class == RT_CORE:
        return normalized
    if gpu_class == DATACENTER_HEADLESS:
        raise SonicRoutingError(
            f"{what} cannot run on the datacenter-headless GPU {gpu_target!r} "
            f"(H100/H200/A100 and datacenter Blackwell B200/B300 have no RT "
            f"cores). {_RENDER_HINT}"
        )
    raise SonicRoutingError(
        f"{what} requires an RT-core GPU; {gpu_target!r} is not recognized as "
        f"RT-core. {_RENDER_HINT}"
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
