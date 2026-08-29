"""Tool-neutral GPU classification for workbench routing guardrails.

Every workbench tool that has to decide "may this workload run on this GPU?"
needs the same first step: turn a GPU or provider target into a capability
class. The classes are physical, not commercial:

* :data:`RT_CORE` parts can rasterize frames (L40S, RTX PRO 6000).
* :data:`DATACENTER_HEADLESS` parts cannot (H100, H200, A100, B200, B300).
* :data:`CPU` is no accelerator at all.

"Blackwell" is a marketing family, not one GPU class. The workstation part
(RTX PRO 6000, ``sm_120``) has RT cores; the datacenter parts (B200 ``sm_100``,
B300 ``sm_103``) do not, exactly like H100/H200. So a bare "blackwell" token is
only read as RT-core when no datacenter model number is present.

The *policy* - which workload may use which class - stays with each tool, because
SONIC, Isaac Lab, and LeIsaac do not agree on it and should not. This module
holds only the classification the policies share, so a tool cannot drift into its
own private idea of what a B200 is. Everything here is pure and dependency-free
so all three tiers (raw YAML materialization, the SDK, and the CLI) can use it.
"""

from __future__ import annotations

CPU = "cpu"
RT_CORE = "rt-core"
DATACENTER_HEADLESS = "datacenter-headless"
UNKNOWN = "unknown"

# Substring tokens, matched against a normalized target.
RT_CORE_TOKENS = (
    "l40s",
    "rtx",
    "rtxpro",
    "rtx-pro",
    "rtx6000",
    "blackwell",
    "sm-120",
    "sm120",
)
DATACENTER_HEADLESS_TOKENS = (
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
CPU_TOKENS = (
    "cpu",
    "none",
    "host",
)

RT_CORE_HINT = (
    "RT-core GPUs (L40S, RTX PRO 6000 Blackwell sm_120) are the only ones that "
    "can rasterize frames; H100, H200, and datacenter Blackwell B200/B300 have "
    "no RT cores."
)


class GpuRoutingError(ValueError):
    """Raised when a workload is routed to an incompatible GPU class."""


def normalize_gpu_target(gpu_target: str | None) -> str:
    return (gpu_target or "").strip().lower().replace("_", "-")


def classify_gpu_target(gpu_target: str | None) -> str:
    """Classify a GPU/provider target into a routing class.

    Returns one of :data:`CPU`, :data:`RT_CORE`, :data:`DATACENTER_HEADLESS`,
    or :data:`UNKNOWN`. An empty target is treated as :data:`UNKNOWN` so callers
    can decide whether to require an explicit selection.
    """

    normalized = normalize_gpu_target(gpu_target)
    if not normalized:
        return UNKNOWN
    # Datacenter models are checked first: "b200"/"b300" are unambiguous, and a
    # name like "blackwell-b300" must not be read as RT-core just because it
    # carries the family name. Datacenter Blackwell has no RT cores.
    if any(token in normalized for token in DATACENTER_HEADLESS_TOKENS):
        return DATACENTER_HEADLESS
    # RT-core tokens then win over the remaining datacenter set so an RTX or
    # workstation-Blackwell part is never misread as headless-only.
    if any(token in normalized for token in RT_CORE_TOKENS):
        return RT_CORE
    if any(token == normalized or token in normalized for token in CPU_TOKENS):
        return CPU
    return UNKNOWN


def is_rt_core_target(gpu_target: str | None) -> bool:
    """Return True when the target is an RT-core (render-capable) GPU."""

    return classify_gpu_target(gpu_target) == RT_CORE


def is_datacenter_headless_target(gpu_target: str | None) -> bool:
    """Return True when the target is a headless datacenter GPU (no RT cores)."""

    return classify_gpu_target(gpu_target) == DATACENTER_HEADLESS


def require_rt_core_target(
    gpu_target: str | None,
    *,
    what: str,
    hint: str = "",
    error_cls: type[GpuRoutingError] = GpuRoutingError,
) -> str:
    """Validate that a rendering workload targets an RT-core GPU.

    An empty target is allowed (the caller falls back to its RT-core default).
    A datacenter-headless or otherwise unrecognized target fails loud, with
    ``error_cls`` so each tool keeps raising its own exception type.
    """

    normalized = normalize_gpu_target(gpu_target)
    if not normalized:
        return ""
    resolved_hint = hint or RT_CORE_HINT
    gpu_class = classify_gpu_target(gpu_target)
    if gpu_class == RT_CORE:
        return normalized
    if gpu_class == DATACENTER_HEADLESS:
        raise error_cls(
            f"{what} cannot run on the datacenter-headless GPU {gpu_target!r} "
            f"(H100/H200/A100 and datacenter Blackwell B200/B300 have no RT "
            f"cores). {resolved_hint}"
        )
    raise error_cls(
        f"{what} requires an RT-core GPU; {gpu_target!r} is not recognized as "
        f"RT-core. {resolved_hint}"
    )


__all__ = [
    "CPU",
    "CPU_TOKENS",
    "DATACENTER_HEADLESS",
    "DATACENTER_HEADLESS_TOKENS",
    "RT_CORE",
    "RT_CORE_HINT",
    "RT_CORE_TOKENS",
    "UNKNOWN",
    "GpuRoutingError",
    "classify_gpu_target",
    "is_datacenter_headless_target",
    "is_rt_core_target",
    "normalize_gpu_target",
    "require_rt_core_target",
]
