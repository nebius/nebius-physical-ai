"""Classify raw SkyPilot / Nebius GPU launch failures.

A ``sky launch`` (or ``sky jobs launch``) that fails because the requested GPU
tier has no free capacity ("NER" -- not-enough-resources) should be retried on
the next GPU tier, whereas a configuration or code failure should fail fast so
it is not masked by cycling through every accelerator. This module centralizes
the text signatures that mark a capacity shortfall so the GPU-chain launch
helpers (and live GPU tests) can make that distinction consistently.
"""

from __future__ import annotations

# Lowercased substrings that indicate a GPU / instance capacity shortfall rather
# than a bad request or a bug. Kept broad on purpose: SkyPilot, the Nebius VM
# backend, and managed Kubernetes all phrase "no capacity right now" differently.
CAPACITY_ERROR_PATTERNS: tuple[str, ...] = (
    # SkyPilot resource resolution / provisioning
    "resourcesunavailableerror",
    "resources unavailable",
    "no resources available",
    "no launchable resource",
    "no resource satisfying",
    "no resources satisfy",
    "quota exceeded",
    "quota limit",
    "try again later",
    "retry later",
    # Nebius / generic cloud capacity
    "insufficient capacity",
    "insufficientinstancecapacity",
    "no capacity available",
    "capacity not available",
    "out of capacity",
    "out of stock",
    "no gpu available",
    "no available gpu",
    "resource not available",
    "scheduling failed",
    # Kubernetes scheduling shortfalls
    "insufficient nvidia.com/gpu",
    "nodes are available",  # "0/3 nodes are available: 3 Insufficient nvidia.com/gpu"
)


def is_capacity_error(text: str | None) -> bool:
    """Return True when ``text`` looks like a retryable GPU capacity shortfall.

    ``text`` is typically the combined stdout+stderr of a failed ``sky launch``.
    """

    if not text:
        return False
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in CAPACITY_ERROR_PATTERNS)
