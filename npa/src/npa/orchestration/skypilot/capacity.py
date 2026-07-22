"""Classify raw SkyPilot / Nebius GPU launch failures.

A ``sky launch`` (or ``sky jobs launch``) that fails because the requested GPU
tier has no free capacity ("NER" -- not-enough-resources) should be retried on
the next GPU tier, whereas a configuration or code failure should fail fast so
it is not masked by cycling through every accelerator. This module centralizes
the text signatures that mark a capacity shortfall so the GPU-chain launch
helpers (and live GPU tests) can make that distinction consistently.
"""

from __future__ import annotations

# Lowercased substrings that specifically indicate a GPU / instance capacity
# shortfall rather than a bad request or a bug. Deliberately high-confidence:
# these must not match generic transient/rate-limit/healthy-scheduler output,
# otherwise a real failure on the last GPU tier would be misreported as
# "no capacity" and the fail-fast contract would be defeated. The Kubernetes
# GPU shortfall is matched via the exact scheduler reason string
# ("insufficient nvidia.com/gpu") rather than the bare "nodes are available",
# which also appears in healthy events like "3/3 nodes are available".
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
    # Nebius / generic cloud capacity
    "insufficient capacity",
    "insufficientinstancecapacity",
    "no capacity available",
    "capacity not available",
    "out of capacity",
    "out of stock",
    "no gpu available",
    "no available gpu",
    # Kubernetes GPU scheduling shortfall (exact scheduler reason)
    "insufficient nvidia.com/gpu",
)


def is_capacity_error(text: str | None) -> bool:
    """Return True when ``text`` looks like a retryable GPU capacity shortfall.

    ``text`` is typically the combined stdout+stderr of a failed ``sky launch``.
    Matching is intentionally conservative: only unambiguous capacity signatures
    trigger a retry on the next GPU tier; everything else fails fast.
    """

    if not text:
        return False
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in CAPACITY_ERROR_PATTERNS)
