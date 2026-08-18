"""Registry for shared mk8s and Soperator cluster backend adapters."""

from __future__ import annotations

from typing import Any

from npa.cluster_backends.base import (
    BackendCapabilities,
    BackendCapabilityError,
    BackendOwnershipError,
    ClusterBackend,
    MaterializedPlan,
    persisted_backend,
    require_backend_ownership,
)


def get_backend(name: str) -> ClusterBackend[Any, Any, Any, Any, Any]:
    """Resolve one adapter lazily, keeping backend packages acyclic."""

    normalized = name.strip().lower() or "mk8s"
    if normalized == "mk8s":
        from npa.cluster_backends.mk8s import MK8sBackend

        return MK8sBackend()
    if normalized == "soperator":
        from npa.cluster_backends.soperator import SoperatorBackend

        return SoperatorBackend()
    raise BackendCapabilityError(
        f"unsupported cluster backend {name!r}; expected 'mk8s' or 'soperator'"
    )


__all__ = [
    "BackendCapabilities",
    "BackendCapabilityError",
    "BackendOwnershipError",
    "ClusterBackend",
    "MaterializedPlan",
    "get_backend",
    "persisted_backend",
    "require_backend_ownership",
]
