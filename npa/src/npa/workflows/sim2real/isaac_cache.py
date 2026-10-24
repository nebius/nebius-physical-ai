"""Explicit custody for the durable Isaac runtime-cache claim.

The public Isaac image intentionally contains no restricted Isaac runtime bytes.
The canonical workflow warms those bytes in its own first SkyPilot state, but a
shared RWX claim must exist before that state can be scheduled.  This module
creates or verifies only that claim; it never creates a Kubernetes Job and it
never changes a claim owned by another tool or operator.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from npa.lifecycle_intent import OperationIntent, require_intent


_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]{0,251}[a-z0-9])?$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_STORAGE_RE = re.compile(r"^[1-9][0-9]*(?:Gi|Ti)$")

MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
MANAGED_BY_VALUE = "npa-workflow"
CACHE_ROLE_LABEL = "sim2real.npa.dev/cache-role"
CACHE_ROLE_VALUE = "isaac-runtime"


class IsaacCacheClaimError(RuntimeError):
    """A durable Isaac cache claim cannot be safely provisioned or reused."""


@dataclass(frozen=True)
class IsaacCacheClaim:
    """One verified cache-claim observation suitable for a workflow receipt."""

    name: str
    namespace: str
    phase: str
    access_modes: tuple[str, ...]
    storage_class: str
    requested_storage: str
    custody: str
    created: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _api_status(exc: BaseException) -> int:
    try:
        return int(getattr(exc, "status", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _require_dns(value: str, *, field: str) -> str:
    resolved = str(value or "").strip()
    if not _DNS_SUBDOMAIN_RE.fullmatch(resolved):
        raise IsaacCacheClaimError(f"{field} must be a DNS-safe Kubernetes name")
    return resolved


def _require_storage(value: str) -> str:
    resolved = str(value or "").strip()
    if not _STORAGE_RE.fullmatch(resolved):
        raise IsaacCacheClaimError(
            "storage must be a positive whole Gi or Ti quantity, for example 64Gi"
        )
    return resolved


def _require_namespace(value: str) -> str:
    resolved = str(value or "").strip()
    if not _DNS_LABEL_RE.fullmatch(resolved):
        raise IsaacCacheClaimError("namespace must be a DNS-label Kubernetes name")
    return resolved


def cache_claim_manifest(
    *, name: str, namespace: str, storage_class: str, storage: str
) -> dict[str, Any]:
    """Build the immutable shape of an NPA-managed shared Isaac cache claim."""

    name = _require_dns(name, field="name")
    namespace = _require_namespace(namespace)
    storage = _require_storage(storage)
    selected_storage_class = str(storage_class or "").strip()
    if selected_storage_class:
        _require_dns(selected_storage_class, field="storage_class")
    spec: dict[str, Any] = {
        "accessModes": ["ReadWriteMany"],
        "resources": {"requests": {"storage": storage}},
    }
    if selected_storage_class:
        spec["storageClassName"] = selected_storage_class
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                MANAGED_BY_LABEL: MANAGED_BY_VALUE,
                CACHE_ROLE_LABEL: CACHE_ROLE_VALUE,
            },
        },
        "spec": spec,
    }


def _claim_value(claim: Any, path: tuple[str, ...], default: Any = "") -> Any:
    current: Any = claim
    for part in path:
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _claim_observation(claim: Any, *, created: bool) -> IsaacCacheClaim:
    name = str(_claim_value(claim, ("metadata", "name")) or "").strip()
    namespace = str(_claim_value(claim, ("metadata", "namespace")) or "").strip()
    labels = dict(_claim_value(claim, ("metadata", "labels"), {}) or {})
    access_modes = tuple(
        str(item)
        for item in (
            _claim_value(claim, ("spec", "access_modes"), None)
            or _claim_value(claim, ("spec", "accessModes"), [])
            or []
        )
    )
    phase = str(_claim_value(claim, ("status", "phase")) or "").strip()
    storage_class = str(
        _claim_value(claim, ("spec", "storage_class_name"), None)
        or _claim_value(claim, ("spec", "storageClassName"), "")
        or ""
    ).strip()
    requested = str(
        _claim_value(claim, ("spec", "resources", "requests", "storage"), "") or ""
    ).strip()
    managed = (
        labels.get(MANAGED_BY_LABEL) == MANAGED_BY_VALUE
        and labels.get(CACHE_ROLE_LABEL) == CACHE_ROLE_VALUE
    )
    partial_npa_ownership = (
        labels.get(MANAGED_BY_LABEL) == MANAGED_BY_VALUE
        or labels.get(CACHE_ROLE_LABEL) == CACHE_ROLE_VALUE
    ) and not managed
    if partial_npa_ownership:
        raise IsaacCacheClaimError(
            f"claim {name!r} has incomplete NPA cache custody labels; refusing adoption"
        )
    return IsaacCacheClaim(
        name=name,
        namespace=namespace,
        phase=phase,
        access_modes=access_modes,
        storage_class=storage_class,
        requested_storage=requested,
        custody="npa-managed" if managed else "operator-existing",
        created=created,
    )


def _verify_claim(
    observation: IsaacCacheClaim,
    *,
    storage_class: str,
    storage: str,
) -> None:
    if "ReadWriteMany" not in observation.access_modes:
        raise IsaacCacheClaimError(
            f"claim {observation.name!r} is not ReadWriteMany; it cannot safely "
            "serve a shared offline Isaac cache"
        )
    if observation.custody == "npa-managed" and (
        observation.storage_class != str(storage_class or "").strip()
        or observation.requested_storage != str(storage or "").strip()
    ):
        raise IsaacCacheClaimError(
            f"managed claim {observation.name!r} does not match the requested "
            "storage class or capacity; PVC spec fields are immutable, so create a "
            "new claim name instead of replacing it"
        )
    if observation.phase in {"Lost", "Failed"}:
        raise IsaacCacheClaimError(
            f"claim {observation.name!r} is terminally unusable (phase={observation.phase})"
        )


def provision_cache_claim(
    *,
    core_api: Any,
    name: str,
    namespace: str,
    storage_class: str,
    storage: str,
    wait: bool = True,
    poll_seconds: float = 2.0,
) -> IsaacCacheClaim:
    """Create one owned RWX claim or verify an existing claim without replacing it.

    Waiting has no built-in deadline.  The caller can choose ``wait=False`` for
    an observable pending claim, but a canonical workflow preflight will still
    refuse to schedule before it becomes Bound.
    """

    require_intent(OperationIntent.MUTATE, primitive="provision Isaac cache PVC")
    manifest = cache_claim_manifest(
        name=name,
        namespace=namespace,
        storage_class=storage_class,
        storage=storage,
    )
    created = False
    try:
        claim = core_api.read_namespaced_persistent_volume_claim(
            name=manifest["metadata"]["name"],
            namespace=manifest["metadata"]["namespace"],
        )
    except Exception as exc:
        if _api_status(exc) != 404:
            raise IsaacCacheClaimError(
                f"could not read Isaac cache claim {name!r}: status={_api_status(exc)}"
            ) from exc
        try:
            claim = core_api.create_namespaced_persistent_volume_claim(
                namespace=manifest["metadata"]["namespace"], body=manifest
            )
            created = True
        except Exception as create_exc:
            if _api_status(create_exc) != 409:
                raise IsaacCacheClaimError(
                    f"could not create Isaac cache claim {name!r}: "
                    f"status={_api_status(create_exc)}"
                ) from create_exc
            claim = core_api.read_namespaced_persistent_volume_claim(
                name=manifest["metadata"]["name"],
                namespace=manifest["metadata"]["namespace"],
            )

    observed = _claim_observation(claim, created=created)
    _verify_claim(observed, storage_class=storage_class, storage=storage)
    if not wait or observed.phase == "Bound":
        return observed
    while True:
        time.sleep(max(0.1, float(poll_seconds)))
        claim = core_api.read_namespaced_persistent_volume_claim(
            name=observed.name, namespace=observed.namespace
        )
        observed = _claim_observation(claim, created=created)
        _verify_claim(observed, storage_class=storage_class, storage=storage)
        if observed.phase == "Bound":
            return observed
