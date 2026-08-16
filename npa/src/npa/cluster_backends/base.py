"""Shared contract for NPA cluster provisioning backends.

Fleet owns target expansion, concurrency, aggregation, and inventory. Backends
own one cluster's desired state and lifecycle. The generic parameters keep the
mk8s and Soperator request models distinct instead of creating a bag of
optional fields that neither backend can validate precisely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


DesiredT = TypeVar("DesiredT")
MaterializedT = TypeVar("MaterializedT")
ApplyRequestT = TypeVar("ApplyRequestT")
StatusRequestT = TypeVar("StatusRequestT")
DestroyRequestT = TypeVar("DestroyRequestT")


class BackendCapabilityError(ValueError):
    """A desired feature is not supported by the selected backend."""


class BackendOwnershipError(RuntimeError):
    """Persisted state belongs to a different backend than the requested one."""


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    supports_mig: bool
    supports_shared_filestore: bool
    supports_slurm: bool
    supports_capacity_blocks: bool
    supports_cuda_verification: bool


@dataclass(frozen=True)
class MaterializedPlan:
    """Provider-free rendering evidence shared by CLI, SDK, agent, and fleet."""

    backend: str
    desired_state: dict[str, Any]
    deployment_inputs: dict[str, Any]
    rendered_path: Path | None = None


@runtime_checkable
class ClusterBackend(
    Protocol,
    Generic[
        DesiredT,
        MaterializedT,
        ApplyRequestT,
        StatusRequestT,
        DestroyRequestT,
    ],
):
    """One complete per-cluster lifecycle behind all NPA surfaces."""

    name: str
    capabilities: BackendCapabilities

    def validate(self, desired: DesiredT) -> None: ...

    def plan(self, desired: DesiredT) -> dict[str, Any]: ...

    def preflight(
        self, desired: DesiredT, request: ApplyRequestT
    ) -> dict[str, Any]: ...

    def materialize(
        self, desired: DesiredT, request: ApplyRequestT
    ) -> MaterializedT: ...

    def apply(self, desired: DesiredT, request: ApplyRequestT) -> dict[str, Any]: ...

    def status(self, desired: DesiredT, request: StatusRequestT) -> dict[str, Any]: ...

    def verify(self, desired: DesiredT, request: StatusRequestT) -> dict[str, Any]: ...

    def destroy(
        self, desired: DesiredT, request: DestroyRequestT
    ) -> dict[str, Any] | None: ...


def persisted_backend(state: dict[str, Any]) -> str:
    """Read backend ownership; old fleet inventory is mk8s by definition."""

    value = str(state.get("backend", "mk8s") or "mk8s").strip().lower()
    if value not in {"mk8s", "soperator"}:
        raise BackendOwnershipError(
            f"persisted cluster state has unsupported backend {value!r}; refusing lifecycle action"
        )
    return value


def require_backend_ownership(state: dict[str, Any], expected: str) -> None:
    actual = persisted_backend(state)
    normalized = expected.strip().lower()
    if actual != normalized:
        raise BackendOwnershipError(
            f"persisted cluster state belongs to backend {actual!r}, but the spec "
            f"selects {normalized!r}; refusing to plan, reconcile, or destroy with "
            "the wrong backend"
        )
