"""Select exact-project quota evidence only after a tenant-scope IAM denial."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from npa.clients.nebius import NebiusError, is_permission_denied


@dataclass(frozen=True)
class QuotaCatalog:
    """One provider response and the permission scope that authorized its read."""

    payload: Mapping[str, Any]
    scope: str


class QuotaCatalogUnavailable(ValueError):
    """A sanitized failure to obtain authoritative quota evidence."""


def _read_project_catalog(project_id: str, reader: Callable) -> QuotaCatalog:
    try:
        return QuotaCatalog(reader(project_id), "project")
    except Exception as exc:  # noqa: BLE001 - provider and decoding boundaries
        raise QuotaCatalogUnavailable(
            "tenant-wide quota query is unavailable due to RBAC and the "
            f"project-scoped quota query failed ({type(exc).__name__})"
        ) from exc


def read_scoped_quota_catalog(
    tenant_id: str,
    project_id: str,
    *,
    reader: Callable[[str], Mapping[str, Any]],
) -> QuotaCatalog:
    """Read tenant quotas, falling back only to the exact existing project.

    Args:
        tenant_id: Selected tenant whose aggregate the provider enforces.
        project_id: Exact existing project; empty for tenant-wide fleet operations.
        reader: Profile-bound provider reader, including complete pagination.
    Returns:
        Unmodified provider evidence with its actual authorization scope.
    Raises:
        QuotaCatalogUnavailable: Reads fail or tenant failure is not an IAM denial.
    """
    try:
        return QuotaCatalog(reader(tenant_id), "tenant")
    except Exception as exc:  # noqa: BLE001 - classify before scope fallback
        if (
            not project_id
            or not isinstance(exc, NebiusError)
            or not is_permission_denied(str(exc))
        ):
            raise QuotaCatalogUnavailable(
                "tenant-wide quota query failed for a reason other than an RBAC "
                f"scope limitation ({type(exc).__name__})"
            ) from exc
    return _read_project_catalog(project_id, reader)
