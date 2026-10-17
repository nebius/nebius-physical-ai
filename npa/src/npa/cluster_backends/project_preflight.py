"""Read project-authorized quota and reservation evidence for one mk8s target."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json

from npa.clients.nebius import NebiusError, is_permission_denied
from npa.quota_scope import (
    QuotaCatalog,
    QuotaCatalogUnavailable,
    read_scoped_quota_catalog,
)


class TenantCapacityAccessDenied(ValueError):
    """Tenant inventory was denied before any exact-project fallback was possible."""


@dataclass(frozen=True)
class ProjectCapacityReader:
    """Bind every capacity read to one explicit provider profile and environment."""

    nebius_bin: str
    tenant_id: str
    project_id: str
    profile: str
    env: dict[str, str]
    run_capture: Callable
    nebius_argv: Callable

    def list_resources(self, resource: tuple[str, ...], parent_id: str) -> dict:
        """Read a complete catalog without substituting another principal.

        Args:
            resource: Provider service and resource command.
            parent_id: Exact tenant or project to read.
        Returns:
            A complete mapping with an items list.
        Raises:
            NebiusError: The provider denies or fails the read.
            ValueError: The response is malformed or incomplete.
        """
        result = self.run_capture(
            [
                *self.nebius_argv(self.nebius_bin, self.profile),
                *resource,
                "list",
                "--parent-id",
                parent_id,
                "--all",
                "--format",
                "json",
            ],
            env=self.env,
            check=False,
            timeout=120,
        )
        if result.returncode:
            raise NebiusError(
                str(getattr(result, "stderr", "") or "provider read failed")
            )
        return _complete_catalog(result.stdout)

    def quota_catalog(self, region: str, names: Mapping[str, int]) -> QuotaCatalog:
        """Select tenant evidence or validated project constraints after RBAC denial.

        Args:
            region: Exact deployment region.
            names: Required quota names and amounts.
        Returns:
            The original complete catalog with verified project-default semantics.
        Raises:
            ValueError: Quota reads are unavailable.
            ValueError: Project evidence is malformed or unusable.
        """
        try:
            catalog = read_scoped_quota_catalog(
                self.tenant_id,
                self.project_id,
                reader=lambda parent: self.list_resources(
                    ("quotas", "quota-allowance"), parent
                ),
            )
        except QuotaCatalogUnavailable as exc:
            if self.project_id:
                raise
            failure = (
                TenantCapacityAccessDenied
                if isinstance(exc.__cause__, NebiusError)
                and is_permission_denied(str(exc.__cause__))
                else ValueError
            )
            raise failure(
                f"could not read tenant quota allowances for {region}; refusing to "
                "create projects or resources without capacity evidence"
            ) from exc
        if catalog.scope == "project":
            _validate_project_defaults(catalog.payload, self.project_id, region, names)
        return catalog

    def reservation_catalog(self, requirements: Mapping, on_status=None) -> dict | None:
        """Verify exact project entitlement when tenant reservations are forbidden.

        Args:
            requirements: Explicit reservation IDs and required GPU counts.
            on_status: Optional non-secret progress sink.
        Returns:
            Tenant catalog, or None after exact-project entitlement verification.
        Raises:
            NebiusError: A read fails outside the supported IAM fallback.
            ValueError: Project entitlement cannot cover the strict request.
        """
        try:
            return self.list_resources(
                ("capacity", "capacity-block-group"), self.tenant_id
            )
        except NebiusError as exc:
            if not self.project_id and is_permission_denied(str(exc)):
                raise TenantCapacityAccessDenied(
                    "tenant reservation inventory is unavailable due to RBAC"
                ) from exc
            if not self.project_id or not is_permission_denied(str(exc)):
                raise ValueError(
                    "could not read capacity block groups; refusing to bypass STRICT "
                    "reservation preflight"
                ) from exc
        self.verify_project_allowances(requirements)
        if on_status:
            on_status(
                "Project reservation allowances verified; tenant inventory is unavailable "
                "due to RBAC. Provider apply enforces actual reserved capacity and compatibility."
            )
        return None

    def verify_project_allowances(self, requirements: Mapping) -> None:
        """Verify exact project entitlement without consulting tenant inventory.

        Args:
            requirements: Cumulative incremental GPU demand by reservation ID.
        Returns:
            None after the complete project catalog verifies every requirement.
        Raises:
            ValueError: Identity, allowance state, usage or headroom is unverified.
        """
        if not self.project_id:
            raise ValueError("project reservation validation requires an exact project")
        try:
            payload = self.list_resources(
                ("capacity", "capacity-allowance"), self.project_id
            )
        except (NebiusError, ValueError) as exc:
            raise ValueError("could not verify project reservation allowances") from exc
        _verify_project_reservations(payload, self.project_id, requirements)

    def verify_reservations(self, requirements: Mapping, on_status=None) -> None:
        """Validate strict placement using the permitted reservation evidence.

        Args:
            requirements: Exact reservations and cumulative GPU demand.
            on_status: Optional non-secret progress sink.
        Returns:
            None after complete tenant or exact-project evidence is verified.
        Raises:
            ValueError: Capacity or compatibility is denied or unverified.
        """
        from npa.cluster_backends.quotas import (
            find_reservation_shortfalls,
            parse_capacity_blocks,
            reservation_shortfall_message,
        )

        payload = self.reservation_catalog(requirements, on_status)
        if payload is None:
            return
        blocks = parse_capacity_blocks(json.dumps(payload))
        shortfalls = find_reservation_shortfalls(requirements, blocks, self.tenant_id)
        if shortfalls:
            raise ValueError(reservation_shortfall_message(shortfalls))
        if on_status:
            reserved = sum(item.required_gpus for item in requirements.values())
            on_status(
                f"validated {reserved} GPU(s) against {len(requirements)} active "
                "capacity block group(s); ordinary GPU quota excluded"
            )


def _validate_project_defaults(payload, project_id, region, names):
    from npa.provisioning_preflight import (
        _project_quota_payload,
        parse_quota_allowances,
    )

    normalized = _project_quota_payload(payload, project_id=project_id, region=region)
    observations = parse_quota_allowances(normalized, region=region, names=tuple(names))
    # Preserve the raw catalog for the backend's stricter unit and fractional
    # usage accounting, after the shared parser has ruled out ambiguous limits.
    if any(item.state == "unknown" for item in observations.values()):
        raise ValueError("project quota has unusable limit or status evidence")


def _project_allowance_rows(payload, project_id):
    indexed = {}
    for row in payload["items"]:
        if not isinstance(row, dict):
            raise ValueError("malformed project reservation allowance")
        metadata, spec, status = (
            row.get(key) for key in ("metadata", "spec", "status")
        )
        if not all(isinstance(part, dict) for part in (metadata, spec, status)):
            raise ValueError("malformed project reservation allowance")
        if metadata.get("parent_id") != project_id:
            raise ValueError("reservation allowance belongs to another project")
        reservation_id = spec.get("capacity_block_group_id")
        if not reservation_id or reservation_id in indexed:
            raise ValueError("missing or duplicate project reservation identity")
        indexed[reservation_id] = row
    return indexed


def _verify_project_reservations(payload, project_id, requirements):
    from npa.cluster_backends.quotas import _finite_allowance, _unlimited_allowance

    allowances = _project_allowance_rows(payload, project_id)
    # This resource proves project entitlement, not the tenant block's region,
    # platform, fabric or remaining physical capacity. Apply still checks those.
    for reservation_id, requirement in requirements.items():
        row = allowances.get(reservation_id)
        if row is None:
            raise ValueError(
                "project has no allowance for the requested strict reservation"
            )
        if row["status"].get("state") != "STATE_ACTIVE":
            raise ValueError("project reservation allowance is not active")
        if row["status"].get("unit") != "gpu":
            raise ValueError("project reservation allowance has an incompatible unit")
        name = "compute.instance.gpu.reserved"
        if "limit" in row["spec"]:
            available = _finite_allowance(name, row)["available"]
            if available < requirement.required_gpus:
                raise ValueError(
                    "project reservation allowance has insufficient GPU headroom"
                )
        else:
            _unlimited_allowance(name, row)


def _complete_catalog(stdout):
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("provider capacity catalog is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("provider capacity catalog has no items list")
    if payload.get("next_page_token"):
        raise ValueError("provider capacity catalog is incomplete")
    return payload
