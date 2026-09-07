"""Network ingress helpers backed by the Nebius CLI."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_network
import re
from typing import Any
from collections.abc import Callable, Sequence

from npa.clients import nebius
from npa.clients.nebius import NebiusError


class NetworkIngressError(Exception):
    """Raised when ingress cannot be resolved or changed."""

    def __init__(self, message: str, *, deleted: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.deleted = tuple(deleted)


class NetworkCleanupError(Exception):
    """Raised when ownership-safe network teardown cannot continue."""


_DEFAULT_SECURITY_GROUP_REFUSAL_MARKERS = (
    "cannotdeletedefaultsecuritygroup",
    "cantdeletedefaultsecuritygroup",
    "defaultsecuritygroupcannotbedeleted",
    "defaultsecuritygroupcantbedeleted",
    "deletionofdefaultsecuritygroupisnotallowed",
)


def is_default_security_group_delete_refusal(message: str) -> bool:
    """Whether *message* narrowly identifies the provider's default-SG refusal.

    Nebius documents that only non-default security groups are directly
    deletable. Match that specific condition without reclassifying ordinary
    dependency, permission, transport, or non-default security-group failures.
    """

    text = str(message or "").lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    if re.search(r"\bnon[- ]?default\b", text) or "nondefaultsecuritygroup" in compact:
        return False
    if any(marker in compact for marker in _DEFAULT_SECURITY_GROUP_REFUSAL_MARKERS):
        return True
    has_default_group = "securitygroup" in compact and "default" in compact
    has_refusal = any(
        marker in compact
        for marker in (
            "cannotdelete",
            "cantdelete",
            "cannotbedeleted",
            "cantbedeleted",
            "deletionisnotallowed",
            "deleteisnotallowed",
        )
    )
    return has_default_group and has_refusal


def recover_default_security_group_delete(
    *,
    error: str,
    parent_network_id: str,
    parent_network_owned: bool,
    cleanup_action: str,
    on_status: Callable[[str], None] | None = None,
) -> bool:
    """Delete an owned parent network after the provider rejects its default SG.

    Returns ``False`` for unrelated failures. A default security group is a
    provider-managed child and cannot be deleted directly; deleting its parent
    network is safe only when Terraform state proves NPA owns that network.
    """

    if not is_default_security_group_delete_refusal(error):
        return False

    action = str(cleanup_action or "the existing NPA cleanup action").strip()
    network_id = str(parent_network_id or "").strip()
    if not parent_network_owned or not network_id:
        raise NetworkCleanupError(
            "Nebius refused direct deletion of a default security group. The "
            "supported recovery is deletion of its parent network, but that requires "
            "proof that NPA owns the whole network. No such Terraform ownership proof "
            "was found, so the reused/shared network and its default security group "
            f"were preserved. Use {action} only for the owning NPA stack, or ask the "
            "network owner to remove the parent network."
        )

    status = on_status or (lambda _message: None)
    status(
        "Nebius default security groups cannot be deleted directly; Terraform "
        f"state proves parent network {network_id} is NPA-owned, so teardown is "
        "deleting that parent network through the supported provider lifecycle."
    )
    try:
        nebius._run(["vpc", "network", "delete", "--id", network_id])
    except NebiusError as exc:
        if nebius.is_not_found(str(exc)):
            status(
                f"NPA-owned parent network {network_id} is already absent; continuing."
            )
            return True
        raise NetworkCleanupError(
            "Nebius refused direct deletion of the default security group, and "
            f"deleting its NPA-owned parent network {network_id} also failed: {exc}. "
            f"Fix the reported provider error and retry {action}."
        ) from exc
    status(
        f"Deleted NPA-owned parent network {network_id}; its default security group "
        "is removed with the network."
    )
    return True


@dataclass(frozen=True)
class SecurityGroupIngressResult:
    security_group_id: str
    security_group_name: str
    network_id: str
    covered_ports: tuple[int, ...]
    missing_ports: tuple[int, ...]
    created_rule_id: str = ""
    created_rule_name: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.created_rule_id)


@dataclass(frozen=True)
class EnsureIngressResult:
    instance_id: str
    project_id: str
    public_ip: str
    ports: tuple[int, ...]
    source: str
    tool: str
    security_groups: tuple[SecurityGroupIngressResult, ...]

    @property
    def changed(self) -> bool:
        return any(group.changed for group in self.security_groups)

    @property
    def warnings(self) -> tuple[str, ...]:
        values: list[str] = []
        for group in self.security_groups:
            values.extend(group.warnings)
        return tuple(values)


@dataclass(frozen=True)
class InstanceNetworkContext:
    instance_id: str
    project_id: str
    public_ip: str
    security_group_ids: tuple[str, ...]

    @property
    def security_group_id(self) -> str:
        return self.security_group_ids[0] if self.security_group_ids else ""


@dataclass(frozen=True)
class _SecurityGroupContext:
    security_group: dict[str, Any]
    rules: tuple[dict[str, Any], ...]
    covered_ports: frozenset[int]
    warnings: tuple[str, ...]


def _validate_ingress_source(source: str, *, allow_world_open: bool) -> str:
    value = str(source or "").strip()
    if not value:
        raise NetworkIngressError("an explicit source CIDR is required")
    try:
        network = ip_network(value, strict=False)
    except ValueError as exc:
        raise NetworkIngressError(f"invalid source CIDR {value!r}") from exc
    if network.version != 4:
        raise NetworkIngressError("source CIDR must be IPv4")
    if network.prefixlen == 0 and not allow_world_open:
        raise NetworkIngressError(
            "world-open ingress requires explicit operator acknowledgement"
        )
    return str(network)


def parse_ports(value: str) -> tuple[int, ...]:
    """Parse a comma-separated TCP port list into sorted unique ports."""
    ports: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            raise ValueError("ports must be a comma-separated list of integers")
        try:
            port = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid port {item!r}") from exc
        if port < 1 or port > 65535:
            raise ValueError(f"port {port} is outside the valid range 1-65535")
        ports.add(port)
    if not ports:
        raise ValueError("at least one port is required")
    return tuple(sorted(ports))


def rule_name(tool: str, ports: tuple[int, ...]) -> str:
    """Return the conventional npa allow rule name for a tool and port set."""
    normalized_tool = re.sub(r"[^a-z0-9-]+", "-", tool.lower()).strip("-") or "manual"
    return f"allow-npa-{normalized_tool}-{'-'.join(str(port) for port in ports)}"


NPA_INGRESS_RULE_PREFIX = "allow-npa-"


def remove_npa_ingress_rules(
    security_group_ids: tuple[str, ...],
    *,
    on_status: Callable[[str], None] | None = None,
) -> list[str]:
    """Delete ingress rules created by :func:`ensure_ingress` (``allow-npa-*``)."""
    deleted: list[str] = []
    for group_id in security_group_ids:
        if not group_id:
            continue
        rules = _list_security_rules(group_id)
        for rule in rules:
            metadata = _metadata(rule)
            name = str(metadata.get("name", ""))
            rule_id = str(metadata.get("id", ""))
            if not rule_id or not name.startswith(NPA_INGRESS_RULE_PREFIX):
                continue
            try:
                nebius._run(["vpc", "security-rule", "delete", "--id", rule_id])
                deleted.append(rule_id)
                if on_status:
                    on_status(f"Removed ingress rule {name!r} ({rule_id}).")
            except NebiusError as exc:
                if on_status:
                    on_status(f"Could not remove ingress rule {rule_id}: {exc}")
    return deleted


def remove_ingress_for_instance(
    instance_id: str,
    *,
    on_status: Callable[[str], None] | None = None,
) -> list[str]:
    """Remove npa-managed ingress rules from the instance security groups."""
    instance = _get_instance(instance_id)
    return remove_npa_ingress_rules(
        _instance_security_group_ids(instance),
        on_status=on_status,
    )


def remove_npa_ingress_for_instance_ports(
    instance_id: str,
    *,
    ports: tuple[int, ...],
    on_status: Callable[[str], None] | None = None,
) -> list[str]:
    """Remove exact NPA-managed ingress rules for internal-only ports.

    Reused agent VMs may retain rules created by older NPA releases that
    exposed the backend port.  Delete only a dedicated ``allow-npa-*`` TCP
    rule whose destination ports are entirely within ``ports``.  An unmanaged,
    mixed-purpose, or all-port rule is not safe to rewrite automatically, so
    fail closed and let the operator resolve it explicitly. Nebius defines an
    empty ``destination_ports`` list as matching any port, rather than none:
    https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_security_rule
    """

    protected = {int(port) for port in ports}
    if not protected:
        raise NetworkIngressError("at least one internal-only port is required")
    instance = _get_instance(instance_id)
    deleted: list[str] = []
    for group_id in _instance_security_group_ids(instance):
        for rule in _list_security_rules(group_id):
            spec = rule.get("spec", {})
            ingress = spec.get("ingress")
            if not ingress or spec.get("access", "").upper() != "ALLOW":
                continue
            protocol = str(spec.get("protocol", "")).upper()
            raw_ports = ingress.get("destination_ports") or []
            destination_ports = {int(port) for port in raw_ports}
            exposes_protected = protocol in {"TCP", "ANY"} and (
                not destination_ports or bool(protected.intersection(destination_ports))
            )
            if not exposes_protected:
                continue
            metadata = _metadata(rule)
            name = str(metadata.get("name", ""))
            rule_id = str(metadata.get("id", ""))
            if (
                protocol != "TCP"
                or not destination_ports
                or not destination_ports.issubset(protected)
                or not name.startswith(NPA_INGRESS_RULE_PREFIX)
                or not rule_id
            ):
                raise NetworkIngressError(
                    f"security rule {name or rule_id or '<unnamed>'!r} exposes "
                    f"internal agent port(s) {sorted(protected)} and is not a "
                    "dedicated NPA-managed rule",
                    deleted=tuple(deleted),
                )
            try:
                nebius._run(["vpc", "security-rule", "delete", "--id", rule_id])
            except NebiusError as exc:
                raise NetworkIngressError(
                    f"Could not remove internal-port ingress rule {rule_id}: {exc}",
                    deleted=tuple(deleted),
                ) from exc
            deleted.append(rule_id)
            if on_status:
                on_status(f"Removed legacy internal-port ingress rule {name!r}.")
    return deleted


def remove_exact_npa_ingress_for_instance(
    instance_id: str,
    *,
    ports: tuple[int, ...],
    source: str,
    tool: str,
    protocol: str = "TCP",
    on_status: Callable[[str], None] | None = None,
) -> list[str]:
    """Remove only the exact NPA-managed rule created for one service contract."""

    normalized_protocol = _normalize_protocol(protocol)
    expected_ports = {int(port) for port in ports}
    expected_name = rule_name(tool, tuple(sorted(expected_ports)))
    instance = _get_instance(instance_id)
    deleted: list[str] = []
    for group_id in _instance_security_group_ids(instance):
        for rule in _list_security_rules(group_id):
            metadata = _metadata(rule)
            spec = rule.get("spec", {})
            ingress = spec.get("ingress") or {}
            rule_id = str(metadata.get("id", ""))
            if (
                rule_id
                and metadata.get("name") == expected_name
                and str(spec.get("access", "")).upper() == "ALLOW"
                and str(spec.get("protocol", "")).upper() == normalized_protocol
                and set(ingress.get("source_cidrs") or []) == {source}
                and {int(port) for port in ingress.get("destination_ports") or []}
                == expected_ports
            ):
                try:
                    nebius._run(["vpc", "security-rule", "delete", "--id", rule_id])
                except NebiusError as exc:
                    raise NetworkIngressError(
                        f"Could not remove ingress rule {rule_id}: {exc}"
                    ) from exc
                deleted.append(rule_id)
                if on_status:
                    on_status(f"Removed ingress rule {expected_name!r} ({rule_id}).")
    return deleted


def ensure_ingress(
    *,
    vm_id: str | None = None,
    ip: str | None = None,
    project_id: str | None = None,
    ports: tuple[int, ...],
    source: str = "",
    allow_world_open: bool = False,
    tool: str = "manual",
    protocol: str = "TCP",
) -> EnsureIngressResult:
    """Ensure TCP or UDP ingress from ``source`` to ``ports`` on the target VM groups."""
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_ingress")
    normalized_protocol = _normalize_protocol(protocol)
    source = _validate_ingress_source(source, allow_world_open=allow_world_open)
    if bool(vm_id) == bool(ip and project_id):
        raise NetworkIngressError("pass exactly one of --vm or (--ip and --project)")
    if ip and not project_id:
        raise NetworkIngressError("--ip requires --project")
    if project_id and not ip and not vm_id:
        raise NetworkIngressError("--project requires --ip unless --vm is used")

    instance = (
        _get_instance(vm_id)
        if vm_id
        else _find_instance_by_ip(ip or "", project_id or "")
    )
    instance_id = _metadata(instance).get("id", "")
    parent_id = _metadata(instance).get("parent_id", project_id or "")
    public_ip = _instance_public_ip(instance)
    security_group_ids = _instance_security_group_ids(instance)
    if not security_group_ids:
        raise NetworkIngressError(
            f"VM {instance_id or vm_id or ip} has no security group references"
        )

    group_contexts: list[_SecurityGroupContext] = []
    covered_ports: set[int] = set()
    for security_group_id in security_group_ids:
        security_group = _get_security_group(security_group_id)
        rules = tuple(_list_security_rules(security_group_id))
        group_covered = frozenset(
            _covered_ports(
                rules,
                requested_ports=ports,
                source=source,
                protocol=normalized_protocol,
            )
        )
        covered_ports.update(group_covered)
        group_contexts.append(
            _SecurityGroupContext(
                security_group=security_group,
                rules=rules,
                covered_ports=group_covered,
                warnings=tuple(
                    _name_collision_warnings(
                        rules,
                        desired_name=rule_name(tool, ports),
                        ports=ports,
                        source=source,
                        protocol=normalized_protocol,
                    )
                ),
            )
        )

    missing_ports = tuple(port for port in ports if port not in covered_ports)
    group_results = _build_group_results(
        group_contexts=group_contexts,
        missing_ports=missing_ports,
        source=source,
        tool=tool,
        protocol=normalized_protocol,
    )

    return EnsureIngressResult(
        instance_id=instance_id,
        project_id=parent_id,
        public_ip=public_ip,
        ports=ports,
        source=source,
        tool=tool,
        security_groups=tuple(group_results),
    )


def resolve_instance_network_context(instance_id: str) -> InstanceNetworkContext:
    """Resolve project, public IP, and attached security groups for an instance."""
    instance = _get_instance(instance_id)
    metadata = _metadata(instance)
    resolved_id = metadata.get("id", instance_id)
    project_id = metadata.get("parent_id", "")
    public_ip = _instance_public_ip(instance)
    security_group_ids = _instance_security_group_ids(instance)

    if not project_id:
        raise NetworkIngressError(f"VM {resolved_id} has no project reference")
    if not public_ip:
        raise NetworkIngressError(f"VM {resolved_id} has no public IP address")
    if not security_group_ids:
        raise NetworkIngressError(f"VM {resolved_id} has no security group references")

    return InstanceNetworkContext(
        instance_id=resolved_id,
        project_id=project_id,
        public_ip=public_ip,
        security_group_ids=security_group_ids,
    )


def _get_instance(vm_id: str | None) -> dict[str, Any]:
    if not vm_id:
        raise NetworkIngressError("VM ID is required")
    try:
        return nebius._run_json(["compute", "instance", "get", vm_id])
    except NebiusError as exc:
        raise NetworkIngressError(f"Could not fetch VM {vm_id}: {exc}") from exc


def _find_instance_by_ip(ip: str, project_id: str) -> dict[str, Any]:
    target = _strip_cidr(ip)
    try:
        data = nebius._run_json(
            [
                "compute",
                "instance",
                "list",
                "--parent-id",
                project_id,
                "--all",
            ]
        )
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not list VMs in project {project_id}: {exc}"
        ) from exc

    for item in data.get("items", []):
        for iface in item.get("status", {}).get("network_interfaces", []) or []:
            public_ip = iface.get("public_ip_address", {}).get("address", "")
            if _strip_cidr(public_ip) == target:
                return item
    raise NetworkIngressError(
        f"No VM with public IP {target} found in project {project_id}"
    )


def _get_security_group(security_group_id: str) -> dict[str, Any]:
    try:
        return nebius._run_json(["vpc", "security-group", "get", security_group_id])
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not fetch security group {security_group_id}: {exc}"
        ) from exc


def _list_security_rules(security_group_id: str) -> list[dict[str, Any]]:
    try:
        data = nebius._run_json(
            [
                "vpc",
                "security-rule",
                "list",
                "--parent-id",
                security_group_id,
                "--all",
            ]
        )
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not list security rules for {security_group_id}: {exc}"
        ) from exc
    return list(data.get("items", []))


def _build_group_results(
    *,
    group_contexts: list[_SecurityGroupContext],
    missing_ports: tuple[int, ...],
    source: str,
    tool: str,
    protocol: str,
) -> list[SecurityGroupIngressResult]:
    if not group_contexts:
        return []

    results: list[SecurityGroupIngressResult] = []
    created_rule_id = ""
    created_rule_name = ""
    if missing_ports:
        target = group_contexts[0]
        created_rule_id, created_rule_name = _create_group_ingress(
            security_group=target.security_group,
            missing_ports=missing_ports,
            source=source,
            tool=tool,
            protocol=protocol,
        )

    for index, context in enumerate(group_contexts):
        security_group = context.security_group
        results.append(
            SecurityGroupIngressResult(
                security_group_id=_metadata(security_group).get("id", ""),
                security_group_name=_metadata(security_group).get("name", ""),
                network_id=security_group.get("spec", {}).get("network_id", ""),
                covered_ports=tuple(sorted(context.covered_ports)),
                missing_ports=missing_ports if index == 0 else (),
                created_rule_id=created_rule_id if index == 0 else "",
                created_rule_name=created_rule_name if index == 0 else "",
                warnings=context.warnings,
            )
        )
    return results


def _create_group_ingress(
    *,
    security_group: dict[str, Any],
    missing_ports: tuple[int, ...],
    source: str,
    tool: str,
    protocol: str,
) -> tuple[str, str]:
    security_group_id = _metadata(security_group).get("id", "")
    create_name = rule_name(tool, missing_ports)
    try:
        created = nebius._run_json(
            [
                "vpc",
                "security-rule",
                "create",
                "--parent-id",
                security_group_id,
                "--name",
                create_name,
                "--access",
                "allow",
                "--protocol",
                protocol.lower(),
                "--type",
                "stateful",
                "--priority",
                "500",
                "--ingress-source-cidrs",
                source,
                *[
                    item
                    for port in missing_ports
                    for item in ("--ingress-destination-ports", str(port))
                ],
            ]
        )
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not create ingress rule on {security_group_id}: {exc}"
        ) from exc

    return (
        _metadata(created).get("id", ""),
        _metadata(created).get("name", create_name),
    )


def _covered_ports(
    rules: Sequence[dict[str, Any]],
    *,
    requested_ports: tuple[int, ...],
    source: str,
    protocol: str = "TCP",
) -> set[int]:
    requested = set(requested_ports)
    covered: set[int] = set()
    for rule in rules:
        spec = rule.get("spec", {})
        ingress = spec.get("ingress")
        if not ingress:
            continue
        if spec.get("access", "").upper() != "ALLOW":
            continue
        if spec.get("protocol", "").upper() != protocol:
            continue
        if source not in (ingress.get("source_cidrs") or []):
            continue
        destination_ports = {
            int(port) for port in ingress.get("destination_ports") or []
        }
        covered.update(
            requested
            if not destination_ports
            else requested.intersection(destination_ports)
        )
    return covered


def _name_collision_warnings(
    rules: Sequence[dict[str, Any]],
    *,
    desired_name: str,
    ports: tuple[int, ...],
    source: str,
    protocol: str = "TCP",
) -> list[str]:
    warnings: list[str] = []
    requested = set(ports)
    for rule in rules:
        metadata = _metadata(rule)
        if metadata.get("name") != desired_name:
            continue
        spec = rule.get("spec", {})
        ingress = spec.get("ingress") or {}
        destination_ports = {
            int(port) for port in ingress.get("destination_ports") or []
        }
        source_cidrs = ingress.get("source_cidrs") or []
        matches = (
            spec.get("access", "").upper() == "ALLOW"
            and spec.get("protocol", "").upper() == protocol
            and source in source_cidrs
            and (not destination_ports or requested.issubset(destination_ports))
        )
        if not matches:
            warnings.append(
                f"security rule {metadata.get('id', '<unknown>')} already uses name "
                f"{desired_name!r} but does not match requested ingress spec"
            )
    return warnings


def _instance_security_group_ids(instance: dict[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for iface in instance.get("spec", {}).get("network_interfaces", []) or []:
        for group in iface.get("security_groups", []) or []:
            group_id = group.get("id", "")
            if group_id and group_id not in ids:
                ids.append(group_id)
    return tuple(ids)


def _instance_public_ip(instance: dict[str, Any]) -> str:
    for iface in instance.get("status", {}).get("network_interfaces", []) or []:
        address = iface.get("public_ip_address", {}).get("address", "")
        if address:
            return address
    return ""


def _metadata(resource: dict[str, Any]) -> dict[str, Any]:
    return resource.get("metadata", {}) or {}


def _strip_cidr(value: str) -> str:
    return value.split("/", 1)[0]


def _normalize_protocol(value: str) -> str:
    protocol = str(value or "").strip().upper()
    if protocol not in ("TCP", "UDP"):
        raise NetworkIngressError("protocol must be TCP or UDP")
    return protocol
