"""Network ingress helpers backed by the Nebius CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from npa.clients import nebius
from npa.clients.nebius import NebiusError


class NetworkIngressError(Exception):
    """Raised when ingress cannot be resolved or changed."""


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


def ensure_ingress(
    *,
    vm_id: str | None = None,
    ip: str | None = None,
    project_id: str | None = None,
    ports: tuple[int, ...],
    source: str = "0.0.0.0/0",
    tool: str = "manual",
) -> EnsureIngressResult:
    """Ensure TCP ingress from ``source`` to ``ports`` on the target VM security groups."""
    if bool(vm_id) == bool(ip and project_id):
        raise NetworkIngressError("pass exactly one of --vm or (--ip and --project)")
    if ip and not project_id:
        raise NetworkIngressError("--ip requires --project")
    if project_id and not ip and not vm_id:
        raise NetworkIngressError("--project requires --ip unless --vm is used")

    instance = _get_instance(vm_id) if vm_id else _find_instance_by_ip(ip or "", project_id or "")
    instance_id = _metadata(instance).get("id", "")
    parent_id = _metadata(instance).get("parent_id", project_id or "")
    public_ip = _instance_public_ip(instance)
    security_group_ids = _instance_security_group_ids(instance)
    if not security_group_ids:
        raise NetworkIngressError(f"VM {instance_id or vm_id or ip} has no security group references")

    group_results: list[SecurityGroupIngressResult] = []
    for security_group_id in security_group_ids:
        security_group = _get_security_group(security_group_id)
        rules = _list_security_rules(security_group_id)
        group_results.append(
            _ensure_group_ingress(
                security_group=security_group,
                rules=rules,
                ports=ports,
                source=source,
                tool=tool,
            )
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
        data = nebius._run_json([
            "compute",
            "instance",
            "list",
            "--parent-id",
            project_id,
            "--all",
        ])
    except NebiusError as exc:
        raise NetworkIngressError(f"Could not list VMs in project {project_id}: {exc}") from exc

    for item in data.get("items", []):
        for iface in item.get("status", {}).get("network_interfaces", []) or []:
            public_ip = iface.get("public_ip_address", {}).get("address", "")
            if _strip_cidr(public_ip) == target:
                return item
    raise NetworkIngressError(f"No VM with public IP {target} found in project {project_id}")


def _get_security_group(security_group_id: str) -> dict[str, Any]:
    try:
        return nebius._run_json(["vpc", "security-group", "get", security_group_id])
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not fetch security group {security_group_id}: {exc}"
        ) from exc


def _list_security_rules(security_group_id: str) -> list[dict[str, Any]]:
    try:
        data = nebius._run_json([
            "vpc",
            "security-rule",
            "list",
            "--parent-id",
            security_group_id,
            "--all",
        ])
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not list security rules for {security_group_id}: {exc}"
        ) from exc
    return list(data.get("items", []))


def _ensure_group_ingress(
    *,
    security_group: dict[str, Any],
    rules: list[dict[str, Any]],
    ports: tuple[int, ...],
    source: str,
    tool: str,
) -> SecurityGroupIngressResult:
    security_group_id = _metadata(security_group).get("id", "")
    security_group_name = _metadata(security_group).get("name", "")
    network_id = security_group.get("spec", {}).get("network_id", "")
    desired_name = rule_name(tool, ports)
    covered = _covered_ports(rules, requested_ports=ports, source=source)
    missing = tuple(port for port in ports if port not in covered)
    warnings = tuple(_name_collision_warnings(rules, desired_name=desired_name, ports=ports, source=source))

    if not missing:
        return SecurityGroupIngressResult(
            security_group_id=security_group_id,
            security_group_name=security_group_name,
            network_id=network_id,
            covered_ports=tuple(sorted(covered)),
            missing_ports=(),
            warnings=warnings,
        )

    create_name = rule_name(tool, missing)
    try:
        created = nebius._run_json([
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
            "tcp",
            "--type",
            "stateful",
            "--priority",
            "500",
            "--ingress-source-cidrs",
            source,
            *[
                item
                for port in missing
                for item in ("--ingress-destination-ports", str(port))
            ],
        ])
    except NebiusError as exc:
        raise NetworkIngressError(
            f"Could not create ingress rule on {security_group_id}: {exc}"
        ) from exc

    return SecurityGroupIngressResult(
        security_group_id=security_group_id,
        security_group_name=security_group_name,
        network_id=network_id,
        covered_ports=tuple(sorted(covered)),
        missing_ports=missing,
        created_rule_id=_metadata(created).get("id", ""),
        created_rule_name=_metadata(created).get("name", create_name),
        warnings=warnings,
    )


def _covered_ports(
    rules: list[dict[str, Any]],
    *,
    requested_ports: tuple[int, ...],
    source: str,
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
        if spec.get("protocol", "").upper() != "TCP":
            continue
        if source not in (ingress.get("source_cidrs") or []):
            continue
        destination_ports = {int(port) for port in ingress.get("destination_ports") or []}
        covered.update(requested.intersection(destination_ports))
    return covered


def _name_collision_warnings(
    rules: list[dict[str, Any]],
    *,
    desired_name: str,
    ports: tuple[int, ...],
    source: str,
) -> list[str]:
    warnings: list[str] = []
    requested = set(ports)
    for rule in rules:
        metadata = _metadata(rule)
        if metadata.get("name") != desired_name:
            continue
        spec = rule.get("spec", {})
        ingress = spec.get("ingress") or {}
        destination_ports = {int(port) for port in ingress.get("destination_ports") or []}
        source_cidrs = ingress.get("source_cidrs") or []
        matches = (
            spec.get("access", "").upper() == "ALLOW"
            and spec.get("protocol", "").upper() == "TCP"
            and source in source_cidrs
            and requested.issubset(destination_ports)
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
