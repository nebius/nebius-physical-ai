"""npa.network - infrastructure network helpers."""

from __future__ import annotations

from collections.abc import Iterable

from npa.clients.network import ensure_ingress as _ensure_ingress
from npa.clients.network import parse_ports


def ensure_ingress(
    *,
    ports: str | Iterable[int],
    vm: str | None = None,
    ip: str | None = None,
    project: str | None = None,
    source: str = "",
    allow_world_open: bool = False,
    tool: str = "manual",
):
    """Ensure TCP ingress to a VM security group."""
    if isinstance(ports, str):
        parsed_ports = list(parse_ports(ports))
    else:
        parsed_ports = [int(port) for port in ports]
    from npa.cli.ingress import validate_ingress_source

    return _ensure_ingress(
        vm_id=vm,
        ip=ip,
        project_id=project,
        ports=parsed_ports,
        source=validate_ingress_source(source, allow_world_open=allow_world_open),
        allow_world_open=allow_world_open,
        tool=tool,
    )


__all__ = ["ensure_ingress"]
