"""Public-network state and URL contract for NPA agent deployments."""

from __future__ import annotations

from typing import Any, Callable

from npa.clients.network import NetworkIngressError
from npa.cli.agent_deployment import (
    AgentConfig as AgentConfig,
    build_agent_urls as build_agent_urls,
    is_routable_public_ip,
    record_customer_url as record_customer_url,
    record_public_https as record_public_https,
    record_tls_verify as record_tls_verify,
)

DEFAULT_AGENT_PORT = 8088


def resolve_record_public_ip(
    record: dict[str, Any],
    *,
    resolver: Callable[[str], Any],
) -> str:
    """Resolve provider state for an existing agent and return its public IP."""

    instance_id = str(record.get("instance_id", "")).strip()
    if instance_id:
        context = resolver(instance_id)
        public_ip = str(context.public_ip or "").strip().split("/", 1)[0]
    else:
        public_ip = str(record.get("public_ip", "")).strip()
    if not is_routable_public_ip(public_ip):
        raise NetworkIngressError("agent VM does not have a routable public IP")
    return public_ip
