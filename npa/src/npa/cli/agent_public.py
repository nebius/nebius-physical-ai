"""Public-network state and URL contract for NPA agent deployments."""

from __future__ import annotations

import ssl
from typing import Any, Callable

from npa.clients.nebius import NebiusError
from npa.clients.network import NetworkIngressError
from npa.clients.ssh import SSHError
from npa.cli.agent_deployment import (
    AgentConfig as AgentConfig,
    build_agent_urls as build_agent_urls,
    is_routable_public_ip,
    record_customer_url as record_customer_url,
    record_public_https as record_public_https,
    record_tls_verify as record_tls_verify,
)

DEFAULT_AGENT_PORT = 8088


def probe_agent_health(
    record: dict[str, Any],
    agent_url: str,
    sim_viz_url: str,
    *,
    user: str,
    password: str,
    tls_verifier: Callable[[dict[str, Any]], bool | ssl.SSLContext],
    health_check: Callable[..., tuple[bool, int]],
    auth_check: Callable[..., tuple[bool, int]],
) -> dict[str, Any]:
    """Authenticate a saved agent before probing health or disclosing endpoints.

    Args:
        record: Saved provider and SSH identity for the selected agent.
        agent_url: Agent UI endpoint.
        sim_viz_url: Agent visualization endpoint.
        user: Basic Auth username used only after trust succeeds.
        password: Basic Auth password used only after trust succeeds.
        tls_verifier: Resolve trust through the saved provider and SSH identity.
        health_check: Probe an authenticated endpoint with the verified context.
        auth_check: Check rejection of unauthenticated access with that context.
    Returns:
        Health and disclosure fields, or sanitized trust failure fields without
        making any HTTP request.
    """
    result: dict[str, Any] = {
        "health": False,
        "basic_auth_enforced": False,
        "unauthenticated_ui_status_code": 0,
        "endpoint_disclosure_allowed": False,
        "ui_status_code": 0,
        "rerun_status_code": 0,
    }
    try:
        tls_verify = tls_verifier(record)
    except (ValueError, SSHError, NetworkIngressError, NebiusError, OSError) as exc:
        result.update(
            error_category=type(exc).__name__,
            detail="Agent HTTPS identity could not be verified; check provider access "
            "and the recorded SSH key before retrying.",
        )
        return result
    ui_ok, ui_code = health_check(
        agent_url, user=user, password=password, verify=tls_verify
    )
    basic_auth_enforced, unauthenticated_ui_code = auth_check(
        agent_url, verify=tls_verify
    )
    rerun_ok, rerun_code = health_check(
        sim_viz_url, user=user, password=password, verify=tls_verify
    )
    result.update(
        health=bool(ui_ok and rerun_ok and basic_auth_enforced),
        basic_auth_enforced=basic_auth_enforced,
        unauthenticated_ui_status_code=unauthenticated_ui_code,
        endpoint_disclosure_allowed=bool(ui_ok and basic_auth_enforced),
        ui_status_code=ui_code,
        rerun_status_code=rerun_code,
    )
    return result


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
