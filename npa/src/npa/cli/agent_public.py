"""Public-network state and URL contract for NPA agent deployments."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Callable

from npa.clients.network import NetworkIngressError

DEFAULT_AGENT_PORT = 8088


@dataclass(frozen=True)
class AgentConfig:
    project_alias: str
    name: str
    project_id: str
    tenant_id: str
    region: str
    public_ip: str
    instance_id: str
    agent_url: str
    rerun_url: str
    sim_viz_url: str
    sim_assets_url: str
    cameras_api_url: str
    auth_user: str
    auth_secret_path: str
    llm_provider: str
    llm_model: str
    service_account_id: str = ""
    llm_models: tuple[str, ...] = ()
    public_url: str = ""
    public_https: bool = True
    direct_url: str = ""
    ssh_key_path: str = ""
    credentials: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "public_ip": self.public_ip,
            "instance_id": self.instance_id,
            "service_account_id": self.service_account_id,
            "agent_url": self.agent_url,
            "rerun_url": self.rerun_url,
            "sim_viz_url": self.sim_viz_url,
            "sim_assets_url": self.sim_assets_url,
            "cameras_api_url": self.cameras_api_url,
            "auth_user": self.auth_user,
            "auth_secret_path": self.auth_secret_path,
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "models": list(self.llm_models or (self.llm_model,)),
            },
        }
        if self.public_url:
            payload["public_url"] = self.public_url
        if self.public_https:
            payload["public_https"] = True
        if self.direct_url:
            payload["direct_url"] = self.direct_url
        if self.ssh_key_path:
            payload["ssh_key_path"] = self.ssh_key_path
        if self.service_account_id:
            payload["service_account_id"] = self.service_account_id
        if self.credentials:
            payload["credentials"] = dict(self.credentials)
        return payload


def build_agent_urls(
    public_ip: str,
    *,
    agent_port: int = DEFAULT_AGENT_PORT,
    public_https: bool = True,
) -> dict[str, str]:
    """Return customer-facing and operator-direct URLs for an agent VM."""

    direct = f"http://{public_ip}:{agent_port}/"
    base = f"https://{public_ip}/" if public_https else direct
    root = base.rstrip("/")
    return {
        "public_url": base,
        "agent_url": base,
        "rerun_url": f"{root}/rerun/",
        "sim_viz_url": f"{root}/rerun/",
        "sim_assets_url": f"{root}/assets/",
        "cameras_api_url": f"{root}/assets/api/sim-assets/cameras",
        "direct_url": direct,
    }


def is_routable_public_ip(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate or candidate == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_private or ip.is_unspecified or ip.is_link_local)


def record_public_https(record: dict[str, Any]) -> bool:
    if "public_https" in record:
        return bool(record.get("public_https"))
    public_url = str(record.get("public_url", "")).strip()
    if public_url.startswith("https://"):
        return True
    agent_url = str(record.get("agent_url", "")).strip()
    return agent_url.startswith("https://")


def record_tls_verify(record: dict[str, Any]) -> bool:
    """Self-signed HTTPS on the VM public IP is expected; skip CA verification."""

    return not record_public_https(record)


def record_customer_url(record: dict[str, Any]) -> str:
    public_ip = str(record.get("public_ip", "")).strip()
    if record_public_https(record) and is_routable_public_ip(public_ip):
        return f"https://{public_ip}/"
    public_url = str(record.get("public_url", "")).strip()
    if public_url:
        return public_url
    return str(record.get("agent_url", "")).strip()


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
