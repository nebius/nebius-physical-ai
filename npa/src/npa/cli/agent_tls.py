"""Authenticate agent HTTPS with the certificate obtained through verified SSH."""

from __future__ import annotations

import ipaddress
import ssl
from pathlib import Path
from typing import Any, Mapping

from npa.clients.config import SSHConfig
from npa.clients.network import resolve_instance_network_context
from npa.clients.ssh import SSHClient


def trusted_agent_tls_context(ssh: SSHClient) -> ssl.SSLContext:
    """Trust the installed agent certificate without learning trust from HTTPS.

    Args:
        ssh: Client that rejects unknown or changed SSH host keys.
    Returns:
        A certificate- and hostname-verifying context with only this trust anchor.
    Raises:
        ValueError: The installed certificate is unavailable or malformed.
        SSHError: The SSH connection or host authentication fails.
    """
    code, certificate, _stderr = ssh.run(
        "openssl x509 -in /etc/nginx/ssl/npa-agent.crt -outform PEM"
    )
    if code != 0:
        raise ValueError(
            "Could not read the agent TLS certificate through verified SSH"
        )
    certificate = certificate.strip()
    if (
        not certificate.startswith("-----BEGIN CERTIFICATE-----\n")
        or not certificate.endswith("\n-----END CERTIFICATE-----")
        or certificate.count("-----BEGIN CERTIFICATE-----") != 1
    ):
        raise ValueError("Verified SSH returned an invalid agent TLS certificate")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    try:
        context.load_verify_locations(cadata=certificate)
    except ssl.SSLError as exc:
        raise ValueError(
            "Verified SSH returned an invalid agent TLS certificate"
        ) from exc
    return context


def record_agent_tls_context(record: Mapping[str, Any]) -> ssl.SSLContext:
    """Resolve the recorded provider instance before authenticating its HTTPS.

    Args:
        record: Saved agent provider identity, public address and SSH settings.
    Returns:
        Trust obtained from the selected instance's verified SSH connection.
    Raises:
        ValueError: Required SSH identity is absent or the provider address changed.
        SSHError: The selected instance cannot be authenticated through SSH.
        NebiusError: Provider identity lookup fails.
    """
    instance_id = str(record.get("instance_id") or "").strip()
    key_path = str(record.get("ssh_key_path") or "").strip()
    if not instance_id or not key_path or not Path(key_path).expanduser().is_file():
        raise ValueError("Agent HTTPS requires a recorded instance and usable SSH key")
    network = resolve_instance_network_context(instance_id)
    host = str(ipaddress.ip_address(str(network.public_ip).split("/", 1)[0]))
    saved_host = str(record.get("public_ip") or "").split("/", 1)[0]
    if saved_host != host:
        raise ValueError("Agent address differs from provider state; refresh bootstrap")
    ssh = SSHClient(
        SSHConfig(
            host=host,
            user=str(record.get("ssh_user") or "ubuntu"),
            key_path=key_path,
        )
    )
    return trusted_agent_tls_context(ssh)
