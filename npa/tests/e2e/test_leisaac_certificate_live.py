"""Compare an existing agent's SSH-authenticated TLS pin with its public listener."""

import hashlib
import os
import secrets
import socket
import ssl

import pytest

from npa.cli.workbench.leisaac import (
    _agent_certificate_sha256,
    _agent_relay_context,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        not all(
            os.environ.get(name)
            for name in (
                "NPA_LEISAAC_CERTIFICATE_LIVE_PROJECT",
                "NPA_LEISAAC_CERTIFICATE_LIVE_AGENT",
            )
        ),
        reason="Requires a selected existing agent and its verified SSH host key",
    ),
]


def test_existing_agents_public_certificate_matches_trusted_ssh_pin():
    _, host, ssh, _, _ = _agent_relay_context(
        os.environ["NPA_LEISAAC_CERTIFICATE_LIVE_PROJECT"],
        os.environ["NPA_LEISAAC_CERTIFICATE_LIVE_AGENT"],
    )
    trusted_pin = _agent_certificate_sha256(ssh)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, 443), timeout=10) as raw:
        with context.wrap_socket(raw, server_hostname=host) as connection:
            certificate = connection.getpeercert(binary_form=True)
    assert certificate, "The public listener returned no certificate"
    assert secrets.compare_digest(
        hashlib.sha256(certificate).hexdigest(), trusted_pin
    ), "The public listener does not match the SSH-authenticated certificate"
