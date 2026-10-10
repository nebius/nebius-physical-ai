"""Verify the actual assigned LB with TLS and no personal client authentication."""

import http.client
import socket
import ssl
from urllib.parse import urlsplit

from .errors import BackendError


def verify_endpoint(request, address):
    """Probe the assigned IP using the configured HTTPS hostname and trusted CA.

    Args:
        request: Operator-owned HTTPS origin and optional installation CA.
        address: Exact provider-verified LB address, never a client override.
    Returns:
        Credential-free health, authentication, and scheduler privacy evidence.
    Raises:
        BackendError: A reached endpoint violates the expected API boundary.
        OSError: The operator's network cannot connect; setup preserves its receipt.
    """
    host = urlsplit(request.endpoint).hostname
    context = ssl.create_default_context(cafile=request.ca_file)
    statuses = {}
    for path, expected in (
        ("/health", 200),
        ("/v1/me", 401),
        ("/api/health", 404),
        ("/api/status", 404),
        ("/status", 404),
    ):
        statuses[path] = _status(context, host, address, path)
        if statuses[path] != expected:
            raise BackendError(
                "HTTPS endpoint failed the Workbench authentication boundary"
            )
    try:
        resolved = {entry[4][0] for entry in socket.getaddrinfo(host, 443)}
    except socket.gaierror:
        resolved = set()
    return {"boundary_statuses": statuses, "dns_points_to_lb": address in resolved}


def _status(context, host, address, path):
    connection = http.client.HTTPConnection(host, port=443, timeout=10)
    raw = socket.create_connection((address, 443), timeout=10)
    try:
        connection.sock = context.wrap_socket(raw, server_hostname=host)
        connection.request("GET", path, headers={"Host": host})
        return connection.getresponse().status
    finally:
        raw.close()
        connection.close()
