"""RRD HTTP(S) proxy allowlist for the agent backend (SSRF hardening).

Pure helpers — no network I/O except optional DNS resolution via
``socket.getaddrinfo``. Embedded into the agent-VM backend the same way as
``agent_visual_feedback`` / ``agent_routing``.

Trust model: ``rrd_uri`` is normally written only by this agent's own
load/submit flows on a single-tenant basic-auth operator VM. The allowlist
still refuses loopback/private/link-local/metadata targets so a widened
writer cannot turn the proxy into an SSRF oracle.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Cap proxied .rrd bodies so a large URI cannot memory-DoS the agent process.
MAX_RRD_PROXY_BYTES = 200 * 1024 * 1024

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "metadata.internal",
    }
)


def is_publicly_routable_ip(value: str) -> bool:
    """Return True iff ``value`` is a public unicast IP (v4 or v6)."""
    candidate = (value or "").strip()
    if not candidate:
        return False
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_unspecified
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    ):
        return False
    return True


def rrd_proxy_uri_allowed(uri: str, *, resolve: bool = True) -> bool:
    """Return True if ``uri`` is safe to fetch server-side for an .rrd proxy.

    Requires http(s), a hostname/IP that is not a blocked metadata/localhost
    name, and (when ``resolve``) that every DNS/address result is publicly
    routable — blocking loopback, RFC1918, link-local, and unique-local IPv6.
    """
    try:
        parsed = urlparse(str(uri or "").strip())
    except Exception:  # noqa: BLE001
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in _BLOCKED_HOSTNAMES or host.endswith(".internal"):
        return False
    # Bare IP literal — no DNS needed.
    try:
        ipaddress.ip_address(host)
        return is_publicly_routable_ip(host)
    except ValueError:
        pass
    if not resolve:
        # Hostname without resolution: refuse (cannot prove public routing).
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not infos:
        return False
    seen = False
    for info in infos:
        addr = str(info[4][0])
        seen = True
        if not is_publicly_routable_ip(addr):
            return False
    return seen
