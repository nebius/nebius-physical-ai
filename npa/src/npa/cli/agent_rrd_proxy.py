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
from pathlib import Path
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
    allowed, _fetch_url, _host = resolve_rrd_proxy_target(uri, resolve=resolve)
    return allowed


def resolve_rrd_proxy_target(
    uri: str, *, resolve: bool = True
) -> tuple[bool, str, str]:
    """Validate ``uri`` and return ``(allowed, fetch_url, host_header)``.

    When allowed, ``fetch_url`` points at a vetted IP (DNS resolved once) so the
    subsequent HTTP client does not re-resolve the hostname (DNS-rebinding
    TOCTOU). Callers should send ``Host: host_header`` when fetching.
    """
    try:
        parsed = urlparse(str(uri or "").strip())
    except Exception:  # noqa: BLE001
        return False, "", ""
    if parsed.scheme not in {"http", "https"}:
        return False, "", ""
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False, "", ""
    if host in _BLOCKED_HOSTNAMES or host.endswith(".internal"):
        return False, "", ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Bare IP literal — no DNS needed.
    try:
        ipaddress.ip_address(host)
        if not is_publicly_routable_ip(host):
            return False, "", ""
        return True, str(uri).strip(), host
    except ValueError:
        pass
    if not resolve:
        return False, "", ""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False, "", ""
    if not infos:
        return False, "", ""
    vetted_ip = ""
    for info in infos:
        addr = str(info[4][0])
        if not is_publicly_routable_ip(addr):
            return False, "", ""
        if not vetted_ip:
            vetted_ip = addr
    if not vetted_ip:
        return False, "", ""
    # Rewrite netloc to the vetted IP; preserve path/query/fragment.
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    # Bracket IPv6 literals for URL netloc.
    ip_netloc = f"[{vetted_ip}]" if ":" in vetted_ip else vetted_ip
    if parsed.port:
        ip_netloc = f"{ip_netloc}:{parsed.port}"
    fetch_url = parsed._replace(netloc=f"{userinfo}{ip_netloc}").geturl()
    return True, fetch_url, host


def file_uri_path_allowed(uri: str, *, allowed_paths: list[str] | tuple[str, ...] | None = None) -> bool:
    """Return True if a ``file://`` URI resolves inside one of ``allowed_paths``.

    ``allowed_paths`` should be absolute directories and/or exact files (e.g.
    RECORDINGS_DIR and RRD_PATH). Symlinks are resolved before the check.
    """
    raw = str(uri or "").strip()
    if not raw.startswith("file://"):
        return False
    try:
        path = Path(raw[len("file://") :]).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return False
    if not path.is_file():
        return False
    for allowed in allowed_paths or ():
        try:
            base = Path(allowed).expanduser().resolve()
        except Exception:  # noqa: BLE001
            continue
        if base.is_file():
            if path == base:
                return True
            continue
        try:
            path.relative_to(base)
            return True
        except ValueError:
            continue
    return False
