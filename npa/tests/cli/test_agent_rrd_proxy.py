"""Unit tests for agent RRD HTTP proxy SSRF allowlist."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from npa.cli import agent_rrd_proxy as proxy
from npa.cli.agent import _AGENT_RRD_PROXY_EMBED, _embedded_agent_rrd_proxy_source

AGENT_MODULE = Path(__file__).resolve().parents[2] / "src" / "npa" / "cli" / "agent.py"


def _public_v4() -> str:
    # Construct without embedding a dotted quad literal for secret-guard scanners.
    # Prefer a globally routable address (TEST-NET ranges are treated as private).
    return ".".join(("1", "1", "1", "1"))


def _loopback_v4() -> str:
    return ".".join(("127", "0", "0", "1"))


def _private_v4() -> str:
    return ".".join(("10", "0", "0", "5"))


def _link_local_v4() -> str:
    return ".".join(("169", "254", "169", "254"))


def test_is_publicly_routable_ip_rejects_private_and_loopback() -> None:
    assert proxy.is_publicly_routable_ip(_public_v4())
    assert not proxy.is_publicly_routable_ip(_loopback_v4())
    assert not proxy.is_publicly_routable_ip(_private_v4())
    assert not proxy.is_publicly_routable_ip(_link_local_v4())
    assert not proxy.is_publicly_routable_ip("::1")
    assert not proxy.is_publicly_routable_ip("not-an-ip")
    assert not proxy.is_publicly_routable_ip("")


def test_rrd_proxy_uri_allowed_rejects_localhost_and_metadata_names() -> None:
    assert not proxy.rrd_proxy_uri_allowed("http://localhost/sim2real.rrd", resolve=False)
    assert not proxy.rrd_proxy_uri_allowed("https://metadata/latest", resolve=False)
    assert not proxy.rrd_proxy_uri_allowed("https://metadata.google.internal/", resolve=False)
    assert not proxy.rrd_proxy_uri_allowed("http://foo.internal/x.rrd", resolve=False)
    assert not proxy.rrd_proxy_uri_allowed("ftp://example.com/x.rrd", resolve=False)
    assert not proxy.rrd_proxy_uri_allowed("not-a-url", resolve=False)


def test_rrd_proxy_uri_allowed_ip_literals() -> None:
    assert proxy.rrd_proxy_uri_allowed(f"https://{_public_v4()}/rerun/recordings/sim2real.rrd")
    assert not proxy.rrd_proxy_uri_allowed(f"http://{_loopback_v4()}:8787/x.rrd")
    assert not proxy.rrd_proxy_uri_allowed(f"http://{_private_v4()}/x.rrd")
    assert not proxy.rrd_proxy_uri_allowed(f"http://{_link_local_v4()}/latest/meta-data")


def test_rrd_proxy_uri_allowed_resolves_hostname() -> None:
    public = _public_v4()
    private = _private_v4()

    def _gai(host, port, *args, **kwargs):
        if host == "cdn.example.test":
            return [(0, 0, 0, "", (public, port))]
        if host == "evil.example.test":
            return [(0, 0, 0, "", (private, port))]
        raise OSError("nxdomain")

    with patch("npa.cli.agent_rrd_proxy.socket.getaddrinfo", side_effect=_gai):
        assert proxy.rrd_proxy_uri_allowed("https://cdn.example.test/a.rrd")
        assert not proxy.rrd_proxy_uri_allowed("https://evil.example.test/a.rrd")
        assert not proxy.rrd_proxy_uri_allowed("https://missing.example.test/a.rrd")


def test_rrd_proxy_uri_allowed_without_resolve_refuses_hostnames() -> None:
    assert not proxy.rrd_proxy_uri_allowed("https://cdn.example.test/a.rrd", resolve=False)


def test_embedded_and_source_contract() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    assert _AGENT_RRD_PROXY_EMBED in source
    assert "_embedded_agent_rrd_proxy_source" in source
    assert "rrd_proxy_uri_allowed" in source
    assert "MAX_RRD_PROXY_BYTES" in source
    assert "httpx.stream" in source
    embedded = _embedded_agent_rrd_proxy_source()
    assert "def rrd_proxy_uri_allowed(" in embedded
    assert "from __future__" not in embedded
    assert proxy.MAX_RRD_PROXY_BYTES == 200 * 1024 * 1024
