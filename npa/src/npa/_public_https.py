"""Anonymous HTTPS downloads for pinned public artifacts, including bootstrap.

This module must stay stdlib-only and importable as a standalone file before
NPA or its dependencies are installed. Callers own artifact hash verification
and removal of partial files. Host policy belongs to each source, not the URL.
"""

from __future__ import annotations

import http.client
import ssl
import time
from typing import BinaryIO
from urllib.parse import urljoin, urlsplit


class PublicDownloadError(RuntimeError):
    """A public download failed; diagnostics never include server URL data."""


def _validate_url(url: str, allowed_hosts: frozenset[str]) -> tuple[str, str]:
    # urlsplit silently strips some whitespace/control characters. Reject them
    # before parsing, along with non-ASCII text (URLs must be percent encoded).
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise PublicDownloadError("public download URL is malformed")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or host not in allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.netloc.lower() not in (host, f"{host}:443")
            or parsed.fragment
        ):
            raise PublicDownloadError("public download URL is not permitted")
    except ValueError:
        raise PublicDownloadError("public download URL is malformed") from None
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return host, target


def _redirect_target(response, url, allowed_hosts, visited):
    location = response.getheader("Location")
    if not location:
        raise PublicDownloadError("public download redirect has no location")
    # Validate before urljoin too: it normalizes away controls.
    if any(ord(c) <= 32 or ord(c) >= 127 for c in location):
        raise PublicDownloadError("public download redirect is malformed")
    next_url = urljoin(url, location)
    host, target = _validate_url(next_url, allowed_hosts)
    # Preserve urllib's existing ten-redirect ceiling for these callers.
    if next_url in visited or len(visited) > 10:
        raise PublicDownloadError("public download redirect loop or too many redirects")
    visited.add(next_url)
    return next_url, host, target


def _download_hops(url, output, allowed_hosts, redirect_hosts):
    host, target = _validate_url(url, allowed_hosts)
    visited = {url}
    context = ssl.create_default_context()
    retries = 0
    while True:
        retry_delay = 0
        connection = http.client.HTTPSConnection(host, port=443, context=context)
        try:
            connection.request("GET", target)
            with connection.getresponse() as response:
                if response.status in (301, 302, 303, 307, 308):
                    url, host, target = _redirect_target(
                        response, url, allowed_hosts | redirect_hosts, visited
                    )
                elif response.status == 200:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                    return
                elif response.status in (429, 500, 502, 503, 504) and retries < 3:
                    retry_delay = 2 ** retries
                    retries += 1
                else:
                    raise PublicDownloadError(
                        "public download returned an unsuccessful HTTP status"
                    )
        finally:
            connection.close()
        if retry_delay:
            time.sleep(retry_delay)


def download_public_https(
    url: str,
    output: BinaryIO,
    *,
    allowed_hosts: frozenset[str],
    redirect_hosts: frozenset[str] = frozenset(),
) -> None:
    """Stream an anonymous public artifact over verified HTTPS.

    Requests use exact approved hosts on port 443, with no caller headers,
    ambient proxy/auth configuration, cookies, or Referer. Signed queries stay
    on their request targets; errors omit URLs and server response data.
    Transient HTTP responses before payload delivery receive three retries with
    exponential backoff. Authentication, TLS, redirects and partial reads retain
    their strict failure behavior; callers still verify the complete artifact.

    Args:
        url: Initial HTTPS URL without userinfo or fragments.
        output: Binary stream receiving the artifact; caller verifies its hash.
        allowed_hosts: Exact hosts permitted initially and after redirects.
        redirect_hosts: Additional exact hosts permitted only after redirects.
    Returns:
        None.
    Raises:
        PublicDownloadError: URL policy, TLS, transport, or HTTP validation failed.
    """
    try:
        _download_hops(url, output, allowed_hosts, redirect_hosts)
    except (OSError, http.client.HTTPException, ValueError):
        # Even parser/transport exceptions can contain a signed URL. Do not
        # expose a chained exception when this reaches a CLI traceback.
        raise PublicDownloadError("public HTTPS download failed") from None
