"""Authenticate the container-native Alpamayo 2 Super readiness probe."""

from __future__ import annotations

import os
from http.client import HTTPConnection, HTTPException


def main() -> int:
    """Probe the local service with its runtime admission credential.

    Returns:
        Zero when the authenticated health endpoint is ready; otherwise one.

    Raises:
        None. Connection and request failures are reported through the return code.
    """

    token = os.environ.get("NPA_ALPAMAYO2_SUPER_TOKEN", "")
    if not token:
        return 1
    connection = HTTPConnection("127.0.0.1", 8080, timeout=3)
    try:
        connection.request("GET", "/health", headers={"Authorization": "Bearer " + token})
        return 0 if connection.getresponse().status == 200 else 1
    except (HTTPException, OSError):
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
