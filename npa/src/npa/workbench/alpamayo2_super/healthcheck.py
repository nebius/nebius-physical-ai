"""Authenticate the container-native Alpamayo 2 Super readiness probe."""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import Request, urlopen


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
    request = Request(
        "http://127.0.0.1:8080/health",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urlopen(request, timeout=3) as response:
            return 0 if response.status == 200 else 1
    except (URLError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
