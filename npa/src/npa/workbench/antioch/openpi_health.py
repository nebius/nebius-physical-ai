"""Bounded policy readiness gate for the Isaac/OpenPI bridge pod."""

from __future__ import annotations

import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen


def wait_for_health(
    host: str,
    *,
    port: int = 8000,
    timeout_seconds: float = 1800,
    request_timeout_seconds: float = 5,
) -> None:
    """Wait for a successful private health endpoint without exposing its body."""

    if not host or "://" in host or "/" in host:
        raise ValueError("policy host must be a DNS name or IP address")
    if not 1 <= port <= 65535:
        raise ValueError("policy port is invalid")
    if timeout_seconds <= 0 or request_timeout_seconds <= 0:
        raise ValueError("health timeouts must be positive")
    deadline = time.monotonic() + timeout_seconds
    delay = 1.0
    while True:
        try:
            with urlopen(  # noqa: S310 - host is an operator-controlled in-cluster Service
                f"http://{host}:{port}/healthz", timeout=request_timeout_seconds
            ) as response:
                if 200 <= response.status < 300:
                    return
        except (TimeoutError, URLError):
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "OpenPI policy did not become healthy before the readiness deadline"
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 15.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    wait_for_health(args.host, port=args.port, timeout_seconds=args.timeout_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
