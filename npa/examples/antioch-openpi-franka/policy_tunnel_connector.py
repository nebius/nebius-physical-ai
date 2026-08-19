"""Connect Antioch's authenticated local tunnel to a local policy port-forward."""

from __future__ import annotations

import argparse
import json
import socket
import time

from reverse_policy_relay import ReversePolicyRelay


def _connect(
    host: str,
    port: int,
    *,
    attempts: int,
    initial_backoff_seconds: float,
    maximum_backoff_seconds: float,
) -> socket.socket:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            return socket.create_connection((host, port), timeout=5)
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(
                    min(
                        maximum_backoff_seconds,
                        initial_backoff_seconds * 2**attempt,
                    )
                )
    raise ConnectionError(
        "private policy connector exhausted connection attempts"
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-host", default="127.0.0.1")
    parser.add_argument("--relay-port", type=int, default=18123)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18000)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--connect-attempts", type=int, default=20)
    parser.add_argument("--initial-backoff-seconds", type=float, default=0.25)
    parser.add_argument("--maximum-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-empty-sessions", type=int, default=20)
    args = parser.parse_args()
    if args.sessions < 1 or args.connect_attempts < 1 or args.max_empty_sessions < 1:
        parser.error("session and attempt limits must be positive")
    if (
        args.initial_backoff_seconds <= 0
        or args.maximum_backoff_seconds < args.initial_backoff_seconds
    ):
        parser.error("connector backoff bounds are invalid")
    completed = 0
    empty = 0

    def connect(host: str, port: int) -> socket.socket:
        return _connect(
            host,
            port,
            attempts=args.connect_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            maximum_backoff_seconds=args.maximum_backoff_seconds,
        )

    while completed < args.sessions:
        relay = connect(args.relay_host, args.relay_port)
        policy = connect(args.policy_host, args.policy_port)
        if ReversePolicyRelay._pipe_pair(relay, policy) > 0:
            completed += 1
        else:
            empty += 1
            if empty >= args.max_empty_sessions:
                raise ConnectionError(
                    "private policy connector exhausted empty tunnel sessions"
                )
            time.sleep(args.initial_backoff_seconds)
    print(
        json.dumps(
            {
                "completed_sessions": completed,
                "empty_sessions": empty,
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
