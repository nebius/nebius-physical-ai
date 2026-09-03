"""Authenticated WSS rendezvous inside the persistent Antioch sim service."""

from __future__ import annotations

import argparse
import hmac
import ssl
import threading
import time
from pathlib import Path

from websockets.sync.server import serve

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
ROLES = frozenset({"operator", "simulation"})


class RelayBridge:
    """Pair one operator relay with one streamed scenario connection."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._condition = threading.Condition()
        self._peers: dict[str, object] = {}

    def handle(self, connection) -> None:  # noqa: ANN001
        authorization = connection.request.headers.get("Authorization", "")
        role = connection.request.headers.get("X-NPA-Relay-Role", "")
        if not hmac.compare_digest(authorization, "Api-Key " + self._token):
            connection.close(code=1008, reason="authentication required")
            return
        if role not in ROLES:
            connection.close(code=1008, reason="relay role required")
            return
        other_role = "simulation" if role == "operator" else "operator"
        with self._condition:
            if role in self._peers:
                connection.close(code=1013, reason="relay role already connected")
                return
            self._peers[role] = connection
            self._condition.notify_all()
            while other_role not in self._peers:
                self._condition.wait(timeout=1)
            print(f"NPA_ANTIOCH_BRIDGE_PAIRED role={role}", flush=True)
        try:
            while True:
                message = connection.recv(timeout=120)
                with self._condition:
                    peer = self._peers.get(other_role)
                if peer is None:
                    raise RuntimeError("relay peer disconnected")
                peer.send(message)
        finally:
            with self._condition:
                if self._peers.get(role) is connection:
                    self._peers.pop(role, None)
                peer = self._peers.pop(other_role, None)
                self._condition.notify_all()
            if peer is not None:
                try:
                    peer.close(code=1012, reason="relay peer reconnected")
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--wait-for-bundle", action="store_true")
    parser.add_argument("service_command", nargs="*")
    args = parser.parse_args()
    if args.service_command not in ([], ["sleep", "infinity"]):
        raise RuntimeError("unexpected service command arguments")
    bundle = Path(args.bundle)
    required = ("relay-api-key", "relay-server.crt", "relay-server.key")
    while args.wait_for_bundle and not all(
        (bundle / name).is_file() for name in required
    ):
        time.sleep(1)
    token = (bundle / "relay-api-key").read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("relay API key is missing or malformed")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(
        str(bundle / "relay-server.crt"), str(bundle / "relay-server.key")
    )
    bridge = RelayBridge(token)
    with serve(
        bridge.handle,
        "0.0.0.0",
        8444,
        ssl=context,
        compression=None,
        max_size=MAX_MESSAGE_BYTES,
        max_queue=2,
        open_timeout=10,
        close_timeout=5,
    ) as server:
        print("NPA_ANTIOCH_BRIDGE_READY", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
