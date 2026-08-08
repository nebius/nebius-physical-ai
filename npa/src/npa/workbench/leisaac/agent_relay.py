"""TLS backhaul endpoint for a private LeIsaac Kubernetes session.

The GPU pod initiates one authenticated TLS connection to this process on the
public agent VM. Status, signaling, and the peer-egress discovery endpoint stay
loopback-only for nginx/the launcher. The public, source-restricted TURN control
socket is carried over the same authenticated backhaul to coturn beside the
simulator; the allocation's media path remains private inside the GPU pod.
"""

from __future__ import annotations

import argparse
import hmac
import http.server
import ipaddress
import json
import signal
import socket
import socketserver
import struct
import threading
import time
from pathlib import Path
from typing import Any

STATUS_LISTEN = ("127.0.0.1", 48080)
SIGNAL_LISTEN = ("127.0.0.1", 49100)
MEDIA_LISTEN = ("0.0.0.0", 3478)
BACKHAUL_LISTEN = ("127.0.0.1", 48081)
CONTROL_LISTEN = ("127.0.0.1", 48082)
HELLO, OPEN, DATA, CLOSE, UDP, UDP_CLOSE = 1, 2, 3, 4, 5, 6
HEADER = struct.Struct("!BII")
MAX_FRAME = 4 * 1024 * 1024
MAX_UDP_FLOWS = 64
UDP_FLOW_TTL_SECONDS = 120.0


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("relay config must be an object")
    nonce = str(data.get("session_nonce") or "")
    if len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise ValueError(
            "relay session nonce must be 64 lowercase hexadecimal characters"
        )
    result: dict[str, Any] = {"session_nonce": nonce}
    media_host = str(data.get("media_target_host") or "").strip()
    media_port = int(data.get("media_target_port") or 0)
    if media_host or media_port:
        try:
            address = ipaddress.ip_address(media_host)
        except ValueError as exc:
            raise ValueError(
                "relay media target must be a private IPv4 address"
            ) from exc
        if (
            address.version != 4
            or not address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError("relay media target must be a private IPv4 address")
        if media_port != MEDIA_LISTEN[1]:
            raise ValueError(f"relay media target port must be {MEDIA_LISTEN[1]}")
        result["media_target_host"] = address.compressed
        result["media_target_port"] = media_port
    return result


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("backhaul closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _receive_frame(connection: socket.socket) -> tuple[int, int, bytes]:
    kind, stream_id, size = HEADER.unpack(_receive_exact(connection, HEADER.size))
    if size > MAX_FRAME:
        raise ValueError("backhaul frame is too large")
    return kind, stream_id, _receive_exact(connection, size)


class Backhaul:
    def __init__(
        self,
        nonce: str,
        media_target: tuple[str, int] | None = None,
    ):
        self.nonce = nonce.encode("ascii")
        self.condition = threading.Condition()
        self.connection: socket.socket | None = None
        self.send_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.streams: dict[int, socket.socket] = {}
        self.next_stream = 1
        self.public_udp: socket.socket | None = None
        self.udp_lock = threading.Lock()
        self.next_udp_stream = 1
        self.udp_by_address: dict[tuple[str, int], tuple[int, float]] = {}
        self.udp_by_stream: dict[int, tuple[str, int]] = {}
        self.media_target = media_target
        self.direct_media: dict[tuple[str, int], tuple[socket.socket, float]] = {}
        self.peer_public_ip = ""

    def attach(self, connection: socket.socket) -> bool:
        kind, stream_id, payload = _receive_frame(connection)
        try:
            hello = json.loads(payload)
            nonce = str(hello.get("nonce") or "").encode("ascii")
            peer = ipaddress.ip_address(str(hello.get("peer_public_ip") or ""))
        except (AttributeError, UnicodeError, ValueError, json.JSONDecodeError):
            return False
        if (
            kind != HELLO
            or stream_id != 0
            or not hmac.compare_digest(nonce, self.nonce)
            or peer.version != 4
            or not peer.is_global
        ):
            return False
        with self.condition:
            if self.connection is not None:
                return False
            self.connection = connection
            self.peer_public_ip = peer.compressed
            self.condition.notify_all()
        return True

    def detach(self, connection: socket.socket) -> None:
        with self.condition:
            if self.connection is connection:
                self.connection = None
                self.peer_public_ip = ""
        with self.stream_lock:
            streams = list(self.streams.values())
            self.streams.clear()
        for stream in streams:
            try:
                stream.close()
            except OSError:
                pass
        with self.udp_lock:
            self.udp_by_address.clear()
            self.udp_by_stream.clear()

    def udp_stream_for(
        self, address: tuple[str, int], *, now: float | None = None
    ) -> int:
        """Map each browser UDP socket to its own pod-side connected socket.

        NVIDIA's browser client creates several ICE transports.  Collapsing
        them onto one pod UDP socket makes every reply look identical and can
        route media to the most recently active browser port.  Preserve the
        flow identity in the backhaul frame's stream id instead.
        """

        observed = time.monotonic() if now is None else now
        expired_streams: list[int] = []
        with self.udp_lock:
            existing = self.udp_by_address.get(address)
            if existing is not None:
                stream_id, _last_seen = existing
                self.udp_by_address[address] = (stream_id, observed)
                return stream_id

            expired = [
                candidate
                for candidate, (_stream_id, last_seen) in self.udp_by_address.items()
                if observed - last_seen > UDP_FLOW_TTL_SECONDS
            ]
            for candidate in expired:
                stream_id, _last_seen = self.udp_by_address.pop(candidate)
                self.udp_by_stream.pop(stream_id, None)
                expired_streams.append(stream_id)
            if len(self.udp_by_address) >= MAX_UDP_FLOWS:
                raise ConnectionError("too many browser UDP flows")

            stream_id = self.next_udp_stream
            self.next_udp_stream += 1
            self.udp_by_address[address] = (stream_id, observed)
            self.udp_by_stream[stream_id] = address
        for expired_stream in expired_streams:
            try:
                self.send(UDP_CLOSE, expired_stream)
            except (ConnectionError, OSError):
                pass
        return stream_id

    def browser_address_for(self, stream_id: int) -> tuple[str, int] | None:
        with self.udp_lock:
            return self.udp_by_stream.get(stream_id)

    def _read_direct_media(
        self,
        address: tuple[str, int],
        media: socket.socket,
    ) -> None:
        try:
            while True:
                payload = media.recv(65536)
                public = self.public_udp
                if public is None:
                    break
                public.sendto(payload, address)
        except OSError:
            pass
        finally:
            with self.udp_lock:
                current = self.direct_media.get(address)
                if current is not None and current[0] is media:
                    self.direct_media.pop(address, None)
            try:
                media.close()
            except OSError:
                pass

    def direct_media_for(
        self,
        address: tuple[str, int],
        *,
        now: float | None = None,
    ) -> socket.socket:
        """Return a native private-VPC UDP flow for one browser ICE socket."""

        if self.media_target is None:
            raise ConnectionError("private media target is unavailable")
        observed = time.monotonic() if now is None else now
        expired: list[socket.socket] = []
        with self.udp_lock:
            existing = self.direct_media.get(address)
            if existing is not None:
                media, _last_seen = existing
                self.direct_media[address] = (media, observed)
                return media
            stale = [
                candidate
                for candidate, (_media, last_seen) in self.direct_media.items()
                if observed - last_seen > UDP_FLOW_TTL_SECONDS
            ]
            for candidate in stale:
                media, _last_seen = self.direct_media.pop(candidate)
                expired.append(media)
            if len(self.direct_media) >= MAX_UDP_FLOWS:
                raise ConnectionError("too many browser UDP flows")
            media = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            media.connect(self.media_target)
            self.direct_media[address] = (media, observed)
        for stale_media in expired:
            stale_media.close()
        threading.Thread(
            target=self._read_direct_media,
            args=(address, media),
            daemon=True,
        ).start()
        return media

    def relay_browser_udp(self, payload: bytes, address: tuple[str, int]) -> None:
        """Send a browser datagram over native VPC UDP or the TLS fallback."""

        if self.media_target is not None:
            self.direct_media_for(address).send(payload)
            return
        stream_id = self.udp_stream_for(address)
        self.send(UDP, stream_id, payload)

    def close_direct_media(self) -> None:
        with self.udp_lock:
            media = [item[0] for item in self.direct_media.values()]
            self.direct_media.clear()
        for flow in media:
            try:
                flow.close()
            except OSError:
                pass

    def send(self, kind: int, stream_id: int, payload: bytes = b"") -> None:
        with self.condition:
            if self.connection is None:
                self.condition.wait_for(lambda: self.connection is not None, timeout=10)
            connection = self.connection
        if connection is None:
            raise ConnectionError("LeIsaac pod backhaul is unavailable")
        frame = HEADER.pack(kind, stream_id, len(payload)) + payload
        with self.send_lock:
            connection.sendall(frame)

    def open_stream(self, client: socket.socket, port: int) -> None:
        with self.stream_lock:
            stream_id = self.next_stream
            self.next_stream += 1
            self.streams[stream_id] = client
        try:
            self.send(OPEN, stream_id, struct.pack("!H", port))
            while True:
                payload = client.recv(65536)
                if not payload:
                    break
                self.send(DATA, stream_id, payload)
        except OSError:
            # detach() closes every loopback stream to unblock these worker
            # threads when the pod backhaul reconnects.
            pass
        finally:
            with self.stream_lock:
                self.streams.pop(stream_id, None)
            try:
                self.send(CLOSE, stream_id)
            except (ConnectionError, OSError):
                pass

    def handle(self, kind: int, stream_id: int, payload: bytes) -> None:
        if kind == DATA:
            with self.stream_lock:
                stream = self.streams.get(stream_id)
            if stream is not None:
                stream.sendall(payload)
        elif kind == CLOSE:
            with self.stream_lock:
                stream = self.streams.pop(stream_id, None)
            if stream is not None:
                stream.close()
        elif kind == UDP and self.public_udp is not None:
            address = self.browser_address_for(stream_id)
            if address is not None:
                self.public_udp.sendto(payload, address)


class _TCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server.backhaul.open_stream(self.request, self.server.target_port)  # type: ignore[attr-defined]


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, listen: tuple[str, int], backhaul: Backhaul, target_port: int):
        self.backhaul = backhaul
        self.target_port = target_port
        super().__init__(listen, _TCPHandler)


class _ControlHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/status":
            self.send_error(404)
            return
        peer = self.server.backhaul.peer_public_ip  # type: ignore[attr-defined]
        payload = (
            json.dumps(
                {"connected": bool(peer), "peer_public_ip": peer}, sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        self.send_response(200 if peer else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _ControlServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, backhaul: Backhaul):
        self.backhaul = backhaul
        super().__init__(CONTROL_LISTEN, _ControlHandler)


def serve_backhaul(
    backhaul: Backhaul,
    *,
    stop: threading.Event,
) -> None:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(BACKHAUL_LISTEN)
    listener.listen(2)
    listener.settimeout(0.5)
    try:
        while not stop.is_set():
            try:
                raw, _address = listener.accept()
            except socket.timeout:
                continue
            try:
                if not backhaul.attach(raw):
                    raw.close()
                    continue
                connection = raw
                try:
                    while not stop.is_set():
                        backhaul.handle(*_receive_frame(connection))
                finally:
                    backhaul.detach(connection)
                    connection.close()
            except (EOFError, OSError, ValueError):
                raw.close()
    finally:
        listener.close()


def relay_udp(backhaul: Backhaul, *, stop: threading.Event) -> None:
    public = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    public.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    public.bind(MEDIA_LISTEN)
    public.settimeout(0.5)
    backhaul.public_udp = public
    try:
        while not stop.is_set():
            try:
                payload, address = public.recvfrom(65536)
            except socket.timeout:
                continue
            try:
                backhaul.relay_browser_udp(payload, address)
            except (ConnectionError, OSError):
                continue
    finally:
        backhaul.public_udp = None
        backhaul.close_direct_media()
        public.close()


def serve(config: dict[str, Any]) -> None:
    media_target = None
    if config.get("media_target_host") and config.get("media_target_port"):
        media_target = (
            str(config["media_target_host"]),
            int(config["media_target_port"]),
        )
    backhaul = Backhaul(str(config["session_nonce"]), media_target)
    status = _TCPServer(STATUS_LISTEN, backhaul, 8080)
    signaling = _TCPServer(SIGNAL_LISTEN, backhaul, 49100)
    control = _ControlServer(backhaul)
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    threads = [
        threading.Thread(target=status.serve_forever, daemon=True),
        threading.Thread(target=signaling.serve_forever, daemon=True),
        threading.Thread(target=control.serve_forever, daemon=True),
        threading.Thread(
            target=serve_backhaul,
            kwargs={
                "backhaul": backhaul,
                "stop": stop,
            },
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        relay_udp(backhaul, stop=stop)
    finally:
        status.shutdown()
        signaling.shutdown()
        control.shutdown()
        status.server_close()
        signaling.server_close()
        control.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    serve(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
