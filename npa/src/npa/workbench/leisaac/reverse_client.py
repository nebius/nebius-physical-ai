"""Private GPU-pod side of the LeIsaac TLS backhaul."""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import socket
import ssl
import struct
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

HELLO, OPEN, DATA, CLOSE, UDP, UDP_CLOSE = 1, 2, 3, 4, 5, 6
HEADER = struct.Struct("!BII")
MAX_FRAME = 4 * 1024 * 1024
MAX_UDP_FLOWS = 64
BACKHAUL_SUBPROTOCOL = "npa.leisaac.backhaul.v1"


def _pod_ipv4() -> str:
    """Resolve this pod's non-loopback IPv4 for WebRTC's server-facing peer."""

    for _family, _kind, _protocol, _canonname, sockaddr in socket.getaddrinfo(
        socket.gethostname(), 0, family=socket.AF_INET, type=socket.SOCK_DGRAM
    ):
        address = ipaddress.ip_address(sockaddr[0])
        if (
            address.version == 4
            and address.is_private
            and not address.is_loopback
            and not address.is_link_local
            and not address.is_unspecified
            and not address.is_multicast
        ):
            return address.compressed
    raise ValueError("LeIsaac relay client could not resolve its private pod IPv4")


def _public_ipv4() -> str:
    request = urllib.request.Request(
        "https://api.ipify.org", headers={"User-Agent": "npa-leisaac-relay/0.4.0"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS URL
        address = ipaddress.ip_address(response.read(64).decode("ascii").strip())
    if address.version != 4 or not address.is_global:
        raise ValueError("LeIsaac relay client did not resolve a public IPv4")
    return address.compressed


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    host = ipaddress.ip_address(str(data.get("agent_host") or ""))
    if not host.is_global:
        raise ValueError("agent host must be a public IP address")
    nonce = str(data.get("session_nonce") or "")
    fingerprint = str(data.get("certificate_sha256") or "").lower()
    auth_user = str(data.get("auth_user") or "")
    auth_password = str(data.get("auth_password") or "")
    if len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise ValueError("session nonce is invalid")
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("certificate fingerprint is invalid")
    if not auth_user or not auth_password or "\n" in auth_user + auth_password:
        raise ValueError("agent basic-auth credential is invalid")
    return {
        "agent_host": host.compressed,
        "session_nonce": nonce,
        "certificate_sha256": fingerprint,
        "auth_user": auth_user,
        "auth_password": auth_password,
    }


def _read_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise EOFError("WebSocket closed")
        data.extend(chunk)
    return bytes(data)


class WebSocketConnection:
    """Small binary WSS client sufficient for the private NPA backhaul."""

    def __init__(self, connection: ssl.SSLSocket):
        self.connection = connection
        self.buffer = bytearray()
        self.send_lock = threading.Lock()

    def _send_message(self, payload: bytes, opcode: int = 2) -> None:
        mask = os.urandom(4)
        size = len(payload)
        if size < 126:
            header = bytes((0x80 | opcode, 0x80 | size))
        elif size <= 65535:
            header = bytes((0x80 | opcode, 0x80 | 126)) + struct.pack("!H", size)
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + struct.pack("!Q", size)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        with self.send_lock:
            self.connection.sendall(header + mask + masked)

    def sendall(self, payload: bytes) -> None:
        self._send_message(payload)

    def _receive_message(self) -> bytes:
        fragments = bytearray()
        while True:
            first, second = _read_exact(self.connection, 2)
            final = bool(first & 0x80)
            opcode = first & 0x0F
            size = second & 0x7F
            if size == 126:
                size = struct.unpack("!H", _read_exact(self.connection, 2))[0]
            elif size == 127:
                size = struct.unpack("!Q", _read_exact(self.connection, 8))[0]
            mask = _read_exact(self.connection, 4) if second & 0x80 else b""
            payload = _read_exact(self.connection, size)
            if mask:
                payload = bytes(
                    value ^ mask[index % 4] for index, value in enumerate(payload)
                )
            if opcode == 8:
                raise EOFError("WebSocket closed")
            if opcode == 9:
                self._send_message(payload, opcode=10)
                continue
            if opcode in (0, 2):
                fragments.extend(payload)
                if final:
                    return bytes(fragments)

    def recv(self, size: int) -> bytes:
        if not self.buffer:
            self.buffer.extend(self._receive_message())
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def close(self) -> None:
        try:
            self._send_message(b"", opcode=8)
        except OSError:
            pass
        self.connection.close()


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


class Client:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.connection: WebSocketConnection | None = None
        self.send_lock = threading.Lock()
        self.stream_lock = threading.Lock()
        self.streams: dict[int, socket.socket] = {}
        self.media_lock = threading.Lock()
        self.media: dict[int, socket.socket] = {}
        # Public TURN control datagrams arrive through the authenticated agent
        # backhaul. Coturn shares this pod's network namespace with Isaac Sim,
        # so its relay allocation reaches the 47998 media peer without NAT.
        self.media_target = (_pod_ipv4(), 3478)
        self.peer_public_ip = _public_ipv4()

    def send(self, kind: int, stream_id: int, payload: bytes = b"") -> None:
        connection = self.connection
        if connection is None:
            raise ConnectionError("backhaul is disconnected")
        with self.send_lock:
            connection.sendall(HEADER.pack(kind, stream_id, len(payload)) + payload)

    def read_stream(self, stream_id: int, stream: socket.socket) -> None:
        try:
            while True:
                payload = stream.recv(65536)
                if not payload:
                    break
                self.send(DATA, stream_id, payload)
        except (ConnectionError, OSError):
            pass
        finally:
            with self.stream_lock:
                self.streams.pop(stream_id, None)
            try:
                self.send(CLOSE, stream_id)
            except (ConnectionError, OSError):
                pass
            stream.close()

    def read_media(self, stream_id: int, media: socket.socket) -> None:
        try:
            while True:
                payload = media.recv(65536)
                self.send(UDP, stream_id, payload)
        except (ConnectionError, OSError):
            pass
        finally:
            with self.media_lock:
                if self.media.get(stream_id) is media:
                    self.media.pop(stream_id, None)
            try:
                media.close()
            except OSError:
                pass

    def media_for(self, stream_id: int) -> socket.socket:
        with self.media_lock:
            existing = self.media.get(stream_id)
            if existing is not None:
                return existing
            if len(self.media) >= MAX_UDP_FLOWS:
                raise ConnectionError("too many browser UDP flows")
            media = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            media.connect(self.media_target)
            self.media[stream_id] = media
        threading.Thread(
            target=self.read_media,
            args=(stream_id, media),
            daemon=True,
        ).start()
        return media

    def handle(self, kind: int, stream_id: int, payload: bytes) -> None:
        if kind == OPEN:
            if len(payload) != 2:
                raise ValueError("invalid open frame")
            port = struct.unpack("!H", payload)[0]
            if port not in (8080, 49100):
                raise ValueError("invalid LeIsaac target port")
            stream = socket.create_connection(("127.0.0.1", port), timeout=10)
            stream.settimeout(None)
            with self.stream_lock:
                self.streams[stream_id] = stream
            threading.Thread(
                target=self.read_stream,
                args=(stream_id, stream),
                daemon=True,
            ).start()
        elif kind == DATA:
            with self.stream_lock:
                stream = self.streams.get(stream_id)
            if stream is not None:
                stream.sendall(payload)
        elif kind == CLOSE:
            with self.stream_lock:
                stream = self.streams.pop(stream_id, None)
            if stream is not None:
                stream.close()
        elif kind == UDP:
            self.media_for(stream_id).send(payload)
        elif kind == UDP_CLOSE:
            with self.media_lock:
                media = self.media.pop(stream_id, None)
            if media is not None:
                media.close()

    def connect(self) -> WebSocketConnection:
        host = str(self.config["agent_host"])
        raw = socket.create_connection((host, 443), timeout=10)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = context.wrap_socket(raw, server_hostname=host)
        certificate = connection.getpeercert(binary_form=True)
        if hashlib.sha256(certificate).hexdigest() != self.config["certificate_sha256"]:
            connection.close()
            raise ssl.SSLError("agent relay certificate fingerprint mismatch")
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        credential = base64.b64encode(
            f"{self.config['auth_user']}:{self.config['auth_password']}".encode("utf-8")
        ).decode("ascii")
        request = (
            "GET /api/leisaac/backhaul HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Protocol: {BACKHAUL_SUBPROTOCOL}\r\n"
            f"Authorization: Basic {credential}\r\n\r\n"
        )
        connection.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response and len(response) < 16384:
            response.extend(connection.recv(4096))
        headers, separator, _remainder = bytes(response).partition(b"\r\n\r\n")
        expected_accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
            ).digest()
        )
        if (
            not separator
            or not headers.startswith(b"HTTP/1.1 101 ")
            or b"sec-websocket-accept: " + expected_accept.lower()
            not in headers.lower()
            or f"sec-websocket-protocol: {BACKHAUL_SUBPROTOCOL}".encode("ascii")
            not in headers.lower()
        ):
            connection.close()
            raise ConnectionError("authenticated agent WebSocket upgrade failed")
        connection.settimeout(None)
        return WebSocketConnection(connection)

    def run(self) -> None:
        while True:
            try:
                self.connection = self.connect()
                self.send(
                    HELLO,
                    0,
                    json.dumps(
                        {
                            "nonce": str(self.config["session_nonce"]),
                            "peer_public_ip": self.peer_public_ip,
                        },
                        separators=(",", ":"),
                    ).encode("ascii"),
                )
                while True:
                    self.handle(*_receive_frame(self.connection))
            except (ConnectionError, EOFError, OSError, ssl.SSLError, ValueError):
                time.sleep(2)
            finally:
                if self.connection is not None:
                    self.connection.close()
                self.connection = None
                with self.stream_lock:
                    streams = list(self.streams.values())
                    self.streams.clear()
                for stream in streams:
                    stream.close()
                with self.media_lock:
                    media = list(self.media.values())
                    self.media.clear()
                for flow in media:
                    flow.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    Client(load_config(args.config)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
