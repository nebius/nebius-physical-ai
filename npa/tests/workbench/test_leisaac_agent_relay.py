from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from npa.workbench.leisaac.agent_relay import (
    DATA,
    BACKHAUL_LISTEN,
    CONTROL_LISTEN,
    HELLO,
    OPEN,
    UDP,
    Backhaul,
    _receive_frame,
    load_config as load_server_config,
)
from npa.workbench.leisaac.reverse_client import (
    Client,
    WebSocketConnection,
    _pod_ipv4,
    _public_ipv4,
    load_config as load_client_config,
)


NONCE = "a" * 64


def test_agent_relay_control_and_raw_backhaul_are_loopback_only() -> None:
    assert CONTROL_LISTEN == ("127.0.0.1", 48082)
    assert BACKHAUL_LISTEN == ("127.0.0.1", 48081)


def test_relay_configs_pin_nonce_public_agent_and_certificate(tmp_path: Path) -> None:
    server_path = tmp_path / "server.json"
    server_path.write_text(json.dumps({"session_nonce": NONCE}), encoding="utf-8")
    assert load_server_config(server_path) == {"session_nonce": NONCE}

    server_path.write_text(
        json.dumps(
            {
                "session_nonce": NONCE,
                "media_target_host": "10.96.0.22",
                "media_target_port": 3478,
            }
        ),
        encoding="utf-8",
    )
    assert load_server_config(server_path) == {
        "session_nonce": NONCE,
        "media_target_host": "10.96.0.22",
        "media_target_port": 3478,
    }
    for host, port in (
        ("8.8.8.8", 3478),
        ("127.0.0.1", 3478),
        ("10.96.0.22", 49100),
    ):
        server_path.write_text(
            json.dumps(
                {
                    "session_nonce": NONCE,
                    "media_target_host": host,
                    "media_target_port": port,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="media target"):
            load_server_config(server_path)

    client_path = tmp_path / "client.json"
    client_path.write_text(
        json.dumps(
            {
                "agent_host": "8.8.8.8",
                "session_nonce": NONCE,
                "certificate_sha256": "b" * 64,
                "auth_user": "npa",
                "auth_password": "secret",
            }
        ),
        encoding="utf-8",
    )
    assert load_client_config(client_path) == {
        "agent_host": "8.8.8.8",
        "session_nonce": NONCE,
        "certificate_sha256": "b" * 64,
        "auth_user": "npa",
        "auth_password": "secret",
    }

    for override in (
        {"agent_host": "127.0.0.1"},
        {"session_nonce": "bad"},
        {"certificate_sha256": "bad"},
        {"auth_password": ""},
    ):
        data = json.loads(client_path.read_text(encoding="utf-8"))
        data.update(override)
        client_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            load_client_config(client_path)


def test_backhaul_rejects_wrong_nonce_and_multiplexes_loopback_tcp() -> None:
    backhaul = Backhaul(NONCE)
    server_connection, pod_connection = socket.socketpair()

    def pod() -> None:
        hello = json.dumps({"nonce": NONCE, "peer_public_ip": "8.8.4.4"}).encode(
            "ascii"
        )
        pod_connection.sendall(
            __import__("struct").pack("!BII", HELLO, 0, len(hello)) + hello
        )
        kind, stream_id, payload = _receive_frame(pod_connection)
        assert kind == OPEN
        assert payload == __import__("struct").pack("!H", 8080)
        pod_connection.sendall(
            __import__("struct").pack("!BII", DATA, stream_id, 5) + b"READY"
        )

    threading.Thread(target=pod, daemon=True).start()
    assert backhaul.attach(server_connection) is True
    assert backhaul.peer_public_ip == "8.8.4.4"
    local_server, local_client = socket.socketpair()
    threading.Thread(
        target=backhaul.open_stream,
        args=(local_server, 8080),
        daemon=True,
    ).start()
    kind, stream_id, payload = _receive_frame(server_connection)
    backhaul.handle(kind, stream_id, payload)
    assert local_client.recv(5) == b"READY"
    local_client.close()
    pod_connection.close()
    server_connection.close()


def test_backhaul_rejects_unauthenticated_hello() -> None:
    backhaul = Backhaul(NONCE)
    server_connection, peer = socket.socketpair()
    hello = json.dumps({"nonce": "b" * 64, "peer_public_ip": "8.8.4.4"}).encode("ascii")
    peer.sendall(__import__("struct").pack("!BII", HELLO, 0, len(hello)) + hello)
    assert backhaul.attach(server_connection) is False
    peer.close()
    server_connection.close()


def test_backhaul_preserves_multiple_browser_udp_flows() -> None:
    backhaul = Backhaul(NONCE)
    first = ("198.51.100.10", 41001)
    second = ("198.51.100.10", 41002)

    first_stream = backhaul.udp_stream_for(first, now=1.0)
    second_stream = backhaul.udp_stream_for(second, now=1.0)

    assert first_stream != second_stream
    assert backhaul.udp_stream_for(first, now=2.0) == first_stream
    assert backhaul.browser_address_for(first_stream) == first
    assert backhaul.browser_address_for(second_stream) == second


def test_private_client_uses_one_connected_media_socket_per_udp_flow(
    monkeypatch,
) -> None:
    created: list[object] = []

    class FakeSocket:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sent: list[bytes] = []
            created.append(self)

        def connect(self, address: tuple[str, int]) -> None:
            assert address == ("10.96.34.76", 3478)

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv(self, _size: int) -> bytes:
            raise OSError("test flow complete")

        def close(self) -> None:
            return None

    class FakeThread:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(
        "npa.workbench.leisaac.reverse_client._pod_ipv4",
        lambda: "10.96.34.76",
    )
    monkeypatch.setattr(
        "npa.workbench.leisaac.reverse_client._public_ipv4",
        lambda: "8.8.4.4",
    )
    monkeypatch.setattr(
        "npa.workbench.leisaac.reverse_client.threading.Thread", FakeThread
    )
    client = Client({})
    client.handle(UDP, 11, b"first")
    client.handle(UDP, 12, b"second")
    client.handle(UDP, 11, b"again")

    assert len(created) == 2
    assert created[0].sent == [b"first", b"again"]  # type: ignore[attr-defined]
    assert created[1].sent == [b"second"]  # type: ignore[attr-defined]


def test_private_client_resolves_only_non_loopback_private_pod_ipv4(
    monkeypatch,
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "leisaac-pod")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("10.96.34.76", 0)),
        ],
    )

    assert _pod_ipv4() == "10.96.34.76"


def test_private_client_resolves_only_global_gpu_egress_ipv4(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return b"8.8.4.4\n"

    monkeypatch.setattr(
        "npa.workbench.leisaac.reverse_client.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(),
    )

    assert _public_ipv4() == "8.8.4.4"


def test_backhaul_rejects_private_peer_address() -> None:
    backhaul = Backhaul(NONCE)
    server_connection, peer = socket.socketpair()
    hello = json.dumps({"nonce": NONCE, "peer_public_ip": "10.96.0.22"}).encode("ascii")
    peer.sendall(__import__("struct").pack("!BII", HELLO, 0, len(hello)) + hello)

    assert backhaul.attach(server_connection) is False
    peer.close()
    server_connection.close()


def test_agent_uses_native_private_udp_per_browser_flow(monkeypatch) -> None:
    created: list[object] = []

    class FakeSocket:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sent: list[bytes] = []
            self.target: tuple[str, int] | None = None
            created.append(self)

        def connect(self, address: tuple[str, int]) -> None:
            self.target = address

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

        def close(self) -> None:
            return None

    class FakeThread:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(
        "npa.workbench.leisaac.agent_relay.threading.Thread", FakeThread
    )
    backhaul = Backhaul(NONCE, ("10.96.0.22", 3478))
    first = ("198.51.100.10", 41001)
    second = ("198.51.100.10", 41002)

    backhaul.relay_browser_udp(b"first", first)
    backhaul.relay_browser_udp(b"second", second)
    backhaul.relay_browser_udp(b"again", first)

    assert len(created) == 2
    assert created[0].target == ("10.96.0.22", 3478)  # type: ignore[attr-defined]
    assert created[0].sent == [b"first", b"again"]  # type: ignore[attr-defined]
    assert created[1].sent == [b"second"]  # type: ignore[attr-defined]


def test_private_websocket_client_masks_outbound_and_reads_binary_reply() -> None:
    client_socket, server_socket = socket.socketpair()
    websocket = WebSocketConnection(client_socket)  # type: ignore[arg-type]

    def server() -> None:
        first, second = server_socket.recv(2)
        assert first == 0x82
        assert second & 0x80
        size = second & 0x7F
        mask = server_socket.recv(4)
        payload = server_socket.recv(size)
        assert (
            bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            == b"hello"
        )
        server_socket.sendall(bytes((0x82, 5)) + b"world")

    threading.Thread(target=server, daemon=True).start()
    websocket.sendall(b"hello")
    assert websocket.recv(5) == b"world"
    client_socket.close()
    server_socket.close()
