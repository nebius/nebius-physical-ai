from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from npa.workbench.leisaac.agent_relay import (
    COMPATIBILITY_PEER_IPV4_FIELD as RELAY_COMPATIBILITY_PEER_IPV4_FIELD,
    DEFAULT_HELLO_TIMEOUT_SECONDS,
    DATA,
    BACKHAUL_LISTEN,
    CONTROL_LISTEN,
    HELLO,
    OPEN,
    UDP,
    UDP_SOCKET_BUFFER_BYTES,
    Backhaul,
    _authenticate_backhaul,
    _tune_udp_socket,
    _receive_frame,
    load_config as load_server_config,
)
from npa.workbench.leisaac.reverse_client import (
    COMPATIBILITY_PEER_IPV4_FIELD as CLIENT_COMPATIBILITY_PEER_IPV4_FIELD,
    Client,
    WebSocketConnection,
    _hello_payload,
    _mask_websocket_payload,
    _pod_ipv4,
    load_config as load_client_config,
)
from npa.workbench.leisaac import agent_relay
from npa.workbench.leisaac import reverse_client


NONCE = "a" * 64


def test_rolling_upgrade_hello_uses_non_authoritative_compatibility_field() -> None:
    assert CLIENT_COMPATIBILITY_PEER_IPV4_FIELD == "0.0.0.0"
    assert RELAY_COMPATIBILITY_PEER_IPV4_FIELD == CLIENT_COMPATIBILITY_PEER_IPV4_FIELD
    assert json.loads(_hello_payload({"session_nonce": NONCE})) == {
        "nonce": NONCE,
        "peer_public_ip": "0.0.0.0",
    }

    backhaul = Backhaul(NONCE)
    server_connection, peer = socket.socketpair()
    payload = _hello_payload({"session_nonce": NONCE})
    peer.sendall(__import__("struct").pack("!BII", HELLO, 0, len(payload)) + payload)
    assert backhaul.attach(server_connection) is True
    assert backhaul.peer_public_ip == "0.0.0.0"
    backhaul.revoke()
    peer.close()


def test_agent_relay_tunes_both_udp_socket_buffers() -> None:
    configured: list[tuple[int, int, int]] = []

    class FakeSocket:
        def setsockopt(self, level: int, option: int, value: int) -> None:
            configured.append((level, option, value))

    _tune_udp_socket(FakeSocket())  # type: ignore[arg-type]

    assert configured == [
        (socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_SOCKET_BUFFER_BYTES),
        (socket.SOL_SOCKET, socket.SO_SNDBUF, UDP_SOCKET_BUFFER_BYTES),
    ]


@pytest.mark.parametrize("size", [0, 1, 3, 4, 5, 127, 1260, 180 * 1024])
def test_private_websocket_mask_is_byte_identical_for_arbitrary_sizes(
    size: int,
) -> None:
    payload = os.urandom(size)
    mask = os.urandom(4)
    expected = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    assert _mask_websocket_payload(payload, mask) == expected
    assert _mask_websocket_payload(expected, mask) == payload


def test_private_websocket_mask_rejects_invalid_key_size() -> None:
    with pytest.raises(ValueError, match="four bytes"):
        _mask_websocket_payload(b"payload", b"bad")


def test_agent_relay_control_and_raw_backhaul_are_loopback_only() -> None:
    assert CONTROL_LISTEN == ("127.0.0.1", 48082)
    assert BACKHAUL_LISTEN == ("127.0.0.1", 48081)


def test_relay_configs_pin_nonce_public_agent_and_certificate(tmp_path: Path) -> None:
    server_path = tmp_path / "server.json"
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    server_path.write_text(
        json.dumps(
            {"run_id": "run-123", "session_nonce": NONCE, "expires_at": expires_at}
        ),
        encoding="utf-8",
    )
    loaded = load_server_config(server_path)
    assert loaded["run_id"] == "run-123"
    assert loaded["session_nonce"] == NONCE
    assert loaded["expires_at"] == expires_at
    assert loaded["expires_epoch"] > datetime.now(timezone.utc).timestamp()
    assert loaded["hello_timeout_seconds"] == DEFAULT_HELLO_TIMEOUT_SECONDS

    server_path.write_text(
        json.dumps({"run_id": "run-123", "session_nonce": NONCE, "expires_at": ""}),
        encoding="utf-8",
    )
    loaded = load_server_config(server_path)
    assert loaded["expires_at"] == ""
    assert loaded["expires_epoch"] == float("inf")

    server_path.write_text(
        json.dumps(
            {
                "session_nonce": NONCE,
                "run_id": "run-123",
                "expires_at": expires_at,
                "media_target_host": "10.96.0.22",
                "media_target_port": 3478,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_server_config(server_path)
    expires_epoch = loaded.pop("expires_epoch")
    assert expires_epoch > datetime.now(timezone.utc).timestamp()
    assert loaded == {
        "session_nonce": NONCE,
        "run_id": "run-123",
        "expires_at": expires_at,
        "hello_timeout_seconds": DEFAULT_HELLO_TIMEOUT_SECONDS,
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
                    "run_id": "run-123",
                    "expires_at": expires_at,
                    "media_target_host": host,
                    "media_target_port": port,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="media target"):
            load_server_config(server_path)

    for invalid in (
        {"run_id": "", "session_nonce": NONCE, "expires_at": expires_at},
        {
            "run_id": "run-123",
            "session_nonce": NONCE,
            "expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
        },
    ):
        server_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="run id|expiry|expired"):
            load_server_config(server_path)

    for invalid_timeout in (0, 61, "not-a-number"):
        server_path.write_text(
            json.dumps(
                {
                    "run_id": "run-123",
                    "session_nonce": NONCE,
                    "expires_at": expires_at,
                    "hello_timeout_seconds": invalid_timeout,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="HELLO timeout"):
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


def test_preauth_rejects_malformed_hello_deterministically() -> None:
    server_connection, peer = socket.socketpair()
    malformed = b"{not-json"
    peer.sendall(
        __import__("struct").pack("!BII", HELLO, 0, len(malformed)) + malformed
    )
    assert not _authenticate_backhaul(
        Backhaul(NONCE), server_connection, hello_timeout_seconds=0.1
    )
    peer.close()
    server_connection.close()


def test_preauth_timeout_is_bounded_and_legitimate_peer_restores_timeout() -> None:
    silent_server, silent_peer = socket.socketpair()
    with pytest.raises(socket.timeout):
        _authenticate_backhaul(
            Backhaul(NONCE), silent_server, hello_timeout_seconds=0.02
        )
    silent_server.close()
    silent_peer.close()

    backhaul = Backhaul(NONCE)
    server_connection, peer = socket.socketpair()
    server_connection.settimeout(7.5)
    hello = json.dumps({"nonce": NONCE, "peer_public_ip": "8.8.4.4"}).encode()
    peer.sendall(__import__("struct").pack("!BII", HELLO, 0, len(hello)) + hello)
    assert server_connection.gettimeout() == 7.5
    assert _authenticate_backhaul(
        backhaul, server_connection, hello_timeout_seconds=0.1
    )
    assert server_connection.gettimeout() == 7.5
    backhaul.revoke()
    peer.close()


def test_revoke_unblocks_a_silent_preauth_socket() -> None:
    backhaul = Backhaul(NONCE)
    server_connection, peer = socket.socketpair()
    outcome: list[type[BaseException] | bool] = []

    def authenticate() -> None:
        try:
            outcome.append(
                _authenticate_backhaul(
                    backhaul, server_connection, hello_timeout_seconds=30.0
                )
            )
        except BaseException as exc:
            outcome.append(type(exc))

    worker = threading.Thread(target=authenticate)
    worker.start()
    for _ in range(100):
        with backhaul.preauth_lock:
            if backhaul.preauth_connections:
                break
        threading.Event().wait(0.001)
    backhaul.revoke()
    worker.join(timeout=1)
    assert not worker.is_alive()
    # revoke() may win either immediately before attach() checks the revoked
    # state (a clean False denial) or while recv() is blocked (an EOF/socket
    # error). Both outcomes prove that the credential is denied and the worker
    # is unblocked; the scheduler must not make this assertion flaky.
    assert outcome == [False] or (outcome and outcome[0] in {OSError, EOFError})
    peer.close()


def test_stale_detach_cannot_clear_replacement_connection_state() -> None:
    backhaul = Backhaul(NONCE)
    stale_connection, stale_peer = socket.socketpair()
    replacement_connection, replacement_peer = socket.socketpair()
    browser_stream, browser_peer = socket.socketpair()
    backhaul.connection = replacement_connection
    backhaul.peer_public_ip = "8.8.4.4"
    backhaul.streams[7] = browser_stream
    backhaul.udp_by_address[("198.51.100.10", 41001)] = (11, 1.0)
    backhaul.udp_by_stream[11] = ("198.51.100.10", 41001)

    backhaul.detach(stale_connection)

    assert backhaul.connection is replacement_connection
    assert backhaul.peer_public_ip == "8.8.4.4"
    assert backhaul.streams == {7: browser_stream}
    assert backhaul.udp_by_stream == {11: ("198.51.100.10", 41001)}
    browser_peer.sendall(b"still-open")
    assert browser_stream.recv(10) == b"still-open"

    stale_connection.close()
    stale_peer.close()
    replacement_connection.close()
    replacement_peer.close()
    browser_stream.close()
    browser_peer.close()


def test_active_backhaul_does_not_block_stale_credential_rejection(
    monkeypatch,
) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    listen = ("127.0.0.1", int(probe.getsockname()[1]))
    probe.close()
    monkeypatch.setattr(agent_relay, "BACKHAUL_LISTEN", listen)

    backhaul = Backhaul(NONCE)
    stop = threading.Event()
    server = threading.Thread(
        target=agent_relay.serve_backhaul,
        kwargs={
            "backhaul": backhaul,
            "stop": stop,
            "hello_timeout_seconds": 0.2,
        },
    )
    server.start()

    def connect() -> socket.socket:
        for _attempt in range(100):
            candidate = socket.socket()
            try:
                candidate.connect(listen)
                return candidate
            except ConnectionRefusedError:
                candidate.close()
                threading.Event().wait(0.005)
        raise AssertionError("relay listener did not start")

    legitimate = connect()
    good_hello = json.dumps({"nonce": NONCE, "peer_public_ip": "8.8.4.4"}).encode(
        "ascii"
    )
    legitimate.sendall(
        __import__("struct").pack("!BII", HELLO, 0, len(good_hello)) + good_hello
    )
    for _attempt in range(100):
        if backhaul.connection is not None:
            break
        threading.Event().wait(0.005)
    assert backhaul.connection is not None

    stale = connect()
    stale.settimeout(1.0)
    stale_hello = json.dumps({"nonce": "b" * 64, "peer_public_ip": "8.8.4.4"}).encode(
        "ascii"
    )
    stale.sendall(
        __import__("struct").pack("!BII", HELLO, 0, len(stale_hello)) + stale_hello
    )
    assert stale.recv(1) == b""

    stop.set()
    server.join(timeout=2.0)
    assert not server.is_alive()
    stale.close()
    legitimate.close()


def test_expired_backhaul_revokes_connection_and_rejects_reuse() -> None:
    backhaul = Backhaul(NONCE, expires_epoch=2.0)
    server_connection, peer = socket.socketpair()
    backhaul.connection = server_connection
    backhaul.peer_public_ip = "8.8.4.4"

    assert backhaul.expired(now=1.999) is False
    assert backhaul.expired(now=2.0) is True
    backhaul.revoke()

    assert backhaul.connection is None
    assert backhaul.peer_public_ip == ""
    with pytest.raises((BrokenPipeError, ConnectionResetError, OSError)):
        peer.sendall(b"credential cannot be reused")
    peer.close()


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


def test_private_client_initialization_has_no_public_ip_discovery(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.workbench.leisaac.reverse_client._pod_ipv4",
        lambda: "10.96.34.76",
    )
    client = Client({})
    assert client.media_target == ("10.96.34.76", 3478)
    assert not hasattr(client, "peer_public_ip")


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

        def setsockopt(self, _level: int, _option: int, _value: int) -> None:
            return None

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


def test_private_websocket_heartbeat_aborts_a_half_open_backhaul(
    monkeypatch,
) -> None:
    client_socket, server_socket = socket.socketpair()
    websocket = WebSocketConnection(client_socket)  # type: ignore[arg-type]
    websocket.last_pong = 10.0
    monkeypatch.setattr(reverse_client, "WEBSOCKET_HEARTBEAT_TIMEOUT_SECONDS", 30.0)

    assert websocket._heartbeat_once(now=40.0) is False
    assert websocket.closed.is_set()
    assert server_socket.recv(1) == b""
    server_socket.close()
