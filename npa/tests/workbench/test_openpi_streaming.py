from __future__ import annotations

import base64
from collections import deque
import hashlib
import socket
import struct
import threading
import time
from typing import Mapping

import numpy as np
import pytest

from npa.workbench.antioch.openpi_bridge import (
    OpenPIWebsocketClient,
    RetryPolicy,
    pack_message,
)
from npa.workbench.antioch.openpi_streaming import (
    StreamingConfig,
    StreamingPolicyLoop,
)


SAFE_ROW = np.asarray([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 0.5])
SAFE_ACTIONS = np.tile(SAFE_ROW, (15, 1))
CURRENT_JOINTS = SAFE_ROW[:7].copy()


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("connection closed")
        payload.extend(chunk)
    return bytes(payload)


def _recv_websocket_frame(connection: socket.socket) -> tuple[int, bytes]:
    first, second = _recv_exact(connection, 2)
    opcode = first & 0x0F
    size = second & 0x7F
    if size == 126:
        size = struct.unpack("!H", _recv_exact(connection, 2))[0]
    elif size == 127:
        size = struct.unpack("!Q", _recv_exact(connection, 8))[0]
    mask = _recv_exact(connection, 4) if second & 0x80 else b""
    payload = _recv_exact(connection, size)
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


def _send_websocket_frame(
    connection: socket.socket, opcode: int, payload: bytes
) -> None:
    if len(payload) < 126:
        header = bytes([0x80 | opcode, len(payload)])
    elif len(payload) < 65536:
        header = bytes([0x80 | opcode, 126]) + struct.pack("!H", len(payload))
    else:
        header = bytes([0x80 | opcode, 127]) + struct.pack("!Q", len(payload))
    connection.sendall(header + payload)


class ControllableFakePolicyServer:
    """Minimal real WebSocket server for latency and fault integration tests."""

    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = deque(outcomes)
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.1)
        self.port = int(self.listener.getsockname()[1])
        self.stop_event = threading.Event()
        self.delayed_request = threading.Event()
        self.requests = 0
        self.in_flight = 0
        self.maximum_in_flight = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.listener.close()
        self.thread.join(2)

    def _handshake(self, connection: socket.socket) -> None:
        request = bytearray()
        while b"\r\n\r\n" not in request:
            request.extend(connection.recv(4096))
        headers = {}
        for line in bytes(request).decode("ascii").split("\r\n")[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.lower()] = value.strip()
        accept = base64.b64encode(
            hashlib.sha1(  # noqa: S324 - required by RFC 6455 handshake
                (
                    headers["sec-websocket-key"]
                    + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                ).encode()
            ).digest()
        ).decode()
        connection.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode("ascii")
        )
        _send_websocket_frame(connection, 2, pack_message({"model": "fake-pi0.5"}))

    def _serve_connection(self, connection: socket.socket) -> None:
        self._handshake(connection)
        while not self.stop_event.is_set():
            opcode, payload = _recv_websocket_frame(connection)
            if opcode == 8:
                return
            if opcode == 9:
                _send_websocket_frame(connection, 10, payload)
                continue
            if opcode != 2:
                continue
            self.requests += 1
            self.in_flight += 1
            self.maximum_in_flight = max(self.maximum_in_flight, self.in_flight)
            outcome = self.outcomes.popleft() if self.outcomes else "valid"
            try:
                if outcome == "delay":
                    self.delayed_request.set()
                    time.sleep(0.08)
                elif outcome == "disconnect":
                    return
                response = (
                    {"actions": np.zeros((3, 8))}
                    if outcome == "malformed"
                    else {"actions": SAFE_ACTIONS}
                )
                _send_websocket_frame(connection, 2, pack_message(response))
            finally:
                self.in_flight -= 1

    def _serve(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, _address = self.listener.accept()
            except (OSError, TimeoutError):
                continue
            connection.settimeout(1)
            try:
                with connection:
                    self._serve_connection(connection)
            except (ConnectionError, OSError, TimeoutError):
                continue


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ScriptedTransport:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = deque(outcomes or [SAFE_ACTIONS])
        self.generation = 1
        self.reconnect_count = 0
        self.pings = 0
        self.calls = 0
        self.closed = False

    def ping(self) -> None:
        self.pings += 1

    def infer(self, _observation: Mapping[str, object]) -> np.ndarray:
        self.calls += 1
        outcome = self.outcomes.popleft() if self.outcomes else SAFE_ACTIONS
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return np.asarray(outcome)

    def close(self) -> None:
        self.closed = True


def _config(**overrides: object) -> StreamingConfig:
    values: dict[str, object] = {
        "policy_request_hz": 10.0,
        "minimum_ready_seconds": 1.0,
        "minimum_ready_cycles": 2,
        "reconnect_initial_backoff_seconds": 0.01,
        "reconnect_maximum_backoff_seconds": 0.02,
    }
    values.update(overrides)
    return StreamingConfig(**values)  # type: ignore[arg-type]


def _observation(sequence: int = 0) -> dict[str, object]:
    return {"frame_sequence": sequence}


def test_steady_state_streaming_and_metrics_are_sustained() -> None:
    clock = FakeClock()
    transport = ScriptedTransport([SAFE_ACTIONS, SAFE_ACTIONS])
    loop = StreamingPolicyLoop(transport, config=_config(), clock=clock)

    for sequence in range(2):
        loop.record_render_tick()
        loop.publish_observation(_observation(sequence))
        assert loop.policy_cycle()
        decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
        assert decision.source == "policy"
        loop.mark_applied(decision)
        clock.advance(0.6)

    metrics = loop.metrics_snapshot()
    assert metrics["ready"] is True
    assert metrics["policy_round_trips"] == 2
    assert metrics["safely_applied_targets"] == 2
    assert metrics["latest_observation_sequence"] == 2
    assert metrics["observation_fps"] > 0
    assert metrics["control_step_fps"] > 0
    assert metrics["render_fps"] > 0


def test_latest_observation_replacement_keeps_memory_bounded() -> None:
    loop = StreamingPolicyLoop(ScriptedTransport(), config=_config())
    for sequence in range(100):
        loop.publish_observation(
            _observation(sequence), monotonic_seconds=float(sequence)
        )
    assert loop.pending_counts() == {"observations": 1, "responses": 0, "actions": 0}
    assert loop.metrics_snapshot()["dropped_observations"] == 99


def test_metric_history_is_bounded_during_long_lived_streaming() -> None:
    clock = FakeClock()
    loop = StreamingPolicyLoop(ScriptedTransport(), config=_config(), clock=clock)
    for sequence in range(2100):
        loop.publish_observation(_observation(sequence))
        assert loop.policy_cycle()
        clock.advance(0.2)
    assert loop.metrics_snapshot()["metric_sample_window"] == {
        "capacity": 2048,
        "inference": 2048,
        "observation_age": 2048,
        "response_age": 2048,
    }


def test_new_chunk_prefetch_supersedes_unexhausted_receding_horizon() -> None:
    clock = FakeClock()
    first = SAFE_ACTIONS.copy()
    second = SAFE_ACTIONS.copy()
    second[:, 0] = 0.04
    loop = StreamingPolicyLoop(
        ScriptedTransport([first, second]),
        config=_config(executed_targets_per_chunk=5),
        clock=clock,
    )
    loop.publish_observation(_observation(1))
    assert loop.policy_cycle()
    first_decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    assert first_decision.target is not None and first_decision.target[0] == 0.0

    clock.advance(0.2)
    loop.publish_observation(_observation(2))
    assert loop.policy_cycle()
    second_decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    assert second_decision.target is not None and second_decision.target[0] == 0.04
    assert loop.pending_counts()["actions"] == 4
    assert loop.metrics_snapshot()["chunks_superseded"] == 1


def test_stale_response_is_rejected_and_safe_hold_is_used() -> None:
    clock = FakeClock()

    def delayed() -> np.ndarray:
        clock.advance(2.0)
        return SAFE_ACTIONS

    loop = StreamingPolicyLoop(
        ScriptedTransport([delayed]),
        config=_config(maximum_response_age_seconds=0.5),
        clock=clock,
    )
    loop.publish_observation(_observation())
    assert loop.policy_cycle() is False
    decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    assert decision.source == "hold"
    assert np.array_equal(decision.target, SAFE_ROW)
    metrics = loop.metrics_snapshot()
    assert metrics["stale_responses"] == 1
    assert metrics["safely_applied_targets"] == 0


def test_timeout_resets_epoch_and_yields_no_action_when_configured() -> None:
    clock = FakeClock()

    def timeout() -> np.ndarray:
        clock.advance(2.0)
        raise TimeoutError

    loop = StreamingPolicyLoop(
        ScriptedTransport([timeout]),
        config=_config(
            inference_deadline_seconds=1.0,
            safe_hold_behavior="no-action",
        ),
        clock=clock,
    )
    loop.publish_observation(_observation())
    assert loop.policy_cycle() is False
    assert loop.epoch == 1
    decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    assert decision.source == "no-action" and decision.target is None
    metrics = loop.metrics_snapshot()
    assert metrics["inference_timeouts"] == 1
    assert metrics["reconnects"] == 1


def test_reconnect_generation_change_rejects_response_and_resets_actions() -> None:
    transport = ScriptedTransport()

    def reconnect() -> np.ndarray:
        transport.generation += 1
        transport.reconnect_count += 1
        return SAFE_ACTIONS

    transport.outcomes = deque([reconnect])
    loop = StreamingPolicyLoop(transport, config=_config())
    loop.publish_observation(_observation())
    assert loop.policy_cycle() is False
    assert loop.epoch == 1
    assert loop.pending_counts() == {"observations": 0, "responses": 0, "actions": 0}
    assert loop.next_control_decision(CURRENT_JOINTS, 0.5).source == "hold"


def test_guarded_application_refuses_a_pre_disconnect_decision() -> None:
    loop = StreamingPolicyLoop(ScriptedTransport(), config=_config())
    loop.publish_observation(_observation())
    assert loop.policy_cycle()
    decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    loop.reset_epoch()
    applied: list[np.ndarray] = []
    with pytest.raises(Exception, match="stale control epoch"):
        loop.apply_if_current(decision, applied.append)
    assert applied == []


@pytest.mark.parametrize(
    "response",
    [np.zeros((3, 8)), np.full((15, 8), np.nan), np.full((15, 8), 99.0)],
)
def test_malformed_and_unsafe_responses_never_reach_control(
    response: np.ndarray,
) -> None:
    loop = StreamingPolicyLoop(ScriptedTransport([response]), config=_config())
    loop.publish_observation(_observation())
    assert loop.policy_cycle() is False
    assert loop.next_control_decision(CURRENT_JOINTS, 0.5).source == "hold"
    metrics = loop.metrics_snapshot()
    assert metrics["malformed_or_unsafe_responses"] == 1
    assert metrics["safely_applied_targets"] == 0


def test_clean_cancellation_stops_worker_and_closes_transport() -> None:
    transport = ScriptedTransport()
    loop = StreamingPolicyLoop(transport, config=_config())
    loop.start()
    assert loop.is_running()
    loop.stop()
    assert not loop.is_running()
    assert transport.closed is True


def test_controllable_fake_policy_latency_does_not_stop_render_ticks() -> None:
    release = threading.Event()
    entered = threading.Event()

    def delayed_server_response() -> np.ndarray:
        entered.set()
        assert release.wait(2)
        return SAFE_ACTIONS

    transport = ScriptedTransport([delayed_server_response])
    loop = StreamingPolicyLoop(transport, config=_config())
    loop.start()
    loop.publish_observation(_observation())
    assert entered.wait(1)
    for _ in range(25):
        loop.record_render_tick()
        time.sleep(0.001)
    assert loop.metrics_snapshot()["render_ticks"] == 25
    release.set()
    deadline = time.monotonic() + 2
    while loop.metrics_snapshot()["policy_round_trips"] == 0:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
    assert decision.source == "policy"
    loop.stop()


def test_real_websocket_faults_do_not_stall_render_or_apply_bad_actions() -> None:
    server = ControllableFakePolicyServer(
        ["delay", "disconnect", "malformed", "valid", "valid"]
    )
    server.start()
    client = OpenPIWebsocketClient(
        "127.0.0.1",
        port=server.port,
        connect_timeout_seconds=0.5,
        inference_timeout_seconds=0.5,
        retry=RetryPolicy(attempts=1),
    )
    loop = StreamingPolicyLoop(
        client,
        config=_config(
            policy_request_hz=30.0,
            maximum_observation_age_seconds=1.0,
            maximum_response_age_seconds=1.0,
            inference_deadline_seconds=0.5,
            ping_interval_seconds=0.02,
        ),
    )
    applied: list[np.ndarray] = []
    loop.start()
    deadline = time.monotonic() + 1.2
    sequence = 0
    try:
        while time.monotonic() < deadline and server.requests < 6:
            sequence += 1
            loop.publish_observation(
                {
                    "observation/exterior_image_1_left": np.zeros(
                        (224, 224, 3), dtype=np.uint8
                    ),
                    "observation/wrist_image_left": np.ones(
                        (224, 224, 3), dtype=np.uint8
                    ),
                    "observation/joint_position": CURRENT_JOINTS.astype(np.float32),
                    "observation/gripper_position": np.asarray([0.5], dtype=np.float32),
                    "prompt": "pick up the fork",
                }
            )
            loop.record_render_tick()
            decision = loop.next_control_decision(CURRENT_JOINTS, 0.5)
            if decision.source == "policy":
                loop.apply_if_current(decision, applied.append)
            time.sleep(0.005)
        assert server.delayed_request.is_set()
    finally:
        loop.stop()
        server.stop()
    metrics = loop.metrics_snapshot()
    assert metrics["render_ticks"] > metrics["policy_requests"]
    assert metrics["policy_round_trips"] >= 2
    assert metrics["transport_failures"] >= 2
    assert metrics["reconnects"] >= 2
    assert metrics["rejected_responses"] >= 2
    assert metrics["safely_applied_targets"] == len(applied) >= 1
    assert server.maximum_in_flight == 1


def test_metrics_emit_no_observation_payload_or_transport_identity() -> None:
    loop = StreamingPolicyLoop(ScriptedTransport(), config=_config())
    loop.publish_observation(
        {"prompt": "sensitive", "image": b"private", "endpoint": "private"}
    )
    metrics = loop.metrics_snapshot()
    serialized = repr(metrics).lower()
    assert "sensitive" not in serialized
    assert "private" not in serialized
    assert "prompt" not in serialized
    assert "endpoint" not in serialized
