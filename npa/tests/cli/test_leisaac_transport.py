"""Deterministic protocol and runtime tests for low-latency LeIsaac transport."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from npa.agent_backend.leisaac_routes import (
    _client_address,
    _mint_ws_session,
    _same_origin_websocket,
    _valid_ws_session,
)
from npa.agent_backend.leisaac_transport import (
    AsyncFrameCreditWindow,
    AsyncLatestByKey,
    AsyncLatestValue,
    CONTROL_SUBPROTOCOL,
    ControlLedger,
    FrameEnvelope,
    MAX_CONTROL_MESSAGE_BYTES,
    TransportMetrics,
    TransportProtocolError,
    VIDEO_SUBPROTOCOL,
    pack_frame,
    parse_control_message,
    parse_video_ack,
    stamp_agent_frame,
    stamp_verified_frame,
    unpack_frame,
)

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "leisaac-transport-test"
NONCE = "n" * 64


def _control(seq: int = 1, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "v": 1,
        "type": "control",
        "run_id": RUN_ID,
        "client_id": "browser-test",
        "seq": seq,
        "key": "W",
        "event": "press",
        "client_mono_ns": 100 + seq,
        "client_wall_ns": 200 + seq,
    }
    payload.update(overrides)
    return payload


def _runtime_module():
    path = ROOT / "npa/docker/workbench/leisaac/session_server.py"
    spec = importlib.util.spec_from_file_location(
        f"npa_leisaac_transport_{id(path)}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_messages_are_bounded_and_exactly_scoped() -> None:
    parsed = parse_control_message(json.dumps(_control()), expected_run_id=RUN_ID)
    assert parsed["key"] == "W"
    assert parsed["seq"] == 1

    for override, code in (
        ({"run_id": "other"}, "run_mismatch"),
        ({"key": "R"}, "invalid_message"),
        ({"seq": -1}, "invalid_message"),
        ({"type": "unknown"}, "invalid_message"),
    ):
        with pytest.raises(TransportProtocolError) as exc_info:
            parse_control_message(
                json.dumps(_control(**override)), expected_run_id=RUN_ID
            )
        assert exc_info.value.code == code

    with pytest.raises(TransportProtocolError, match="size"):
        parse_control_message(
            b"{" + b"x" * MAX_CONTROL_MESSAGE_BYTES, expected_run_id=RUN_ID
        )


def test_control_ledger_is_ordered_idempotent_and_recovers_state() -> None:
    ledger = ControlLedger(history_limit=2)
    first = _control()
    accepted, queued = ledger.accept(first, received_mono_ns=301, received_wall_ns=401)
    assert accepted["phase"] == "accepted"
    assert queued is not None and queued["seq"] == 1
    assert ledger.keys_down("browser-test") == ("W",)

    duplicate, duplicate_queue = ledger.accept(first)
    assert duplicate["duplicate"] is True
    assert duplicate_queue is None

    with pytest.raises(TransportProtocolError) as reused:
        ledger.accept(_control(event="release"))
    assert reused.value.code == "sequence_reused"
    with pytest.raises(TransportProtocolError) as gap:
        ledger.accept(_control(3))
    assert gap.value.code == "out_of_order"
    assert gap.value.expected_seq == 2

    applied = {
        "client_id": "browser-test",
        "seq": 1,
        "simulator_applied_mono_ns": "501",
        "simulator_applied_wall_ns": "601",
        "simulator_step": 7,
    }
    assert ledger.mark_applied(applied) == applied
    assert ledger.applied("browser-test", 1) == applied
    resume = ledger.resume("browser-test")
    assert resume["next_seq"] == 2
    assert resume["last_applied_seq"] == 1
    assert resume["keys_down"] == ["W"]

    ledger.accept(_control(2, event="release"))
    ledger.accept(_control(3, key="A"))
    assert ledger.keys_down("browser-test") == ("A",)
    with pytest.raises(TransportProtocolError) as stale:
        ledger.accept(first)
    assert stale.value.code == "sequence_too_old"


def test_direct_so101_actions_share_ordering_and_reject_unsafe_values() -> None:
    action = {
        "v": 1,
        "type": "action",
        "run_id": RUN_ID,
        "client_id": "custom-device",
        "seq": 1,
        "device": "custom-so101",
        "action": [0.1, -0.2, 0.0, 0.3, 0.0, 0.0, -0.4, 1.0],
        "client_mono_ns": 101,
        "client_wall_ns": 201,
    }
    parsed = parse_control_message(json.dumps(action), expected_run_id=RUN_ID)
    assert parsed["action"] == pytest.approx(action["action"])
    ledger = ControlLedger()
    accepted, queued = ledger.accept(parsed)
    assert accepted["device"] == "custom-so101"
    assert queued is not None and queued["type"] == "action"
    assert ledger.keys_down("custom-device") == ()
    duplicate, duplicate_queue = ledger.accept(parsed)
    assert duplicate["duplicate"] is True and duplicate_queue is None

    for invalid in (
        {**action, "action": [0.0] * 7},
        {**action, "action": [0.0] * 7 + [1.01]},
        {**action, "action": [0.0] * 7 + [float("nan")]},
        {**action, "device": "untrusted-script"},
        {**action, "unexpected": True},
    ):
        with pytest.raises(TransportProtocolError, match="direct action"):
            parse_control_message(json.dumps(invalid), expected_run_id=RUN_ID)


def test_binary_frame_envelope_round_trips_and_detects_tampering() -> None:
    jpeg = b"\xff\xd8" + b"frame-data" * 20 + b"\xff\xd9"
    envelope = FrameEnvelope(
        sequence=9,
        capture_wall_ns=100,
        capture_monotonic_ns=101,
        encoded_wall_ns=102,
        encoded_monotonic_ns=103,
        runtime_send_monotonic_ns=104,
        causal_action_sequence=7,
        causal_applied_monotonic_ns=99,
        dropped_before=2,
    )
    packed = pack_frame(envelope, jpeg)
    decoded, content = unpack_frame(packed)
    assert content == jpeg
    assert decoded.sequence == 9
    assert decoded.causal_action_sequence == 7
    assert decoded.causal_applied_monotonic_ns == 99
    assert decoded.dropped_before == 2
    assert decoded.sha256 == hashlib.sha256(jpeg).digest()

    stamped = stamp_agent_frame(
        packed, received_mono_ns=105, send_mono_ns=106, additional_dropped=3
    )
    decoded, _content = unpack_frame(stamped)
    assert decoded.agent_receive_monotonic_ns == 105
    assert decoded.agent_send_monotonic_ns == 106
    assert decoded.dropped_before == 5

    restamped = stamp_verified_frame(
        decoded, content, received_mono_ns=107, send_mono_ns=108
    )
    restamped_envelope, restamped_content = unpack_frame(restamped)
    assert restamped_content == content
    assert restamped_envelope.agent_receive_monotonic_ns == 107

    tampered = bytearray(stamped)
    tampered[-2] ^= 0x01
    with pytest.raises(TransportProtocolError, match="digest"):
        unpack_frame(bytes(tampered))


def test_binary_frame_envelope_accepts_v1_as_zero_causal_compatibility() -> None:
    jpeg = b"\xff\xd8legacy\xff\xd9"
    legacy = struct.Struct("!4sBBHQQQQQQQQII32s").pack(
        b"NPAF",
        1,
        0,
        112,
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        len(jpeg),
        2,
        hashlib.sha256(jpeg).digest(),
    ) + jpeg

    envelope, content = unpack_frame(legacy)

    assert content == jpeg
    assert envelope.sequence == 3
    assert envelope.causal_action_sequence == 0
    assert envelope.causal_applied_monotonic_ns == 0
    assert envelope.dropped_before == 2


def test_verified_relay_stamps_a_frame_without_hashing_the_jpeg_twice(
    monkeypatch,
) -> None:
    import npa.agent_backend.leisaac_transport as transport

    jpeg = b"\xff\xd8" + b"large-frame" * 300_000 + b"\xff\xd9"
    packed = pack_frame(
        FrameEnvelope(
            sequence=1,
            capture_wall_ns=2,
            capture_monotonic_ns=3,
            encoded_wall_ns=4,
            encoded_monotonic_ns=5,
            runtime_send_monotonic_ns=6,
        ),
        jpeg,
    )
    real_sha256 = transport.hashlib.sha256
    calls = 0

    def counted_sha256(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(transport.hashlib, "sha256", counted_sha256)
    envelope, verified_jpeg = unpack_frame(packed, verify_digest=True)
    stamped = stamp_verified_frame(
        envelope,
        verified_jpeg,
        received_mono_ns=7,
        send_mono_ns=8,
    )

    assert calls == 1
    assert stamped.endswith(jpeg)


def test_video_receipt_ack_is_bounded_exact_and_run_scoped() -> None:
    acknowledgement = parse_video_ack(
        json.dumps({"v": 1, "type": "frame-ack", "run_id": RUN_ID, "sequence": 17}),
        expected_run_id=RUN_ID,
    )
    assert acknowledgement["sequence"] == 17

    with pytest.raises(TransportProtocolError, match="run ID"):
        parse_video_ack(
            json.dumps(
                {"v": 1, "type": "frame-ack", "run_id": "other", "sequence": 17}
            ),
            expected_run_id=RUN_ID,
        )
    with pytest.raises(TransportProtocolError, match="invalid video"):
        parse_video_ack(
            json.dumps(
                {
                    "v": 1,
                    "type": "frame-ack",
                    "run_id": RUN_ID,
                    "sequence": 17,
                    "unexpected": True,
                }
            ),
            expected_run_id=RUN_ID,
        )
    with pytest.raises(TransportProtocolError, match="size"):
        parse_video_ack("x" * 513, expected_run_id=RUN_ID)


@pytest.mark.anyio
async def test_video_credit_window_pipelines_but_never_grows_unbounded() -> None:
    window = AsyncFrameCreditWindow(limit=2)
    assert await window.reserve(7) == 1
    assert await window.reserve(8) == 2
    assert window.high_water == 2

    blocked = asyncio.create_task(window.reserve(9))
    await asyncio.sleep(0)
    assert not blocked.done()
    with pytest.raises(TransportProtocolError) as mismatch:
        window.acknowledge(8)
    assert mismatch.value.code == "out_of_order"
    assert mismatch.value.expected_seq == 7

    assert window.acknowledge(7) == 1
    assert await asyncio.wait_for(blocked, timeout=0.1) == 2
    assert window.acknowledge(8) == 1
    assert window.acknowledge(9) == 0
    with pytest.raises(TransportProtocolError, match="no in-flight"):
        window.acknowledge(9)


@pytest.mark.anyio
async def test_latest_frame_wins_for_a_slow_consumer() -> None:
    latest = AsyncLatestValue()
    await latest.publish("frame-1")
    await latest.publish("frame-2")
    generation, value, skipped = await latest.wait_after(0, timeout=0.1)
    assert (generation, value, skipped) == (2, "frame-2", 1)

    waiter = asyncio.create_task(latest.wait_after(generation, timeout=0.1))
    await asyncio.sleep(0)
    await latest.publish("frame-3")
    assert await waiter == (3, "frame-3", 0)
    with pytest.raises(asyncio.TimeoutError):
        await latest.wait_after(3, timeout=0.001)


@pytest.mark.anyio
async def test_camera_latest_values_are_bounded_and_serviced_fairly() -> None:
    latest = AsyncLatestByKey(("workspace", "overview"))
    await latest.publish("workspace", "workspace-1")
    await latest.publish("overview", "overview-1")
    await latest.publish("workspace", "workspace-2")

    generations: dict[str, int] = {}
    camera, generation, value, skipped, next_index = await latest.wait_after(
        generations, timeout=0.1
    )
    assert (camera, value, skipped) == (
        "workspace",
        "workspace-2",
        1,
    )
    generations[camera] = generation
    camera, generation, value, skipped, next_index = await latest.wait_after(
        generations, next_index=next_index, timeout=0.1
    )
    assert (camera, value, skipped) == ("overview", "overview-1", 0)
    generations[camera] = generation

    await latest.publish("overview", "overview-2")
    await latest.publish("overview", "overview-3")
    camera, generation, value, skipped, next_index = await latest.wait_after(
        generations, next_index=next_index, timeout=0.1
    )
    assert (camera, value, skipped) == ("overview", "overview-3", 1)
    generations[camera] = generation
    with pytest.raises(asyncio.TimeoutError):
        await latest.wait_after(generations, next_index=next_index, timeout=0.001)


@pytest.mark.anyio
async def test_camera_latest_values_prefer_primary_once_without_starvation() -> None:
    latest = AsyncLatestByKey(("workspace", "overview"))
    await latest.publish("overview", "overview-1")
    await latest.publish("workspace", "workspace-1")
    generations: dict[str, int] = {}

    camera, generation, value, skipped, next_index = await latest.wait_after(
        generations,
        next_index=1,
        preferred_key="workspace",
        timeout=0.1,
    )
    assert (camera, generation, value, skipped) == (
        "workspace",
        1,
        "workspace-1",
        0,
    )
    generations[camera] = generation

    camera, generation, value, skipped, _ = await latest.wait_after(
        generations,
        next_index=next_index,
        preferred_key="workspace",
        timeout=0.1,
    )
    assert (camera, generation, value, skipped) == (
        "overview",
        1,
        "overview-1",
        0,
    )


def test_transport_metrics_are_low_cardinality() -> None:
    metrics = TransportMetrics()
    metrics.increment("frames_sent", 2)
    assert metrics.snapshot()["frames_sent"] == 2
    metrics.increment("frames_relay_acked")
    metrics.increment("frames_browser_acked")
    assert metrics.snapshot()["frames_relay_acked"] == 1
    assert metrics.snapshot()["frames_browser_acked"] == 1
    with pytest.raises(ValueError):
        metrics.increment("run-id-as-a-label")


@pytest.mark.parametrize(
    "headers,allowed",
    [
        (
            {
                "x-forwarded-proto": "https",
                "origin": "https://agent.example",
                "host": "agent.example",
                "sec-websocket-protocol": CONTROL_SUBPROTOCOL,
            },
            True,
        ),
        (
            {
                "x-forwarded-proto": "https",
                "origin": "https://evil.example",
                "host": "agent.example",
                "sec-websocket-protocol": CONTROL_SUBPROTOCOL,
            },
            False,
        ),
        (
            {
                "x-forwarded-proto": "http",
                "origin": "https://agent.example",
                "host": "agent.example",
                "sec-websocket-protocol": CONTROL_SUBPROTOCOL,
            },
            False,
        ),
        (
            {
                "x-forwarded-proto": "https",
                "origin": "https://agent.example",
                "host": "agent.example",
                "sec-websocket-protocol": f"{CONTROL_SUBPROTOCOL}, extra",
            },
            False,
        ),
    ],
)
def test_public_websocket_requires_exact_origin_and_subprotocol(
    headers, allowed
) -> None:
    websocket = SimpleNamespace(headers=headers)
    assert _same_origin_websocket(websocket, CONTROL_SUBPROTOCOL) is allowed


def test_short_lived_ws_session_is_signed_and_bound_to_run_address_and_time() -> None:
    secret = b"deterministic-test-secret"
    token = _mint_ws_session(secret, RUN_ID, "8.8.8.8", "control", now=1_000)

    assert _valid_ws_session(secret, token, RUN_ID, "8.8.8.8", "control", now=1_000)
    assert _valid_ws_session(secret, token, RUN_ID, "8.8.8.8", "control", now=1_120)
    assert not _valid_ws_session(
        secret, token, "other-run", "8.8.8.8", "control", now=1_000
    )
    assert not _valid_ws_session(secret, token, RUN_ID, "8.8.4.4", "control", now=1_000)
    assert not _valid_ws_session(secret, token, RUN_ID, "8.8.8.8", "video", now=1_000)
    assert not _valid_ws_session(secret, token, RUN_ID, "8.8.8.8", "control", now=1_121)
    assert not _valid_ws_session(
        secret,
        token[:-1] + ("A" if token[-1] != "A" else "B"),
        RUN_ID,
        "8.8.8.8",
        "control",
        now=1_000,
    )

    consumed: dict[str, int] = {}
    assert _valid_ws_session(
        secret,
        token,
        RUN_ID,
        "8.8.8.8",
        "control",
        now=1_000,
        consumed_nonces=consumed,
    )
    assert not _valid_ws_session(
        secret,
        token,
        RUN_ID,
        "8.8.8.8",
        "control",
        now=1_000,
        consumed_nonces=consumed,
    )
    reconnect = _mint_ws_session(secret, RUN_ID, "8.8.8.8", "control", now=1_001)
    assert _valid_ws_session(
        secret,
        reconnect,
        RUN_ID,
        "8.8.8.8",
        "control",
        now=1_001,
        consumed_nonces=consumed,
    )


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"x-real-ip": "8.8.8.8"}, "8.8.8.8"),
        ({}, ""),
        ({"x-real-ip": "invalid"}, ""),
        ({"x-real-ip": "127.0.0.1"}, ""),
        ({"x-real-ip": "10.0.0.2"}, ""),
    ],
)
def test_client_address_requires_nginx_attested_public_ip(headers, expected) -> None:
    assert _client_address(headers, SimpleNamespace(host="127.0.0.1")) == expected


def _prepare_runtime(monkeypatch, tmp_path: Path):
    runtime = _runtime_module()
    paths = {
        "INPUT_COUNTER_PATH": tmp_path / "input-count",
        "APPLIED_COUNTER_PATH": tmp_path / "applied-count",
        "INPUT_QUEUE_PATH": tmp_path / "input.jsonl",
        "FRAME_PATH": tmp_path / "frame.jpg",
        "FRAME_META_PATH": tmp_path / "frame.json",
        "SECONDARY_FRAME_PATH": tmp_path / "frame-overview.jpg",
        "SECONDARY_FRAME_META_PATH": tmp_path / "frame-overview.json",
        "VIEW_COMMAND_PATH": tmp_path / "view-command.json",
        "APPLIED_ACK_PATH": tmp_path / "applied.jsonl",
        "RECORDER_ROOT": tmp_path / "recorder",
        "RECORDER_STATUS_PATH": tmp_path / "recorder/status.json",
        "RECORDER_CONTROL_PATH": tmp_path / "recorder/control.jsonl",
        "RECORDER_PENDING_PATH": tmp_path / "recorder/pending.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runtime, name, path)
    monkeypatch.setattr(
        runtime,
        "CAMERA_PATHS",
        {
            "workspace": (paths["FRAME_PATH"], paths["FRAME_META_PATH"]),
            "overview": (
                paths["SECONDARY_FRAME_PATH"],
                paths["SECONDARY_FRAME_META_PATH"],
            ),
        },
    )
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", RUN_ID)
    monkeypatch.setenv("NPA_LEISAAC_SESSION_NONCE", NONCE)
    runtime.STATE.update(state="ready", detail="ready", webrtc_ready=True, pid=123)
    runtime.CONTROL_LEDGER = ControlLedger()
    runtime.TRANSPORT_METRICS = TransportMetrics()
    runtime.FRAME_LATEST = AsyncLatestByKey(("workspace", "overview"))
    runtime.APPLIED_ACK_OFFSET = 0
    return runtime


def _runtime_headers() -> dict[str, str]:
    return {
        "x-npa-leisaac-nonce": NONCE,
        "x-npa-leisaac-run-id": RUN_ID,
    }


def test_runtime_retries_an_unexpected_clean_simulator_exit(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    launches = 0

    class StopEvent:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def set(self) -> None:
            self.stopped = True

        def wait(self, _timeout: float) -> bool:
            return self.stopped

    stop = StopEvent()

    class Child:
        pid = 123
        returncode = 0

        def poll(self) -> int:
            return 0

    def popen(*_args, **_kwargs):
        nonlocal launches
        launches += 1
        if launches == 2:
            stop.set()
        return Child()

    monkeypatch.setattr(runtime, "SERVER_STOP", stop)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime, "_simulation_launch", lambda: (["leisaac"], {}))
    monkeypatch.setattr(runtime, "detect_gpu", lambda: "RTX test GPU")

    runtime.run_simulation()

    assert launches == 2


def test_runtime_restart_resets_applied_ack_reader_offset(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    runtime.APPLIED_ACK_PATH.write_text('{"seq":1}\n', encoding="utf-8")
    runtime.APPLIED_ACK_OFFSET = 4096

    runtime._reset_runtime_files()

    assert runtime.APPLIED_ACK_OFFSET == 0
    assert runtime.APPLIED_ACK_PATH.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("_stress_iteration", range(25))
def test_runtime_control_ack_ordering_application_and_disconnect_cleanup(
    monkeypatch, tmp_path: Path, _stress_iteration: int
) -> None:
    assert 0 <= _stress_iteration < 25
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    with TestClient(runtime.build_app()) as client:
        with client.websocket_connect(
            "/transport/control",
            headers=_runtime_headers(),
            subprotocols=[CONTROL_SUBPROTOCOL],
        ) as websocket:
            websocket.send_json(
                {
                    "v": 1,
                    "type": "resume",
                    "run_id": RUN_ID,
                    "client_id": "browser-test",
                    "last_acked_seq": 0,
                    "keys_down": [],
                    "client_mono_ns": 1,
                    "client_wall_ns": 2,
                }
            )
            assert websocket.receive_json()["next_seq"] == 1

            websocket.send_json(_control(2))
            error = websocket.receive_json()
            assert error["code"] == "out_of_order"
            assert error["expected_seq"] == 1

            websocket.send_json(_control())
            accepted = websocket.receive_json()
            assert accepted["phase"] == "accepted"
            record = json.loads(runtime.INPUT_QUEUE_PATH.read_text().splitlines()[0])
            runtime.APPLIED_ACK_PATH.write_text(
                json.dumps(
                    {
                        **record,
                        "simulator_applied_mono_ns": "700",
                        "simulator_applied_wall_ns": "800",
                        "simulator_step": 9,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            applied = websocket.receive_json()
            assert applied["phase"] == "applied"
            assert applied["seq"] == 1
            assert applied["simulator_step"] == 9

            websocket.send_json(_control())
            assert websocket.receive_json()["duplicate"] is True
            assert websocket.receive_json()["phase"] == "applied"

            websocket.send_json(_control(2, key="A"))
            assert websocket.receive_json()["phase"] == "accepted"

    records = [
        json.loads(line)
        for line in runtime.INPUT_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["seq"], item["event"]) for item in records] == [
        (1, "press"),
        (2, "press"),
        (3, "release"),
        (4, "release"),
    ]
    assert [item["key"] for item in records] == ["W", "A", "A", "W"]


def test_runtime_disconnect_cleanup_is_idempotent_after_explicit_release_all(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    with TestClient(runtime.build_app()) as client:
        with client.websocket_connect(
            "/transport/control",
            headers=_runtime_headers(),
            subprotocols=[CONTROL_SUBPROTOCOL],
        ) as websocket:
            websocket.send_json(_control())
            assert websocket.receive_json()["phase"] == "accepted"
            websocket.send_json(
                {
                    "v": 1,
                    "type": "release-all",
                    "run_id": RUN_ID,
                    "client_id": "browser-test",
                    "client_mono_ns": 3,
                    "client_wall_ns": 4,
                }
            )
            released = websocket.receive_json()
            assert released["type"] == "released"
            assert released["released_count"] == 1
    records = [
        json.loads(line) for line in runtime.INPUT_QUEUE_PATH.read_text().splitlines()
    ]
    assert [(item["seq"], item["event"]) for item in records] == [
        (1, "press"),
        (2, "release"),
    ]


def test_runtime_abrupt_client_close_durably_releases_every_held_control(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    with TestClient(runtime.build_app()) as client:
        with client.websocket_connect(
            "/transport/control",
            headers=_runtime_headers(),
            subprotocols=[CONTROL_SUBPROTOCOL],
        ) as websocket:
            for seq, key in ((1, "W"), (2, "A"), (3, "D")):
                websocket.send_json(_control(seq, key=key))
                assert websocket.receive_json()["phase"] == "accepted"
            websocket.close(code=1001)

    records = [
        json.loads(line)
        for line in runtime.INPUT_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["seq"], item["key"], item["event"]) for item in records] == [
        (1, "W", "press"),
        (2, "A", "press"),
        (3, "D", "press"),
        (4, "A", "release"),
        (5, "D", "release"),
        (6, "W", "release"),
    ]
    runtime.APPLIED_ACK_PATH.write_text(
        "".join(
            json.dumps(
                {
                    **record,
                    "simulator_applied_mono_ns": str(700 + record["seq"]),
                    "simulator_applied_wall_ns": str(800 + record["seq"]),
                    "simulator_step": 9 + record["seq"],
                }
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    with TestClient(runtime.build_app()) as client:
        with client.websocket_connect(
            "/transport/control",
            headers=_runtime_headers(),
            subprotocols=[CONTROL_SUBPROTOCOL],
        ) as websocket:
            websocket.send_json(
                {
                    "v": 1,
                    "type": "resume",
                    "run_id": RUN_ID,
                    "client_id": "browser-test",
                    "last_acked_seq": 3,
                    "keys_down": ["W", "A", "D"],
                    "client_mono_ns": 10,
                    "client_wall_ns": 11,
                }
            )
            resumed = websocket.receive_json()
            assert resumed["next_seq"] == 7
            assert resumed["last_applied_seq"] == 6
            assert resumed["keys_down"] == []


def test_runtime_reliable_datachannel_uses_shared_ordering_and_disconnect_release(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)

    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.callbacks = {}
            self.sent: list[dict] = []

        def on(self, name: str):
            def register(callback):
                self.callbacks[name] = callback
                return callback

            return register

        def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        def close(self) -> None:
            self.readyState = "closed"
            callback = self.callbacks.get("close")
            if callback:
                callback()

    async def exercise() -> Channel:
        channel = Channel()
        task = asyncio.create_task(runtime._serve_control_datachannel(channel))
        await asyncio.sleep(0)
        channel.callbacks["message"](json.dumps(_control()))
        for _attempt in range(100):
            if channel.sent:
                break
            await asyncio.sleep(0.001)
        assert channel.sent[0]["phase"] == "accepted"
        channel.close()
        with pytest.raises(ConnectionError, match="control channel closed"):
            await task
        return channel

    channel = asyncio.run(exercise())
    assert channel.sent[0]["seq"] == 1
    records = [
        json.loads(line)
        for line in runtime.INPUT_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["seq"], item["event"]) for item in records] == [
        (1, "press"),
        (2, "release"),
    ]


def test_runtime_client_exception_still_waits_for_disconnect_release(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    with TestClient(runtime.build_app()) as client:
        with pytest.raises(RuntimeError, match="client failure"):
            with client.websocket_connect(
                "/transport/control",
                headers=_runtime_headers(),
                subprotocols=[CONTROL_SUBPROTOCOL],
            ) as websocket:
                websocket.send_json(_control())
                assert websocket.receive_json()["phase"] == "accepted"
                raise RuntimeError("client failure")

    records = [
        json.loads(line)
        for line in runtime.INPUT_QUEUE_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert [(item["seq"], item["event"]) for item in records] == [
        (1, "press"),
        (2, "release"),
    ]


@pytest.mark.parametrize(
    "accept_omni,accept_isaac,missing",
    [
        (None, None, {"OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"}),
        ("YES", None, {"ISAACSIM_ACCEPT_EULA"}),
        (None, "YES", {"OMNI_KIT_ACCEPT_EULA"}),
    ],
)
def test_runtime_eula_gate_refuses_unless_both_acceptances_are_explicit(
    monkeypatch, capsys, accept_omni, accept_isaac, missing
) -> None:
    runtime = _runtime_module()
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.delenv("ISAACSIM_ACCEPT_EULA", raising=False)
    if accept_omni is not None:
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", accept_omni)
    if accept_isaac is not None:
        monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", accept_isaac)
    with pytest.raises(SystemExit) as exc_info:
        runtime.require_operator_eula()
    assert exc_info.value.code == 78
    message = capsys.readouterr().err
    assert all(f"{name}=YES" in message for name in missing)
    assert "token" not in message.lower() and "secret" not in message.lower()


def test_runtime_eula_gate_accepts_only_both_explicit_yes_values(monkeypatch) -> None:
    runtime = _runtime_module()
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "YES")
    runtime.require_operator_eula()


def test_runtime_rejects_bad_auth_and_preserves_polling_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    with TestClient(runtime.build_app()) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/transport/control",
                headers={
                    "x-npa-leisaac-nonce": "wrong",
                    "x-npa-leisaac-run-id": RUN_ID,
                },
                subprotocols=[CONTROL_SUBPROTOCOL],
            ):
                pass
        assert exc_info.value.code == 1008

        fallback = client.post(
            "/input",
            headers={"x-npa-leisaac-nonce": NONCE},
            json={"key": "A", "event": "press"},
        )
        assert fallback.status_code == 202
        assert fallback.json()["phase"] == "accepted"
        assert json.loads(runtime.INPUT_QUEUE_PATH.read_text())["key"] == "A"


def test_runtime_fsyncs_safety_releases_but_not_transient_presses(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    syncs: list[int] = []
    monkeypatch.setattr(runtime.os, "fsync", syncs.append)

    runtime._append_inputs([_control(1, event="press")])
    assert syncs == []
    runtime._append_inputs([_control(2, event="release")])
    assert len(syncs) == 1
    runtime._append_inputs(
        [
            {
                "v": 1,
                "type": "action",
                "run_id": RUN_ID,
                "client_id": "browser-test",
                "seq": 3,
                "device": "browser-gamepad",
                "action": [0.0] * 8,
            }
        ]
    )
    assert len(syncs) == 2


def test_runtime_video_envelope_is_binary_and_nonblank(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    jpeg = b"\xff\xd8" + b"real-frame" * 30 + b"\xff\xd9"
    runtime.FRAME_PATH.write_bytes(jpeg)
    runtime.FRAME_META_PATH.write_text(
        json.dumps(
            {
                "schema": "npa.leisaac.frame.v1",
                "sequence": 4,
                "capture_wall_ns": 100,
                "capture_monotonic_ns": 101,
                "encoded_wall_ns": 102,
                "encoded_monotonic_ns": 103,
                "bytes": len(jpeg),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with TestClient(runtime.build_app()) as client:
        with client.websocket_connect(
            "/transport/video",
            headers=_runtime_headers(),
            subprotocols=[VIDEO_SUBPROTOCOL],
        ) as websocket:
            envelope, content = unpack_frame(websocket.receive_bytes())
            assert envelope.sequence == 4
            assert envelope.flags == 0
            assert envelope.runtime_send_monotonic_ns > 0
            assert content == jpeg
            next_jpeg = b"\xff\xd8" + b"new-frame" * 30 + b"\xff\xd9"
            runtime.FRAME_PATH.write_bytes(next_jpeg)
            next_metadata = {
                "schema": "npa.leisaac.frame.v1",
                "sequence": 5,
                "capture_wall_ns": 200,
                "capture_monotonic_ns": 201,
                "encoded_wall_ns": 202,
                "encoded_monotonic_ns": 203,
                "bytes": len(next_jpeg),
                "sha256": hashlib.sha256(next_jpeg).hexdigest(),
            }
            runtime.FRAME_META_PATH.write_text(
                json.dumps(next_metadata), encoding="utf-8"
            )
            client.portal.call(
                runtime.FRAME_LATEST.publish,
                "workspace",
                ("workspace", next_metadata, next_jpeg),
            )
            # The bounded sliding window permits the next frame without waiting
            # one full runtime-to-relay acknowledgement round trip.
            next_envelope, next_content = unpack_frame(websocket.receive_bytes())
            assert next_envelope.sequence == 5
            assert next_content == next_jpeg
            websocket.send_json(
                {"v": 1, "type": "frame-ack", "run_id": RUN_ID, "sequence": 4}
            )
            websocket.send_json(
                {"v": 1, "type": "frame-ack", "run_id": RUN_ID, "sequence": 5}
            )


def test_runtime_frame_watcher_remains_live_across_both_cameras(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    publications: list[tuple[str, dict, bytes]] = []
    reads = [
        [("workspace", {"sequence": 1}, b"workspace-1")],
        [("overview", {"sequence": 1}, b"overview-1")],
        [("workspace", {"sequence": 2}, b"workspace-2")],
        [("overview", {"sequence": 2}, b"overview-2")],
    ]

    def read(_sequences: dict[str, int], _producers: dict[str, int]):
        return reads.pop(0) if reads else []

    async def publish(_camera, item):
        publications.append(item)

    monkeypatch.setattr(runtime, "_read_new_frames", read)
    monkeypatch.setattr(runtime.FRAME_LATEST, "publish", publish)

    async def exercise() -> None:
        watcher = asyncio.create_task(runtime._watch_frames())
        while len(publications) < 4:
            await asyncio.sleep(0.001)
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher

    asyncio.run(exercise())

    assert [camera for camera, _metadata, _jpeg in publications] == [
        "workspace",
        "overview",
        "workspace",
        "overview",
    ]
    assert runtime.TRANSPORT_METRICS.snapshot()["frames_published"] == 4


def test_runtime_frame_reader_skips_unchanged_jpeg_integrity_work(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    jpeg = b"\xff\xd8" + b"sequence-gated" * 30 + b"\xff\xd9"
    runtime.FRAME_PATH.write_bytes(jpeg)
    runtime.FRAME_META_PATH.write_text(
        json.dumps(
            {
                "sequence": 7,
                "bytes": len(jpeg),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert runtime._read_consistent_frame("workspace", 7) is None
    metadata, content = runtime._read_consistent_frame("workspace", 6)
    assert metadata["sequence"] == 7
    assert content == jpeg


def test_runtime_frame_reader_accepts_sequence_reset_from_new_producer(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    jpeg = b"\xff\xd8" + b"new-producer" * 30 + b"\xff\xd9"
    runtime.FRAME_PATH.write_bytes(jpeg)
    runtime.FRAME_META_PATH.write_text(
        json.dumps(
            {
                "producer_pid": 202,
                "sequence": 1,
                "bytes": len(jpeg),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert runtime._read_consistent_frame("workspace", 99, 202) is None
    metadata, content = runtime._read_consistent_frame("workspace", 99, 101)
    assert metadata["producer_pid"] == 202
    assert metadata["sequence"] == 1
    assert content == jpeg


def test_runtime_detects_only_stale_complete_dual_camera_stream(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    assert runtime._frame_stream_stalled(now=100.0) is False
    runtime.FRAME_PATH.write_bytes(b"workspace")
    assert runtime._frame_stream_stalled(now=100.0) is False
    runtime.SECONDARY_FRAME_PATH.write_bytes(b"overview")
    frame_time = runtime.FRAME_PATH.stat().st_mtime
    assert runtime._frame_stream_stalled(now=frame_time + 29) is False
    assert runtime._frame_stream_stalled(now=frame_time + 31) is True


def test_runtime_secondary_camera_and_orbit_are_bounded_and_authenticated(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    jpeg = b"\xff\xd8" + b"distinct-overview" * 30 + b"\xff\xd9"
    runtime.SECONDARY_FRAME_PATH.write_bytes(jpeg)
    runtime.SECONDARY_FRAME_META_PATH.write_text(
        json.dumps(
            {
                "schema": "npa.leisaac.frame.v1",
                "camera": "overview",
                "sequence": 8,
                "capture_wall_ns": 200,
                "capture_monotonic_ns": 201,
                "encoded_wall_ns": 202,
                "encoded_monotonic_ns": 203,
                "bytes": len(jpeg),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with TestClient(runtime.build_app()) as client:
        forbidden = client.post(
            "/view",
            json={
                "camera": "overview",
                "sequence": 1,
                "yaw_delta": 0.1,
                "pitch_delta": 0.1,
                "distance_delta": 0,
            },
        )
        assert forbidden.status_code == 403
        invalid = client.post(
            "/view",
            headers={"x-npa-leisaac-nonce": NONCE},
            json={
                "camera": "overview",
                "sequence": 1,
                "yaw_delta": 4,
                "pitch_delta": 0,
                "distance_delta": 0,
            },
        )
        assert invalid.status_code == 400
        accepted = client.post(
            "/view",
            headers={"x-npa-leisaac-nonce": NONCE},
            json={
                "camera": "overview",
                "sequence": 2,
                "yaw_delta": 0.1,
                "pitch_delta": -0.2,
                "distance_delta": 0.3,
            },
        )
        assert accepted.status_code == 202
        assert json.loads(runtime.VIEW_COMMAND_PATH.read_text())["sequence"] == 2
        frame = client.get(
            "/frame.jpg?camera=overview",
            headers={"x-npa-leisaac-nonce": NONCE},
        )
        assert frame.status_code == 200
        assert frame.headers["x-npa-camera"] == "overview"
        assert frame.content == jpeg
