"""Deterministic protocol and runtime tests for low-latency LeIsaac transport."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
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
    DEFAULT_RECORDING_CAMERA_MODE,
    DEFAULT_VIEW_MODE,
    FrameEnvelope,
    MAX_CONTROL_MESSAGE_BYTES,
    TransportMetrics,
    TransportProtocolError,
    VIDEO_SUBPROTOCOL,
    RecordingCameraMode,
    ViewMode,
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
        ({"unexpected": True}, "invalid_message"),
    ):
        with pytest.raises(TransportProtocolError) as exc_info:
            parse_control_message(
                json.dumps(_control(**override)), expected_run_id=RUN_ID
            )
        assert exc_info.value.code == code


def test_authoritative_view_mode_defaults_and_exact_parser() -> None:
    assert DEFAULT_VIEW_MODE is ViewMode.SINGLE_FAST
    assert DEFAULT_RECORDING_CAMERA_MODE is RecordingCameraMode.PRIMARY_ONLY
    base = {
        "v": 1,
        "type": "view-mode",
        "run_id": RUN_ID,
        "client_id": "browser-test",
        "revision": 7,
        "mode": ViewMode.DUAL_SLOW.value,
        "client_mono_ns": 100,
        "client_wall_ns": 200,
    }
    parsed = parse_control_message(json.dumps(base), expected_run_id=RUN_ID)
    assert parsed["mode"] == ViewMode.DUAL_SLOW.value
    assert parsed["revision"] == 7
    for mutation in (
        {**base, "mode": "fast-ish"},
        {**base, "observer": True},
        {**base, "revision": -1},
    ):
        with pytest.raises(TransportProtocolError, match="invalid"):
            parse_control_message(json.dumps(mutation), expected_run_id=RUN_ID)

    recording = dict(base)
    recording.update(
        type="recording-cameras",
        mode=RecordingCameraMode.PRIMARY_AND_SECONDARY.value,
    )
    assert (
        parse_control_message(json.dumps(recording), expected_run_id=RUN_ID)["mode"]
        == RecordingCameraMode.PRIMARY_AND_SECONDARY.value
    )


def test_single_fast_reader_performs_zero_secondary_work(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime, "_mode_state", lambda: {"applied_view_mode": "single_fast"}
    )
    reads: list[str] = []

    def read(camera: str, *_args, **_kwargs):
        reads.append(camera)
        return ({"sequence": 1}, b"jpeg")

    monkeypatch.setattr(runtime, "_read_consistent_frame", read)
    frames = runtime._read_new_frames({}, {})
    assert reads == ["workspace"]
    assert [camera for camera, _metadata, _jpeg in frames] == ["workspace"]


def test_mode_applied_ack_is_scheduler_bound_and_stale_requests_coalesce(
    monkeypatch,
) -> None:
    runtime = _runtime_module()
    requested = {
        "v": 1,
        "type": "view-mode",
        "run_id": RUN_ID,
        "client_id": "browser-test",
        "revision": 8,
        "mode": "dual_slow",
    }
    snapshots = iter(
        [
            {
                "view_revision": 8,
                "applied_view_revision": 7,
                "applied_view_mode": "single_fast",
            },
            {
                "view_revision": 8,
                "applied_view_revision": 8,
                "applied_view_mode": "dual_slow",
                "mode_transition_latency_ms": 3.5,
            },
        ]
    )
    monkeypatch.setattr(runtime, "_mode_state", lambda: next(snapshots))
    applied = asyncio.run(runtime._wait_for_mode_applied(requested))
    assert applied["phase"] == "applied"
    assert applied["mode_transition_latency_ms"] == 3.5

    monkeypatch.setattr(
        runtime,
        "_mode_state",
        lambda: {
            "view_revision": 9,
            "applied_view_revision": 7,
            "applied_view_mode": "single_fast",
        },
    )
    superseded = asyncio.run(runtime._wait_for_mode_applied(requested))
    assert superseded["phase"] == "superseded"

    with pytest.raises(TransportProtocolError, match="size"):
        parse_control_message(
            b"{" + b"x" * MAX_CONTROL_MESSAGE_BYTES, expected_run_id=RUN_ID
        )


def test_only_one_authenticated_control_transport_owns_mode_changes(
    monkeypatch,
) -> None:
    runtime = _runtime_module()
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", RUN_ID)
    monkeypatch.setattr(runtime, "_queue_mode_request", lambda _message: None)
    runtime.CONTROL_OWNER.update(token="", client_id="", lease_id="")

    async def exercise() -> None:
        first_queue: asyncio.Queue[str] = asyncio.Queue()
        second_queue: asyncio.Queue[str] = asyncio.Queue()
        first_emitted: list[dict[str, object]] = []
        second_emitted: list[dict[str, object]] = []
        first_ready = asyncio.Event()
        second_ready = asyncio.Event()

        async def first_receive() -> str:
            return await first_queue.get()

        async def second_receive() -> str:
            return await second_queue.get()

        async def first_emit(payload: dict[str, object]) -> None:
            first_emitted.append(payload)
            first_ready.set()

        async def second_emit(payload: dict[str, object]) -> None:
            second_emitted.append(payload)
            second_ready.set()

        first = asyncio.create_task(
            runtime._serve_control_protocol(first_receive, first_emit)
        )
        await first_queue.put(
            json.dumps(
                {
                    "v": 1,
                    "type": "resume",
                    "run_id": RUN_ID,
                    "client_id": "active-controller",
                    "last_acked_seq": 0,
                    "client_mono_ns": 1,
                    "client_wall_ns": 2,
                }
            )
        )
        await asyncio.wait_for(first_ready.wait(), timeout=1.0)
        assert first_emitted[0]["type"] == "resumed"

        second = asyncio.create_task(
            runtime._serve_control_protocol(second_receive, second_emit)
        )
        await second_queue.put(
            json.dumps(
                {
                    "v": 1,
                    "type": "view-mode",
                    "run_id": RUN_ID,
                    "client_id": "observer",
                    "revision": 1,
                    "mode": "dual_slow",
                    "client_mono_ns": 3,
                    "client_wall_ns": 4,
                }
            )
        )
        await asyncio.wait_for(second_ready.wait(), timeout=1.0)
        assert second_emitted[0]["code"] == "controller_busy"
        assert runtime.CONTROL_OWNER["client_id"] == "active-controller"
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(exercise())


def test_server_lease_replaces_half_open_transport_and_forged_clock_cannot_steal(
    monkeypatch,
) -> None:
    runtime = _runtime_module()
    monkeypatch.setenv("NPA_LEISAAC_RUN_ID", RUN_ID)
    runtime.CONTROL_OWNER.update(token="", client_id="", lease_id="")
    released: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "_append_inputs", lambda rows: released.extend(rows))

    async def exercise() -> None:
        first_queue: asyncio.Queue[str] = asyncio.Queue()
        replacement_queue: asyncio.Queue[str] = asyncio.Queue()
        first_emitted: list[dict[str, object]] = []
        replacement_emitted: list[dict[str, object]] = []
        first_ready = asyncio.Event()
        replacement_ready = asyncio.Event()

        async def first_receive() -> str:
            return await first_queue.get()

        async def replacement_receive() -> str:
            return await replacement_queue.get()

        async def first_emit(payload: dict[str, object]) -> None:
            first_emitted.append(payload)
            first_ready.set()

        async def replacement_emit(payload: dict[str, object]) -> None:
            replacement_emitted.append(payload)
            replacement_ready.set()

        first = asyncio.create_task(
            runtime._serve_control_protocol(first_receive, first_emit)
        )
        replacement = asyncio.create_task(
            runtime._serve_control_protocol(replacement_receive, replacement_emit)
        )
        resume = {
            "v": 1,
            "type": "resume",
            "run_id": RUN_ID,
            "client_id": "stable-browser-controller",
            "last_acked_seq": 0,
            "client_mono_ns": 1,
            "client_wall_ns": 2,
        }
        await first_queue.put(json.dumps(resume))
        await asyncio.wait_for(first_ready.wait(), timeout=1.0)
        assert first_emitted[0]["type"] == "resumed"
        first_lease = str(first_emitted[0]["lease_id"])

        replacement_resume = dict(resume)
        replacement_resume.update(
            client_mono_ns=3,
            client_wall_ns=2**63 - 1,
            lease_id=first_lease,
        )
        await replacement_queue.put(json.dumps(replacement_resume))
        await asyncio.wait_for(replacement_ready.wait(), timeout=1.0)
        assert replacement_emitted[0]["type"] == "resumed"

        first_ready.clear()
        await first_queue.put(json.dumps(resume))
        await asyncio.wait_for(first_ready.wait(), timeout=1.0)
        assert first_emitted[-1]["code"] == "controller_busy"

        first_ready.clear()
        forged_resume = dict(resume)
        forged_resume.update(
            client_id="forged-second-browser",
            client_wall_ns=2**63 - 1,
        )
        await first_queue.put(json.dumps(forged_resume))
        await asyncio.wait_for(first_ready.wait(), timeout=1.0)
        assert first_emitted[-1]["code"] == "controller_busy"
        assert runtime.CONTROL_OWNER["client_id"] == "stable-browser-controller"

        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        assert runtime.CONTROL_OWNER["client_id"] == "stable-browser-controller"
        assert released == []

        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)
        assert runtime.CONTROL_OWNER["token"] == ""
        assert runtime.CONTROL_OWNER["client_id"] == "stable-browser-controller"
        assert runtime.CONTROL_OWNER["lease_id"]

    asyncio.run(exercise())


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

    assert ledger.reset_for_runtime_restart() == 1
    reset = ledger.resume("browser-test")
    assert reset["next_seq"] == 1
    assert reset["last_applied_seq"] == 0
    assert reset["keys_down"] == []


def test_control_ledger_prunes_idle_lru_clients_without_evicting_active_keys() -> None:
    ledger = ControlLedger(client_limit=2, client_ttl_seconds=1)
    action = {
        "type": "action",
        "device": "browser-gamepad",
        "action": [0.0] * 8,
    }
    ledger.accept(_control(client_id="idle-a", **action), received_mono_ns=1)
    ledger.accept(_control(client_id="active-b", key="A"), received_mono_ns=2)
    ledger.accept(
        _control(client_id="idle-c", **action), received_mono_ns=2_000_000_000
    )
    assert ledger.client_count == 2
    assert ledger.keys_down("active-b") == ("A",)
    ledger.accept(
        _control(client_id="idle-d", **action), received_mono_ns=2_000_000_001
    )
    assert ledger.client_count == 2
    assert ledger.keys_down("active-b") == ("A",)

    fully_active = ControlLedger(client_limit=2)
    fully_active.accept(_control(client_id="active-a", key="A"))
    fully_active.accept(_control(client_id="active-b", key="D"))
    with pytest.raises(TransportProtocolError) as capacity:
        fully_active.accept(_control(client_id="blocked-c", key="W"))
    assert capacity.value.code == "client_capacity"
    assert fully_active.client_count == 2


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
        view_revision=11,
        dropped_before=2,
    )
    packed = pack_frame(envelope, jpeg)
    decoded, content = unpack_frame(packed)
    assert content == jpeg
    assert decoded.sequence == 9
    assert decoded.causal_action_sequence == 7
    assert decoded.causal_applied_monotonic_ns == 99
    assert decoded.view_revision == 11
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
    legacy = (
        struct.Struct("!4sBBHQQQQQQQQII32s").pack(
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
        )
        + jpeg
    )

    envelope, content = unpack_frame(legacy)

    assert content == jpeg
    assert envelope.sequence == 3
    assert envelope.causal_action_sequence == 0
    assert envelope.causal_applied_monotonic_ns == 0
    assert envelope.view_revision == 0
    assert envelope.dropped_before == 2


@pytest.mark.parametrize("length", [8, 111, 112, 127, 135])
def test_binary_frame_claimed_v3_truncation_is_a_protocol_error(length: int) -> None:
    payload = bytearray(max(8, length))
    struct.pack_into("!4sBBH", payload, 0, b"NPAF", 3, 0, 136)
    with pytest.raises(TransportProtocolError) as malformed:
        unpack_frame(bytes(payload))
    assert malformed.value.code == "invalid_frame"
    assert "truncated" in malformed.value.detail


def test_binary_frame_envelope_accepts_v2_as_zero_view_revision() -> None:
    jpeg = b"\xff\xd8v2-frame\xff\xd9"
    legacy = (
        struct.Struct("!4sBBHQQQQQQQQQQII32s").pack(
            b"NPAF",
            2,
            0,
            128,
            4,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            len(jpeg),
            0,
            hashlib.sha256(jpeg).digest(),
        )
        + jpeg
    )

    envelope, content = unpack_frame(legacy)

    assert content == jpeg
    assert envelope.causal_action_sequence == 17
    assert envelope.causal_applied_monotonic_ns == 18
    assert envelope.view_revision == 0


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

    signal = _mint_ws_session(secret, RUN_ID, "8.8.8.8", "signal", now=1_001)
    bounded_signal: dict[str, int | tuple[int, int]] = {}
    for _use in range(2):
        assert _valid_ws_session(
            secret,
            signal,
            RUN_ID,
            "8.8.8.8",
            "signal",
            now=1_001,
            consumed_nonces=bounded_signal,
            max_uses=2,
        )
    assert not _valid_ws_session(
        secret,
        signal,
        RUN_ID,
        "8.8.8.8",
        "signal",
        now=1_001,
        consumed_nonces=bounded_signal,
        max_uses=2,
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
        "MODE_COMMAND_PATH": tmp_path / "mode-command.json",
        "MODE_STATUS_PATH": tmp_path / "mode-status.json",
        "APPLIED_ACK_PATH": tmp_path / "applied.jsonl",
        "IPC_EVENT_PATH": tmp_path / "events.sock",
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
    runtime.MODE_OWNER["client_id"] = ""
    return runtime


def _runtime_headers() -> dict[str, str]:
    return {
        "x-npa-leisaac-nonce": NONCE,
        "x-npa-leisaac-run-id": RUN_ID,
    }


def _http_controller_headers(
    runtime, *, client_id: str = "browser-test", lease_id: str = "a" * 64
) -> dict[str, str]:
    runtime.CONTROL_OWNER.update(
        token="",
        client_id=client_id,
        lease_id=lease_id,
        leased_at_monotonic_ns=1,
    )
    return {
        **_runtime_headers(),
        "x-npa-leisaac-client-id": client_id,
        "x-npa-leisaac-lease-id": lease_id,
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


def test_runtime_restart_retains_only_the_controller_for_mode_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    first = {
        "v": 1,
        "type": "view-mode",
        "run_id": RUN_ID,
        "client_id": "bundle-restart-controller",
        "revision": 4,
        "mode": "dual_slow",
        "client_mono_ns": 1,
        "client_wall_ns": 2,
    }
    assert runtime._queue_mode_request(first) is True
    second = {
        **first,
        "type": "recording-cameras",
        "revision": 3,
        "mode": "primary_and_secondary",
    }
    assert runtime._queue_mode_request(second) is True

    runtime._reset_runtime_files()
    runtime.CONTROL_OWNER.update(token="", client_id="", lease_id="")

    restarted = json.loads(runtime.MODE_COMMAND_PATH.read_text(encoding="utf-8"))
    assert restarted["requested_view_mode"] == "dual_slow"
    assert restarted["view_revision"] == 4
    assert restarted["requested_recording_camera_mode"] == "primary_and_secondary"
    assert restarted["recording_revision"] == 3
    assert restarted["applied_view_mode"] == "single_fast"
    assert restarted["applied_view_revision"] == 0
    assert restarted["applied_recording_camera_mode"] == "primary_only"
    assert restarted["applied_recording_revision"] == 0
    assert restarted["owner_client_id"] == "bundle-restart-controller"

    with TestClient(runtime.build_app()) as client:
        controller_headers = _http_controller_headers(
            runtime, client_id="bundle-restart-controller"
        )
        retained = client.post(
            "/input",
            headers=controller_headers,
            json={
                **first,
                "type": "recording-cameras",
                "revision": 4,
                "mode": "primary_and_secondary",
            },
        )
        assert retained.status_code == 202
        assert retained.json()["phase"] == "accepted"

        observer = client.post(
            "/input",
            headers={
                **controller_headers,
                "x-npa-leisaac-client-id": "observer",
            },
            json={**first, "client_id": "observer", "revision": 5},
        )
        assert observer.status_code == 409
        assert observer.json()["code"] == "controller_busy"


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
            resumed = websocket.receive_json()
            assert resumed["next_seq"] == 1
            assert resumed["view_revision"] == 0
            assert resumed["recording_revision"] == 0

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
    safe_mode = json.loads(runtime.MODE_COMMAND_PATH.read_text(encoding="utf-8"))
    assert safe_mode["requested_view_mode"] == "single_fast"
    assert safe_mode["view_revision"] == 1


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


@pytest.mark.parametrize("value", ["", "N", "no", "0", "FALSE"])
def test_runtime_eula_gate_refuses_recognized_opt_out_before_download(
    monkeypatch, capsys, value
) -> None:
    runtime = _runtime_module()
    monkeypatch.setenv("ACCEPT_EULA", value)
    with pytest.raises(SystemExit) as exc_info:
        runtime.require_operator_eula()
    assert exc_info.value.code == 78
    message = capsys.readouterr().err
    assert "explicitly opts out" in message
    assert "ACCEPT_EULA=Y" in message
    assert "token" not in message.lower() and "secret" not in message.lower()


@pytest.mark.parametrize("value", [None, "Y", "YES", "yes", "1", "TRUE"])
def test_runtime_eula_gate_defaults_and_normalizes_affirmative_values(
    monkeypatch, value
) -> None:
    runtime = _runtime_module()
    if value is None:
        monkeypatch.delenv("ACCEPT_EULA", raising=False)
    else:
        monkeypatch.setenv("ACCEPT_EULA", value)
    runtime.require_operator_eula()
    assert os.environ["ACCEPT_EULA"] == "Y"


def test_runtime_eula_gate_rejects_invalid_value_distinctly(
    monkeypatch, capsys
) -> None:
    runtime = _runtime_module()
    monkeypatch.setenv("ACCEPT_EULA", "maybe")
    with pytest.raises(SystemExit) as exc_info:
        runtime.require_operator_eula()
    assert exc_info.value.code == 78
    message = capsys.readouterr().err
    assert "Invalid ACCEPT_EULA" in message
    assert "Nothing has been downloaded" in message


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

        controller_headers = _http_controller_headers(runtime)
        fallback = client.post(
            "/input",
            headers=controller_headers,
            json={"key": "A", "event": "press"},
        )
        assert fallback.status_code == 202
        assert fallback.json()["phase"] == "accepted"
        assert json.loads(runtime.INPUT_QUEUE_PATH.read_text())["key"] == "A"

        mode_payload = {
            "v": 1,
            "type": "view-mode",
            "run_id": RUN_ID,
            "client_id": "fallback-browser",
            "revision": 9,
            "mode": "dual_slow",
            "client_mono_ns": 300,
            "client_wall_ns": 400,
        }
        mode_headers = _http_controller_headers(
            runtime, client_id="fallback-browser", lease_id="b" * 64
        )
        mode = client.post(
            "/input",
            headers=mode_headers,
            json=mode_payload,
        )
        assert mode.status_code == 202
        assert mode.json() == {
            "v": 1,
            "type": "ack",
            "phase": "accepted",
            "request_type": "view-mode",
            "run_id": RUN_ID,
            "client_id": "fallback-browser",
            "revision": 9,
            "mode": "dual_slow",
        }
        queued = json.loads(runtime.MODE_COMMAND_PATH.read_text(encoding="utf-8"))
        assert queued["requested_view_mode"] == "dual_slow"
        assert queued["view_revision"] == 9
        assert queued["owner_client_id"] == "fallback-browser"

        stale = client.post(
            "/input",
            headers=mode_headers,
            json={**mode_payload, "revision": 8, "mode": "single_fast"},
        )
        assert stale.status_code == 202
        assert stale.json()["phase"] == "superseded"
        still_queued = json.loads(runtime.MODE_COMMAND_PATH.read_text(encoding="utf-8"))
        assert still_queued["requested_view_mode"] == "dual_slow"
        assert still_queued["view_revision"] == 9

        observer = client.post(
            "/input",
            headers={
                **mode_headers,
                "x-npa-leisaac-client-id": "observer",
            },
            json={**mode_payload, "client_id": "observer", "revision": 10},
        )
        assert observer.status_code == 409
        assert observer.json()["code"] == "controller_busy"


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


def test_runtime_batches_counter_reservation_once_per_append(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    reservations: list[int] = []

    def reserve(amount: int) -> int:
        reservations.append(amount)
        return 100 + amount

    monkeypatch.setattr(runtime, "_advance_input_counter", reserve)
    monkeypatch.setattr(runtime.os, "fsync", lambda _fd: None)
    counts = runtime._append_inputs(
        [_control(index, event="release") for index in range(1, 9)]
    )

    assert reservations == [8]
    assert counts == list(range(101, 109))


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
    monkeypatch.setattr(
        runtime, "_mode_state", lambda: {"applied_view_mode": "dual_slow"}
    )
    publications: list[tuple[str, dict, bytes]] = []
    frames = {
        ("workspace", 1): ({"sequence": 1}, b"workspace-1"),
        ("overview", 1): ({"sequence": 1}, b"overview-1"),
        ("workspace", 2): ({"sequence": 2}, b"workspace-2"),
        ("overview", 2): ({"sequence": 2}, b"overview-2"),
    }

    def read(camera, _sequence, _producer, metadata=None, _jpeg=None):
        return frames[(camera, int(metadata["sequence"]))]

    async def publish(_camera, item):
        publications.append(item)

    monkeypatch.setattr(runtime, "_read_new_frames", lambda *_args: [])
    monkeypatch.setattr(runtime, "_read_consistent_frame", read)
    monkeypatch.setattr(runtime.FRAME_LATEST, "publish", publish)

    async def exercise() -> None:
        runtime.RUNTIME_EVENT_QUEUE = asyncio.Queue()
        watcher = asyncio.create_task(runtime._watch_frames())
        for camera, sequence in frames:
            await runtime.RUNTIME_EVENT_QUEUE.put(
                {"type": "frame", "camera": camera, "metadata": {"sequence": sequence}}
            )
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
    metrics = runtime.TRANSPORT_METRICS.snapshot()
    assert metrics["frames_published"] == 4
    assert metrics["workspace_frames_published"] == 2
    assert metrics["overview_frames_published"] == 2


def test_runtime_trusted_ipc_frame_skips_redundant_digest(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    jpeg = b"\xff\xd8" + b"trusted-ipc" * 30 + b"\xff\xd9"
    runtime.FRAME_PATH.write_bytes(jpeg)
    metadata = {
        "producer_pid": 9,
        "sequence": 3,
        "bytes": len(jpeg),
        "sha256": hashlib.sha256(jpeg).hexdigest(),
    }
    monkeypatch.setattr(
        runtime.hashlib,
        "sha256",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected hash")),
    )

    assert runtime._read_consistent_frame(
        "workspace", 2, 9, trusted_metadata=metadata
    ) == (metadata, jpeg)


def test_runtime_applied_ipc_wakes_waiter_without_file_poll(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    acknowledgement = {
        **_control(),
        "simulator_applied_mono_ns": "700",
        "simulator_applied_wall_ns": "800",
        "simulator_step": 9,
    }
    runtime.CONTROL_LEDGER.accept(_control())

    async def exercise() -> None:
        runtime.APPLIED_EVENT = asyncio.Event()
        stop = __import__("threading").Event()
        waiter = asyncio.create_task(
            runtime._wait_for_applied("browser-test", 1, stop, timeout=1.0)
        )
        await asyncio.sleep(0)
        runtime._RuntimeEventProtocol().datagram_received(
            json.dumps(
                {"type": "applied", "acknowledgement": acknowledgement}
            ).encode(),
            None,
        )
        assert await asyncio.wait_for(waiter, timeout=0.1) == acknowledgement

    monkeypatch.setattr(
        runtime,
        "_scan_applied_acks",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected file poll")),
    )
    asyncio.run(exercise())


def test_runtime_lifespan_receives_unix_datagrams_on_uvloop(
    monkeypatch, tmp_path: Path
) -> None:
    uvloop = pytest.importorskip("uvloop")
    runtime = _prepare_runtime(monkeypatch, tmp_path)
    acknowledgement = {
        **_control(),
        "simulator_applied_mono_ns": "700",
        "simulator_applied_wall_ns": "800",
        "simulator_step": 9,
    }
    runtime.CONTROL_LEDGER.accept(_control())

    async def exercise() -> None:
        async with runtime._lifespan(None):
            sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                sender.sendto(
                    json.dumps(
                        {"type": "applied", "acknowledgement": acknowledgement}
                    ).encode(),
                    str(runtime.IPC_EVENT_PATH),
                )
                for _ in range(100):
                    if runtime.CONTROL_LEDGER.applied("browser-test", 1) is not None:
                        break
                    await asyncio.sleep(0.001)
                assert (
                    runtime.CONTROL_LEDGER.applied("browser-test", 1) == acknowledgement
                )
            finally:
                sender.close()

    loop = uvloop.new_event_loop()
    try:
        loop.run_until_complete(exercise())
    finally:
        loop.close()


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
    monkeypatch.setattr(
        runtime, "_mode_state", lambda: {"applied_view_mode": "dual_slow"}
    )
    jpeg = b"\xff\xd8" + b"distinct-overview" * 30 + b"\xff\xd9"
    runtime.SECONDARY_FRAME_PATH.write_bytes(jpeg)
    runtime.SECONDARY_FRAME_META_PATH.write_text(
        json.dumps(
            {
                "schema": "npa.leisaac.frame.v1",
                "camera": "workspace",
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
        controller_headers = _http_controller_headers(runtime)
        forbidden = client.post(
            "/view",
            json={
                "camera": "workspace",
                "client_id": "browser-test",
                "sequence": 1,
                "yaw_delta": 0.1,
                "pitch_delta": 0.1,
                "distance_delta": 0,
            },
        )
        assert forbidden.status_code == 403
        invalid = client.post(
            "/view",
            headers=controller_headers,
            json={
                "camera": "workspace",
                "client_id": "browser-test",
                "sequence": 1,
                "yaw_delta": 4,
                "pitch_delta": 0,
                "distance_delta": 0,
            },
        )
        assert invalid.status_code == 400
        accepted = client.post(
            "/view",
            headers=controller_headers,
            json={
                "camera": "workspace",
                "client_id": "browser-test",
                "sequence": 2,
                "yaw_delta": 0.1,
                "pitch_delta": -0.2,
                "distance_delta": 0.3,
            },
        )
        assert accepted.status_code == 202
        assert json.loads(runtime.VIEW_COMMAND_PATH.read_text())["sequence"] == 2
        observer = client.post(
            "/view",
            headers={
                **controller_headers,
                "x-npa-leisaac-client-id": "observer",
            },
            json={
                "camera": "workspace",
                "client_id": "observer",
                "sequence": 3,
                "yaw_delta": 0.1,
                "pitch_delta": 0,
                "distance_delta": 0,
            },
        )
        assert observer.status_code == 409
        assert observer.json()["code"] == "controller_busy"
        frame = client.get(
            "/frame.jpg?camera=overview",
            headers={"x-npa-leisaac-nonce": NONCE},
        )
        assert frame.status_code == 200
        assert frame.headers["x-npa-camera"] == "overview"
        assert frame.content == jpeg
