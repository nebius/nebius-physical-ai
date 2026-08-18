"""Focused contract tests for the optional low-latency video data channel."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from npa.agent_backend.leisaac_datachannel import (
    CONTROL_DATACHANNEL_LABEL,
    CONTROL_DATACHANNEL_PROTOCOL,
    ControlDataChannelPeerPool,
    MAX_VIDEO_DATACHANNEL_OFFER_BYTES,
    VIDEO_DATACHANNEL_BUFFER_LOW_BYTES,
    VideoDataChannelError,
    VideoDataChannelPeerPool,
    _wait_for_buffer_low,
    parse_video_datachannel_offer,
    require_aiortc_runtime_version,
    valid_control_datachannel,
    valid_video_datachannel,
)
from npa.agent_backend.leisaac_transport import (
    FrameEnvelope,
    TransportMetrics,
    pack_frame,
    unpack_frame,
)


RUN_ID = "leisaac-live-1"
SDP = "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"


def test_datachannel_offer_requires_exact_bounded_run_bound_sdp() -> None:
    assert (
        parse_video_datachannel_offer(
            {"v": 1, "run_id": RUN_ID, "type": "offer", "sdp": SDP},
            expected_run_id=RUN_ID,
        )
        == SDP
    )

    invalid = (
        {"v": 1, "run_id": "other", "type": "offer", "sdp": SDP},
        {"v": 1, "run_id": RUN_ID, "type": "answer", "sdp": SDP},
        {"v": 1, "run_id": RUN_ID, "type": "offer", "sdp": "v=0\r\n"},
        {"v": 1, "run_id": RUN_ID, "type": "offer", "sdp": SDP, "token": "x"},
        {
            "v": 1,
            "run_id": RUN_ID,
            "type": "offer",
            "sdp": SDP + "x" * MAX_VIDEO_DATACHANNEL_OFFER_BYTES,
        },
    )
    for payload in invalid:
        with pytest.raises(VideoDataChannelError, match="invalid WebRTC video offer"):
            parse_video_datachannel_offer(payload, expected_run_id=RUN_ID)


def test_aiortc_runtime_version_is_exact_and_fails_clearly(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.agent_backend.leisaac_datachannel.version", lambda _name: "1.15.0"
    )
    require_aiortc_runtime_version()
    monkeypatch.setattr(
        "npa.agent_backend.leisaac_datachannel.version", lambda _name: "1.16.0"
    )
    with pytest.raises(
        VideoDataChannelError, match=r"requires aiortc 1\.15\.0; found 1\.16\.0"
    ):
        require_aiortc_runtime_version()


def test_datachannel_contract_is_unordered_and_never_retransmits_stale_frames() -> None:
    channel = SimpleNamespace(
        label="npa-leisaac-video",
        protocol="npa.leisaac.video.v1",
        ordered=False,
        maxRetransmits=0,
    )
    assert valid_video_datachannel(channel)
    for override in (
        {"ordered": True},
        {"maxRetransmits": None},
        {"maxRetransmits": 1},
        {"label": "other"},
        {"protocol": "other"},
    ):
        rejected = SimpleNamespace(**{**channel.__dict__, **override})
        assert not valid_video_datachannel(rejected)


def test_control_datachannel_contract_is_reliable_and_ordered() -> None:
    channel = SimpleNamespace(
        label=CONTROL_DATACHANNEL_LABEL,
        protocol=CONTROL_DATACHANNEL_PROTOCOL,
        ordered=True,
        maxRetransmits=None,
        maxPacketLifeTime=None,
    )
    assert valid_control_datachannel(channel)
    for override in (
        {"ordered": False},
        {"maxRetransmits": 0},
        {"maxPacketLifeTime": 100},
        {"label": "other"},
        {"protocol": "other"},
    ):
        rejected = SimpleNamespace(**{**channel.__dict__, **override})
        assert not valid_control_datachannel(rejected)


def test_datachannel_peer_pool_is_bounded() -> None:
    assert VideoDataChannelPeerPool(limit=4).active == 0
    with pytest.raises(ValueError, match="invalid WebRTC video peer limit"):
        VideoDataChannelPeerPool(limit=0)
    with pytest.raises(ValueError, match="invalid WebRTC video peer limit"):
        VideoDataChannelPeerPool(limit=17)


def test_datachannel_relay_is_bounded_and_preserves_causal_stamp() -> None:
    assert VIDEO_DATACHANNEL_BUFFER_LOW_BYTES == 0
    asyncio.run(_assert_datachannel_relay_contract())


def test_control_datachannel_pool_invokes_one_shared_handler() -> None:
    asyncio.run(_assert_control_datachannel_handler())


def test_both_peer_pool_variants_use_shared_negotiation_with_own_validator() -> None:
    asyncio.run(_assert_shared_negotiation_contract())


async def _assert_shared_negotiation_contract() -> None:
    calls: list[tuple[str, bool]] = []

    async def no_frames():
        if False:
            yield b""

    async def exercise_video() -> None:
        pool = VideoDataChannelPeerPool()
        channel = SimpleNamespace(
            label="npa-leisaac-video",
            protocol="npa.leisaac.video.v1",
            ordered=False,
            maxRetransmits=0,
        )

        async def negotiate(**kwargs):
            valid = kwargs["channel_validator"](channel)
            calls.append((kwargs["channel_kind"], valid))
            ready = asyncio.get_running_loop().create_future()
            ready.set_result(channel)
            return object(), ready, "video-answer"

        async def serve(**_kwargs) -> None:
            return None

        pool._negotiate = negotiate  # type: ignore[method-assign]
        pool._serve = serve  # type: ignore[method-assign]
        answer = await pool.create_answer(
            offer_sdp=SDP,
            ice_server=None,
            frame_source=lambda: no_frames(),
            metrics=TransportMetrics(),
        )
        await asyncio.gather(*tuple(pool._tasks))
        assert answer == {"v": 1, "type": "answer", "sdp": "video-answer"}
        assert channel.bufferedAmountLowThreshold == 0

    async def exercise_control() -> None:
        pool = ControlDataChannelPeerPool()
        channel = SimpleNamespace(
            label=CONTROL_DATACHANNEL_LABEL,
            protocol=CONTROL_DATACHANNEL_PROTOCOL,
            ordered=True,
            maxRetransmits=None,
            maxPacketLifeTime=None,
        )

        async def negotiate(**kwargs):
            valid = kwargs["channel_validator"](channel)
            calls.append((kwargs["channel_kind"], valid))
            ready = asyncio.get_running_loop().create_future()
            ready.set_result(channel)
            return object(), ready, "control-answer"

        async def serve_control(**_kwargs) -> None:
            return None

        pool._negotiate = negotiate  # type: ignore[method-assign]
        pool._serve_control = serve_control  # type: ignore[method-assign]
        answer = await pool.create_answer(
            offer_sdp=SDP,
            ice_server=None,
            channel_handler=lambda _channel: asyncio.sleep(0),
            metrics=TransportMetrics(),
        )
        await asyncio.gather(*tuple(pool._tasks))
        assert answer == {"v": 1, "type": "answer", "sdp": "control-answer"}

    await exercise_video()
    await exercise_control()
    assert calls == [("video", True), ("control", True)]


async def _assert_control_datachannel_handler() -> None:
    handled: list[object] = []

    async def handler(channel: object) -> None:
        handled.append(channel)

    class Peer:
        async def close(self) -> None:
            return None

    channel = object()
    ready = asyncio.get_running_loop().create_future()
    ready.set_result(channel)
    await ControlDataChannelPeerPool()._serve_control(
        peer=Peer(),
        channel_ready=ready,
        channel_handler=handler,
        metrics=TransportMetrics(),
    )
    assert handled == [channel]


async def _assert_datachannel_relay_contract() -> None:
    def frame(sequence: int) -> bytes:
        jpeg = b"\xff\xd8" + bytes([sequence]) * 8 + b"\xff\xd9"
        return pack_frame(
            FrameEnvelope(
                sequence=sequence,
                capture_wall_ns=100 + sequence,
                capture_monotonic_ns=200 + sequence,
                encoded_wall_ns=300 + sequence,
                encoded_monotonic_ns=400 + sequence,
                runtime_send_monotonic_ns=500 + sequence,
                causal_action_sequence=40 + sequence,
                causal_applied_monotonic_ns=600 + sequence,
                dropped_before=1,
            ),
            jpeg,
        )

    async def frames():
        yield frame(2)

    class Channel:
        readyState = "open"

        def __init__(self) -> None:
            self.frames: list[bytes] = []
            self.buffer_checks = 0
            self.listeners: dict[str, list] = {}

        def on(self, name: str, callback) -> None:
            self.listeners.setdefault(name, []).append(callback)

        def remove_listener(self, name: str, callback) -> None:
            self.listeners[name].remove(callback)

        @property
        def bufferedAmount(self) -> int:
            self.buffer_checks += 1
            return 1 if self.buffer_checks == 1 else 0

        def send(self, raw: bytes) -> None:
            self.frames.append(bytes(raw))
            self.readyState = "closed"

    class Peer:
        async def close(self) -> None:
            return None

    channel = Channel()
    ready = asyncio.get_running_loop().create_future()
    ready.set_result(channel)
    metrics = TransportMetrics()
    await VideoDataChannelPeerPool()._serve(
        peer=Peer(),
        channel_ready=ready,
        frame_source=lambda: frames(),
        metrics=metrics,
    )

    assert len(channel.frames) == 1
    envelope, content = unpack_frame(channel.frames[0])
    assert envelope.sequence == 2
    assert envelope.causal_action_sequence == 42
    assert envelope.causal_applied_monotonic_ns == 602
    assert envelope.dropped_before == 1
    assert content == b"\xff\xd8" + bytes([2]) * 8 + b"\xff\xd9"
    snapshot = metrics.snapshot()
    assert snapshot["datachannel_window_saturated"] == 1
    assert snapshot["datachannel_frames_sent"] == 1


def test_datachannel_buffer_wait_wakes_on_aiortc_low_event() -> None:
    async def exercise() -> None:
        class Channel:
            readyState = "open"
            bufferedAmount = 1

            def __init__(self) -> None:
                self.listeners: dict[str, list] = {}

            def on(self, name: str, callback) -> None:
                self.listeners.setdefault(name, []).append(callback)

            def remove_listener(self, name: str, callback) -> None:
                self.listeners[name].remove(callback)

            def emit(self, name: str) -> None:
                for callback in tuple(self.listeners.get(name, ())):
                    callback()

        channel = Channel()
        waiter = asyncio.create_task(
            _wait_for_buffer_low(channel, TransportMetrics(), fallback_seconds=1.0)
        )
        await asyncio.sleep(0)
        channel.bufferedAmount = 0
        channel.emit("bufferedamountlow")
        assert await asyncio.wait_for(waiter, timeout=0.1) is True

    asyncio.run(exercise())


def test_datachannel_buffer_wait_wakes_on_close() -> None:
    async def exercise() -> None:
        class Channel:
            readyState = "open"
            bufferedAmount = 1

            def __init__(self) -> None:
                self.listeners: dict[str, list] = {}

            def on(self, name: str, callback) -> None:
                self.listeners.setdefault(name, []).append(callback)

            def remove_listener(self, name: str, callback) -> None:
                self.listeners[name].remove(callback)

            def emit(self, name: str) -> None:
                for callback in tuple(self.listeners.get(name, ())):
                    callback()

        channel = Channel()
        waiter = asyncio.create_task(
            _wait_for_buffer_low(channel, TransportMetrics(), fallback_seconds=1.0)
        )
        await asyncio.sleep(0)
        channel.readyState = "closed"
        channel.emit("close")
        assert await asyncio.wait_for(waiter, timeout=0.1) is False

    asyncio.run(exercise())


def test_datachannel_buffer_wait_recovers_from_lost_event() -> None:
    async def exercise() -> None:
        class Channel:
            readyState = "open"
            bufferedAmount = 1

            def __init__(self) -> None:
                self.listeners: dict[str, list] = {}

            def on(self, name: str, callback) -> None:
                self.listeners.setdefault(name, []).append(callback)

            def remove_listener(self, name: str, callback) -> None:
                self.listeners[name].remove(callback)

        channel = Channel()
        waiter = asyncio.create_task(
            _wait_for_buffer_low(channel, TransportMetrics(), fallback_seconds=0.01)
        )
        await asyncio.sleep(0)
        channel.bufferedAmount = 0
        assert await asyncio.wait_for(waiter, timeout=0.1) is True

    asyncio.run(exercise())
