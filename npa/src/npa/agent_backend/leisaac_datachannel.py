"""Bounded WebRTC data-channel relay for causal LeIsaac video frames.

Control and safety acknowledgements deliberately stay on the reliable ordered
WebSocket.  This module carries only independently decodable, sequence-stamped
video frames over an unordered partial-reliability SCTP channel so loss cannot
build a stale presentation queue.  The existing binary WebSocket remains the
fallback when WebRTC negotiation is unavailable.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Awaitable, Callable

try:  # workbench image: sibling modules are on sys.path
    from leisaac_transport import TransportMetrics
except ImportError:
    try:  # agent VM: /opt/npa-agent is on sys.path
        from agent_backend.leisaac_transport import TransportMetrics
    except ImportError:  # repository tests
        from npa.agent_backend.leisaac_transport import TransportMetrics


VIDEO_DATACHANNEL_LABEL = "npa-leisaac-video"
VIDEO_DATACHANNEL_PROTOCOL = "npa.leisaac.video.v1"
MAX_VIDEO_DATACHANNEL_OFFER_BYTES = 65_536
MAX_VIDEO_DATACHANNEL_FRAME_BYTES = 65_536
MAX_VIDEO_DATACHANNEL_PEERS = 4
# One independently decodable JPEG at a time.  A positive threshold let a full
# workspace frame plus an overview frame enter SCTP while the previous message
# was still draining; on a TURN path that converted latest-value delivery into
# roughly one second of user-visible queue age.  Pull the next latest frame only
# after aiortc has handed the current message out of its application buffer.
VIDEO_DATACHANNEL_BUFFER_LOW_BYTES = 0
VIDEO_DATACHANNEL_LOST_EVENT_FALLBACK_SECONDS = 0.1
CONTROL_DATACHANNEL_LABEL = "npa-leisaac-control"
CONTROL_DATACHANNEL_PROTOCOL = "npa.leisaac.control.v1"
AIORTC_RUNTIME_VERSION = "1.15.0"


class VideoDataChannelError(RuntimeError):
    """A safe, client-facing WebRTC video negotiation failure."""


def require_aiortc_runtime_version() -> None:
    """Pin semantics used for SCTP buffering and codec negotiation."""

    try:
        actual = version("aiortc")
    except PackageNotFoundError as exc:
        raise VideoDataChannelError("WebRTC runtime aiortc is unavailable") from exc
    if actual != AIORTC_RUNTIME_VERSION:
        raise VideoDataChannelError(
            f"WebRTC runtime requires aiortc {AIORTC_RUNTIME_VERSION}; found {actual}"
        )


def parse_video_datachannel_offer(payload: Any, *, expected_run_id: str) -> str:
    """Validate one bounded, exact browser SDP offer payload."""

    if not isinstance(payload, dict) or set(payload) != {"v", "run_id", "type", "sdp"}:
        raise VideoDataChannelError("invalid WebRTC video offer")
    sdp = payload.get("sdp")
    if (
        payload.get("v") != 1
        or payload.get("type") != "offer"
        or payload.get("run_id") != expected_run_id
        or not isinstance(sdp, str)
        or not 1 <= len(sdp.encode("utf-8")) <= MAX_VIDEO_DATACHANNEL_OFFER_BYTES
        or "m=application" not in sdp
        or "UDP/DTLS/SCTP" not in sdp
    ):
        raise VideoDataChannelError("invalid WebRTC video offer")
    return sdp


def valid_video_datachannel(channel: Any) -> bool:
    """Require the browser's explicit stale-frame-dropping channel contract."""

    return (
        str(getattr(channel, "label", "")) == VIDEO_DATACHANNEL_LABEL
        and str(getattr(channel, "protocol", "")) == VIDEO_DATACHANNEL_PROTOCOL
        and getattr(channel, "ordered", True) is False
        and getattr(channel, "maxRetransmits", None) == 0
    )


def valid_control_datachannel(channel: Any) -> bool:
    """Require reliable, ordered SCTP for robot controls and safety releases."""

    return (
        str(getattr(channel, "label", "")) == CONTROL_DATACHANNEL_LABEL
        and str(getattr(channel, "protocol", "")) == CONTROL_DATACHANNEL_PROTOCOL
        and getattr(channel, "ordered", False) is True
        and getattr(channel, "maxRetransmits", None) is None
        and getattr(channel, "maxPacketLifeTime", None) is None
    )


async def _wait_for_buffer_low(
    channel: Any,
    metrics: TransportMetrics,
    *,
    fallback_seconds: float = VIDEO_DATACHANNEL_LOST_EVENT_FALLBACK_SECONDS,
) -> bool:
    """Wait for aiortc's low-buffer event with a race-safe bounded fallback."""

    threshold = VIDEO_DATACHANNEL_BUFFER_LOW_BYTES
    while str(channel.readyState) == "open" and int(channel.bufferedAmount) > threshold:
        metrics.increment("datachannel_window_saturated")
        ready = asyncio.Event()

        def wake() -> None:
            ready.set()

        channel.on("bufferedamountlow", wake)
        channel.on("close", wake)
        try:
            # aiortc can cross the threshold immediately before registration.
            if int(channel.bufferedAmount) <= threshold:
                return True
            try:
                await asyncio.wait_for(ready.wait(), timeout=fallback_seconds)
            except asyncio.TimeoutError:
                # aiortc 1.15.0 emits bufferedamountlow on a downward threshold
                # crossing. This one bounded recheck covers a lost event or a
                # test/future implementation that updates the counter silently.
                pass
        finally:
            remove = getattr(channel, "remove_listener", None)
            if callable(remove):
                remove("bufferedamountlow", wake)
                remove("close", wake)
    return str(channel.readyState) == "open"


class _DataChannelPeerPool:
    """Own bounded peers and share authenticated SDP/ICE negotiation."""

    def __init__(self, *, limit: int = MAX_VIDEO_DATACHANNEL_PEERS) -> None:
        if limit < 1 or limit > 16:
            raise ValueError("invalid WebRTC video peer limit")
        self.limit = limit
        self._peers: set[Any] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return len(self._peers)

    async def _discard(self, peer: Any) -> None:
        async with self._lock:
            self._peers.discard(peer)

    async def _close(self, peer: Any) -> None:
        try:
            await peer.close()
        finally:
            await self._discard(peer)

    async def _negotiate(
        self,
        *,
        offer_sdp: str,
        ice_server: dict[str, Any] | None,
        channel_kind: str,
        channel_validator: Callable[[Any], bool],
    ) -> tuple[Any, asyncio.Future[Any], str]:
        """Share bounded SDP/ICE negotiation while varying channel semantics."""

        require_aiortc_runtime_version()
        try:
            from aiortc import (  # type: ignore[import-not-found]
                RTCConfiguration,
                RTCIceServer,
                RTCPeerConnection,
                RTCSessionDescription,
            )
        except ImportError as exc:
            raise VideoDataChannelError(
                f"WebRTC {channel_kind} relay is unavailable"
            ) from exc

        ice_servers = []
        if ice_server is not None:
            urls = ice_server.get("urls")
            if not isinstance(urls, list) or len(urls) != 1:
                raise VideoDataChannelError("WebRTC relay configuration is unavailable")
            ice_servers.append(
                RTCIceServer(
                    urls=[str(urls[0])],
                    username=str(ice_server.get("username") or ""),
                    credential=str(ice_server.get("credential") or ""),
                )
            )
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        async with self._lock:
            closed = {
                item
                for item in self._peers
                if str(getattr(item, "connectionState", "")) in {"closed", "failed"}
            }
            self._peers.difference_update(closed)
            if len(self._peers) >= self.limit:
                await peer.close()
                raise VideoDataChannelError(
                    f"WebRTC {channel_kind} peer capacity is busy"
                )
            self._peers.add(peer)

        channel_ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            if channel_ready.done() or not channel_validator(channel):
                channel.close()
                if not channel_ready.done():
                    channel_ready.set_exception(
                        VideoDataChannelError(f"invalid WebRTC {channel_kind} channel")
                    )
                return
            channel_ready.set_result(channel)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if str(peer.connectionState) in {"closed", "failed"}:
                await self._discard(peer)

        try:
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=offer_sdp, type="offer")
            )
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            local = peer.localDescription
            if local is None or local.type != "answer" or not local.sdp:
                raise VideoDataChannelError(
                    f"WebRTC {channel_kind} answer is unavailable"
                )
        except Exception as exc:
            await self._close(peer)
            if isinstance(exc, VideoDataChannelError):
                raise
            raise VideoDataChannelError(
                f"WebRTC {channel_kind} negotiation failed"
            ) from exc
        return peer, channel_ready, str(local.sdp)


class VideoDataChannelPeerPool(_DataChannelPeerPool):
    """Own a small bounded set of authenticated browser video peers."""

    async def create_answer(
        self,
        *,
        offer_sdp: str,
        ice_server: dict[str, Any] | None,
        frame_source: Callable[[], AsyncIterator[bytes]],
        metrics: TransportMetrics,
    ) -> dict[str, Any]:
        """Negotiate and retain one peer; its relay starts when the channel opens."""

        def validate_channel(channel: Any) -> bool:
            valid = valid_video_datachannel(channel)
            if valid:
                channel.bufferedAmountLowThreshold = VIDEO_DATACHANNEL_BUFFER_LOW_BYTES
            return valid

        peer, channel_ready, answer_sdp = await self._negotiate(
            offer_sdp=offer_sdp,
            ice_server=ice_server,
            channel_kind="video",
            channel_validator=validate_channel,
        )

        task = asyncio.create_task(
            self._serve(
                peer=peer,
                channel_ready=channel_ready,
                frame_source=frame_source,
                metrics=metrics,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        metrics.increment("datachannel_connections")
        return {"v": 1, "type": "answer", "sdp": answer_sdp}

    async def _serve(
        self,
        *,
        peer: Any,
        channel_ready: asyncio.Future[Any],
        frame_source: Callable[[], AsyncIterator[bytes]],
        metrics: TransportMetrics,
    ) -> None:
        try:
            channel = await asyncio.wait_for(channel_ready, timeout=10.0)
            source = frame_source().__aiter__()
            while str(channel.readyState) == "open":
                if not await _wait_for_buffer_low(channel, metrics):
                    break
                # Pull from the runtime's latest-value source only after SCTP
                # has room.  Frames that arrived during backpressure are then
                # coalesced before serialization instead of becoming a local
                # stale send queue.
                frame = await anext(source)
                if (
                    not isinstance(frame, bytes)
                    or len(frame) > MAX_VIDEO_DATACHANNEL_FRAME_BYTES
                ):
                    raise VideoDataChannelError(
                        "WebRTC video frame exceeds the bounded channel size"
                    )
                channel.send(frame)
                metrics.increment("datachannel_frames_sent")
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.increment("datachannel_errors")
        finally:
            await self._close(peer)


class ControlDataChannelPeerPool(_DataChannelPeerPool):
    """Negotiate bounded direct peers for reliable latency-critical control."""

    async def create_answer(
        self,
        *,
        offer_sdp: str,
        ice_server: dict[str, Any] | None,
        channel_handler: Callable[[Any], Awaitable[None]],
        metrics: TransportMetrics,
    ) -> dict[str, Any]:
        peer, channel_ready, answer_sdp = await self._negotiate(
            offer_sdp=offer_sdp,
            ice_server=ice_server,
            channel_kind="control",
            channel_validator=valid_control_datachannel,
        )

        task = asyncio.create_task(
            self._serve_control(
                peer=peer,
                channel_ready=channel_ready,
                channel_handler=channel_handler,
                metrics=metrics,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        metrics.increment("control_datachannel_connections")
        return {"v": 1, "type": "answer", "sdp": answer_sdp}

    async def _serve_control(
        self,
        *,
        peer: Any,
        channel_ready: asyncio.Future[Any],
        channel_handler: Callable[[Any], Awaitable[None]],
        metrics: TransportMetrics,
    ) -> None:
        try:
            channel = await asyncio.wait_for(channel_ready, timeout=10.0)
            await channel_handler(channel)
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.increment("control_datachannel_errors")
        finally:
            await self._close(peer)
