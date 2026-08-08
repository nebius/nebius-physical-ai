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
CONTROL_DATACHANNEL_LABEL = "npa-leisaac-control"
CONTROL_DATACHANNEL_PROTOCOL = "npa.leisaac.control.v1"


class VideoDataChannelError(RuntimeError):
    """A safe, client-facing WebRTC video negotiation failure."""


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


class VideoDataChannelPeerPool:
    """Own a small bounded set of authenticated browser video peers."""

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

    async def create_answer(
        self,
        *,
        offer_sdp: str,
        ice_server: dict[str, Any] | None,
        frame_source: Callable[[], AsyncIterator[bytes]],
        metrics: TransportMetrics,
    ) -> dict[str, Any]:
        """Negotiate and retain one peer; its relay starts when the channel opens."""

        try:
            from aiortc import (  # type: ignore[import-not-found]
                RTCConfiguration,
                RTCIceServer,
                RTCPeerConnection,
                RTCSessionDescription,
            )
        except ImportError as exc:
            raise VideoDataChannelError("WebRTC video relay is unavailable") from exc

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
        configuration = RTCConfiguration(iceServers=ice_servers)
        peer = RTCPeerConnection(configuration=configuration)
        async with self._lock:
            closed = {
                item
                for item in self._peers
                if str(getattr(item, "connectionState", "")) in {"closed", "failed"}
            }
            self._peers.difference_update(closed)
            if len(self._peers) >= self.limit:
                await peer.close()
                raise VideoDataChannelError("WebRTC video peer capacity is busy")
            self._peers.add(peer)

        loop = asyncio.get_running_loop()
        channel_ready: asyncio.Future[Any] = loop.create_future()

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            if channel_ready.done():
                channel.close()
                return
            if not valid_video_datachannel(channel):
                channel.close()
                channel_ready.set_exception(
                    VideoDataChannelError("invalid WebRTC video channel")
                )
                return
            channel.bufferedAmountLowThreshold = VIDEO_DATACHANNEL_BUFFER_LOW_BYTES
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
                raise VideoDataChannelError("WebRTC video answer is unavailable")
        except Exception as exc:
            await self._close(peer)
            if isinstance(exc, VideoDataChannelError):
                raise
            raise VideoDataChannelError("WebRTC video negotiation failed") from exc

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
        return {"v": 1, "type": "answer", "sdp": str(local.sdp)}

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
                while (
                    str(channel.readyState) == "open"
                    and int(channel.bufferedAmount)
                    > VIDEO_DATACHANNEL_BUFFER_LOW_BYTES
                ):
                    metrics.increment("datachannel_window_saturated")
                    await asyncio.sleep(0.002)
                if str(channel.readyState) != "open":
                    break
                # Pull from the runtime's latest-value source only after SCTP
                # has room.  Frames that arrived during backpressure are then
                # coalesced before serialization instead of becoming a local
                # stale send queue.
                frame = await anext(source)
                if not isinstance(frame, bytes) or len(frame) > MAX_VIDEO_DATACHANNEL_FRAME_BYTES:
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


class ControlDataChannelPeerPool(VideoDataChannelPeerPool):
    """Negotiate bounded direct peers for reliable latency-critical control."""

    async def create_answer(
        self,
        *,
        offer_sdp: str,
        ice_server: dict[str, Any] | None,
        channel_handler: Callable[[Any], Awaitable[None]],
        metrics: TransportMetrics,
    ) -> dict[str, Any]:
        try:
            from aiortc import (  # type: ignore[import-not-found]
                RTCConfiguration,
                RTCIceServer,
                RTCPeerConnection,
                RTCSessionDescription,
            )
        except ImportError as exc:
            raise VideoDataChannelError("WebRTC control relay is unavailable") from exc

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
                raise VideoDataChannelError("WebRTC control peer capacity is busy")
            self._peers.add(peer)

        loop = asyncio.get_running_loop()
        channel_ready: asyncio.Future[Any] = loop.create_future()

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            if channel_ready.done() or not valid_control_datachannel(channel):
                channel.close()
                if not channel_ready.done():
                    channel_ready.set_exception(
                        VideoDataChannelError("invalid WebRTC control channel")
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
                raise VideoDataChannelError("WebRTC control answer is unavailable")
        except Exception as exc:
            await self._close(peer)
            if isinstance(exc, VideoDataChannelError):
                raise
            raise VideoDataChannelError("WebRTC control negotiation failed") from exc

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
        return {"v": 1, "type": "answer", "sdp": str(local.sdp)}

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
