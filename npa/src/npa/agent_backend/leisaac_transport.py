"""Shared protocol and bounded-state primitives for LeIsaac teleoperation.

FastAPI routes are adapters around this module.  The same source is shipped to
the public agent and copied into the LeIsaac runtime image so sequence handling,
frame envelopes, limits, and backpressure have one implementation.
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
import struct
import threading
import time
from typing import Any


PROTOCOL_VERSION = 1
FRAME_PROTOCOL_VERSION = 2
CONTROL_SUBPROTOCOL = "npa.leisaac.control.v1"
VIDEO_SUBPROTOCOL = "npa.leisaac.video.v1"
MAX_CONTROL_MESSAGE_BYTES = 4096
MAX_VIDEO_ACK_BYTES = 512
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_CLIENT_HISTORY = 1024
MAX_VIDEO_IN_FLIGHT = 4
CLIENT_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,96}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")
ALLOWED_KEYS = frozenset({"W", "S", "A", "D", "Q", "E", "J", "L", "I", "K", "U", "O"})
ALLOWED_EVENTS = frozenset({"press", "release"})
ALLOWED_ACTION_DEVICES = frozenset({"browser-gamepad", "custom-so101"})
FRAME_MAGIC = b"NPAF"
FRAME_HEADER_V1 = struct.Struct("!4sBBHQQQQQQQQII32s")
FRAME_HEADER = struct.Struct("!4sBBHQQQQQQQQQQII32s")


class TransportProtocolError(ValueError):
    """Bounded protocol violation safe to return to a client."""

    def __init__(self, code: str, detail: str, *, expected_seq: int | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.expected_seq = expected_seq

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "type": "error",
            "code": self.code,
            "detail": self.detail,
        }
        if self.expected_seq is not None:
            result["expected_seq"] = self.expected_seq
        return result


class AsyncFrameCreditWindow:
    """Bound ordered video sends without imposing one network RTT per frame."""

    def __init__(self, limit: int = MAX_VIDEO_IN_FLIGHT) -> None:
        if limit < 1 or limit > 32:
            raise ValueError("invalid video credit-window limit")
        self.limit = limit
        self._credits = asyncio.Semaphore(limit)
        self._sequences: deque[int] = deque()
        self.high_water = 0

    @property
    def depth(self) -> int:
        return len(self._sequences)

    async def reserve(self, sequence: int) -> int:
        normalized = _sequence(sequence, name="frame sequence")
        await self._credits.acquire()
        self._sequences.append(normalized)
        self.high_water = max(self.high_water, self.depth)
        return self.depth

    def acknowledge(self, sequence: int) -> int:
        normalized = _sequence(sequence, name="frame sequence")
        if not self._sequences:
            raise TransportProtocolError(
                "out_of_order", "video acknowledgement has no in-flight frame"
            )
        expected = self._sequences[0]
        if normalized != expected:
            raise TransportProtocolError(
                "out_of_order",
                "video acknowledgement is not for the oldest in-flight frame",
                expected_seq=expected,
            )
        self._sequences.popleft()
        self._credits.release()
        return self.depth


def _bounded_text(value: Any, *, pattern: re.Pattern[str], name: str) -> str:
    text = str(value or "")
    if not pattern.fullmatch(text):
        raise TransportProtocolError("invalid_message", f"invalid {name}")
    return text


def _sequence(value: Any, *, name: str = "sequence") -> int:
    if isinstance(value, bool):
        raise TransportProtocolError("invalid_message", f"invalid {name}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TransportProtocolError("invalid_message", f"invalid {name}") from exc
    if result < 0 or result > 2**63 - 1:
        raise TransportProtocolError("invalid_message", f"invalid {name}")
    return result


def parse_control_message(raw: str | bytes, *, expected_run_id: str) -> dict[str, Any]:
    """Parse and validate one bounded browser/runtime control message."""

    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if not encoded or len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
        raise TransportProtocolError(
            "message_too_large", "control message size is invalid"
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportProtocolError(
            "invalid_message", "control message is not valid JSON"
        ) from exc
    if not isinstance(payload, dict) or payload.get("v") != PROTOCOL_VERSION:
        raise TransportProtocolError(
            "invalid_message", "unsupported control protocol version"
        )
    message_type = str(payload.get("type") or "")
    if message_type not in {"control", "action", "resume", "ping", "release-all"}:
        raise TransportProtocolError(
            "invalid_message", "unsupported control message type"
        )
    run_id = _bounded_text(payload.get("run_id"), pattern=RUN_ID_PATTERN, name="run ID")
    if run_id != expected_run_id:
        raise TransportProtocolError(
            "run_mismatch", "control run ID does not match the selected session"
        )
    client_id = _bounded_text(
        payload.get("client_id"), pattern=CLIENT_ID_PATTERN, name="client ID"
    )
    result: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "type": message_type,
        "run_id": run_id,
        "client_id": client_id,
        "client_mono_ns": _sequence(
            payload.get("client_mono_ns", 0), name="client timestamp"
        ),
        "client_wall_ns": _sequence(
            payload.get("client_wall_ns", 0), name="client wall timestamp"
        ),
    }
    if message_type == "control":
        key = str(payload.get("key") or "").upper()
        event = str(payload.get("event") or "")
        if key not in ALLOWED_KEYS or event not in ALLOWED_EVENTS:
            raise TransportProtocolError("invalid_message", "invalid keyboard control")
        result.update(seq=_sequence(payload.get("seq")), key=key, event=event)
    elif message_type == "action":
        action = payload.get("action")
        device = str(payload.get("device") or "")
        if (
            set(payload)
            != {
                "v",
                "type",
                "run_id",
                "client_id",
                "seq",
                "device",
                "action",
                "client_mono_ns",
                "client_wall_ns",
            }
            or device not in ALLOWED_ACTION_DEVICES
            or not isinstance(action, list)
            or len(action) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > 1.0
                for value in action
            )
        ):
            raise TransportProtocolError("invalid_message", "invalid direct action")
        result.update(
            seq=_sequence(payload.get("seq")),
            device=device,
            action=[float(value) for value in action],
        )
    elif message_type == "resume":
        keys = payload.get("keys_down", [])
        if not isinstance(keys, list) or len(keys) > len(ALLOWED_KEYS):
            raise TransportProtocolError("invalid_message", "invalid resumed key state")
        normalized = [str(key).upper() for key in keys]
        if len(set(normalized)) != len(normalized) or any(
            key not in ALLOWED_KEYS for key in normalized
        ):
            raise TransportProtocolError("invalid_message", "invalid resumed key state")
        result.update(
            last_acked_seq=_sequence(
                payload.get("last_acked_seq", 0), name="last acknowledged sequence"
            ),
            keys_down=normalized,
        )
    elif message_type == "ping":
        nonce = str(payload.get("nonce") or "")
        if len(nonce) > 64 or any(character in nonce for character in "\r\n"):
            raise TransportProtocolError("invalid_message", "invalid ping nonce")
        result["nonce"] = nonce
    return result


def parse_video_ack(raw: str | bytes, *, expected_run_id: str) -> dict[str, Any]:
    """Parse the bounded receipt credit that prevents stale socket-frame backlogs."""

    encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if not encoded or len(encoded) > MAX_VIDEO_ACK_BYTES:
        raise TransportProtocolError(
            "message_too_large", "video acknowledgement size is invalid"
        )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportProtocolError(
            "invalid_message", "video acknowledgement is not valid JSON"
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"v", "type", "run_id", "sequence"}
        or payload.get("v") != PROTOCOL_VERSION
        or payload.get("type") != "frame-ack"
    ):
        raise TransportProtocolError("invalid_message", "invalid video acknowledgement")
    run_id = _bounded_text(payload.get("run_id"), pattern=RUN_ID_PATTERN, name="run ID")
    if run_id != expected_run_id:
        raise TransportProtocolError(
            "run_mismatch", "video acknowledgement run ID does not match the session"
        )
    return {
        "v": PROTOCOL_VERSION,
        "type": "frame-ack",
        "run_id": run_id,
        "sequence": _sequence(payload.get("sequence"), name="frame sequence"),
    }


@dataclass
class _ControlRecord:
    key: str
    event: str
    accepted: dict[str, Any]
    applied: dict[str, Any] | None = None


@dataclass
class _ClientState:
    next_seq: int = 1
    keys_down: set[str] = field(default_factory=set)
    history: OrderedDict[int, _ControlRecord] = field(default_factory=OrderedDict)


class ControlLedger:
    """Thread-safe ordered/idempotent control state shared by HTTP and WS adapters."""

    def __init__(self, *, history_limit: int = MAX_CLIENT_HISTORY):
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.history_limit = history_limit
        self._lock = threading.Lock()
        self._clients: dict[str, _ClientState] = {}

    def accept(
        self,
        message: dict[str, Any],
        *,
        received_mono_ns: int | None = None,
        received_wall_ns: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return an accepted ack and a queue record, or ``None`` for a duplicate."""

        if message.get("type") not in {"control", "action"}:
            raise TransportProtocolError(
                "invalid_message", "expected a control or action message"
            )
        is_action = message.get("type") == "action"
        signature_key = str(message["device"]) if is_action else str(message["key"])
        signature_event = (
            json.dumps(message["action"], separators=(",", ":"))
            if is_action
            else str(message["event"])
        )
        client_id = str(message["client_id"])
        seq = int(message["seq"])
        mono_ns = time.monotonic_ns() if received_mono_ns is None else received_mono_ns
        wall_ns = time.time_ns() if received_wall_ns is None else received_wall_ns
        with self._lock:
            state = self._clients.setdefault(client_id, _ClientState())
            if seq < state.next_seq:
                existing = state.history.get(seq)
                if existing is None:
                    raise TransportProtocolError(
                        "sequence_too_old",
                        "control sequence is outside the idempotency window",
                        expected_seq=state.next_seq,
                    )
                if existing.key != signature_key or existing.event != signature_event:
                    raise TransportProtocolError(
                        "sequence_reused",
                        "control sequence was reused with different content",
                    )
                duplicate_accepted = dict(existing.accepted)
                duplicate_accepted["duplicate"] = True
                if existing.applied is not None:
                    duplicate_accepted["already_applied"] = True
                return duplicate_accepted, None
            if seq != state.next_seq:
                raise TransportProtocolError(
                    "out_of_order",
                    "control sequence is not the next expected value",
                    expected_seq=state.next_seq,
                )
            accepted: dict[str, Any] = {
                "v": PROTOCOL_VERSION,
                "type": "ack",
                "phase": "accepted",
                "run_id": message["run_id"],
                "client_id": client_id,
                "seq": seq,
                "client_mono_ns": str(message["client_mono_ns"]),
                "client_wall_ns": str(message["client_wall_ns"]),
                "runtime_received_mono_ns": str(mono_ns),
                "runtime_received_wall_ns": str(wall_ns),
                "duplicate": False,
            }
            queue_record: dict[str, Any] = {
                "v": PROTOCOL_VERSION,
                "run_id": message["run_id"],
                "client_id": client_id,
                "seq": seq,
                "client_mono_ns": str(message["client_mono_ns"]),
                "client_wall_ns": str(message["client_wall_ns"]),
                "runtime_received_mono_ns": str(mono_ns),
                "runtime_received_wall_ns": str(wall_ns),
            }
            if is_action:
                accepted.update(
                    device=message["device"], action=list(message["action"])
                )
                queue_record.update(
                    type="action",
                    device=message["device"],
                    action=list(message["action"]),
                )
            else:
                accepted.update(key=message["key"], event=message["event"])
                queue_record.update(key=message["key"], event=message["event"])
            state.history[seq] = _ControlRecord(
                key=signature_key,
                event=signature_event,
                accepted=dict(accepted),
            )
            state.next_seq += 1
            if not is_action:
                if message["event"] == "press":
                    state.keys_down.add(str(message["key"]))
                else:
                    state.keys_down.discard(str(message["key"]))
            while len(state.history) > self.history_limit:
                state.history.popitem(last=False)
            return accepted, queue_record

    def mark_applied(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Attach a simulator acknowledgement if the control is still retained."""

        client_id = str(payload.get("client_id") or "")
        try:
            seq = int(str(payload.get("seq") or ""))
        except (TypeError, ValueError):
            return None
        with self._lock:
            state = self._clients.get(client_id)
            record = state.history.get(seq) if state is not None else None
            if record is None:
                return None
            if record.applied is None:
                record.applied = dict(payload)
            return dict(record.applied)

    def applied(self, client_id: str, seq: int) -> dict[str, Any] | None:
        with self._lock:
            state = self._clients.get(client_id)
            record = state.history.get(seq) if state is not None else None
            return (
                dict(record.applied)
                if record is not None and record.applied is not None
                else None
            )

    def resume(self, client_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._clients.setdefault(client_id, _ClientState())
            applied = [
                seq
                for seq, record in state.history.items()
                if record.applied is not None
            ]
            return {
                "v": PROTOCOL_VERSION,
                "type": "resumed",
                "client_id": client_id,
                "next_seq": state.next_seq,
                "last_accepted_seq": state.next_seq - 1,
                "last_applied_seq": max(applied, default=0),
                "keys_down": sorted(state.keys_down),
                "runtime_mono_ns": str(time.monotonic_ns()),
                "runtime_wall_ns": str(time.time_ns()),
            }

    def keys_down(self, client_id: str) -> tuple[str, ...]:
        with self._lock:
            state = self._clients.get(client_id)
            return tuple(sorted(state.keys_down)) if state is not None else ()


@dataclass(frozen=True)
class FrameEnvelope:
    sequence: int
    capture_wall_ns: int
    capture_monotonic_ns: int
    encoded_wall_ns: int
    encoded_monotonic_ns: int
    runtime_send_monotonic_ns: int
    agent_receive_monotonic_ns: int = 0
    agent_send_monotonic_ns: int = 0
    causal_action_sequence: int = 0
    causal_applied_monotonic_ns: int = 0
    dropped_before: int = 0
    flags: int = 0
    sha256: bytes = b""


def pack_frame(envelope: FrameEnvelope, jpeg: bytes) -> bytes:
    content = bytes(jpeg)
    if (
        not content.startswith(b"\xff\xd8")
        or not content.endswith(b"\xff\xd9")
        or len(content) > MAX_FRAME_BYTES
    ):
        raise TransportProtocolError("invalid_frame", "invalid JPEG frame")
    digest = envelope.sha256 or hashlib.sha256(content).digest()
    if len(digest) != 32:
        raise TransportProtocolError("invalid_frame", "invalid frame digest")
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        FRAME_PROTOCOL_VERSION,
        envelope.flags,
        FRAME_HEADER.size,
        envelope.sequence,
        envelope.capture_wall_ns,
        envelope.capture_monotonic_ns,
        envelope.encoded_wall_ns,
        envelope.encoded_monotonic_ns,
        envelope.runtime_send_monotonic_ns,
        envelope.agent_receive_monotonic_ns,
        envelope.agent_send_monotonic_ns,
        envelope.causal_action_sequence,
        envelope.causal_applied_monotonic_ns,
        len(content),
        envelope.dropped_before,
        digest,
    )
    return header + content


def unpack_frame(
    payload: bytes, *, verify_digest: bool = True
) -> tuple[FrameEnvelope, bytes]:
    if len(payload) < FRAME_HEADER_V1.size:
        raise TransportProtocolError("invalid_frame", "frame envelope is truncated")
    prefix = struct.unpack("!4sBBH", payload[:8])
    magic, version, flags, header_size = prefix
    if magic != FRAME_MAGIC or (version, header_size) not in {
        (PROTOCOL_VERSION, FRAME_HEADER_V1.size),
        (FRAME_PROTOCOL_VERSION, FRAME_HEADER.size),
    }:
        raise TransportProtocolError(
            "invalid_frame", "frame envelope header is invalid"
        )
    header = FRAME_HEADER if version == FRAME_PROTOCOL_VERSION else FRAME_HEADER_V1
    unpacked = header.unpack(payload[:header_size])
    jpeg_size_index = 14 if version == FRAME_PROTOCOL_VERSION else 12
    dropped_index = jpeg_size_index + 1
    digest_index = jpeg_size_index + 2
    jpeg_size = unpacked[jpeg_size_index]
    jpeg = payload[header_size:]
    if jpeg_size != len(jpeg) or jpeg_size > MAX_FRAME_BYTES:
        raise TransportProtocolError(
            "invalid_frame", "frame envelope length is invalid"
        )
    digest = unpacked[digest_index]
    if verify_digest and not hashlib.sha256(jpeg).digest() == digest:
        raise TransportProtocolError("invalid_frame", "frame digest mismatch")
    envelope = FrameEnvelope(
        sequence=unpacked[4],
        capture_wall_ns=unpacked[5],
        capture_monotonic_ns=unpacked[6],
        encoded_wall_ns=unpacked[7],
        encoded_monotonic_ns=unpacked[8],
        runtime_send_monotonic_ns=unpacked[9],
        agent_receive_monotonic_ns=unpacked[10],
        agent_send_monotonic_ns=unpacked[11],
        causal_action_sequence=(unpacked[12] if version == FRAME_PROTOCOL_VERSION else 0),
        causal_applied_monotonic_ns=(
            unpacked[13] if version == FRAME_PROTOCOL_VERSION else 0
        ),
        dropped_before=unpacked[dropped_index],
        flags=flags,
        sha256=digest,
    )
    return envelope, jpeg


def stamp_agent_frame(
    payload: bytes,
    *,
    received_mono_ns: int,
    send_mono_ns: int,
    additional_dropped: int = 0,
) -> bytes:
    envelope, jpeg = unpack_frame(payload)
    return stamp_verified_frame(
        envelope,
        jpeg,
        received_mono_ns=received_mono_ns,
        send_mono_ns=send_mono_ns,
        additional_dropped=additional_dropped,
    )


def stamp_verified_frame(
    envelope: FrameEnvelope,
    jpeg: bytes,
    *,
    received_mono_ns: int,
    send_mono_ns: int,
    additional_dropped: int = 0,
) -> bytes:
    """Stamp a previously verified envelope/JPEG pair without hashing it again."""

    if envelope.sha256 is None or len(envelope.sha256) != 32:
        raise TransportProtocolError(
            "invalid_frame", "verified frame digest is missing"
        )
    return pack_frame(
        replace(
            envelope,
            agent_receive_monotonic_ns=received_mono_ns,
            agent_send_monotonic_ns=send_mono_ns,
            dropped_before=envelope.dropped_before + max(0, additional_dropped),
        ),
        jpeg,
    )


class AsyncLatestValue:
    """A one-slot async publication primitive: every waiter receives only latest."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._generation = 0
        self._value: Any = None

    @property
    def generation(self) -> int:
        return self._generation

    async def publish(self, value: Any) -> int:
        async with self._condition:
            self._generation += 1
            self._value = value
            self._condition.notify_all()
            return self._generation

    async def wait_after(
        self, generation: int, *, timeout: float | None = None
    ) -> tuple[int, Any, int]:
        async def wait() -> None:
            async with self._condition:
                await self._condition.wait_for(lambda: self._generation > generation)

        if self._generation <= generation:
            await asyncio.wait_for(wait(), timeout=timeout)
        observed = self._generation
        return observed, self._value, max(0, observed - generation - 1)


class AsyncLatestByKey:
    """Bounded per-key latest values with fair, broadcast-safe observation."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("latest-value keys must be unique and non-empty")
        self._keys = keys
        self._condition = asyncio.Condition()
        self._generations = {key: 0 for key in keys}
        self._values: dict[str, Any] = {}

    async def publish(self, key: str, value: Any) -> int:
        if key not in self._keys:
            raise ValueError("unknown latest-value key")
        async with self._condition:
            self._generations[key] += 1
            self._values[key] = value
            self._condition.notify_all()
            return self._generations[key]

    async def wait_after(
        self,
        generations: dict[str, int],
        *,
        next_index: int = 0,
        preferred_key: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, int, Any, int, int]:
        if preferred_key is not None and preferred_key not in self._keys:
            raise ValueError("unknown preferred latest-value key")
        observed = {key: max(0, int(generations.get(key, 0))) for key in self._keys}

        def available() -> bool:
            return any(self._generations[key] > observed[key] for key in self._keys)

        async def wait() -> None:
            async with self._condition:
                await self._condition.wait_for(available)

        if not available():
            await asyncio.wait_for(wait(), timeout=timeout)
        # Put the user-critical viewport on the wire first when both cameras are
        # ready. Its generation is then observed, so the secondary value is
        # selected on the immediately following dequeue rather than starved.
        if (
            preferred_key is not None
            and self._generations[preferred_key] > observed[preferred_key]
        ):
            index = self._keys.index(preferred_key)
            generation = self._generations[preferred_key]
            return (
                preferred_key,
                generation,
                self._values[preferred_key],
                max(0, generation - observed[preferred_key] - 1),
                (index + 1) % len(self._keys),
            )
        start = max(0, int(next_index)) % len(self._keys)
        for offset in range(len(self._keys)):
            index = (start + offset) % len(self._keys)
            key = self._keys[index]
            generation = self._generations[key]
            if generation <= observed[key]:
                continue
            return (
                key,
                generation,
                self._values[key],
                max(0, generation - observed[key] - 1),
                (index + 1) % len(self._keys),
            )
        raise RuntimeError("latest-value publication disappeared")


class TransportMetrics:
    """Low-cardinality counters safe for status/telemetry surfaces."""

    ALLOWED = frozenset(
        {
            "control_connections",
            "video_connections",
            "controls_accepted",
            "controls_duplicate",
            "controls_applied",
            "control_errors",
            "control_datachannel_connections",
            "control_datachannel_errors",
            "frames_published",
            "frames_relay_acked",
            "frames_browser_acked",
            "frames_sent",
            "frames_coalesced",
            "datachannel_connections",
            "datachannel_frames_sent",
            "datachannel_window_saturated",
            "datachannel_errors",
            "video_window_saturated",
            "slow_client_disconnects",
            "reconnects",
        }
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self.ALLOWED or amount < 0:
            raise ValueError("invalid transport metric")
        with self._lock:
            self._counts[name] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                name: int(self._counts.get(name, 0)) for name in sorted(self.ALLOWED)
            }
