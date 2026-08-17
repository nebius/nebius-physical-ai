#!/usr/bin/env python3
"""Verify LeIsaac relay restart and credential expiry on one existing run.

This is an explicitly mutating live check.  It never creates another workload:
the selected Deployment is restarted in place once to rotate its session
credential after the old credential expires.  The immutable dataset is not
modified.  Evidence deliberately excludes credentials, cookies, and provider
identifiers and is written owner-only.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import struct
import sys
import threading
import time
from typing import Any, NoReturn
from urllib.parse import urlencode, urlparse

import boto3

from npa.agent_backend.leisaac_transport import (
    CONTROL_SUBPROTOCOL,
    VIDEO_SUBPROTOCOL,
    unpack_frame,
)
from npa.cli.workbench.leisaac import (
    _LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    _LIFECYCLE_LOCK_RENEW_SECONDS,
    _LIFECYCLE_LOCK_STALE_SECONDS,
    _TransientRelayStatusError,
    _agent_artifact_storage,
    _agent_relay_context,
    _apply,
    _install_agent_relay,
    _kubectl,
    _lifecycle_lock_name,
    _relay_source,
    _relay_status,
    _require_lifecycle_lock_permissions,
    _wait_ready,
)
from npa.workbench.leisaac import (
    relay_client_secret_manifest,
    resource_name,
    session_attestation,
    split_s3_uri,
    validate_run_id,
)
from npa.workbench.leisaac.agent_relay import CONTROL_LISTEN
from npa.workbench.leisaac.reverse_client import (
    BACKHAUL_SUBPROTOCOL,
    HEADER,
    HELLO,
    _hello_payload,
)


_READ_LIMIT = 4 * 1024 * 1024 + 512
_WEBSOCKET_OPERATION_TIMEOUT_SECONDS = 20.0
_SAFETY_RELEASE_TIMEOUT_SECONDS = 30.0


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _json_result(result: Any, label: str) -> dict[str, Any]:
    if result.returncode:
        detail = " ".join((result.stderr or result.stdout or "").split())
        _fail(f"{label} failed: {detail[:500] or 'no provider detail'}")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} returned a non-object document")
    return payload


class _WebSocket:
    """Small RFC 6455 client for the live verifier's fixed message surface."""

    def __init__(self, connection: ssl.SSLSocket, remainder: bytes = b"") -> None:
        self.connection = connection
        self.buffer = bytearray(remainder)

    @classmethod
    def connect(
        cls,
        *,
        host: str,
        path: str,
        subprotocol: str,
        authorization: str,
        certificate_sha256: str,
        cookie: str = "",
        origin: str | None,
        timeout: float = 20.0,
    ) -> _WebSocket:
        if timeout <= 0:
            raise TimeoutError("WebSocket connection deadline expired")
        deadline = time.monotonic() + timeout

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("WebSocket connection deadline expired")
            return value

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        raw = socket.create_connection((host, 443), timeout=remaining())
        try:
            raw.settimeout(remaining())
            connection = context.wrap_socket(raw, server_hostname=host)
        except BaseException:
            raw.close()
            raise
        _verify_certificate(connection, certificate_sha256)
        connection.settimeout(remaining())
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            f"Sec-WebSocket-Protocol: {subprotocol}",
            f"Authorization: Basic {authorization}",
        ]
        if origin is not None:
            headers.append(f"Origin: {origin}")
        if cookie:
            headers.append(f"Cookie: {cookie}")
        connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response and len(response) < 16_384:
            connection.settimeout(remaining())
            chunk = connection.recv(4096)
            if not chunk:
                connection.close()
                raise EOFError("WebSocket peer closed during HTTP upgrade")
            response.extend(chunk)
        raw_headers, separator, remainder = bytes(response).partition(b"\r\n\r\n")
        expected_accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
            ).digest()
        )
        lowered = raw_headers.lower()
        if (
            not separator
            or not raw_headers.startswith(b"HTTP/1.1 101 ")
            or b"sec-websocket-accept: " + expected_accept.lower() not in lowered
            or f"sec-websocket-protocol: {subprotocol}".encode("ascii") not in lowered
        ):
            connection.close()
            status = raw_headers.split(b"\r\n", 1)[0].decode("ascii", "replace")
            raise ConnectionError(f"WebSocket upgrade rejected: {status}")
        remaining()
        # The aggregate deadline governs only connection setup. Established
        # control/video/backhaul sockets use the normal bounded operational
        # timeout so a successful setup near its deadline does not poison the
        # first frame or protocol response with a tiny leftover timeout.
        connection.settimeout(_WEBSOCKET_OPERATION_TIMEOUT_SECONDS)
        return cls(connection, remainder)

    def _read(self, size: int, *, deadline: float | None = None) -> bytes:
        while len(self.buffer) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("WebSocket operation deadline expired")
                self.connection.settimeout(remaining)
            chunk = self.connection.recv(max(4096, size - len(self.buffer)))
            if not chunk:
                raise EOFError("WebSocket closed")
            self.buffer.extend(chunk)
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value

    def send(self, payload: str | bytes, *, opcode: int | None = None) -> None:
        content = (
            payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        )
        selected_opcode = (
            (1 if isinstance(payload, str) else 2) if opcode is None else opcode
        )
        mask = os.urandom(4)
        size = len(content)
        if size < 126:
            header = bytes((0x80 | selected_opcode, 0x80 | size))
        elif size <= 65_535:
            header = bytes((0x80 | selected_opcode, 0x80 | 126)) + struct.pack(
                "!H", size
            )
        else:
            header = bytes((0x80 | selected_opcode, 0x80 | 127)) + struct.pack(
                "!Q", size
            )
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(content))
        self.connection.sendall(header + mask + masked)

    def receive(self, *, deadline: float | None = None) -> tuple[int, bytes]:
        fragments = bytearray()
        initial_opcode = 0
        while True:
            first, second = self._read(2, deadline=deadline)
            final = bool(first & 0x80)
            opcode = first & 0x0F
            size = second & 0x7F
            if size == 126:
                size = struct.unpack("!H", self._read(2, deadline=deadline))[0]
            elif size == 127:
                size = struct.unpack("!Q", self._read(8, deadline=deadline))[0]
            if size > _READ_LIMIT:
                raise ValueError("WebSocket message exceeded the live proof bound")
            mask = self._read(4, deadline=deadline) if second & 0x80 else b""
            payload = self._read(size, deadline=deadline)
            if mask:
                payload = bytes(
                    value ^ mask[index % 4] for index, value in enumerate(payload)
                )
            if opcode == 8:
                raise EOFError("WebSocket closed")
            if opcode == 9:
                self.send(payload, opcode=10)
                continue
            if opcode == 10:
                continue
            if opcode in {1, 2}:
                initial_opcode = opcode
                fragments = bytearray(payload)
            elif opcode == 0 and initial_opcode:
                fragments.extend(payload)
            else:
                raise ValueError("unexpected WebSocket opcode")
            if final:
                return initial_opcode, bytes(fragments)

    def receive_json(self, *, deadline: float | None = None) -> dict[str, Any]:
        opcode, payload = self.receive(deadline=deadline)
        if opcode != 1:
            raise ValueError("expected a text WebSocket message")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value

    def close(self) -> None:
        try:
            self.send(b"", opcode=8)
        except OSError:
            pass
        self.connection.close()


def _basic_authorization(user: str, password: str) -> str:
    return base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def _verify_certificate(connection: Any, expected_sha256: str) -> None:
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        connection.close()
        _fail("agent TLS certificate fingerprint is missing or invalid")
    certificate = connection.getpeercert(binary_form=True)
    actual = hashlib.sha256(certificate or b"").hexdigest()
    if not certificate or not hmac.compare_digest(actual, expected):
        connection.close()
        _fail("agent TLS certificate fingerprint changed")


def _pinned_https_request(
    *,
    host: str,
    path: str,
    method: str,
    user: str,
    password: str,
    certificate_sha256: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, list[tuple[str, str]], bytes]:
    if timeout <= 0:
        raise TimeoutError("agent HTTPS request deadline expired")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("agent HTTPS request deadline expired")
        return value

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    connection = http.client.HTTPSConnection(
        host, 443, context=context, timeout=remaining()
    )
    try:
        connection.connect()
        if connection.sock is None:
            _fail("agent TLS connection did not expose a peer certificate")
        connection.sock.settimeout(remaining())
        _verify_certificate(connection.sock, certificate_sha256)
        request_headers = {
            "Authorization": f"Basic {_basic_authorization(user, password)}",
            **(headers or {}),
        }
        connection.request(
            method,
            path,
            body=b"" if method == "POST" else None,
            headers=request_headers,
        )
        if connection.sock is not None:
            connection.sock.settimeout(remaining())
        response = connection.getresponse()
        response_socket = connection.sock
        if response_socket is None:
            response_socket = getattr(
                getattr(getattr(response, "fp", None), "raw", None), "_sock", None
            )
        if response_socket is None:
            _fail("agent HTTPS response did not expose a bounded socket")
        body = bytearray()
        while len(body) <= 131_072:
            response_socket.settimeout(remaining())
            chunk = response.read1(131_073 - len(body))
            if not chunk:
                break
            body.extend(chunk)
        remaining()
        if len(body) > 131_072:
            _fail("agent HTTPS response exceeded the live-proof read bound")
        return int(response.status), response.getheaders(), bytes(body)
    finally:
        connection.close()


def _browser_sockets(
    *,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    timeout: float = 30.0,
) -> tuple[_WebSocket, _WebSocket, dict[str, Any]]:
    if timeout <= 0:
        raise TimeoutError("browser transport deadline expired")
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("browser transport deadline expired")
        return value

    origin = f"https://{host}"
    query = urlencode({"run_id": run_id})
    auth_status, auth_headers, _auth_body = _pinned_https_request(
        host=host,
        path=f"/api/leisaac/ws-session?{query}",
        method="POST",
        user=user,
        password=password,
        certificate_sha256=certificate_sha256,
        headers={"Origin": origin, "X-NPA-LeIsaac-Control": "1"},
        timeout=remaining(),
    )
    if auth_status != 204:
        _fail(f"browser transport authorization returned HTTP {auth_status}")
    cookie = "; ".join(
        value.split(";", 1)[0]
        for name, value in auth_headers
        if name.lower() == "set-cookie" and "=" in value.split(";", 1)[0]
    )
    status_http, _status_headers, status_body = _pinned_https_request(
        host=host,
        path=f"/api/leisaac/status?{query}",
        method="GET",
        user=user,
        password=password,
        certificate_sha256=certificate_sha256,
        timeout=remaining(),
    )
    if status_http != 200:
        _fail(f"LeIsaac status returned HTTP {status_http}")
    status = json.loads(status_body)
    if not isinstance(status, dict):
        _fail("LeIsaac status returned a non-object document")
    authorization = _basic_authorization(user, password)
    control = _WebSocket.connect(
        host=host,
        path=f"/api/leisaac/transport/control?{query}",
        subprotocol=CONTROL_SUBPROTOCOL,
        authorization=authorization,
        certificate_sha256=certificate_sha256,
        cookie=cookie,
        origin=origin,
        timeout=remaining(),
    )
    try:
        video = _WebSocket.connect(
            host=host,
            path=f"/api/leisaac/transport/video?{query}",
            subprotocol=VIDEO_SUBPROTOCOL,
            authorization=authorization,
            certificate_sha256=certificate_sha256,
            cookie=cookie,
            origin=origin,
            timeout=remaining(),
        )
    except Exception:
        control.close()
        raise
    return control, video, status


def _browser_status(
    *,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    timeout: float,
) -> dict[str, Any]:
    """Read runtime status without acquiring or changing controller ownership."""

    query = urlencode({"run_id": run_id})
    status_http, _status_headers, status_body = _pinned_https_request(
        host=host,
        path=f"/api/leisaac/status?{query}",
        method="GET",
        user=user,
        password=password,
        certificate_sha256=certificate_sha256,
        timeout=timeout,
    )
    if status_http != 200:
        _fail(f"LeIsaac status returned HTTP {status_http}")
    status = json.loads(status_body)
    if not isinstance(status, dict):
        _fail("LeIsaac status returned a non-object document")
    return status


def _input_event_count(status: dict[str, Any], *, phase: str) -> int:
    try:
        value = int(str(status.get("input_events")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{phase} status has no valid input-event count") from exc
    if value < 0:
        _fail(f"{phase} status has no valid input-event count")
    return value


def _applied_input_count(status: dict[str, Any], *, phase: str) -> int:
    try:
        value = int(str(status.get("applied_inputs")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{phase} status has no valid applied-input count") from exc
    if value < 0:
        _fail(f"{phase} status has no valid applied-input count")
    return value


def _wait_browser_sockets(
    *,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    timeout: float = 15.0,
    poll_interval: float = 0.5,
    progress_check: Callable[[], None] | None = None,
) -> tuple[_WebSocket, _WebSocket, dict[str, Any]]:
    """Wait through the agent's bounded negative manifest-cache window."""

    deadline = time.monotonic() + timeout
    while True:
        if progress_check is not None:
            progress_check()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("browser transport deadline expired")
            result = _browser_sockets(
                host=host,
                run_id=run_id,
                user=user,
                password=password,
                certificate_sha256=certificate_sha256,
                timeout=remaining,
            )
            if time.monotonic() > deadline:
                result[0].close()
                result[1].close()
                raise TimeoutError("browser transport deadline expired")
            return result
        except Exception as exc:  # noqa: BLE001 - bounded readiness retry
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"browser transport did not recover within {timeout:g}s; "
                    f"last error: {type(exc).__name__}: {exc}"
                ) from exc
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _wait_controller_release(
    *,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    after_input_events: int,
    timeout: float = 30.0,
    poll_interval: float = 0.25,
    progress_check: Callable[[], None] | None = None,
) -> float:
    """Observe the durable held-key release without claiming controller ownership."""

    started = time.monotonic()
    deadline = started + timeout
    last_error = "release input was not observed"
    while True:
        if progress_check is not None:
            progress_check()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("controller release deadline expired")
            status = _browser_status(
                host=host,
                run_id=run_id,
                user=user,
                password=password,
                certificate_sha256=certificate_sha256,
                timeout=remaining,
            )
            observed = _input_event_count(status, phase="controller release")
            applied = _applied_input_count(status, phase="controller release")
            now = time.monotonic()
            if (
                observed > after_input_events
                and applied >= observed
                and now <= deadline
            ):
                return now - started
            last_error = (
                f"input/applied counts were {observed}/{applied}; expected more "
                f"than {after_input_events} queued and every queued input applied"
            )
        except Exception as exc:  # noqa: BLE001 - bounded readiness retry
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            _fail(
                f"controller safety release did not settle within {timeout:g}s; "
                f"last error: {last_error}"
            )
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _wait_resumed_browser_sockets(
    *,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    client_id: str,
    last_acked_seq: int,
    lease_id: str,
    timeout: float = 30.0,
    poll_interval: float = 0.25,
    progress_check: Callable[[], None] | None = None,
) -> tuple[_WebSocket, _WebSocket, dict[str, Any], dict[str, Any]]:
    """Resume the original controller lease within one end-to-end deadline."""

    deadline = time.monotonic() + timeout
    last_error = "controller remained busy"
    while True:
        if progress_check is not None:
            progress_check()
        control: _WebSocket | None = None
        video: _WebSocket | None = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("controller lease deadline expired")
            control, video, status = _browser_sockets(
                host=host,
                run_id=run_id,
                user=user,
                password=password,
                certificate_sha256=certificate_sha256,
                timeout=remaining,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("controller lease deadline expired")
            control.connection.settimeout(remaining)
            resumed = _resume(
                control,
                run_id=run_id,
                client_id=client_id,
                last_acked_seq=last_acked_seq,
                lease_id=lease_id,
                deadline=deadline,
            )
            if time.monotonic() > deadline:
                raise TimeoutError("controller lease deadline expired")
            control.connection.settimeout(_WEBSOCKET_OPERATION_TIMEOUT_SECONDS)
            video.connection.settimeout(_WEBSOCKET_OPERATION_TIMEOUT_SECONDS)
            return control, video, status, resumed
        except Exception as exc:  # noqa: BLE001 - bounded ownership retry
            last_error = f"{type(exc).__name__}: {exc}"
            for connection in (control, video):
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:  # noqa: BLE001 - next retry remains safe
                        pass
        if time.monotonic() >= deadline:
            _fail(
                f"controller lease did not resume within {timeout:g}s; "
                f"last error: {last_error}"
            )
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def _resume(
    control: _WebSocket,
    *,
    run_id: str,
    client_id: str,
    last_acked_seq: int,
    lease_id: str = "",
    deadline: float | None = None,
) -> dict[str, Any]:
    operation_deadline = (
        deadline
        if deadline is not None
        else time.monotonic() + _WEBSOCKET_OPERATION_TIMEOUT_SECONDS
    )
    now = time.time_ns()
    control.send(
        json.dumps(
            {
                "v": 1,
                "type": "resume",
                "run_id": run_id,
                "client_id": client_id,
                "last_acked_seq": last_acked_seq,
                "keys_down": [],
                "lease_id": lease_id,
                "client_mono_ns": time.monotonic_ns(),
                "client_wall_ns": now,
            },
            separators=(",", ":"),
        )
    )
    response = control.receive_json(deadline=operation_deadline)
    if response.get("type") != "resumed":
        _fail(
            f"control resume failed with {response.get('code') or 'invalid response'}"
        )
    return response


def _press(
    control: _WebSocket, *, run_id: str, client_id: str, sequence: int
) -> dict[str, Any]:
    deadline = time.monotonic() + _WEBSOCKET_OPERATION_TIMEOUT_SECONDS
    control.send(
        json.dumps(
            {
                "v": 1,
                "type": "control",
                "run_id": run_id,
                "client_id": client_id,
                "seq": sequence,
                "key": "W",
                "event": "press",
                "client_mono_ns": time.monotonic_ns(),
                "client_wall_ns": time.time_ns(),
            },
            separators=(",", ":"),
        )
    )
    accepted = control.receive_json(deadline=deadline)
    if accepted.get("phase") != "accepted" or accepted.get("seq") != sequence:
        _fail("runtime did not accept the live safety-proof control")
    while True:
        applied = control.receive_json(deadline=deadline)
        if applied.get("phase") == "applied" and applied.get("seq") == sequence:
            return applied
        if applied.get("type") == "error":
            _fail(f"runtime rejected the live control: {applied.get('code')}")


def _frame(video: _WebSocket, run_id: str) -> int:
    deadline = time.monotonic() + _WEBSOCKET_OPERATION_TIMEOUT_SECONDS
    opcode, payload = video.receive(deadline=deadline)
    if opcode != 2:
        _fail("browser video path did not return a binary frame")
    envelope, jpeg = unpack_frame(payload)
    video.send(
        json.dumps(
            {
                "v": 1,
                "type": "frame-ack",
                "run_id": run_id,
                "sequence": envelope.sequence,
            },
            separators=(",", ":"),
        )
    )
    if len(jpeg) <= 10_000:
        _fail("browser video path returned a non-substantive frame")
    return len(jpeg)


def _press_and_read_frame(
    control: _WebSocket,
    video: _WebSocket,
    *,
    run_id: str,
    client_id: str,
    sequence: int,
) -> int:
    """Apply a bounded control and release its transport if video proof fails."""

    try:
        _press(control, run_id=run_id, client_id=client_id, sequence=sequence)
        return _frame(video, run_id)
    except BaseException:
        # Disconnecting the controller is the runtime's safety-release signal.
        # Preserve the primary failure even when one of the best-effort closes
        # also fails; the outer rollback restarts the relay as a second guard.
        for connection in (control, video):
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - preserve the primary failure
                pass
        raise


def _require_healthy_idle(
    *, health_http: int, health: dict[str, Any], status: dict[str, Any]
) -> None:
    recorder_state = str((status.get("recorder") or {}).get("state") or "")
    if (
        health_http != 200
        or not health.get("ok")
        or status.get("available") is not True
        or recorder_state != "idle"
    ):
        _fail("live proof requires a healthy available run whose recorder is idle")


def _forced_release_count(
    resume: dict[str, Any], *, pressed_sequence: int, phase: str
) -> int:
    try:
        next_sequence = int(resume.get("next_seq") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{phase} returned an invalid control sequence") from exc
    if resume.get("keys_down") or next_sequence <= pressed_sequence + 1:
        _fail(f"{phase} did not durably release the held control")
    return next_sequence - pressed_sequence - 1


def _require_rotated_lease(
    resume: dict[str, Any], *, prior_lease_id: str, phase: str
) -> str:
    lease_id = str(resume.get("lease_id") or "")
    if (
        len(lease_id) != 64
        or any(character not in "0123456789abcdef" for character in lease_id)
        or hmac.compare_digest(lease_id, prior_lease_id)
    ):
        _fail(f"{phase} did not rotate the original controller lease")
    return lease_id


def _wait_closed(
    connection: _WebSocket,
    *,
    timeout: float = 30.0,
    progress_check: Callable[[], None] | None = None,
) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if progress_check is not None:
            progress_check()
        try:
            # Heartbeats or a silent peer must not suppress lifecycle-lock
            # health checks until the much longer aggregate disconnect deadline.
            poll_deadline = min(deadline, time.monotonic() + 1.0)
            connection.receive(deadline=poll_deadline)
        except socket.timeout:
            continue
        except TimeoutError:
            continue
        except (EOFError, OSError, ssl.SSLError):
            if progress_check is not None:
                progress_check()
            return time.monotonic() - started
    _fail(f"WebSocket remained open for more than {timeout:g}s after relay revoke")


def _wait_relay_ready(
    ssh: Any,
    nonce: str,
    *,
    timeout: float = 300.0,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not observed"
    while True:
        if progress_check is not None:
            progress_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            status = _relay_status(
                ssh,
                session_nonce=nonce,
                timeout_seconds=min(10.0, remaining),
            )
        except _TransientRelayStatusError as exc:
            last_error = type(exc).__name__
        else:
            now = time.monotonic()
            if now >= deadline:
                last_error = f"{status.get('state') or 'invalid status'} after deadline"
                break
            if status.get("state") == "ready":
                return status
            last_error = str(status.get("state") or "invalid status")
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
    _fail(f"relay did not recover within {timeout:g}s; last state: {last_error}")


def _release_deadline_remaining(deadline: float, *, phase: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail(
            f"{phase} safety release was not proven within "
            f"{_SAFETY_RELEASE_TIMEOUT_SECONDS:g}s of disconnect"
        )
    return remaining


def _wait_relay_and_release(
    ssh: Any,
    nonce: str,
    *,
    release_deadline: float,
    host: str,
    run_id: str,
    user: str,
    password: str,
    certificate_sha256: str,
    after_input_events: int,
    phase: str,
    progress_check: Callable[[], None] | None = None,
) -> None:
    """Share one post-disconnect deadline across recovery and release proof."""

    _wait_relay_ready(
        ssh,
        nonce,
        timeout=_release_deadline_remaining(release_deadline, phase=phase),
        progress_check=progress_check,
    )
    _release_deadline_remaining(release_deadline, phase=phase)
    _wait_controller_release(
        host=host,
        run_id=run_id,
        user=user,
        password=password,
        certificate_sha256=certificate_sha256,
        after_input_events=after_input_events,
        timeout=_release_deadline_remaining(release_deadline, phase=phase),
        progress_check=progress_check,
    )
    _release_deadline_remaining(release_deadline, phase=phase)


def _restart_relay_for_release_proof(
    ssh: Any,
    control: _WebSocket,
    *,
    run_id: str,
    session_nonce: str,
    manifest_uri: str,
    progress_check: Callable[[], None] | None = None,
) -> tuple[float, float, float]:
    """Start the safety deadline before the restart that causes disconnect."""

    started = time.monotonic()
    deadline = started + _SAFETY_RELEASE_TIMEOUT_SECONDS
    _install_agent_relay(
        ssh,
        run_id=run_id,
        session_nonce=session_nonce,
        expires_at="",
        manifest_uri=manifest_uri,
    )
    disconnected = _wait_closed(
        control,
        timeout=_release_deadline_remaining(deadline, phase="relay restart"),
        progress_check=progress_check,
    )
    _release_deadline_remaining(deadline, phase="relay restart")
    return disconnected, started, deadline


def _relay_metadata(ssh: Any) -> dict[str, str]:
    command = """sudo /usr/bin/python3 - <<'PY'
import json
data=json.load(open('/etc/npa/leisaac-relay.json'))
print(json.dumps({key: str(data.get(key) or '') for key in ('run_id','expires_at','manifest_uri')}))
PY"""
    _code, stdout, _stderr = ssh.run_or_raise(
        command, label="read nonsecret LeIsaac relay metadata"
    )
    value = json.loads(stdout)
    if not isinstance(value, dict):
        _fail("relay metadata is invalid")
    return {str(key): str(item) for key, item in value.items()}


def _secret_config(context: str, namespace: str, name: str) -> dict[str, str]:
    secret = _json_result(
        _kubectl(context, namespace, ["get", "secret", name, "-o", "json"]),
        "relay Secret lookup",
    )
    encoded = str((secret.get("data") or {}).get("config.json") or "")
    try:
        config = json.loads(base64.b64decode(encoded, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("relay Secret has invalid config") from exc
    if not isinstance(config, dict):
        _fail("relay Secret config is not an object")
    return {str(key): str(value) for key, value in config.items()}


def _patch_runtime_nonce(
    context: str, namespace: str, deployment: str, nonce: str
) -> None:
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "leisaac",
                            "env": [
                                {
                                    "name": "NPA_LEISAAC_SESSION_NONCE",
                                    "value": nonce,
                                }
                            ],
                        }
                    ]
                }
            }
        }
    }
    result = _kubectl(
        context,
        namespace,
        [
            "patch",
            "deployment",
            deployment,
            "--type=strategic",
            "--patch-file=/dev/stdin",
        ],
        stdin=json.dumps(patch),
    )
    if result.returncode:
        detail = " ".join((result.stderr or result.stdout or "").split())
        _fail(f"runtime credential patch failed: {detail[:500]}")


def _put_manifest(*, storage: dict[str, str], manifest_uri: str, body: bytes) -> None:
    bucket, key = split_s3_uri(manifest_uri)
    if bucket != storage["bucket"]:
        _fail("relay manifest is outside the selected agent storage bucket")
    client = boto3.client(
        "s3",
        endpoint_url=storage["endpoint"],
        aws_access_key_id=storage["access_key"],
        aws_secret_access_key=storage["secret_key"],
        region_name=storage.get("region") or None,
    )
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        IfNoneMatch="*",
    )


def _get_manifest(
    *, storage: dict[str, str], manifest_uri: str
) -> tuple[dict[str, Any], bytes]:
    bucket, key = split_s3_uri(manifest_uri)
    if bucket != storage["bucket"]:
        _fail("relay manifest is outside the selected agent storage bucket")
    client = boto3.client(
        "s3",
        endpoint_url=storage["endpoint"],
        aws_access_key_id=storage["access_key"],
        aws_secret_access_key=storage["secret_key"],
        region_name=storage.get("region") or None,
    )
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read(131_073)
    if len(body) > 131_072:
        _fail("relay manifest exceeds the live-proof read bound")
    payload = json.loads(body)
    if not isinstance(payload, dict) or "session_nonce" in payload:
        _fail("relay manifest violates the nonsecret discovery contract")
    return payload, body


def _rotated_manifest(original: dict[str, Any], nonce: str) -> bytes:
    payload = dict(original)
    payload.pop("session_nonce", None)
    payload["session_attestation"] = session_attestation(nonce)
    payload["expires_at"] = ""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _rotated_manifest_uri(original_uri: str, nonce: str) -> str:
    bucket, key = split_s3_uri(original_uri)
    leaf = "reports/leisaac-session.json"
    if not key.endswith(leaf):
        _fail("relay manifest is not a canonical LeIsaac capability leaf")
    parent = key[: -len(leaf)].rstrip("/")
    generation = session_attestation(nonce)[:24]
    rotated_key = "/".join(
        part for part in (parent, f"credential-{generation}", leaf) if part
    )
    return f"s3://{bucket}/{rotated_key}"


def _rotate(
    *,
    context: str,
    namespace: str,
    run_id: str,
    project: str,
    agent_name: str,
    deployment: str,
    host: str,
    ssh: Any,
    auth_user: str,
    auth_password: str,
    secret_config: dict[str, str],
    storage: dict[str, str],
    manifest_uri: str,
    manifest_body: bytes | None,
    nonce: str,
    wait_for_runtime: bool = True,
    progress_check: Callable[[], None] | None = None,
) -> float:
    started = time.monotonic()
    certificate = str(secret_config.get("certificate_sha256") or "")
    secret = relay_client_secret_manifest(
        run_id=run_id,
        namespace=namespace,
        agent_host=host,
        session_nonce=nonce,
        certificate_sha256=certificate,
        auth_user=auth_user,
        auth_password=auth_password,
        client_source=_relay_source("reverse_client.py").decode("utf-8"),
    )
    if manifest_body is not None:
        if progress_check is not None:
            progress_check()
        _put_manifest(storage=storage, manifest_uri=manifest_uri, body=manifest_body)
    if progress_check is not None:
        progress_check()
    _apply(context, namespace, [secret])
    if progress_check is not None:
        progress_check()
    _patch_runtime_nonce(context, namespace, deployment, nonce)
    if progress_check is not None:
        progress_check()
    _install_agent_relay(
        ssh,
        run_id=run_id,
        session_nonce=nonce,
        expires_at="",
        manifest_uri=manifest_uri,
    )
    if wait_for_runtime:
        _wait_ready(
            context,
            namespace,
            deployment,
            progress_check=progress_check,
        )
        _wait_relay_ready(ssh, nonce, progress_check=progress_check)
    return time.monotonic() - started


def _open_backhaul_probe(
    *,
    host: str,
    user: str,
    password: str,
    nonce: str,
    certificate_sha256: str,
) -> _WebSocket:
    connection = _WebSocket.connect(
        host=host,
        path="/api/leisaac/backhaul",
        subprotocol=BACKHAUL_SUBPROTOCOL,
        authorization=_basic_authorization(user, password),
        certificate_sha256=certificate_sha256,
        origin=None,
    )
    payload = _hello_payload({"session_nonce": nonce})
    connection.send(HEADER.pack(HELLO, 0, len(payload)) + payload)
    return connection


def _stale_backhaul_denied(
    *,
    host: str,
    user: str,
    password: str,
    stale_nonce: str,
    certificate_sha256: str,
    relay_connected: Callable[[], bool],
) -> bool:
    if relay_connected():
        _fail("relay was already connected before the stale credential probe")
    connection = _open_backhaul_probe(
        host=host,
        user=user,
        password=password,
        nonce=stale_nonce,
        certificate_sha256=certificate_sha256,
    )
    rejected = False
    try:
        connection.connection.settimeout(10.0)
        opcode, _payload = connection.receive()
        rejected = opcode == 8
    except socket.timeout:
        return False
    except (EOFError, OSError, ssl.SSLError):
        rejected = True
    finally:
        connection.close()
    # A dropped socket is evidence of credential rejection only when the relay's
    # authenticated control endpoint remains reachable and explicitly reports
    # that no backhaul attached.  This prevents a relay outage from passing the
    # stale-credential proof.
    return rejected and not relay_connected()


def _relay_connection_state(ssh: Any) -> bool:
    _code, stdout, _stderr = ssh.run_or_raise(
        "curl --silent --show-error --max-time 5 "
        f"http://127.0.0.1:{CONTROL_LISTEN[1]}/status",
        label="read LeIsaac relay connection state",
    )
    payload = json.loads(stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("connected"), bool):
        _fail("relay connection state is invalid")
    return bool(payload["connected"])


def _wait_relay_connection(
    ssh: Any,
    *,
    connected: bool,
    timeout: float = 30.0,
    progress_check: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_state: bool | None = None
    while time.monotonic() < deadline:
        if progress_check is not None:
            progress_check()
        try:
            last_state = _relay_connection_state(ssh)
        except Exception:  # noqa: BLE001 - bounded live readiness loop
            last_state = None
        if last_state is connected:
            return
        time.sleep(0.25)
    _fail(
        f"relay connection did not become {str(connected).lower()} within "
        f"{timeout:g}s; last state: {last_state}"
    )


def _scale_deployment(
    context: str,
    namespace: str,
    deployment: str,
    replicas: int,
    *,
    progress_check: Callable[[], None] | None = None,
) -> None:
    if progress_check is not None:
        progress_check()
    result = _kubectl(
        context,
        namespace,
        ["scale", "deployment", deployment, f"--replicas={replicas}"],
    )
    if result.returncode:
        detail = " ".join((result.stderr or result.stdout or "").split())
        _fail(f"Deployment scale to {replicas} failed: {detail[:500]}")
    if replicas == 1:
        _wait_ready(
            context,
            namespace,
            deployment,
            progress_check=progress_check,
        )
        return
    deadline = time.monotonic() + 300.0
    last = "not observed"
    while time.monotonic() < deadline:
        if progress_check is not None:
            progress_check()
        status = _kubectl(
            context,
            namespace,
            ["get", "deployment", deployment, "-o", "json"],
        )
        if status.returncode == 0:
            payload = json.loads(status.stdout)
            spec = payload.get("spec") or {}
            observed = payload.get("status") or {}
            desired = int(spec.get("replicas") or 0)
            active = max(
                int(observed.get("replicas") or 0),
                int(observed.get("readyReplicas") or 0),
                int(observed.get("availableReplicas") or 0),
            )
            last = f"desired={desired}, active={active}"
            if desired == 0 and active == 0:
                return
        else:
            last = " ".join((status.stderr or status.stdout or "").split())[:400]
        time.sleep(1.0)
    _fail(f"Deployment did not scale to zero within 300s; last observation: {last}")


def _http_status(
    *,
    host: str,
    user: str,
    password: str,
    path: str,
    certificate_sha256: str,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    status, _headers, body = _pinned_https_request(
        host=host,
        path=path,
        method="GET",
        user=user,
        password=password,
        certificate_sha256=certificate_sha256,
        timeout=timeout,
    )
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def _write_evidence(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        if not parent.is_dir():
            _fail("evidence parent exists but is not a directory")
    else:
        parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _evidence_failure(exc: BaseException, code: str) -> dict[str, str]:
    """Return useful failure classification without persisting provider data."""

    return {"code": code, "type": type(exc).__name__}


def _best_effort_cleanup(
    actions: tuple[tuple[str, Callable[[], None]], ...],
) -> list[tuple[str, Exception]]:
    """Run every cleanup action even when an earlier recovery step fails."""

    failures: list[tuple[str, Exception]] = []
    for label, action in actions:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - cleanup must continue
            failures.append((label, exc))
    return failures


def _close_browser_connections(connections: list[Any]) -> list[tuple[str, Exception]]:
    """Close every tracked browser socket before any slower rollback work."""

    actions = tuple(
        (f"close browser connection {index}", connection.close)
        for index, connection in enumerate(reversed(connections), start=1)
    )
    connections.clear()
    return _best_effort_cleanup(actions)


def _publish_evidence(
    path: Path,
    payload: dict[str, Any],
    *,
    active_failure: BaseException | None,
) -> None:
    """Do not replace an operational failure with a secondary evidence error."""

    try:
        _write_evidence(path, payload)
    except Exception:  # noqa: BLE001 - preserve the active operational failure
        if active_failure is None:
            raise


class _ReplicaRestoration:
    """Track replica restoration before Kubernetes can observe a scale-down."""

    def __init__(
        self,
        context: str,
        namespace: str,
        deployment: str,
        *,
        progress_check: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.deployment = deployment
        self.progress_check = progress_check
        self.required = False

    def scale_down(self) -> None:
        self.required = True
        _scale_deployment(
            self.context,
            self.namespace,
            self.deployment,
            0,
            progress_check=self.progress_check,
        )

    def scale_up(self, *, enforce_lease: bool = True) -> None:
        _scale_deployment(
            self.context,
            self.namespace,
            self.deployment,
            1,
            # Cleanup must restore the customer's existing Deployment even
            # after the verifier loses its exclusion lease. Forward progress
            # remains lease-guarded; only best-effort rollback opts out.
            progress_check=self.progress_check if enforce_lease else None,
        )
        self.required = False


def _lifecycle_lock_document(
    *,
    namespace: str,
    name: str,
    holder: str,
    resource_version: str,
    acquired_epoch: float,
    renewed_epoch: float,
    uid: str = "",
) -> dict[str, Any]:
    annotations = {
        "npa.nebius.com/lifecycle-holder": holder,
        "npa.nebius.com/lifecycle-acquired-epoch": f"{acquired_epoch:.6f}",
        "npa.nebius.com/lifecycle-renewed-epoch": f"{renewed_epoch:.6f}",
    }
    document: dict[str, Any] = {
        "apiVersion": "v1",
        # LeIsaac already requires Secret CRUD for relay and recorder state.
        # Keeping the lock on that resource avoids an extra RBAC requirement.
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": annotations,
        },
    }
    if resource_version:
        document["metadata"]["resourceVersion"] = resource_version
    if uid:
        document["metadata"]["uid"] = uid
    return document


def _lifecycle_lock_values(document: dict[str, Any]) -> dict[str, str]:
    annotations = (document.get("metadata") or {}).get("annotations") or {}
    return {
        "holder": str(annotations.get("npa.nebius.com/lifecycle-holder") or ""),
        "acquired_epoch": str(
            annotations.get("npa.nebius.com/lifecycle-acquired-epoch") or ""
        ),
        "renewed_epoch": str(
            annotations.get("npa.nebius.com/lifecycle-renewed-epoch") or ""
        ),
    }


def _acquire_lifecycle_lock(
    context: str,
    namespace: str,
    deployment: str,
    holder: str,
) -> str:
    """Atomically exclude another mutating proof for the selected run."""

    name = _lifecycle_lock_name(deployment)
    acquired_epoch = time.time()
    document = _lifecycle_lock_document(
        namespace=namespace,
        name=name,
        holder=holder,
        resource_version="",
        acquired_epoch=acquired_epoch,
        renewed_epoch=acquired_epoch,
    )
    result = _kubectl(
        context,
        namespace,
        ["create", "-f", "-"],
        stdin=json.dumps(document, sort_keys=True),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if not result.returncode:
        return name
    detail = " ".join((result.stderr or result.stdout or "").split()).lower()
    if "alreadyexists" not in detail and "already exists" not in detail:
        _fail("could not acquire the selected run lifecycle lock")

    current = _json_result(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "existing lifecycle lock lookup",
    )
    metadata = current.get("metadata") or {}
    data = _lifecycle_lock_values(current)
    resource_version = str(metadata.get("resourceVersion") or "")
    try:
        prior_epoch = float(str(data.get("renewed_epoch") or ""))
    except (TypeError, ValueError):
        _fail("existing lifecycle lock has no valid acquisition time")
    age = acquired_epoch - prior_epoch
    if not resource_version or age < 0 or age <= _LIFECYCLE_LOCK_STALE_SECONDS:
        _fail("another lifecycle verification already holds the selected run lock")

    replacement = _lifecycle_lock_document(
        namespace=namespace,
        name=name,
        holder=holder,
        resource_version=resource_version,
        acquired_epoch=acquired_epoch,
        renewed_epoch=acquired_epoch,
    )
    reclaimed = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(replacement, sort_keys=True),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if reclaimed.returncode:
        _fail("stale lifecycle lock changed before it could be reclaimed")
    return name


def _renew_lifecycle_lock(
    context: str,
    namespace: str,
    name: str,
    holder: str,
) -> None:
    current = _json_result(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "lifecycle lock renewal lookup",
    )
    metadata = current.get("metadata") or {}
    data = _lifecycle_lock_values(current)
    if str(data.get("holder") or "") != holder:
        _fail("selected run lifecycle lock ownership changed")
    resource_version = str(metadata.get("resourceVersion") or "")
    try:
        acquired_epoch = float(str(data.get("acquired_epoch") or ""))
    except (TypeError, ValueError):
        _fail("selected run lifecycle lock has no valid acquisition time")
    if not resource_version:
        _fail("selected run lifecycle lock has no resource version")
    replacement = _lifecycle_lock_document(
        namespace=namespace,
        name=name,
        holder=holder,
        resource_version=resource_version,
        acquired_epoch=acquired_epoch,
        renewed_epoch=time.time(),
    )
    renewed = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(replacement, sort_keys=True),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if renewed.returncode:
        _fail("could not renew the selected run lifecycle lock")


class _LifecycleLockHeartbeat:
    def __init__(
        self,
        context: str,
        namespace: str,
        name: str,
        holder: str,
    ) -> None:
        self.context = context
        self.namespace = namespace
        self.name = name
        self.holder = holder
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(_LIFECYCLE_LOCK_RENEW_SECONDS):
            try:
                _renew_lifecycle_lock(
                    self.context,
                    self.namespace,
                    self.name,
                    self.holder,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced by assert_healthy
                with self.state_lock:
                    self.failure = exc
                # A missed renewal makes exclusive ownership uncertain even if a
                # later API request might recover. Latch the failure and stop.
                return

    def assert_healthy(self) -> None:
        with self.state_lock:
            failure = self.failure
        if failure is not None:
            raise RuntimeError(
                "selected run lifecycle lock renewal failed"
            ) from failure

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS + 5.0)
        if self.thread.is_alive():
            _fail("selected run lifecycle lock renewal did not stop")
        self.assert_healthy()


def _record_lock_cleanup_failures(
    evidence: dict[str, Any],
    failures: list[tuple[str, Exception]],
) -> None:
    if not failures:
        return
    evidence["outcome"] = "failure"
    evidence.setdefault("failure", "lifecycle lock cleanup failed")
    evidence["lock_cleanup_failure"] = [
        {"operation": label, **_evidence_failure(lock_exc, "lock_cleanup_error")}
        for label, lock_exc in failures
    ]


def _release_lifecycle_lock(
    context: str,
    namespace: str,
    name: str,
    holder: str,
) -> None:
    current = _json_result(
        _kubectl(
            context,
            namespace,
            ["get", "secret", name, "-o", "json"],
            timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
        ),
        "lifecycle lock lookup",
    )
    if _lifecycle_lock_values(current)["holder"] != holder:
        _fail("selected run lifecycle lock ownership changed")
    metadata = current.get("metadata") or {}
    resource_version = str(metadata.get("resourceVersion") or "")
    uid = str(metadata.get("uid") or "")
    if not resource_version or not uid:
        _fail("selected run lifecycle lock has no release preconditions")
    # Match the CLI protocol exactly: atomically replace the owned lock with a
    # fresh quarantine marker, then delete it. A contender cannot acquire the
    # fresh marker between those calls. If deletion fails, the marker ages into
    # a safely reclaimable lock instead of becoming permanent.
    released_epoch = time.time()
    released = _kubectl(
        context,
        namespace,
        ["replace", "-f", "-"],
        stdin=json.dumps(
            _lifecycle_lock_document(
                namespace=namespace,
                name=name,
                holder="",
                resource_version=resource_version,
                acquired_epoch=released_epoch,
                renewed_epoch=released_epoch,
                uid=uid,
            ),
            sort_keys=True,
        ),
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if released.returncode:
        _fail("could not release the selected run lifecycle lock")
    deleted = _kubectl(
        context,
        namespace,
        ["delete", "secret", name, "--ignore-not-found=true"],
        timeout=_LIFECYCLE_LOCK_IO_TIMEOUT_SECONDS,
    )
    if deleted.returncode:
        _fail("could not remove the released run lifecycle lock")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expiry-delay-seconds", type=float, default=25.0)
    args = parser.parse_args()
    if not 15.0 <= args.expiry_delay_seconds <= 120.0:
        parser.error("--expiry-delay-seconds must be between 15 and 120")

    run_id = validate_run_id(args.run_id)
    deployment = resource_name(run_id)
    evidence: dict[str, Any] = {
        "schema": "npa.leisaac.relay-lifecycle-live.v1",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "outcome": "failure",
    }
    original_nonce = ""
    recovery_nonce = secrets.token_hex(32)
    original_manifest_body = b""
    original_manifest: dict[str, Any] = {}
    manifest_uri = ""
    mutated = False
    restored = False
    replica_restore = _ReplicaRestoration(args.context, args.namespace, deployment)
    lock_holder = secrets.token_hex(16)
    lock_name = ""
    lock_heartbeat: _LifecycleLockHeartbeat | None = None
    browser_connections: list[Any] = []
    ssh: Any = None
    host = ""
    auth_user = ""
    auth_password = ""
    secret_config: dict[str, str] = {}
    storage: dict[str, str] = {}
    try:
        service = _json_result(
            _kubectl(
                args.context,
                args.namespace,
                ["get", "service", f"{deployment}-relay", "-o", "json"],
            ),
            "relay Service lookup",
        )
        annotations = (service.get("metadata") or {}).get("annotations") or {}
        if (
            annotations.get("npa.nebius.com/agent-project") != args.project
            or annotations.get("npa.nebius.com/agent-name") != args.name
        ):
            _fail("relay Service ownership does not match the selected agent")

        # Acquire and start renewing before reading any credential-bound state.
        # Otherwise another verifier could rotate and restore the relay between
        # our snapshot and lock acquisition, leaving this proof with stale data.
        _require_lifecycle_lock_permissions(args.context, args.namespace)
        lock_name = _acquire_lifecycle_lock(
            args.context,
            args.namespace,
            deployment,
            lock_holder,
        )
        lock_heartbeat = _LifecycleLockHeartbeat(
            args.context,
            args.namespace,
            lock_name,
            lock_holder,
        )
        lock_heartbeat.start()
        replica_restore.progress_check = lock_heartbeat.assert_healthy

        _instance, host, ssh, auth_user, auth_password = _agent_relay_context(
            args.project, args.name
        )
        storage = _agent_artifact_storage(args.project, args.name)
        relay_metadata = _relay_metadata(ssh)
        if relay_metadata.get("run_id") != run_id:
            _fail("agent relay is owned by another run")
        if relay_metadata.get("expires_at"):
            _fail("existing relay already has an operator-defined expiry")
        manifest_uri = str(relay_metadata.get("manifest_uri") or "")
        parsed_uri = urlparse(manifest_uri)
        if parsed_uri.scheme != "s3" or not parsed_uri.netloc:
            _fail("agent relay has no valid manifest URI")
        original_manifest, original_manifest_body = _get_manifest(
            storage=storage, manifest_uri=manifest_uri
        )
        if original_manifest.get("run_id") != run_id:
            _fail("relay manifest belongs to another run")
        secret_config = _secret_config(
            args.context, args.namespace, f"{deployment}-relay-client"
        )
        original_nonce = str(secret_config.get("session_nonce") or "")
        certificate_sha256 = (
            str(secret_config.get("certificate_sha256") or "").strip().lower()
        )
        if len(original_nonce) != 64 or original_manifest.get(
            "session_attestation"
        ) != session_attestation(original_nonce):
            _fail("relay Secret and public manifest attestation do not agree")
        if len(certificate_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in certificate_sha256
        ):
            _fail("relay Secret has no valid agent certificate fingerprint")

        health_http, health = _http_status(
            host=host,
            user=auth_user,
            password=auth_password,
            path="/api/health",
            certificate_sha256=certificate_sha256,
        )
        control, video, status = _browser_sockets(
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
        )
        browser_connections.extend((control, video))
        try:
            _require_healthy_idle(
                health_http=health_http,
                health=health,
                status=status,
            )
        except Exception:
            control.close()
            video.close()
            raise
        client_id = f"relay-live-proof-{secrets.token_hex(8)}"
        resume = _resume(
            control,
            run_id=run_id,
            client_id=client_id,
            last_acked_seq=0,
        )
        lease_id = str(resume.get("lease_id") or "")
        if len(lease_id) != 64 or any(
            character not in "0123456789abcdef" for character in lease_id
        ):
            _fail("runtime did not issue a valid controller lease")
        sequence = int(resume.get("next_seq") or 1)
        # Treat the browser control connection as mutating before its first send:
        # a transport failure can occur after the runtime accepted the key but
        # before the client observed the acknowledgement.
        mutated = True
        baseline_frame_bytes = _press_and_read_frame(
            control,
            video,
            run_id=run_id,
            client_id=client_id,
            sequence=sequence,
        )
        restart_input_events = _input_event_count(
            _browser_status(
                host=host,
                run_id=run_id,
                user=auth_user,
                password=auth_password,
                certificate_sha256=certificate_sha256,
                timeout=30.0,
            ),
            phase="relay restart baseline",
        )
        lock_heartbeat.assert_healthy()

        (
            restart_disconnect,
            restart_started,
            restart_release_deadline,
        ) = _restart_relay_for_release_proof(
            ssh,
            control,
            run_id=run_id,
            session_nonce=original_nonce,
            manifest_uri=manifest_uri,
            progress_check=lock_heartbeat.assert_healthy,
        )
        video.close()
        _wait_relay_and_release(
            ssh,
            original_nonce,
            release_deadline=restart_release_deadline,
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
            after_input_events=restart_input_events,
            phase="relay restart",
            progress_check=lock_heartbeat.assert_healthy,
        )
        restart_release_seconds = time.monotonic() - restart_started
        control, video, recovered_status, resume = _wait_resumed_browser_sockets(
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
            client_id=client_id,
            last_acked_seq=sequence,
            lease_id=lease_id,
            progress_check=lock_heartbeat.assert_healthy,
        )
        browser_connections.extend((control, video))
        restart_forced_releases = _forced_release_count(
            resume, pressed_sequence=sequence, phase="relay restart"
        )
        lease_id = _require_rotated_lease(
            resume, prior_lease_id=lease_id, phase="relay restart"
        )
        restart_frame_bytes = _frame(video, run_id)
        evidence["restart"] = {
            "disconnect_ms": round(restart_disconnect * 1000, 3),
            "release_settle_seconds": round(restart_release_seconds, 3),
            "recovery_seconds": round(time.monotonic() - restart_started, 3),
            "forced_release_count": restart_forced_releases,
            "frame_bytes_after_reconnect": restart_frame_bytes,
            "recorder_state": str(
                (recovered_status.get("recorder") or {}).get("state") or ""
            ),
        }
        control.close()
        video.close()

        expires_at = (
            datetime.now(timezone.utc)
            + timedelta(seconds=float(args.expiry_delay_seconds))
        ).isoformat()
        lock_heartbeat.assert_healthy()
        _install_agent_relay(
            ssh,
            run_id=run_id,
            session_nonce=original_nonce,
            expires_at=expires_at,
            manifest_uri=manifest_uri,
        )
        _wait_relay_ready(
            ssh,
            original_nonce,
            progress_check=lock_heartbeat.assert_healthy,
        )
        control, video, _expiry_status = _browser_sockets(
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
        )
        browser_connections.extend((control, video))
        resume = _resume(
            control,
            run_id=run_id,
            client_id=client_id,
            last_acked_seq=int(resume.get("last_accepted_seq") or sequence + 1),
            lease_id=str(resume.get("lease_id") or lease_id),
        )
        expiry_sequence = int(resume.get("next_seq") or 1)
        expiry_frame_bytes = _press_and_read_frame(
            control,
            video,
            run_id=run_id,
            client_id=client_id,
            sequence=expiry_sequence,
        )
        # _expiry_status was fetched before the credential lifetime was armed
        # with any mutable input. The acknowledged press appends exactly one
        # input, so derive its baseline locally: another HTTPS request here
        # could consume the intentionally short expiry window being tested.
        expiry_input_events = (
            _input_event_count(_expiry_status, phase="credential expiry baseline") + 1
        )
        expiry_disconnect = _wait_closed(
            control,
            timeout=float(args.expiry_delay_seconds) + 20.0,
            progress_check=lock_heartbeat.assert_healthy,
        )
        expiry_release_started = time.monotonic()
        expiry_release_deadline = (
            expiry_release_started + _SAFETY_RELEASE_TIMEOUT_SECONDS
        )
        video.close()
        if (
            _release_deadline_remaining(
                expiry_release_deadline, phase="credential expiry"
            )
            < 6.0
        ):
            _fail("credential expiry safety-release observation window was exhausted")
        time.sleep(6.0)
        expired_http, expired_status = _http_status(
            host=host,
            user=auth_user,
            password=auth_password,
            path=f"/api/leisaac/status?run_id={run_id}",
            certificate_sha256=certificate_sha256,
            timeout=_release_deadline_remaining(
                expiry_release_deadline, phase="credential expiry"
            ),
        )
        if expired_http != 200 or expired_status.get("available") is not False:
            _fail("expired capability remained available through the agent")

        # Restore the same relay credential only long enough to resume the same
        # browser lease.  This proves the runtime's disconnect handler durably
        # synthesized the held-key release before any pod restart can clear the
        # in-memory ledger.  The credential is rotated immediately afterwards,
        # and the old value is then tested as stale in isolation.
        lock_heartbeat.assert_healthy()
        _install_agent_relay(
            ssh,
            run_id=run_id,
            session_nonce=original_nonce,
            expires_at="",
            manifest_uri=manifest_uri,
        )
        _wait_relay_and_release(
            ssh,
            original_nonce,
            release_deadline=expiry_release_deadline,
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
            after_input_events=expiry_input_events,
            phase="credential expiry",
            progress_check=lock_heartbeat.assert_healthy,
        )
        expiry_release_seconds = time.monotonic() - expiry_release_started
        (
            expiry_control,
            expiry_video,
            _restored_expiry_status,
            expiry_resume,
        ) = _wait_resumed_browser_sockets(
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
            client_id=client_id,
            last_acked_seq=expiry_sequence,
            lease_id=str(resume.get("lease_id") or lease_id),
            progress_check=lock_heartbeat.assert_healthy,
        )
        browser_connections.extend((expiry_control, expiry_video))
        expiry_forced_releases = _forced_release_count(
            expiry_resume,
            pressed_sequence=expiry_sequence,
            phase="credential expiry",
        )
        _require_rotated_lease(
            expiry_resume,
            prior_lease_id=str(resume.get("lease_id") or lease_id),
            phase="credential expiry",
        )
        _frame(expiry_video, run_id)
        expiry_control.close()
        expiry_video.close()

        recovery_body = _rotated_manifest(original_manifest, recovery_nonce)
        recovery_manifest_uri = _rotated_manifest_uri(manifest_uri, recovery_nonce)
        recovery_started = time.monotonic()
        lock_heartbeat.assert_healthy()
        replica_restore.scale_down()
        _wait_relay_connection(
            ssh,
            connected=False,
            progress_check=lock_heartbeat.assert_healthy,
        )
        _rotate(
            context=args.context,
            namespace=args.namespace,
            run_id=run_id,
            project=args.project,
            agent_name=args.name,
            deployment=deployment,
            host=host,
            ssh=ssh,
            auth_user=auth_user,
            auth_password=auth_password,
            secret_config=secret_config,
            storage=storage,
            manifest_uri=recovery_manifest_uri,
            manifest_body=recovery_body,
            nonce=recovery_nonce,
            wait_for_runtime=False,
            progress_check=lock_heartbeat.assert_healthy,
        )

        # With the Deployment scaled to zero there is no legitimate sidecar to
        # occupy the relay. Prove the new credential attaches, disconnect it,
        # then prove the expired credential is rejected in the same isolated
        # state. This distinguishes nonce authentication from the relay's
        # single-active-backhaul policy.
        current_probe = _open_backhaul_probe(
            host=host,
            user=auth_user,
            password=auth_password,
            nonce=recovery_nonce,
            certificate_sha256=certificate_sha256,
        )
        try:
            _wait_relay_connection(
                ssh,
                connected=True,
                progress_check=lock_heartbeat.assert_healthy,
            )
        finally:
            current_probe.close()
        _wait_relay_connection(
            ssh,
            connected=False,
            progress_check=lock_heartbeat.assert_healthy,
        )
        stale_denied = _stale_backhaul_denied(
            host=host,
            user=auth_user,
            password=auth_password,
            stale_nonce=original_nonce,
            certificate_sha256=certificate_sha256,
            relay_connected=lambda: _relay_connection_state(ssh),
        )
        if not stale_denied:
            _fail("the expired relay credential authenticated after rotation")
        _wait_relay_connection(
            ssh,
            connected=False,
            progress_check=lock_heartbeat.assert_healthy,
        )
        replica_restore.scale_up()
        _wait_relay_ready(
            ssh,
            recovery_nonce,
            progress_check=lock_heartbeat.assert_healthy,
        )
        recovery_seconds = time.monotonic() - recovery_started
        time.sleep(6.0)
        final_control, final_video, final_status = _browser_sockets(
            host=host,
            run_id=run_id,
            user=auth_user,
            password=auth_password,
            certificate_sha256=certificate_sha256,
        )
        browser_connections.extend((final_control, final_video))
        final_resume = _resume(
            final_control,
            run_id=run_id,
            client_id=f"relay-restored-{secrets.token_hex(8)}",
            last_acked_seq=0,
        )
        final_frame_bytes = _frame(final_video, run_id)
        final_control.close()
        final_video.close()
        final_health_http, _final_health = _http_status(
            host=host,
            user=auth_user,
            password=auth_password,
            path="/api/health",
            certificate_sha256=certificate_sha256,
        )
        if (
            final_health_http != 200
            or final_status.get("available") is not True
            or str((final_status.get("recorder") or {}).get("state") or "") != "idle"
            or not final_resume.get("lease_id")
        ):
            _fail("restored relay did not return the complete healthy browser contract")
        restored = True
        evidence.update(
            baseline={
                "agent_health_http": health_http,
                "agent_health_ok": bool(health.get("ok")),
                "leisaac_available": bool(status.get("available")),
                "frame_bytes": baseline_frame_bytes,
                "recorder_state": str(
                    (status.get("recorder") or {}).get("state") or ""
                ),
            },
            expiry={
                "disconnect_ms": round(expiry_disconnect * 1000, 3),
                "release_settle_seconds": round(expiry_release_seconds, 3),
                "frame_bytes_before_expiry": expiry_frame_bytes,
                "expired_status_http": expired_http,
                "expired_capability_available": bool(expired_status.get("available")),
                "forced_release_count": expiry_forced_releases,
                "current_credential_accepted_in_isolation": True,
                "stale_credential_denied_after_rotation": stale_denied,
                "recovery_seconds": round(recovery_seconds, 3),
            },
            final={
                "agent_health_http": final_health_http,
                "leisaac_status_http": 200,
                "leisaac_available": True,
                "frame_bytes": final_frame_bytes,
                "controls_unlocked": True,
                "recorder_state": "idle",
            },
            outcome="success",
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - restore must span every live failure
        # The original exception continues to stderr via the re-raise. Evidence
        # is durable and may be shared, so retain only a stable classification,
        # never provider hosts, namespaces, buckets, paths, or credentials.
        evidence["failure"] = "operational error"
        evidence["failure_detail"] = _evidence_failure(exc, "operational_error")
        raise
    finally:
        active_failure = sys.exc_info()[1]
        # Disconnect is the runtime's safety-release signal. Do this before any
        # credential, Deployment, or lock recovery that can block or retry.
        browser_cleanup_failures = _close_browser_connections(browser_connections)
        if browser_cleanup_failures:
            evidence["browser_cleanup_failure"] = [
                {
                    "operation": label,
                    **_evidence_failure(close_exc, "browser_cleanup_error"),
                }
                for label, close_exc in browser_cleanup_failures
            ]
        if (
            mutated
            and not restored
            and all(
                (
                    original_nonce,
                    original_manifest_body,
                    manifest_uri,
                    host,
                    auth_user,
                    auth_password,
                    secret_config,
                    storage,
                    ssh,
                )
            )
        ):
            rollback_started = time.monotonic()

            def restore_credential() -> None:
                _rotate(
                    context=args.context,
                    namespace=args.namespace,
                    run_id=run_id,
                    project=args.project,
                    agent_name=args.name,
                    deployment=deployment,
                    host=host,
                    ssh=ssh,
                    auth_user=auth_user,
                    auth_password=auth_password,
                    secret_config=secret_config,
                    storage=storage,
                    manifest_uri=manifest_uri,
                    manifest_body=None,
                    nonce=original_nonce,
                    wait_for_runtime=not replica_restore.required,
                )

            def restore_replica() -> None:
                if replica_restore.required:
                    replica_restore.scale_up(enforce_lease=False)

            def verify_restored_relay() -> None:
                if not replica_restore.required:
                    _wait_relay_ready(ssh, original_nonce)

            rollback_failures = _best_effort_cleanup(
                (
                    ("restore original credential", restore_credential),
                    ("restore Deployment replica", restore_replica),
                    ("verify restored relay", verify_restored_relay),
                )
            )
            if not rollback_failures:
                evidence["rollback_seconds"] = round(
                    time.monotonic() - rollback_started,
                    3,
                )
                evidence["rollback_restored_original_credential"] = True
            else:
                evidence["rollback_failure"] = [
                    {
                        "operation": label,
                        **_evidence_failure(rollback_exc, "rollback_error"),
                    }
                    for label, rollback_exc in rollback_failures
                ]
        lock_cleanup_failures: list[tuple[str, Exception]] = []
        if lock_heartbeat is not None:
            try:
                lock_heartbeat.stop()
            except Exception as lock_stop_exc:  # noqa: BLE001 - release still runs
                lock_cleanup_failures.append(("stop lock renewal", lock_stop_exc))
        if lock_name:
            try:
                _release_lifecycle_lock(
                    args.context,
                    args.namespace,
                    lock_name,
                    lock_holder,
                )
            except Exception as lock_exc:  # noqa: BLE001 - preserve prior failure
                lock_cleanup_failures.append(("release lifecycle lock", lock_exc))
        _record_lock_cleanup_failures(evidence, lock_cleanup_failures)
        if browser_cleanup_failures:
            evidence["outcome"] = "failure"
            evidence.setdefault("failure", "browser safety cleanup failed")
        evidence["completed_at"] = datetime.now(timezone.utc).isoformat()
        _publish_evidence(
            args.evidence,
            evidence,
            active_failure=(
                active_failure
                or (
                    browser_cleanup_failures[0][1] if browser_cleanup_failures else None
                )
                or (lock_cleanup_failures[0][1] if lock_cleanup_failures else None)
            ),
        )
        if active_failure is None and browser_cleanup_failures:
            raise browser_cleanup_failures[0][1]
        if active_failure is None and lock_cleanup_failures:
            raise lock_cleanup_failures[0][1]


if __name__ == "__main__":
    raise SystemExit(main())
