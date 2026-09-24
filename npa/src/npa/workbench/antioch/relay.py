"""Bounded double-WSS relay for an Antioch declared localhost port."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from pathlib import Path
from typing import Any, Sequence

from websockets.sync.client import connect

from .cluster_runtime import _state_ready
from .health import StateHealthServer

MAX_MESSAGE_BYTES = 32 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_REQUESTS_PER_CONNECTION = 1_000_000


class AntiochRelayError(RuntimeError):
    """The authenticated live relay failed closed."""


def _private_text(path: Path, *, minimum_length: int = 1) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise AntiochRelayError(f"private relay file {path.name!r} is unavailable")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < minimum_length:
        raise AntiochRelayError(f"private relay file {path.name!r} is malformed")
    return value


def _remote_settings(bundle: Path) -> tuple[str, str, ssl.SSLContext]:
    endpoint_path = bundle / "endpoint.json"
    if (
        not endpoint_path.is_file()
        or endpoint_path.is_symlink()
        or endpoint_path.stat().st_mode & 0o077
    ):
        raise AntiochRelayError("private policy endpoint file is unavailable")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    if set(endpoint) != {"scheme", "host", "port"} or endpoint["scheme"] != "wss":
        raise AntiochRelayError("policy endpoint contract is malformed")
    host = str(endpoint["host"]).strip()
    port = int(endpoint["port"])
    if not host or port != 443:
        raise AntiochRelayError("policy endpoint must be WSS on port 443")
    token = _private_text(bundle / "api-key", minimum_length=32)
    context = ssl.create_default_context(cafile=str(bundle / "ca.crt"))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return f"wss://{host}:{port}", token, context


def _local_settings(
    bundle: Path, *, local_port: int
) -> tuple[str, str, ssl.SSLContext]:
    if not 1 <= local_port <= 65535:
        raise AntiochRelayError("declared relay port is invalid")
    token = _private_text(bundle / "relay-api-key", minimum_length=32)
    context = ssl.create_default_context(cafile=str(bundle / "relay-ca.crt"))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return f"wss://127.0.0.1:{local_port}", token, context


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["heartbeat_unix"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        payload = (json.dumps(state, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def run_relay(
    *,
    bundle: Path,
    local_port: int,
    stop_file: Path,
    state_path: Path,
    owner_identity: str,
) -> dict[str, Any]:
    """Reconnect indefinitely and forward one bounded request/reply stream."""

    remote_uri, policy_token, remote_context = _remote_settings(bundle)
    local_uri, relay_token, local_context = _local_settings(
        bundle, local_port=local_port
    )
    state: dict[str, Any] = {
        "schema": "npa.workbench.antioch-live-relay.v2",
        "schema_version": 2,
        "owner_identity": owner_identity,
        "status": "starting",
        "connections": 0,
        "reconnects": 0,
        "forwarded_requests": 0,
        "failures": 0,
        "last_round_trip_ms": None,
        "last_error_type": None,
        "last_failed_phase": None,
    }
    _write_state(state_path, state)
    backoff = 1.0
    while not stop_file.exists():
        try:
            state["status"] = "connecting_simulation"
            _write_state(state_path, state)
            with connect(
                local_uri,
                ssl=local_context,
                compression=None,
                max_size=MAX_MESSAGE_BYTES,
                max_queue=2,
                open_timeout=10,
                close_timeout=5,
                additional_headers={
                    "Authorization": f"Api-Key {relay_token}",
                    "X-NPA-Relay-Role": "operator",
                },
                proxy=None,
            ) as simulation:
                state["status"] = "connecting_policy"
                _write_state(state_path, state)
                with connect(
                    remote_uri,
                    ssl=remote_context,
                    compression=None,
                    max_size=MAX_MESSAGE_BYTES,
                    max_queue=2,
                    open_timeout=10,
                    close_timeout=5,
                    additional_headers={"Authorization": f"Api-Key {policy_token}"},
                    proxy=None,
                ) as policy:
                    greeting = policy.recv(timeout=30)
                    simulation.send(greeting)
                    state["connections"] += 1
                    state["status"] = "connected"
                    state["last_error_type"] = None
                    state["last_failed_phase"] = None
                    _write_state(state_path, state)
                    print(
                        "NPA_ANTIOCH_RELAY_CONNECTED "
                        f"connections={state['connections']}",
                        flush=True,
                    )
                    backoff = 1.0
                    for _ in range(MAX_REQUESTS_PER_CONNECTION):
                        request = simulation.recv(timeout=REQUEST_TIMEOUT_SECONDS)
                        started = time.monotonic()
                        policy.send(request)
                        response = policy.recv(timeout=REQUEST_TIMEOUT_SECONDS)
                        simulation.send(response)
                        latency_ms = (time.monotonic() - started) * 1000.0
                        state["forwarded_requests"] += 1
                        state["last_round_trip_ms"] = round(latency_ms, 3)
                        _write_state(state_path, state)
                        print(
                            "NPA_ANTIOCH_RELAY_ROUND_TRIP "
                            f"requests={state['forwarded_requests']} "
                            f"latency_ms={latency_ms:.3f}",
                            flush=True,
                        )
        except Exception as exc:
            if stop_file.exists():
                break
            failed_phase = state.get("status")
            state["status"] = "reconnecting"
            state["reconnects"] += 1
            state["failures"] += 1
            state["last_error_type"] = type(exc).__name__
            state["last_failed_phase"] = failed_phase
            _write_state(state_path, state)
            print(
                "NPA_ANTIOCH_RELAY_RECONNECT "
                f"reconnects={state['reconnects']} reason={type(exc).__name__}",
                flush=True,
            )
            deadline = time.monotonic() + backoff
            while not stop_file.exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            backoff = min(backoff * 2.0, 5.0)
    state["status"] = "stopped"
    _write_state(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--local-port", type=int, default=18_444)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--owner-identity", required=True)
    parser.add_argument("--health-port", type=int, default=0)
    parser.add_argument("--resume-after-stop", action="store_true")
    return parser


def _publish_stopped_until_resumable(
    *, stop_file: Path, state_path: Path, state: dict[str, Any]
) -> None:
    """Keep terminal relay evidence fresh without restart-looping."""

    while stop_file.exists():
        _write_state(state_path, state)
        time.sleep(5)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stop_file = Path(args.stop_file)
    state_path = Path(args.state_path)
    health = None
    if args.health_port:
        health = StateHealthServer(
            port=args.health_port,
            checks={
                "/ready": lambda: _state_ready(
                    json.loads(state_path.read_text(encoding="utf-8")),
                    component="relay",
                    expected_owner_identity=args.owner_identity,
                    max_age_seconds=150.0,
                ),
                "/live": lambda: _state_ready(
                    json.loads(state_path.read_text(encoding="utf-8")),
                    component="relay-liveness",
                    expected_owner_identity=args.owner_identity,
                    max_age_seconds=240.0,
                ),
            },
        )
        health.start()
    try:
        while True:
            result = run_relay(
                bundle=Path(args.bundle),
                local_port=args.local_port,
                stop_file=stop_file,
                state_path=state_path,
                owner_identity=args.owner_identity,
            )
            print(json.dumps(result, sort_keys=True))
            if not args.resume_after_stop:
                return 0
            _publish_stopped_until_resumable(
                stop_file=stop_file, state_path=state_path, state=result
            )
    finally:
        if health is not None:
            health.close()


if __name__ == "__main__":
    raise SystemExit(main())
