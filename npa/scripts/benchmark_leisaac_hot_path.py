#!/usr/bin/env python3
"""Reproducible local microbenchmarks for LeIsaac's latency-critical plumbing."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import tempfile
import time

from npa.agent_backend.leisaac_transport import AsyncLatestByKey
from npa.workbench.leisaac.reverse_client import _mask_websocket_payload


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_BYTES = 180 * 1024
SAMPLES = 200


def distribution(samples: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered) / 1_000_000,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] / 1_000_000,
    }


def timed(function, samples: int = SAMPLES) -> dict[str, float | int]:
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        function()
        elapsed.append(time.perf_counter_ns() - started)
    return distribution(elapsed)


def runtime_module():
    path = ROOT / "npa/docker/workbench/leisaac/session_server.py"
    spec = importlib.util.spec_from_file_location("leisaac_benchmark_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counter_benchmark() -> dict[str, float | int]:
    runtime = runtime_module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime.INPUT_COUNTER_PATH = root / "counter"
        runtime.INPUT_QUEUE_PATH = root / "inputs.jsonl"
        runtime.INPUT_COUNTER_PATH.write_text("0\n", encoding="utf-8")
        rows = [
            {"type": "control", "event": "release", "seq": index}
            for index in range(8)
        ]
        return timed(lambda: runtime._append_inputs(rows), samples=40)


def ipc_benchmark(payload: bytes, *, samples: int = SAMPLES) -> dict[str, float | int]:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        return timed(
            lambda: (sender.send(payload), receiver.recv(PAYLOAD_BYTES + 4096)),
            samples=samples,
        )
    finally:
        sender.close()
        receiver.close()


async def relay_benchmark() -> dict[str, float | int]:
    latest = AsyncLatestByKey(("workspace", "overview"))
    elapsed: list[int] = []
    sequences = {"workspace": 0, "overview": 0}
    for sequence in range(1, SAMPLES + 1):
        started = time.perf_counter_ns()
        await latest.publish("workspace", (sequence, b"frame"))
        key, generation, _value, _dropped, _next_index = await latest.wait_after(
            sequences
        )
        sequences[key] = generation
        elapsed.append(time.perf_counter_ns() - started)
    return distribution(elapsed)


def main() -> int:
    payload = os.urandom(PAYLOAD_BYTES)
    mask = os.urandom(4)
    report = {
        "schema": "npa.leisaac.hot-path-benchmark.v1",
        "payload_bytes": PAYLOAD_BYTES,
        "mask_180k": timed(lambda: _mask_websocket_payload(payload, mask)),
        "sha256_180k": timed(lambda: hashlib.sha256(payload).digest()),
        "counter_append_8_releases": counter_benchmark(),
        "unix_datagram_wakeup": ipc_benchmark(
            b'{"type":"frame","camera":"workspace"}'
        ),
        "unix_datagram_frame_180k": ipc_benchmark(payload, samples=40),
        "latest_value_relay_schedule": asyncio.run(relay_benchmark()),
        "browser_decode_paint": "measured by agent_leisaac_performance_live.cy.js",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
