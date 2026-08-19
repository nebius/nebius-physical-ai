#!/usr/bin/env python3
"""Run one real pi0.5-DROID request from a verified runtime cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np
from openpi_client import websocket_client_policy


CACHE_ROOT = os.environ.get("NPA_OPENPI_CACHE_ROOT", "/opt/npa-model-cache/openpi")
PORT = int(os.environ.get("NPA_OPENPI_SMOKE_PORT", "8000"))


def _observation() -> dict[str, object]:
    frame = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    return {
        "observation/exterior_image_1_left": np.asarray(frame % 251, dtype=np.uint8),
        "observation/wrist_image_left": np.asarray(
            np.flip(frame, axis=1) % 253, dtype=np.uint8
        ),
        "observation/joint_position": np.asarray(
            [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
            dtype=np.float32,
        ),
        "observation/gripper_position": np.asarray([0.04], dtype=np.float32),
        "prompt": "pick up the fork",
    }


def _wait_ready(process: subprocess.Popen[bytes], deadline: float) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("OpenPI server exited before its health endpoint was ready")
        try:
            with opener.open(f"http://127.0.0.1:{PORT}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("OpenPI server health deadline expired")


def main() -> int:
    if not Path(CACHE_ROOT).is_dir():
        raise RuntimeError("the verified runtime cache mount is absent")
    subprocess.run(["/usr/local/bin/npa-openpi-sm100-probe"], check=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "/opt/npa-openpi/openpi_policy_server.py",
            "--cache-root",
            CACHE_ROOT,
            "--port",
            str(PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_ready(process, time.monotonic() + 900)
        started = time.perf_counter()
        response = websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1", PORT
        ).infer(_observation())
        elapsed_ms = (time.perf_counter() - started) * 1000
        actions = np.asarray(response.get("actions"))
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise RuntimeError("policy response must contain finite [15,8] actions")
        print(
            json.dumps(
                {
                    "schema": "npa.openpi-policy.golden-eval.v1",
                    "status": "passed",
                    "action_shape": [15, 8],
                    "finite": True,
                    "round_trip_ms": round(elapsed_ms, 3),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
