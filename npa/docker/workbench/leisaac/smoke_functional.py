#!/usr/bin/env python3
"""Golden eval: start real PickOrange keyboard teleoperation and WebRTC."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from leisaac_registry import REGISTRY_FINGERPRINT


def main() -> int:
    environment = os.environ.copy()
    environment.setdefault("NPA_LEISAAC_RUN_ID", "leisaac-golden-eval")
    environment.setdefault("NPA_LEISAAC_SESSION_NONCE", "a" * 64)
    environment.setdefault("NPA_LEISAAC_MEDIA_HOST", "127.0.0.1")
    environment.setdefault("NPA_LEISAAC_TASK", "LeIsaac-SO101-PickOrange-v0")
    environment.setdefault("NPA_LEISAAC_ENVIRONMENT_ID", "golden-eval")
    environment.setdefault("NPA_LEISAAC_ENVIRONMENT_INDEX", "0")
    environment.setdefault("NPA_LEISAAC_SEED", "42")
    environment.setdefault("NPA_LEISAAC_NUM_ENVS", "1")
    environment.setdefault("NPA_LEISAAC_REGISTRY_FINGERPRINT", REGISTRY_FINGERPRINT)
    environment.setdefault("NPA_LEISAAC_OUTPUT_PATH", "s3://npa-smoke/leisaac")
    process = subprocess.Popen(
        [
            "/opt/npa/sim/venv/bin/python",
            "/opt/npa/leisaac/session_server.py",
        ],
        env=environment,
    )
    try:
        while process.poll() is None:
            try:
                nonce = environment.get("NPA_LEISAAC_SESSION_NONCE", "a" * 64)
                req = urllib.request.Request("http://127.0.0.1:8080/status")
                req.add_header("x-npa-leisaac-nonce", nonce)
                with urllib.request.urlopen(req) as response:
                    status = json.loads(response.read().decode("utf-8"))
                if (
                    status.get("state") == "ready"
                    and status.get("webrtc_ready") is True
                ):
                    if status.get("task") != "LeIsaac-SO101-PickOrange-v0":
                        raise RuntimeError(f"wrong real task: {status}")
                    if status.get("teleop_device") != "keyboard":
                        raise RuntimeError(f"wrong teleoperation device: {status}")
                    if "RTX PRO 6000" not in str(status.get("gpu") or ""):
                        raise RuntimeError(
                            f"current LeIsaac launcher requires RTX PRO 6000: {status}"
                        )
                    print(json.dumps(status, indent=2, sort_keys=True))
                    print("NPA_LEISAAC_PICK_ORANGE_KEYBOARD_WEBRTC_OK")
                    return 0
            except (OSError, urllib.error.HTTPError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError(f"LeIsaac service exited before ready: {process.returncode}")
    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
