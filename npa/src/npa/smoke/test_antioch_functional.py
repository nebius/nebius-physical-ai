"""Credential-free functional smoke for the Antioch control-plane image."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib import metadata


def main() -> int:
    token = "synthetic-golden-eval-token"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    env = os.environ.copy()
    env.update(
        {
            "ANTIOCH_WORKBENCH_AUTH_MODE": "token",
            "ANTIOCH_WORKBENCH_TOKEN": token,
        }
    )
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "npa.workbench.antioch.service:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        payload = None
        for _ in range(50):
            try:
                request = urllib.request.Request(
                    f"{base}/system-info",
                    headers={"Authorization": f"Bearer {token}"},
                )
                with urllib.request.urlopen(request) as response:  # noqa: S310
                    payload = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                if server.poll() is not None:
                    raise RuntimeError("Antioch control-plane server exited early")
                time.sleep(0.1)
        if payload is None:
            raise RuntimeError("Antioch control-plane server did not become ready")
        try:
            urllib.request.urlopen(f"{base}/system-info")  # noqa: S310
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise RuntimeError(
                    "Antioch control plane returned the wrong authentication status"
                ) from exc
        else:
            raise RuntimeError("Antioch control plane did not enforce authentication")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    if payload.get("cpu_only") is not True:
        raise RuntimeError("Antioch control plane did not report CPU-only operation")
    if payload.get("proprietary_payload_baked") is not False:
        raise RuntimeError("Antioch proprietary distribution is present in the image")
    try:
        metadata.version("antioch-sim")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("antioch-sim must not be baked into the public adapter")
    print(
        json.dumps(
            {
                "status": "passed",
                "auth_boundary": True,
                "cpu_only": True,
                "proprietary_payload_baked": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
