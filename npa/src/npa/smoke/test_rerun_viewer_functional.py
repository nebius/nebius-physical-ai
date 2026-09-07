"""Functional CPU validation for robotics conversion and Rerun serving."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import rerun as rr


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[bytes]) -> bytes:
    last_error: Exception | None = None
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"Rerun server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return response.read(4096)
        except Exception as exc:  # service readiness is necessarily transient
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Rerun web viewer did not become ready: {last_error}")


def main() -> int:
    """Convert a robotics trace to RRD, read it, and serve it over HTTP."""

    source_sha = os.environ.get("NPA_IMAGE_SOURCE_SHA", "")
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise RuntimeError("NPA_IMAGE_SOURCE_SHA must be an exact lowercase 40-hex commit")

    output_dir = Path(os.environ.get("NPA_SMOKE_OUTPUT_DIR", "/tmp/npa-golden"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "robot-joint-trace.json"
    trace = {
        "schema": "npa.robotics.joint-trace.v1",
        "joint_names": ["shoulder", "elbow", "wrist"],
        "samples": [
            {"step": 0, "positions": [0.0, 0.25, -0.1]},
            {"step": 1, "positions": [0.1, 0.4, -0.2]},
            {"step": 2, "positions": [0.2, 0.55, -0.3]},
        ],
    }
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")

    rrd_path = output_dir / "robot-joint-trace.rrd"
    recording = rr.RecordingStream("npa_rerun_viewer_golden")
    recording.save(rrd_path)
    recording.log(
        "robot/joints/names",
        rr.TextDocument(",".join(trace["joint_names"])),
        static=True,
    )
    for sample in trace["samples"]:
        recording.set_time("control_step", sequence=int(sample["step"]))
        recording.log("robot/joints/positions", rr.Scalars(sample["positions"]))
    recording.flush()
    recording.disconnect()
    if not rrd_path.is_file() or rrd_path.stat().st_size == 0:
        raise RuntimeError("robotics trace conversion did not produce an RRD")

    verify = subprocess.run(
        ["rerun", "rrd", "verify", str(rrd_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    printed = subprocess.run(
        ["rerun", "rrd", "print", str(rrd_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "robot/joints/positions" not in printed.stdout:
        raise RuntimeError("Rerun CLI readback omitted the robotics joint entity")

    web_port = _free_port()
    grpc_port = _free_port()
    process = subprocess.Popen(
        [
            "rerun",
            str(rrd_path),
            "--serve-web",
            "--web-viewer",
            "--bind",
            "127.0.0.1",
            "--web-viewer-port",
            str(web_port),
            "--port",
            str(grpc_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        response = _wait_for_http(f"http://127.0.0.1:{web_port}/", process)
        if not response:
            raise RuntimeError("Rerun web viewer returned an empty response")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    report = {
        "schema": "npa.golden.rerun-viewer.v1",
        "source_sha": source_sha,
        "input_schema": trace["schema"],
        "sample_count": len(trace["samples"]),
        "rrd_bytes": rrd_path.stat().st_size,
        "rrd_sha256": hashlib.sha256(rrd_path.read_bytes()).hexdigest(),
        "verify_stdout": verify.stdout.strip(),
        "readback_entity": "robot/joints/positions",
        "web_read_verified": True,
        "status": "passed",
    }
    report_path = output_dir / "rerun-viewer-functional.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
