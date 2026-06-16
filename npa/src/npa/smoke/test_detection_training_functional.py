"""Detection-training container functional golden eval.

Starts the detection-training FastAPI service and proves the read-only service
contract: health and system-info. It deliberately does not launch a training run
so the golden eval stays fast and GPU-optional; the heavier train/eval paths are
covered by the BDD100K pipeline tests.

Run inside the npa-detection-training image with:
    python -m npa.smoke.test_detection_training_functional
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SERVER_TARGET = os.environ.get(
    "DETECTION_TRAINING_SMOKE_APP", "npa.workbench.detection_training.service:app"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    server_log: Path
    port: int
    process: subprocess.Popen[str] | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _request_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    if len(text) <= limit:
        return text.strip()
    return "...<truncated>...\n" + text[-limit:].strip()


def _base_url(state: SmokeState) -> str:
    return f"http://127.0.0.1:{state.port}"


def check_start_server(state: SmokeState) -> CheckResult:
    uvicorn = shutil.which("uvicorn")
    if uvicorn is None:
        return CheckResult("start detection-training service", False, "uvicorn not found on PATH")

    env = {**os.environ, "DETECTION_TRAINING_AUTH_MODE": "none"}
    log_handle = state.server_log.open("w")
    state.process = subprocess.Popen(
        [uvicorn, SERVER_TARGET, "--host", "127.0.0.1", "--port", str(state.port)],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 90
    last_error = ""
    while time.time() < deadline:
        if state.process.poll() is not None:
            return CheckResult(
                "start detection-training service",
                False,
                f"service exited with {state.process.returncode}; log:\n{_tail(state.server_log)}",
            )
        try:
            health = _request_json(f"{_base_url(state)}/health", timeout=5)
            if health.get("status") != "ok":
                return CheckResult(
                    "start detection-training service", False, f"unhealthy: {health}"
                )
            return CheckResult(
                "start detection-training service", True, json.dumps(health, sort_keys=True)
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = _format_exception(exc)
            time.sleep(1)
    return CheckResult(
        "start detection-training service",
        False,
        f"timed out waiting for /health; last error: {last_error}",
    )


def check_system_info(state: SmokeState) -> CheckResult:
    try:
        info = _request_json(f"{_base_url(state)}/system-info", timeout=30)
    except Exception as exc:
        return CheckResult("system-info", False, _format_exception(exc))
    if not isinstance(info, dict) or not info:
        return CheckResult("system-info", False, f"empty payload: {info}")
    return CheckResult("system-info", True, json.dumps(info, sort_keys=True)[:300])


def _stop_server(state: SmokeState) -> None:
    process = state.process
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="npa_detection_training_smoke_"))
    state = SmokeState(
        server_log=root / "service.log",
        port=int(os.environ.get("DETECTION_TRAINING_SMOKE_PORT", "8790")),
    )
    print(f"Smoke workspace: {root}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_start_server,
        check_system_info,
    ]
    results: list[CheckResult] = []
    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
            if not result.ok and check is check_start_server:
                break
    finally:
        _stop_server(state)
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(result.ok for result in results)
    total = len(checks)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
