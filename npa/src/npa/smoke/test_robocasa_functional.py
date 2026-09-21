"""RoboCasa container functional golden eval.

Starts the RoboCasa FastAPI service, proves the read-only service contract, and
checks real upstream task registration in a fresh process. It deliberately does
not launch an asset-heavy simulation run so the golden eval stays fast and
GPU-optional; EGL reset and rollout remain live workflow gates.

Run inside the npa-robocasa image with:
    python -m npa.smoke.test_robocasa_functional
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SERVER_TARGET = os.environ.get(
    "ROBOCASA_SMOKE_APP", "npa.workbench.robocasa.service:app"
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


def _registration_worker(sender: Any, _request_payload: dict[str, Any]) -> None:
    """Exercise production spawn and process-group cleanup without S3."""
    try:
        os.setsid()
        from npa.workbench.robocasa.capabilities import kitchen_task_registration
        from npa.workbench.robocasa.service import _send_worker_message

        result = kitchen_task_registration(download_assets=False)
        _send_worker_message(sender, {"kind": "result", "result": result})
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    except Exception as exc:
        try:
            from npa.workbench.robocasa.service import _send_worker_message

            _send_worker_message(
                sender,
                {"kind": "error", "error": f"{type(exc).__name__}: {exc}"},
            )
            sender.recv_bytes(1)
        except (EOFError, OSError):
            pass
    finally:
        sender.close()


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
        return CheckResult("start robocasa service", False, "uvicorn not found on PATH")

    env = {**os.environ, "ROBOCASA_AUTH_MODE": "none"}
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
                "start robocasa service",
                False,
                f"service exited with {state.process.returncode}; log:\n{_tail(state.server_log)}",
            )
        try:
            health = _request_json(f"{_base_url(state)}/health", timeout=5)
            if health.get("status") != "ok":
                return CheckResult(
                    "start robocasa service", False, f"unhealthy: {health}"
                )
            return CheckResult(
                "start robocasa service", True, json.dumps(health, sort_keys=True)
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = _format_exception(exc)
            time.sleep(1)
    return CheckResult(
        "start robocasa service",
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


def check_non_root_asset_writability(_state: SmokeState) -> CheckResult:
    script = """
import json
import os
import tempfile
from npa.workbench.robocasa.capabilities import _assets_root

root = _assets_root()
if os.geteuid() == 0:
    raise RuntimeError("RoboCasa golden eval must run as a non-root user")
root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(prefix=".npa-write-", dir=root, delete=True) as handle:
    handle.write(b"ok")
    handle.flush()
print(json.dumps({"euid": os.geteuid(), "assets_root": str(root)}))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("non-root asset writability", False, _format_exception(exc))
    if completed.returncode != 0:
        return CheckResult(
            "non-root asset writability",
            False,
            (completed.stderr or completed.stdout).strip(),
        )
    return CheckResult(
        "non-root asset writability", True, completed.stdout.strip()[:300]
    )


def check_task_registration(_state: SmokeState) -> CheckResult:
    try:
        from npa.workbench.robocasa.schemas import RoboCasaRunRequest
        from npa.workbench.robocasa.service import _execute_capability_in_worker

        request = RoboCasaRunRequest(
            capability="kitchen_task_registration",
            output_uri="s3://robocasa-golden.invalid/registration",
            download_assets=False,
            timeout_seconds=90,
        )
        outcome = _execute_capability_in_worker(
            request, worker_target=_registration_worker
        )
    except Exception as exc:
        return CheckResult("fresh task registration", False, _format_exception(exc))
    if outcome.error is not None or outcome.result is None or not outcome.stopped:
        return CheckResult(
            "fresh task registration",
            False,
            str(outcome.error or "worker did not return a stopped result"),
        )
    if int(outcome.result.get("registered_env_count") or 0) < 1:
        return CheckResult(
            "fresh task registration",
            False,
            f"no RoboCasa environments registered: {outcome.result}",
        )
    return CheckResult(
        "fresh task registration",
        True,
        json.dumps(outcome.result, sort_keys=True)[:300],
    )


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
    root = Path(tempfile.mkdtemp(prefix="npa_robocasa_smoke_"))
    state = SmokeState(
        server_log=root / "service.log",
        port=int(os.environ.get("ROBOCASA_SMOKE_PORT", "8791")),
    )
    print(f"Smoke workspace: {root}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_start_server,
        check_system_info,
        check_non_root_asset_writability,
        check_task_registration,
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
