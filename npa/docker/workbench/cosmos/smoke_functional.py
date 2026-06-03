"""Cosmos container functional smoke checks."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_COSMOS_VERSION = os.environ.get("COSMOS_VERSION", "1.0.9")
DEFAULT_MODEL = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
DEFAULT_PROMPT = "A robot arm places a red cube on a table in a clean lab."
DEFAULT_MODEL_DIR = Path("/opt/cosmos-data/models")
DEFAULT_OUTPUT_DIR = Path("/opt/cosmos-data/outputs")
SERVER_CWD = Path("/opt/cosmos")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    output_dir: Path
    server_log: Path
    port: int
    process: subprocess.Popen[str] | None = None
    job_id: str = ""
    output_path: Path | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _model_slug(model: str) -> str:
    return model.replace("/", "--").replace(":", "--")


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    if len(text) <= limit:
        return text.strip()
    return "...<truncated>...\n" + text[-limit:].strip()


def check_cosmos_version(state: SmokeState) -> CheckResult:
    try:
        version = metadata.version("cosmos-predict2")
    except Exception as exc:
        return CheckResult("check cosmos version", False, _format_exception(exc))
    if version != EXPECTED_COSMOS_VERSION:
        return CheckResult(
            "check cosmos version",
            False,
            f"expected version: {EXPECTED_COSMOS_VERSION}; found: {version}",
        )
    return CheckResult("check cosmos version", True, f"version: {version}")


def check_model_weights(state: SmokeState) -> CheckResult:
    model = os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL)
    model_dir = Path(os.environ.get("COSMOS_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_path = model_dir / _model_slug(model)
    if not model_path.is_dir():
        return CheckResult("check mounted model weights", False, f"missing {model_path}")
    return CheckResult("check mounted model weights", True, str(model_path))


def check_start_server(state: SmokeState) -> CheckResult:
    uvicorn = shutil.which("uvicorn")
    if uvicorn is None:
        return CheckResult("start Cosmos server", False, "uvicorn not found on PATH")

    env = {
        **os.environ,
        "COSMOS_MODEL_ID": os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL),
        "COSMOS_MODEL_DIR": os.environ.get("COSMOS_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        "COSMOS_OUTPUT_DIR": str(state.output_dir),
        "COSMOS_DISABLE_SAFETY": os.environ.get("COSMOS_DISABLE_SAFETY", "0"),
        "HF_HOME": os.environ.get("HF_HOME", "/opt/cosmos-data/hf_cache"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", "/opt/cosmos-data/hf_cache"),
    }
    state.output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = state.server_log.open("w")
    state.process = subprocess.Popen(
        [uvicorn, "server:app", "--host", "127.0.0.1", "--port", str(state.port)],
        cwd=str(SERVER_CWD),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 120
    last_error = ""
    while time.time() < deadline:
        if state.process.poll() is not None:
            return CheckResult(
                "start Cosmos server",
                False,
                f"server exited with {state.process.returncode}; log:\n{_tail(state.server_log)}",
            )
        try:
            health = _request_json("GET", f"http://127.0.0.1:{state.port}/health", timeout=5)
            return CheckResult("start Cosmos server", True, json.dumps(health, sort_keys=True))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = _format_exception(exc)
            time.sleep(2)
    return CheckResult("start Cosmos server", False, f"timed out waiting for /health; last error: {last_error}")


def check_submit_inference(state: SmokeState) -> CheckResult:
    prompt = os.environ.get("COSMOS_SMOKE_PROMPT", DEFAULT_PROMPT)
    try:
        response = _request_json(
            "POST",
            f"http://127.0.0.1:{state.port}/infer",
            {"prompt": prompt},
            timeout=30,
        )
    except Exception as exc:
        return CheckResult("submit async inference job", False, _format_exception(exc))

    job_id = str(response.get("job_id") or "")
    if not job_id:
        return CheckResult("submit async inference job", False, f"missing job_id: {response}")
    state.job_id = job_id
    return CheckResult("submit async inference job", True, f"job_id: {job_id}; model: {response.get('model')}")


def check_poll_inference(state: SmokeState) -> CheckResult:
    if not state.job_id:
        return CheckResult("poll async inference job", False, "skipped because submit failed")

    timeout = int(os.environ.get("COSMOS_SMOKE_TIMEOUT_SECONDS", "1200"))
    interval = float(os.environ.get("COSMOS_SMOKE_POLL_INTERVAL_SECONDS", "10"))
    deadline = time.time() + timeout
    last_status: dict[str, Any] = {}

    while time.time() < deadline:
        try:
            last_status = _request_json(
                "GET",
                f"http://127.0.0.1:{state.port}/jobs/{state.job_id}",
                timeout=30,
            )
        except Exception as exc:
            return CheckResult("poll async inference job", False, _format_exception(exc))

        status = last_status.get("status")
        if status == "completed":
            output_path = Path(str(last_status.get("output_path", "")))
            state.output_path = output_path
            return CheckResult("poll async inference job", True, f"output_path: {output_path}")
        if status == "failed":
            return CheckResult(
                "poll async inference job",
                False,
                f"job failed: {last_status.get('error')}; server log:\n{_tail(state.server_log)}",
            )
        time.sleep(interval)

    return CheckResult(
        "poll async inference job",
        False,
        f"timed out after {timeout}s; last_status: {last_status}; server log:\n{_tail(state.server_log)}",
    )


def check_mp4_output(state: SmokeState) -> CheckResult:
    if state.output_path is None:
        return CheckResult("verify MP4 output", False, "skipped because inference did not complete")
    if not state.output_path.exists():
        return CheckResult("verify MP4 output", False, f"missing {state.output_path}")
    if state.output_path.suffix.lower() != ".mp4":
        return CheckResult("verify MP4 output", False, f"expected .mp4 output, got {state.output_path}")
    size = state.output_path.stat().st_size
    if size <= 0:
        return CheckResult("verify MP4 output", False, f"zero-size output: {state.output_path}")
    return CheckResult("verify MP4 output", True, f"{state.output_path} ({size} bytes)")


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
    run_id = uuid.uuid4().hex[:10]
    root = DEFAULT_OUTPUT_DIR / f"npa_cosmos_smoke_{run_id}"
    state = SmokeState(
        root=root,
        output_dir=root,
        server_log=root / "cosmos_server.log",
        port=int(os.environ.get("COSMOS_SMOKE_PORT", "8080")),
    )
    state.root.mkdir(parents=True, exist_ok=True)
    print(f"Smoke workspace: {state.root}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_cosmos_version,
        check_model_weights,
        check_start_server,
        check_submit_inference,
        check_poll_inference,
        check_mp4_output,
    ]
    results = []
    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        _stop_server(state)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
