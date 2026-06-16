"""LanceDB container functional golden eval.

Starts the LanceDB FastAPI wrapper against a throwaway storage path, then proves
the core vector-store contract end to end: health, create table, vector query,
and table listing. No external data or S3 access is required.

Run inside the npa-lancedb image with:
    python -m npa.smoke.test_lancedb_functional
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

SERVER_TARGET = os.environ.get("LANCEDB_SMOKE_APP", "npa.workbench.lancedb.server:app")
TABLE_NAME = "npa_golden_eval"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    storage_path: Path
    server_log: Path
    port: int
    process: subprocess.Popen[str] | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


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
        return CheckResult("start LanceDB server", False, "uvicorn not found on PATH")

    env = {
        **os.environ,
        "LANCEDB_STORAGE_PATH": str(state.storage_path),
        "LANCEDB_AUTH_MODE": "none",
    }
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
                "start LanceDB server",
                False,
                f"server exited with {state.process.returncode}; log:\n{_tail(state.server_log)}",
            )
        try:
            health = _request_json("GET", f"{_base_url(state)}/health", timeout=5)
            return CheckResult("start LanceDB server", True, json.dumps(health, sort_keys=True))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = _format_exception(exc)
            time.sleep(1)
    return CheckResult(
        "start LanceDB server", False, f"timed out waiting for /health; last error: {last_error}"
    )


def check_create_table(state: SmokeState) -> CheckResult:
    payload = {
        "rows": [
            {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4]},
            {"id": "b", "vector": [0.9, 0.8, 0.7, 0.6]},
        ],
        "id_column": "id",
        "vector_column": "vector",
        "mode": "overwrite",
    }
    try:
        response = _request_json(
            "POST", f"{_base_url(state)}/tables/{TABLE_NAME}", payload, timeout=30
        )
    except Exception as exc:
        return CheckResult("create table", False, _format_exception(exc))
    if response.get("rows") != 2:
        return CheckResult("create table", False, f"unexpected response: {response}")
    return CheckResult("create table", True, json.dumps(response, sort_keys=True))


def check_vector_query(state: SmokeState) -> CheckResult:
    payload = {"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 2}
    try:
        response = _request_json(
            "POST", f"{_base_url(state)}/tables/{TABLE_NAME}/query", payload, timeout=30
        )
    except Exception as exc:
        return CheckResult("vector query", False, _format_exception(exc))
    count = response.get("count", 0)
    if count < 1:
        return CheckResult("vector query", False, f"no results returned: {response}")
    return CheckResult("vector query", True, f"count: {count}")


def check_list_tables(state: SmokeState) -> CheckResult:
    try:
        response = _request_json("GET", f"{_base_url(state)}/tables", timeout=30)
    except Exception as exc:
        return CheckResult("list tables", False, _format_exception(exc))
    if TABLE_NAME not in response.get("tables", []):
        return CheckResult("list tables", False, f"{TABLE_NAME} missing: {response}")
    return CheckResult("list tables", True, json.dumps(response, sort_keys=True))


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
    root = Path(tempfile.mkdtemp(prefix="npa_lancedb_smoke_"))
    state = SmokeState(
        root=root,
        storage_path=root / "storage",
        server_log=root / "lancedb_server.log",
        port=int(os.environ.get("LANCEDB_SMOKE_PORT", "8686")),
    )
    state.storage_path.mkdir(parents=True, exist_ok=True)
    print(f"Smoke workspace: {root}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_start_server,
        check_create_table,
        check_vector_query,
        check_list_tables,
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
