"""Authenticated FastAPI service for OpenArm simulator workloads."""

from __future__ import annotations

import hmac
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from npa.workbench.openarm.runtime import manifest_sha256, run, run_id, system_info
from npa.workbench.openarm.schemas import (
    OpenArmRunListResponse,
    OpenArmRunRequest,
    OpenArmRunResponse,
    OpenArmStatusResponse,
    OpenArmSystemInfo,
)


class RunRegistry:
    """Small concurrency-safe registry; active runs are never evicted."""

    def __init__(self, max_terminal: int = 256) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, OpenArmStatusResponse] = {}
        self._max_terminal = max_terminal

    def put(self, value: OpenArmStatusResponse) -> None:
        with self._lock:
            self._runs[value.run_id] = value
            terminal = [
                key
                for key, row in self._runs.items()
                if row.status in {"completed", "failed"}
            ]
            for key in terminal[: -self._max_terminal]:
                self._runs.pop(key, None)

    def claim(self, value: OpenArmStatusResponse) -> tuple[OpenArmStatusResponse, bool]:
        """Atomically claim a missing or failed run for one executor submission."""
        with self._lock:
            current = self._runs.get(value.run_id)
            if current is not None and current.status != "failed":
                return current, False
            self._runs[value.run_id] = value
            return value, True

    def get(self, key: str) -> OpenArmStatusResponse | None:
        with self._lock:
            return self._runs.get(key)

    def values(self) -> list[OpenArmStatusResponse]:
        with self._lock:
            return list(self._runs.values())


RUNS = RunRegistry()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openarm")


def _authorized(authorization: str) -> bool:
    if os.environ.get("OPENARM_AUTH_MODE", "token").lower() == "none":
        return True
    expected = os.environ.get("OPENARM_TOKEN", "")
    provided = authorization.removeprefix("Bearer ").strip()
    return bool(expected) and hmac.compare_digest(provided, expected)


async def _require_auth(request: Request, authorization: str) -> None:
    if request.url.path == "/health":
        return
    if not _authorized(authorization):
        raise HTTPException(status_code=401, detail="invalid token")


def _execute(body: OpenArmRunRequest, identifier: str, registry: RunRegistry) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="npa-openarm-service-") as temporary:
            result = run(body, output_dir=Path(temporary))
        registry.put(
            OpenArmStatusResponse(
                run_id=identifier,
                status="completed",
                simulator=body.simulator,
                output_uri=body.output_uri,
                result=result,
            )
        )
    except Exception as exc:
        registry.put(
            OpenArmStatusResponse(
                run_id=identifier,
                status="failed",
                simulator=body.simulator,
                output_uri=body.output_uri,
                error=str(exc),
            )
        )


def _registry(request: Request) -> RunRegistry:
    return request.app.state.openarm_registry


async def _health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "runs": len(_registry(request).values())}


async def _info(
    request: Request, authorization: str = Header(default="")
) -> OpenArmSystemInfo:
    await _require_auth(request, authorization)
    return await run_in_threadpool(system_info)


async def _list_runs(
    request: Request, authorization: str = Header(default="")
) -> OpenArmRunListResponse:
    await _require_auth(request, authorization)
    return OpenArmRunListResponse(runs=_registry(request).values())


async def _start(
    body: OpenArmRunRequest,
    request: Request,
    authorization: str = Header(default=""),
) -> OpenArmRunResponse:
    await _require_auth(request, authorization)
    runs = _registry(request)
    identifier = run_id(body)
    current, claimed = runs.claim(
        OpenArmStatusResponse(
            run_id=identifier,
            status="running",
            simulator=body.simulator,
            output_uri=body.output_uri,
        )
    )
    if claimed:
        EXECUTOR.submit(_execute, body, identifier, runs)
    status = current.status if current.status == "completed" else "running"
    return OpenArmRunResponse(
        run_id=identifier,
        status=status,
        simulator=body.simulator,
        output_uri=body.output_uri,
        manifest_sha256=manifest_sha256(body),
    )


async def _status(
    run_id: str, request: Request, authorization: str = Header(default="")
) -> OpenArmStatusResponse:
    await _require_auth(request, authorization)
    value = _registry(request).get(run_id)
    if value is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return value


def create_app(*, registry: RunRegistry | None = None) -> FastAPI:
    """Create an independently testable OpenArm service application."""
    service = FastAPI(title="NPA OpenArm", version="1")
    service.state.openarm_registry = registry or RUNS
    service.add_api_route("/health", _health, methods=["GET"])
    service.add_api_route(
        "/system-info", _info, methods=["GET"], response_model=OpenArmSystemInfo
    )
    service.add_api_route(
        "/runs", _list_runs, methods=["GET"], response_model=OpenArmRunListResponse
    )
    service.add_api_route(
        "/run", _start, methods=["POST"], response_model=OpenArmRunResponse
    )
    service.add_api_route(
        "/status", _status, methods=["GET"], response_model=OpenArmStatusResponse
    )
    return service


app = create_app()
