"""FastAPI service for the RoboCasa workbench."""

from __future__ import annotations

import hmac
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from npa.workbench.robocasa.capabilities import (
    RoboCasaError,
    compute_manifest_sha256,
    make_run_id,
    run_capability_with_output,
    system_info,
)
from npa.workbench.robocasa.schemas import (
    RoboCasaRunListResponse,
    RoboCasaRunRequest,
    RoboCasaRunResponse,
    RoboCasaStatusResponse,
    RoboCasaSystemInfo,
)

LOGGER = logging.getLogger(__name__)
_TERMINAL_STATUSES = frozenset({"completed", "failed"})


class RunRegistry:
    """Concurrency-safe bounded run status registry.

    TTL and size eviction apply only to terminal runs. Active runs are never
    discarded merely to satisfy the bound, so the registry may temporarily
    exceed ``max_entries`` while all retained work is active.
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 24 * 60 * 60,
        clock: Any = time.monotonic,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, tuple[RoboCasaStatusResponse, float, int]] = {}
        self._sequence = 0

    def __len__(self) -> int:
        with self._lock:
            self._evict_locked()
            return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self._evict_locked()
            return iter(tuple(self._entries))

    def __setitem__(self, run_id: str, status: RoboCasaStatusResponse) -> None:
        with self._lock:
            self._sequence += 1
            self._entries[run_id] = (status, self._clock(), self._sequence)
            self._evict_locked()

    def get(self, run_id: str) -> RoboCasaStatusResponse | None:
        with self._lock:
            self._evict_locked()
            entry = self._entries.get(run_id)
            return entry[0] if entry else None

    def values(self) -> list[RoboCasaStatusResponse]:
        with self._lock:
            self._evict_locked()
            return [entry[0] for entry in self._entries.values()]

    def update(
        self,
        run_id: str,
        *,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self._lock:
            current = self._entries.get(run_id)
            if current is None:
                return
            self._sequence += 1
            updated = current[0].model_copy(
                update={"status": status, "result": result, "error": error}
            )
            self._entries[run_id] = (updated, self._clock(), self._sequence)
            self._evict_locked()

    def _evict_locked(self) -> None:
        now = self._clock()
        expired = [
            run_id
            for run_id, (record, updated_at, _sequence) in self._entries.items()
            if record.status in _TERMINAL_STATUSES
            and now - updated_at >= self.ttl_seconds
        ]
        for run_id in expired:
            self._entries.pop(run_id, None)

        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return
        terminal = sorted(
            (
                (updated_at, sequence, run_id)
                for run_id, (record, updated_at, sequence) in self._entries.items()
                if record.status in _TERMINAL_STATUSES
            )
        )
        for _updated_at, _sequence, run_id in terminal[:overflow]:
            self._entries.pop(run_id, None)


RUNS = RunRegistry(
    max_entries=int(os.environ.get("ROBOCASA_RUNS_MAX_ENTRIES", "256")),
    ttl_seconds=float(os.environ.get("ROBOCASA_RUNS_TTL_SECONDS", str(24 * 60 * 60))),
)


def create_app(
    *,
    auth_mode: str | None = None,
    token: str | None = None,
    runs: RunRegistry | None = None,
) -> FastAPI:
    """Create the RoboCasa FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("ROBOCASA_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("ROBOCASA_TOKEN", "")
    registry = runs if runs is not None else RUNS
    app = FastAPI(title="NPA RoboCasa")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "robocasa service started with auth disabled; every endpoint is reachable "
            "without a token. Set ROBOCASA_AUTH_MODE=token and ROBOCASA_TOKEN."
        )

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="ROBOCASA_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "runs": len(registry)}

    @app.get("/system-info", response_model=RoboCasaSystemInfo)
    async def system_info_endpoint(
        request: Request, authorization: str = Header(default="")
    ) -> RoboCasaSystemInfo:
        await require_auth(request, authorization)
        # torch/CUDA and Gymnasium discovery can import native extensions and
        # probe drivers. Keep that blocking work off the ASGI event-loop thread.
        return await run_in_threadpool(system_info)

    @app.get("/runs", response_model=RoboCasaRunListResponse)
    async def runs(request: Request, authorization: str = Header(default="")) -> RoboCasaRunListResponse:
        await require_auth(request, authorization)
        return RoboCasaRunListResponse(runs=registry.values())

    @app.post("/run", response_model=RoboCasaRunResponse)
    async def run(
        body: RoboCasaRunRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RoboCasaRunResponse:
        await require_auth(request, authorization)
        manifest = compute_manifest_sha256("run", body.model_dump(mode="json"))
        run_id = make_run_id(body.capability, manifest)
        response = RoboCasaRunResponse(
            run_id=run_id,
            status="running",
            env_id=body.env_id,
            capability=body.capability,
            output_uri=body.output_uri,
            manifest_sha256=manifest,
        )
        registry[run_id] = RoboCasaStatusResponse(
            run_id=run_id,
            status="running",
            capability=body.capability,
            env_id=body.env_id,
            output_uri=body.output_uri,
        )
        background_tasks.add_task(_run_capability, body, run_id, registry)
        return response

    @app.get("/status", response_model=RoboCasaStatusResponse)
    async def status(
        run_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RoboCasaStatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id, runs=registry)

    return app


def status_for_run(
    run_id: str, *, runs: RunRegistry | None = None
) -> RoboCasaStatusResponse:
    registry = runs if runs is not None else RUNS
    status = registry.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return status


def _run_capability(
    body: RoboCasaRunRequest, run_id: str, runs: RunRegistry | None = None
) -> None:
    registry = runs if runs is not None else RUNS

    def update(status: str, result: dict[str, Any] | None, error: str | None) -> None:
        registry.update(run_id, status=status, result=result, error=error)

    try:
        LOGGER.info(
            "starting robocasa run_id=%s capability=%s env_id=%s",
            run_id,
            body.capability,
            body.env_id,
        )
        with tempfile.TemporaryDirectory(prefix="robocasa_") as tmp:
            result = run_capability_with_output(body, output_dir=Path(tmp))
        update("completed", result, None)
    except RoboCasaError as exc:
        update("failed", None, str(exc))
    except Exception as exc:  # pragma: no cover - defensive service boundary.
        update("failed", None, str(exc))


app = create_app()
