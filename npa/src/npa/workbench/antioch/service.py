"""Authenticated CPU-only FastAPI control plane for Antioch operations."""

from __future__ import annotations

import hmac
import os
import platform

from fastapi import FastAPI, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .manager import AntiochManager, AntiochOperationError
from .runtime import (
    AntiochRuntimeError,
    ensure_runtime,
    runtime_cache_root,
    runtime_has_proprietary_distribution,
)
from .schemas import (
    CollectRequest,
    HealthResponse,
    OperationListResponse,
    OperationRecord,
    ResumeRequest,
    SubmitRequest,
)
from .vendor_cli import AntiochCli, AntiochCliError


def create_app(
    *,
    manager: AntiochManager | None = None,
    auth_mode: str | None = None,
    token: str | None = None,
) -> FastAPI:
    resolved_mode = auth_mode or os.environ.get("ANTIOCH_WORKBENCH_AUTH_MODE", "token")
    resolved_token = (
        token if token is not None else os.environ.get("ANTIOCH_WORKBENCH_TOKEN", "")
    )
    operations = manager
    app = FastAPI(title="NPA Antioch Workbench", version="1")

    def operation_manager() -> AntiochManager:
        nonlocal operations
        if operations is None:
            operations = AntiochManager()
        return operations

    async def require_auth(
        request: Request, authorization: str = Header(default="")
    ) -> None:
        del request
        if resolved_mode == "none":
            return
        if resolved_mode != "token" or not resolved_token:
            raise HTTPException(
                status_code=503,
                detail="Antioch Workbench service authentication is not configured",
            )
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid service token")

    async def call(fn, *args):  # noqa: ANN001, ANN202
        try:
            return await run_in_threadpool(fn, *args)
        except AntiochOperationError as exc:
            status = (
                503
                if exc.retryable
                else (404 if exc.error_type == "not_found" else 409)
            )
            raise HTTPException(
                status_code=status,
                detail={
                    "type": exc.error_type,
                    "message": str(exc),
                    "retryable": exc.retryable,
                },
            ) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health(
        request: Request, authorization: str = Header(default="")
    ) -> HealthResponse:
        await require_auth(request, authorization)
        try:
            executable = await run_in_threadpool(ensure_runtime)
            details = await run_in_threadpool(AntiochCli(executable).health)
            return HealthResponse(status="ok", cli_installed=True, **details)
        except (AntiochRuntimeError, AntiochCliError) as exc:
            return HealthResponse(
                status="degraded",
                cli_installed=not isinstance(exc, AntiochRuntimeError),
                authenticated=False,
                detail=str(exc),
            )

    @app.get("/system-info")
    async def system_info(
        request: Request, authorization: str = Header(default="")
    ) -> dict[str, object]:
        await require_auth(request, authorization)
        return {
            "status": "ok",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_only": True,
            "runtime_cache": str(runtime_cache_root()),
            "proprietary_payload_baked": await run_in_threadpool(
                runtime_has_proprietary_distribution
            ),
        }

    @app.post("/submit", response_model=OperationRecord)
    async def submit(
        body: SubmitRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().submit, body)

    @app.post("/run", response_model=OperationRecord)
    async def run(
        body: SubmitRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().run, body)

    @app.post("/status", response_model=OperationRecord)
    async def status(
        body: ResumeRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().reconcile, body)

    @app.post("/resume", response_model=OperationRecord)
    async def resume(
        body: ResumeRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().resume, body)

    @app.post("/reconcile", response_model=OperationRecord)
    async def reconcile(
        body: ResumeRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().reconcile, body)

    @app.post("/cancel", response_model=OperationRecord)
    async def cancel(
        body: ResumeRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().cancel, body)

    @app.post("/collect", response_model=OperationRecord)
    async def collect(
        body: CollectRequest, request: Request, authorization: str = Header(default="")
    ) -> OperationRecord:
        await require_auth(request, authorization)
        return await call(operation_manager().collect, body)

    @app.get("/list", response_model=OperationListResponse)
    async def list_operations(
        output_path: str, request: Request, authorization: str = Header(default="")
    ) -> OperationListResponse:
        await require_auth(request, authorization)
        records = await run_in_threadpool(operation_manager().states.list, output_path)
        return OperationListResponse(operations=records)

    return app


app = create_app()
