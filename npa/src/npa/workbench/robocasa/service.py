"""FastAPI service for the RoboCasa workbench."""

from __future__ import annotations

import hmac
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

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

RUNS: dict[str, RoboCasaStatusResponse] = {}
LOGGER = logging.getLogger(__name__)


def create_app(*, auth_mode: str | None = None, token: str | None = None) -> FastAPI:
    """Create the RoboCasa FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("ROBOCASA_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("ROBOCASA_TOKEN", "")
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
        return {"status": "ok", "runs": len(RUNS)}

    @app.get("/system-info", response_model=RoboCasaSystemInfo)
    async def system_info_endpoint(
        request: Request, authorization: str = Header(default="")
    ) -> RoboCasaSystemInfo:
        await require_auth(request, authorization)
        return system_info()

    @app.get("/runs", response_model=RoboCasaRunListResponse)
    async def runs(request: Request, authorization: str = Header(default="")) -> RoboCasaRunListResponse:
        await require_auth(request, authorization)
        return RoboCasaRunListResponse(runs=list(RUNS.values()))

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
        RUNS[run_id] = RoboCasaStatusResponse(
            run_id=run_id,
            status="running",
            capability=body.capability,
            env_id=body.env_id,
            output_uri=body.output_uri,
        )
        background_tasks.add_task(_run_capability, body, run_id)
        return response

    @app.get("/status", response_model=RoboCasaStatusResponse)
    async def status(
        run_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RoboCasaStatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id)

    return app


def status_for_run(run_id: str) -> RoboCasaStatusResponse:
    status = RUNS.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return status


def _run_capability(body: RoboCasaRunRequest, run_id: str) -> None:
    def update(status: str, result: dict[str, Any] | None, error: str | None) -> None:
        current = RUNS[run_id]
        RUNS[run_id] = current.model_copy(
            update={"status": status, "result": result, "error": error}
        )

    try:
        LOGGER.info(
            "starting robocasa run_id=%s capability=%s env_id=%s output_uri=%s",
            run_id,
            body.capability,
            body.env_id,
            body.output_uri,
        )
        with tempfile.TemporaryDirectory(prefix="robocasa_") as tmp:
            result = run_capability_with_output(body, output_dir=Path(tmp))
        update("completed", result, None)
    except RoboCasaError as exc:
        update("failed", None, str(exc))
    except Exception as exc:  # pragma: no cover - defensive service boundary.
        update("failed", None, str(exc))


app = create_app()
