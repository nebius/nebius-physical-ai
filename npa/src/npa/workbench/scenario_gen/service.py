"""FastAPI service for the adversarial scenario-generation workbench."""

from __future__ import annotations

import hmac
import logging
import os
import platform
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from .generation import ScenarioGenError, generate_scenarios
from .ranking import ScenarioRankError, rank_scenarios
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    RankRequest,
    RankResponse,
    RunListResponse,
    StatusResponse,
)

RUNS: dict[str, StatusResponse] = {}
LOGGER = logging.getLogger(__name__)


def create_app(*, auth_mode: str | None = None, token: str | None = None) -> FastAPI:
    """Create the scenario-generation FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("SCENARIO_GEN_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("SCENARIO_GEN_TOKEN", "")
    app = FastAPI(title="NPA Adversarial Scenario Generation")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "scenario-gen service started with auth disabled; every endpoint is reachable "
            "without a token. Set SCENARIO_GEN_AUTH_MODE=token and SCENARIO_GEN_TOKEN."
        )

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="SCENARIO_GEN_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return {"status": "ok", "runs": len(RUNS)}

    @app.get("/system-info")
    async def system_info(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return system_info_payload()

    @app.get("/list", response_model=RunListResponse)
    async def list_runs(request: Request, authorization: str = Header(default="")) -> RunListResponse:
        await require_auth(request, authorization)
        return RunListResponse(runs=list(RUNS.values()))

    @app.get("/status", response_model=StatusResponse)
    async def status(
        run_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> StatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(
        body: GenerateRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> GenerateResponse:
        await require_auth(request, authorization)
        try:
            response = generate_scenarios(body)
        except ScenarioGenError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        RUNS[response.run_id] = StatusResponse(
            run_id=response.run_id,
            status=response.status,
            kind="generate",
            scenario_count=response.scenario_count,
            manifest_uri=response.manifest_uri,
            top_severity=response.top_severity,
            manifest_sha256=response.manifest_sha256,
        )
        return response

    @app.post("/rank", response_model=RankResponse)
    async def rank(
        body: RankRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RankResponse:
        await require_auth(request, authorization)
        try:
            response = rank_scenarios(body)
        except ScenarioRankError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        RUNS[response.run_id] = StatusResponse(
            run_id=response.run_id,
            status=response.status,
            kind="rank",
            scenario_count=response.ranked_count,
            manifest_uri=response.ranked_manifest_uri,
            manifest_sha256=response.manifest_sha256,
        )
        return response

    return app


def status_for_run(run_id: str) -> StatusResponse:
    status = RUNS.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return status


def system_info_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "tool": "scenario_gen",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rl_backend": "isaac_lab",
        "gpu_routing": "RTX PRO 6000 or L40S (RT-core capable)",
    }
    try:
        import torch

        payload.update(
            {
                "torch": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        payload["torch_error"] = str(exc)
    return payload


app = create_app()
