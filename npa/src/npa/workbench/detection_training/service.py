"""FastAPI service for the detection-training workbench."""

from __future__ import annotations

import hmac
import logging
import os
import platform
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from .evaluation import DetectionEvaluationError, evaluate_detector
from .schemas import EvalRequest, EvalResponse, RunListResponse, StatusResponse, TrainRequest, TrainResponse
from .training import (
    DetectionTrainingError,
    checkpoint_uri_pattern,
    compute_manifest_sha256,
    make_run_id,
    metrics_uri,
    resolve_num_classes,
    train_detector,
)

RUNS: dict[str, StatusResponse] = {}
LOGGER = logging.getLogger(__name__)


def create_app(*, auth_mode: str | None = None, token: str | None = None) -> FastAPI:
    """Create the detection-training FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("DETECTION_TRAINING_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("DETECTION_TRAINING_TOKEN", "")
    app = FastAPI(title="NPA Detection Training")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "detection-training service started with auth disabled; every endpoint is reachable "
            "without a token. Set DETECTION_TRAINING_AUTH_MODE=token and DETECTION_TRAINING_TOKEN."
        )

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="DETECTION_TRAINING_TOKEN is not configured")
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

    @app.get("/runs", response_model=RunListResponse)
    async def runs(request: Request, authorization: str = Header(default="")) -> RunListResponse:
        await require_auth(request, authorization)
        return RunListResponse(runs=list(RUNS.values()))

    @app.post("/train", response_model=TrainResponse)
    async def train(
        body: TrainRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        authorization: str = Header(default=""),
    ) -> TrainResponse:
        await require_auth(request, authorization)
        manifest = compute_manifest_sha256("train", body.model_dump(mode="json"))
        run_id = make_run_id("train", manifest)
        response = TrainResponse(
            run_id=run_id,
            status="running",
            checkpoint_uri_pattern=checkpoint_uri_pattern(body.output_uri, run_id),
            metrics_uri=metrics_uri(body.output_uri, run_id),
            total_epochs=body.epochs,
            manifest_sha256=manifest,
        )
        RUNS[run_id] = StatusResponse(
            run_id=run_id,
            status="running",
            epochs_completed=0,
            total_epochs=body.epochs,
            checkpoint_uri_pattern=response.checkpoint_uri_pattern,
            metrics_uri=response.metrics_uri,
            manifest_sha256=manifest,
        )
        background_tasks.add_task(_run_training, body, run_id)
        return response

    @app.post("/eval", response_model=EvalResponse)
    async def evaluate(
        body: EvalRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> EvalResponse:
        await require_auth(request, authorization)
        try:
            return evaluate_detector(body)
        except DetectionEvaluationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/status", response_model=StatusResponse)
    async def status(
        run_id: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> StatusResponse:
        await require_auth(request, authorization)
        return status_for_run(run_id)

    return app


def status_for_run(run_id: str) -> StatusResponse:
    status = RUNS.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return status


def system_info_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
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


def _run_training(body: TrainRequest, run_id: str) -> None:
    def update(status: str, epochs_completed: int, metrics: dict[str, Any], error: str | None) -> None:
        current = RUNS[run_id]
        RUNS[run_id] = current.model_copy(
            update={
                "status": status,
                "epochs_completed": epochs_completed,
                "last_metrics": metrics,
                "error": error,
            }
        )

    try:
        LOGGER.info(
            "starting detection training run_id=%s view=%s num_classes=%s label_map=%s",
            run_id,
            body.view,
            resolve_num_classes(body),
            "provided" if body.label_map is not None else "absent",
        )
        result = train_detector(body, run_id=run_id, status_callback=update)
        current = RUNS[run_id]
        RUNS[run_id] = current.model_copy(
            update={
                "status": result.status,
                "epochs_completed": result.total_epochs,
                "error": None,
            }
        )
    except DetectionTrainingError as exc:
        update("failed", RUNS[run_id].epochs_completed, RUNS[run_id].last_metrics, str(exc))
    except Exception as exc:  # pragma: no cover - defensive service boundary.
        update("failed", RUNS[run_id].epochs_completed, RUNS[run_id].last_metrics, str(exc))


app = create_app()
