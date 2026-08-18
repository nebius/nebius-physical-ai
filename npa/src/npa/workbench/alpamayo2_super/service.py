"""FastAPI service for the shared Alpamayo 2 Super runtime."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from npa.workbench.alpamayo2_super.runtime import (
    ARTIFACT_SCHEMA,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    run_inference,
)
from npa.workbench.alpamayo2_super.schemas import InferenceBody


def create_app() -> FastAPI:
    """Create the deployable Alpamayo service application."""

    service = FastAPI(title="NPA Alpamayo 2 Super", version="1.0")

    @service.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema": ARTIFACT_SCHEMA}

    @service.get("/system-info")
    def system_info() -> dict[str, Any]:
        return {"model_revision": DEFAULT_MODEL_REVISION, "weights_baked": False}

    @service.post("/run")
    def run(body: InferenceBody) -> dict[str, Any]:
        try:
            return run_inference(Alpamayo2SuperRequest(**body.model_dump()))
        except Alpamayo2SuperError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @service.get("/status")
    def status() -> dict[str, str]:
        return {"status": "ready"}

    @service.get("/list")
    def list_capabilities() -> list[dict[str, Any]]:
        return [{"name": "trajectory-inference", "schema": ARTIFACT_SCHEMA}]

    return service


app = create_app()
