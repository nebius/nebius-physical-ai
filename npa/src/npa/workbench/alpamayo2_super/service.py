"""FastAPI service for the shared Alpamayo 2 Super runtime."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from npa.workbench.alpamayo2_super.runtime import (
    ARTIFACT_SCHEMA,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    run_inference,
)


class InferenceBody(BaseModel):
    """Validated HTTP inference body."""

    model_config = ConfigDict(extra="forbid")
    output_path: str
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    dataset_revision: str = DEFAULT_DATASET_REVISION
    manifest: str = DEFAULT_MANIFEST
    sample_index: int = 0
    diffusion_steps: int = 10
    seed: int = 42
    figure_style: str = "blog"
    require_camera_projection: bool = True
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


app = FastAPI(title="NPA Alpamayo 2 Super", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "schema": ARTIFACT_SCHEMA}


@app.get("/system-info")
def system_info() -> dict[str, Any]:
    return {"model_revision": DEFAULT_MODEL_REVISION, "weights_baked": False}


@app.post("/run")
def run(body: InferenceBody) -> dict[str, Any]:
    try:
        return run_inference(Alpamayo2SuperRequest(**body.model_dump()))
    except Alpamayo2SuperError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/list")
def list_capabilities() -> list[dict[str, Any]]:
    return [{"name": "trajectory-inference", "schema": ARTIFACT_SCHEMA}]
