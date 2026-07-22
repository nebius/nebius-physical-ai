"""FastAPI service for the dataset-of-record workbench."""

from __future__ import annotations

import hmac
import logging
import os
import platform
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from .curation import DatasetCurateError, curate_dataset, query_dataset
from .ingestion import DatasetIngestError, ingest_dataset
from .integrations import DatasetIntegrationError
from .schemas import (
    CurateRequest,
    CurateResponse,
    DatasetListResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ValidateRequest,
    ValidateResponse,
)
from .validation import DatasetValidationError, validate_manifest

# Registered dataset versions keyed by "dataset_id@version".
DATASETS: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger(__name__)


def create_app(*, auth_mode: str | None = None, token: str | None = None) -> FastAPI:
    """Create the dataset-of-record FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("DATASET_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("DATASET_TOKEN", "")
    app = FastAPI(title="NPA Dataset of Record")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "dataset service started with auth disabled; every endpoint is reachable "
            "without a token. Set DATASET_AUTH_MODE=token and DATASET_TOKEN."
        )

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="DATASET_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return {"status": "ok", "datasets": len(DATASETS)}

    @app.get("/system-info")
    async def system_info(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return system_info_payload()

    @app.get("/list", response_model=DatasetListResponse)
    async def list_datasets(request: Request, authorization: str = Header(default="")) -> DatasetListResponse:
        await require_auth(request, authorization)
        return DatasetListResponse(datasets=list(DATASETS.values()))

    @app.get("/status")
    async def status(
        dataset_id: str,
        version: str,
        request: Request,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        return status_for_version(dataset_id, version)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        body: IngestRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> IngestResponse:
        await require_auth(request, authorization)
        try:
            response = ingest_dataset(body)
        except DatasetIngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DatasetIntegrationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _record(response.dataset_id, response.version, response.manifest_uri, response.record_count, "ingest")
        return response

    @app.post("/validate", response_model=ValidateResponse)
    async def validate(
        body: ValidateRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> ValidateResponse:
        await require_auth(request, authorization)
        try:
            return validate_manifest(body)
        except DatasetValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/curate", response_model=CurateResponse)
    async def curate(
        body: CurateRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> CurateResponse:
        await require_auth(request, authorization)
        try:
            response = curate_dataset(body)
        except DatasetCurateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _record(response.dataset_id, response.version, response.manifest_uri, response.record_count, "curate")
        return response

    @app.get("/query", response_model=QueryResponse)
    async def query(
        input_uri: str,
        request: Request,
        event: str = "",
        location: str = "",
        modality: str = "",
        quality_metric: str = "completeness",
        min_quality: float | None = None,
        limit: int = 100,
        lancedb_endpoint: str = "",
        authorization: str = Header(default=""),
    ) -> QueryResponse:
        await require_auth(request, authorization)
        try:
            return query_dataset(
                QueryRequest(
                    input_uri=input_uri,
                    event=event,
                    location=location,
                    modality=modality,
                    quality_metric=quality_metric,
                    min_quality=min_quality,
                    limit=limit,
                    lancedb_endpoint=lancedb_endpoint,
                )
            )
        except DatasetCurateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DatasetIntegrationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def _record(dataset_id: str, version: str, manifest_uri: str, record_count: int, kind: str) -> None:
    key = f"{dataset_id}@{version}"
    DATASETS[key] = {
        "dataset_id": dataset_id,
        "version": version,
        "manifest_uri": manifest_uri,
        "record_count": record_count,
        "kind": kind,
    }


def status_for_version(dataset_id: str, version: str) -> dict[str, Any]:
    entry = DATASETS.get(f"{dataset_id}@{version}")
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset version: {dataset_id}@{version}")
    return entry


def system_info_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "tool": "dataset",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "manifest_schema": "npa.dataset.manifest.v1",
        "query_backend": "lancedb (manifest fallback)",
        "curation_handoff": "fiftyone",
    }


app = create_app()
