"""Compatibility SDK for the dataset-of-record workbench."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.workbench.dataset.schemas import (
    DEFAULT_COMPLETENESS_MIN,
    DEFAULT_MAX_CORRUPTION_RATE,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_TOKEN_ENV,
    DEFAULT_VERSION,
    CurateRequest,
    CurateResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SensorSchema,
    ValidateRequest,
    ValidateResponse,
)

ENDPOINT_ENV = "NPA_DATASET_ENDPOINT"


class DatasetServiceError(RuntimeError):
    """Raised when a dataset service request fails."""


class DatasetValidationInputError(ValueError):
    """Raised when local SDK inputs are invalid."""


def ingest(
    *,
    input_uri: str,
    output_uri: str,
    dataset_id: str,
    version: str = DEFAULT_VERSION,
    sensor_schema: dict[str, Any] | None = None,
    source: str = "",
    workflow_run: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 120.0,
) -> IngestResponse:
    """Ingest raw sensor data and register a versioned dataset manifest."""
    request = IngestRequest(
        input_uri=input_uri,
        output_uri=output_uri,
        dataset_id=dataset_id,
        version=version,
        sensor_schema=SensorSchema(**(sensor_schema or {})),
        source=source,
        workflow_run=workflow_run,
    )
    if _resolve_mode(mode=mode, service=service):
        return IngestResponse.model_validate(
            _request_json("POST", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/ingest", payload=request.model_dump(mode="json"), token_env=token_env, timeout=timeout)
        )
    from npa.workbench.dataset.ingestion import ingest_dataset

    return ingest_dataset(request)


def validate(
    *,
    input_uri: str,
    output_uri: str,
    completeness_min: float = DEFAULT_COMPLETENESS_MIN,
    max_corruption_rate: float = DEFAULT_MAX_CORRUPTION_RATE,
    workflow_run: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> ValidateResponse:
    """Validate a dataset manifest against schema + quality thresholds."""
    request = ValidateRequest(
        input_uri=input_uri,
        output_uri=output_uri,
        completeness_min=completeness_min,
        max_corruption_rate=max_corruption_rate,
        workflow_run=workflow_run,
    )
    if _resolve_mode(mode=mode, service=service):
        return ValidateResponse.model_validate(
            _request_json("POST", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/validate", payload=request.model_dump(mode="json"), token_env=token_env, timeout=timeout)
        )
    from npa.workbench.dataset.validation import validate_manifest

    return validate_manifest(request)


def curate(
    *,
    input_uri: str,
    output_uri: str,
    event: str = "",
    location: str = "",
    quality_metric: str = "completeness",
    min_quality: float | None = None,
    workflow_run: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> CurateResponse:
    """Slice a dataset version by event/location/quality with lineage."""
    request = CurateRequest(
        input_uri=input_uri,
        output_uri=output_uri,
        event=event,
        location=location,
        quality_metric=quality_metric,
        min_quality=min_quality,
        workflow_run=workflow_run,
    )
    if _resolve_mode(mode=mode, service=service):
        return CurateResponse.model_validate(
            _request_json("POST", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/curate", payload=request.model_dump(mode="json"), token_env=token_env, timeout=timeout)
        )
    from npa.workbench.dataset.curation import curate_dataset

    return curate_dataset(request)


def query(
    *,
    input_uri: str,
    event: str = "",
    location: str = "",
    modality: str = "",
    quality_metric: str = "completeness",
    min_quality: float | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    lancedb_endpoint: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> QueryResponse:
    """Query dataset records by event/location/quality facets."""
    request = QueryRequest(
        input_uri=input_uri,
        event=event,
        location=location,
        modality=modality,
        quality_metric=quality_metric,
        min_quality=min_quality,
        limit=limit,
        lancedb_endpoint=lancedb_endpoint,
    )
    if _resolve_mode(mode=mode, service=service):
        params = {k: v for k, v in request.model_dump(mode="json").items() if v not in ("", None)}
        return QueryResponse.model_validate(
            _request_json("GET", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/query", params=params, token_env=token_env, timeout=timeout)
        )
    from npa.workbench.dataset.curation import query_dataset

    return query_dataset(request)


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise DatasetValidationInputError("mode must be either 'local' or 'service'")


def _request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = endpoint.strip().rstrip("/")
    if not resolved:
        raise DatasetValidationInputError("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{resolved}{path}", headers=headers, json=payload, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise DatasetServiceError(f"Dataset service request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise DatasetServiceError(f"Cannot reach dataset service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise DatasetServiceError("Dataset service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise DatasetServiceError("Dataset service returned an unexpected response")
    return data


__all__ = [
    "DatasetServiceError",
    "DatasetValidationInputError",
    "CurateResponse",
    "IngestResponse",
    "QueryResponse",
    "ValidateResponse",
    "curate",
    "ingest",
    "query",
    "validate",
]
