"""Compatibility SDK for the insights lineage + metrics backbone."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.workbench.insights.schemas import (
    DEFAULT_QUERY_LIMIT,
    DEFAULT_TOKEN_ENV,
    CompareRequest,
    CompareResponse,
    DashboardRequest,
    DashboardResponse,
    IngestRunRequest,
    IngestRunResponse,
    LineageRequest,
    LineageResponse,
    QueryRequest,
    QueryResponse,
    RecordRequest,
    RecordResponse,
)

ENDPOINT_ENV = "NPA_INSIGHTS_ENDPOINT"


class InsightsServiceError(RuntimeError):
    """Raised when an insights service request fails."""


class InsightsValidationInputError(ValueError):
    """Raised when local SDK inputs are invalid."""


def record(
    *,
    output_uri: str,
    input_uri: str = "",
    records: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    workflow_run: str = "",
    lancedb_endpoint: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> RecordResponse:
    """Record metric emissions + lineage edges into the store."""
    request = RecordRequest(
        output_uri=output_uri,
        input_uri=input_uri,
        records=records or [],
        edges=edges or [],
        workflow_run=workflow_run,
        lancedb_endpoint=lancedb_endpoint,
    )
    if _resolve_mode(mode=mode, service=service):
        return RecordResponse.model_validate(
            _request_json("POST", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/record", payload=request.model_dump(mode="json"), token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.store import record_metrics

    return record_metrics(request)


def ingest_run(
    *,
    input_uri: str,
    output_uri: str,
    workflow: str = "",
    workflow_run: str = "",
    lancedb_endpoint: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 120.0,
) -> IngestRunResponse:
    """Scan an S3 run prefix for known manifests and extract metrics + lineage."""
    request = IngestRunRequest(
        input_uri=input_uri,
        output_uri=output_uri,
        workflow=workflow,
        workflow_run=workflow_run,
        lancedb_endpoint=lancedb_endpoint,
    )
    if _resolve_mode(mode=mode, service=service):
        return IngestRunResponse.model_validate(
            _request_json("POST", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/ingest-run", payload=request.model_dump(mode="json"), token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.store import ingest_run as _ingest_run

    return _ingest_run(request)


def query(
    *,
    input_uri: str,
    workflow: str = "",
    run_id: str = "",
    tool: str = "",
    stage: str = "",
    dataset_version: str = "",
    model_version: str = "",
    metric_name: str = "",
    time_start: str = "",
    time_end: str = "",
    threshold_metric: str = "",
    threshold_op: str = "",
    threshold_value: float | None = None,
    limit: int = DEFAULT_QUERY_LIMIT,
    lancedb_endpoint: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> QueryResponse:
    """Query metric records by facet."""
    request = QueryRequest(
        input_uri=input_uri,
        workflow=workflow,
        run_id=run_id,
        tool=tool,
        stage=stage,
        dataset_version=dataset_version,
        model_version=model_version,
        metric_name=metric_name,
        time_start=time_start,
        time_end=time_end,
        threshold_metric=threshold_metric,
        threshold_op=threshold_op,
        threshold_value=threshold_value,
        limit=limit,
        lancedb_endpoint=lancedb_endpoint,
    )
    if _resolve_mode(mode=mode, service=service):
        params = {k: v for k, v in request.model_dump(mode="json").items() if v not in ("", None)}
        return QueryResponse.model_validate(
            _request_json("GET", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/query", params=params, token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.analytics import query_metrics

    return query_metrics(request)


def lineage(
    *,
    input_uri: str,
    uri: str,
    version: str = "",
    direction: str = "both",
    depth: int = -1,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> LineageResponse:
    """Traverse the provenance graph for an artifact/version."""
    request = LineageRequest(input_uri=input_uri, uri=uri, version=version, direction=direction, depth=depth)
    if _resolve_mode(mode=mode, service=service):
        params = {k: v for k, v in request.model_dump(mode="json").items() if v not in ("", None)}
        return LineageResponse.model_validate(
            _request_json("GET", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/lineage", params=params, token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.analytics import traverse_lineage

    return traverse_lineage(request)


def compare(
    *,
    input_uri: str,
    base_run: str,
    candidate_run: str,
    metric_names: list[str] | None = None,
    lower_is_better: list[str] | None = None,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> CompareResponse:
    """Compare a metric set between two run ids; flag regressed/improved."""
    request = CompareRequest(
        input_uri=input_uri,
        base_run=base_run,
        candidate_run=candidate_run,
        metric_names=metric_names or [],
        lower_is_better=lower_is_better or [],
    )
    if _resolve_mode(mode=mode, service=service):
        params: dict[str, Any] = {"input_uri": input_uri, "base_run": base_run, "candidate_run": candidate_run}
        if request.metric_names:
            params["metric_names"] = request.metric_names
        if request.lower_is_better:
            params["lower_is_better"] = request.lower_is_better
        return CompareResponse.model_validate(
            _request_json("GET", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/compare", params=params, token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.analytics import compare_runs

    return compare_runs(request)


def dashboard(
    *,
    input_uri: str,
    output_path: str = "",
    workflow: str = "",
    group_by: str = "metric_name",
    latest_run: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> DashboardResponse:
    """Build a dashboard rollup + optional static HTML report."""
    request = DashboardRequest(
        input_uri=input_uri,
        output_path=output_path,
        workflow=workflow,
        group_by=group_by,
        latest_run=latest_run,
    )
    if _resolve_mode(mode=mode, service=service):
        params = {k: v for k, v in request.model_dump(mode="json").items() if v not in ("", None)}
        return DashboardResponse.model_validate(
            _request_json("GET", endpoint or os.environ.get(ENDPOINT_ENV, ""), "/dashboard", params=params, token_env=token_env, timeout=timeout)
        )
    from npa.workbench.insights.analytics import build_dashboard

    return build_dashboard(request)


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise InsightsValidationInputError("mode must be either 'local' or 'service'")


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
        raise InsightsValidationInputError("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{resolved}{path}", headers=headers, json=payload, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise InsightsServiceError(f"Insights service request failed ({exc.response.status_code}): {detail}") from exc
    except httpx.HTTPError as exc:
        raise InsightsServiceError(f"Cannot reach insights service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise InsightsServiceError("Insights service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise InsightsServiceError("Insights service returned an unexpected response")
    return data


__all__ = [
    "InsightsServiceError",
    "InsightsValidationInputError",
    "CompareResponse",
    "DashboardResponse",
    "IngestRunResponse",
    "LineageResponse",
    "QueryResponse",
    "RecordResponse",
    "compare",
    "dashboard",
    "ingest_run",
    "lineage",
    "query",
    "record",
]
