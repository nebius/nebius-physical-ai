"""FastAPI service for the insights lineage + metrics backbone."""

from __future__ import annotations

import hmac
import logging
import os
import platform
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request

from .analytics import (
    InsightsQueryError,
    build_dashboard,
    compare_runs,
    query_metrics,
    traverse_lineage,
)
from .integrations import InsightsIntegrationError
from .schemas import (
    CompareRequest,
    CompareResponse,
    DashboardRequest,
    DashboardResponse,
    IngestRunRequest,
    IngestRunResponse,
    InsightsListResponse,
    LineageRequest,
    LineageResponse,
    QueryRequest,
    QueryResponse,
    RecordRequest,
    RecordResponse,
)
from .store import InsightsStoreError, ingest_run, read_edges, read_records, record_metrics

# Service-tracked stores keyed by store URI.
STORES: dict[str, dict[str, Any]] = {}
LOGGER = logging.getLogger(__name__)


def create_app(*, auth_mode: str | None = None, token: str | None = None) -> FastAPI:
    """Create the insights FastAPI application."""
    resolved_auth_mode = auth_mode or os.environ.get("INSIGHTS_AUTH_MODE", "none")
    resolved_token = token if token is not None else os.environ.get("INSIGHTS_TOKEN", "")
    app = FastAPI(title="NPA Insights")
    if resolved_auth_mode == "none":
        LOGGER.warning(
            "insights service started with auth disabled; every endpoint is reachable "
            "without a token. Set INSIGHTS_AUTH_MODE=token and INSIGHTS_TOKEN."
        )

    async def require_auth(request: Request, authorization: str = Header(default="")) -> None:
        if resolved_auth_mode == "none":
            return
        if not resolved_token:
            raise HTTPException(status_code=500, detail="INSIGHTS_TOKEN is not configured")
        if not hmac.compare_digest(authorization, f"Bearer {resolved_token}"):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return {"status": "ok", "stores": len(STORES)}

    @app.get("/system-info")
    async def system_info(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
        await require_auth(request, authorization)
        return system_info_payload()

    @app.get("/list", response_model=InsightsListResponse)
    async def list_stores(request: Request, authorization: str = Header(default="")) -> InsightsListResponse:
        await require_auth(request, authorization)
        return InsightsListResponse(stores=list(STORES.values()))

    @app.get("/status")
    async def status(
        input_uri: str,
        request: Request,
        run_id: str = "",
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        await require_auth(request, authorization)
        return status_for_store(input_uri, run_id)

    @app.post("/record", response_model=RecordResponse)
    async def record(
        body: RecordRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> RecordResponse:
        await require_auth(request, authorization)
        try:
            response = record_metrics(body)
        except InsightsStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InsightsIntegrationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _track(response.store_uri, response.total_records, response.total_edges)
        return response

    @app.post("/ingest-run", response_model=IngestRunResponse)
    async def ingest(
        body: IngestRunRequest,
        request: Request,
        authorization: str = Header(default=""),
    ) -> IngestRunResponse:
        await require_auth(request, authorization)
        try:
            response = ingest_run(body)
        except InsightsStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InsightsIntegrationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _track(response.store_uri, response.total_records, response.total_edges)
        return response

    @app.get("/lineage", response_model=LineageResponse)
    async def lineage(
        input_uri: str,
        uri: str,
        request: Request,
        version: str = "",
        direction: str = "both",
        depth: int = -1,
        authorization: str = Header(default=""),
    ) -> LineageResponse:
        await require_auth(request, authorization)
        try:
            return traverse_lineage(
                LineageRequest(
                    input_uri=input_uri, uri=uri, version=version, direction=direction, depth=depth
                )
            )
        except InsightsQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/query", response_model=QueryResponse)
    async def query(
        input_uri: str,
        request: Request,
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
        limit: int = 100,
        lancedb_endpoint: str = "",
        authorization: str = Header(default=""),
    ) -> QueryResponse:
        await require_auth(request, authorization)
        try:
            return query_metrics(
                QueryRequest(
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
            )
        except InsightsQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InsightsIntegrationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/compare", response_model=CompareResponse)
    async def compare(
        input_uri: str,
        base_run: str,
        candidate_run: str,
        request: Request,
        metric_names: list[str] = Query(default=[]),
        lower_is_better: list[str] = Query(default=[]),
        authorization: str = Header(default=""),
    ) -> CompareResponse:
        await require_auth(request, authorization)
        try:
            return compare_runs(
                CompareRequest(
                    input_uri=input_uri,
                    base_run=base_run,
                    candidate_run=candidate_run,
                    metric_names=list(metric_names),
                    lower_is_better=list(lower_is_better),
                )
            )
        except InsightsQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/dashboard", response_model=DashboardResponse)
    async def dashboard(
        input_uri: str,
        request: Request,
        output_path: str = "",
        workflow: str = "",
        group_by: str = "metric_name",
        latest_run: str = "",
        authorization: str = Header(default=""),
    ) -> DashboardResponse:
        await require_auth(request, authorization)
        try:
            return build_dashboard(
                DashboardRequest(
                    input_uri=input_uri,
                    output_path=output_path,
                    workflow=workflow,
                    group_by=group_by,
                    latest_run=latest_run,
                )
            )
        except InsightsQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _track(store_uri: str, total_records: int, total_edges: int) -> None:
    STORES[store_uri] = {
        "store_uri": store_uri,
        "total_records": total_records,
        "total_edges": total_edges,
    }


def status_for_store(input_uri: str, run_id: str = "") -> dict[str, Any]:
    records = read_records(input_uri)
    edges = read_edges(input_uri)
    runs = sorted({str(record.get("run_id", "")) for record in records if record.get("run_id")})
    payload: dict[str, Any] = {
        "store_uri": input_uri,
        "total_records": len(records),
        "total_edges": len(edges),
        "runs": runs,
    }
    if run_id:
        run_records = [record for record in records if record.get("run_id") == run_id]
        payload["run_id"] = run_id
        payload["run_record_count"] = len(run_records)
        payload["run_metrics"] = sorted({str(r.get("metric_name", "")) for r in run_records})
    return payload


def system_info_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "tool": "insights",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "metric_record_schema": "npa.insights.metric_record.v1",
        "lineage_edge_schema": "npa.insights.lineage_edge.v1",
        "store_backend": "s3 jsonl (append-only)",
        "query_backend": "lancedb (jsonl fallback)",
        "gpu_routing": "cpu",
    }


app = create_app()
