"""Typer CLI for `npa workbench insights`."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import httpx
import typer

from npa.workbench.insights.schemas import DEFAULT_QUERY_LIMIT, DEFAULT_TOKEN_ENV

app = typer.Typer(
    name="insights",
    help="Insights: lineage graph + common metrics store over workflow-run artifacts.",
    no_args_is_help=True,
)

ENDPOINT_ENV = "NPA_INSIGHTS_ENDPOINT"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text if text is not None else "\n".join(f"{key}: {value}" for key, value in payload.items()))


@app.command("record")
def record_cmd(
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix of the insights store."),
    input_path: str = typer.Option("", "--input-path", "--input-uri", help="Optional JSON of records/edges to record."),
    run_id: str = typer.Option("", "--run-id", help="Run id for an inline single metric record."),
    metric: str = typer.Option("", "--metric", help="Metric name for an inline single metric record."),
    value: float = typer.Option(0.0, "--value", help="Metric value for an inline single metric record."),
    tool: str = typer.Option("", "--tool", help="Tool label for the inline metric."),
    stage: str = typer.Option("", "--stage", help="Stage label for the inline metric."),
    workflow: str = typer.Option("", "--workflow", help="Workflow label for the inline metric."),
    unit: str = typer.Option("", "--unit", help="Unit label for the inline metric."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id; default run id for records without one."),
    lancedb_endpoint: str = typer.Option("", "--lancedb-endpoint", help="Optional LanceDB endpoint backing the query index."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Record metric emissions + lineage into the store."""
    records: list[dict[str, Any]] = []
    if metric:
        records.append(
            {
                "run_id": run_id or workflow_run or "unknown",
                "metric_name": metric,
                "value": value,
                "tool": tool,
                "stage": stage,
                "workflow": workflow,
                "unit": unit,
            }
        )
    payload = {
        "output_uri": output_path,
        "input_uri": input_path,
        "records": records,
        "workflow_run": workflow_run,
        "lancedb_endpoint": lancedb_endpoint,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/record", payload=payload, token_env=token_env, timeout=120.0)
    else:
        from npa.sdk.workbench.insights import record

        result = record(
            output_uri=output_path,
            input_uri=input_path,
            records=records,
            workflow_run=workflow_run,
            lancedb_endpoint=lancedb_endpoint,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"store_uri: {result.get('store_uri')}\nrecorded_count: {result.get('recorded_count')}\ntotal_records: {result.get('total_records')}")


@app.command("ingest-run")
def ingest_run_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of a run to scan for known manifests."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix of the insights store."),
    workflow: str = typer.Option("", "--workflow", help="Workflow label threaded into extracted metrics."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into extracted metrics."),
    lancedb_endpoint: str = typer.Option("", "--lancedb-endpoint", help="Optional LanceDB endpoint backing the query index."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Non-invasively ingest a run prefix into the store."""
    payload = {
        "input_uri": input_path,
        "output_uri": output_path,
        "workflow": workflow,
        "workflow_run": workflow_run,
        "lancedb_endpoint": lancedb_endpoint,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/ingest-run", payload=payload, token_env=token_env, timeout=120.0)
    else:
        from npa.sdk.workbench.insights import ingest_run

        result = ingest_run(
            input_uri=input_path,
            output_uri=output_path,
            workflow=workflow,
            workflow_run=workflow_run,
            lancedb_endpoint=lancedb_endpoint,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"store_uri: {result.get('store_uri')}\nscanned: {result.get('scanned')}\nrecorded_count: {result.get('recorded_count')}\nedge_count: {result.get('edge_count')}")


@app.command("query")
def query_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of the insights store to query."),
    workflow: str = typer.Option("", "--workflow", help="Filter by workflow name."),
    run_id: str = typer.Option("", "--run-id", help="Filter by run id."),
    tool: str = typer.Option("", "--tool", help="Filter by tool."),
    stage: str = typer.Option("", "--stage", help="Filter by stage."),
    dataset_version: str = typer.Option("", "--dataset-version", help="Filter by dataset version."),
    model_version: str = typer.Option("", "--model-version", help="Filter by model/checkpoint version."),
    metric_name: str = typer.Option("", "--metric-name", help="Filter by metric name."),
    time_start: str = typer.Option("", "--time-start", help="ISO timestamp lower bound (inclusive)."),
    time_end: str = typer.Option("", "--time-end", help="ISO timestamp upper bound (inclusive)."),
    threshold_metric: str = typer.Option("", "--threshold-metric", help="Metric name the threshold predicate applies to."),
    threshold_op: str = typer.Option("", "--threshold-op", help="Threshold predicate: gt|ge|lt|le|eq."),
    threshold_value: float = typer.Option(None, "--threshold-value", help="Threshold predicate value."),
    limit: int = typer.Option(DEFAULT_QUERY_LIMIT, "--limit", help="Maximum records to return."),
    lancedb_endpoint: str = typer.Option("", "--lancedb-endpoint", help="LanceDB endpoint backing the query index."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Query metric records by facet."""
    if service:
        params = {
            "input_uri": input_path,
            "workflow": workflow,
            "run_id": run_id,
            "tool": tool,
            "stage": stage,
            "dataset_version": dataset_version,
            "model_version": model_version,
            "metric_name": metric_name,
            "time_start": time_start,
            "time_end": time_end,
            "threshold_metric": threshold_metric,
            "threshold_op": threshold_op,
            "threshold_value": threshold_value,
            "limit": limit,
            "lancedb_endpoint": lancedb_endpoint,
        }
        params = {k: v for k, v in params.items() if v not in ("", None)}
        result = request_json("GET", resolve_endpoint(endpoint), "/query", params=params, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.insights import query

        result = query(
            input_uri=input_path,
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
        ).model_dump(mode="json")
    emit(result, output=output, text=f"backend: {result.get('backend')}\ncount: {result.get('count')}")


@app.command("lineage")
def lineage_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of the insights store."),
    uri: str = typer.Option(..., "--uri", help="Artifact URI to trace lineage for."),
    version: str = typer.Option("", "--version", help="Artifact version."),
    direction: str = typer.Option("both", "--direction", help="both|ancestors|descendants."),
    depth: int = typer.Option(-1, "--depth", help="Traversal depth; -1 is unbounded."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Traverse the provenance graph for an artifact/version."""
    if service:
        params = {"input_uri": input_path, "uri": uri, "version": version, "direction": direction, "depth": depth}
        params = {k: v for k, v in params.items() if v not in ("", None)}
        result = request_json("GET", resolve_endpoint(endpoint), "/lineage", params=params, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.insights import lineage

        result = lineage(input_uri=input_path, uri=uri, version=version, direction=direction, depth=depth).model_dump(mode="json")
    emit(result, output=output, text=f"nodes: {len(result.get('nodes', []))}\nancestors: {len(result.get('ancestors', []))}\ndescendants: {len(result.get('descendants', []))}")


@app.command("compare")
def compare_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of the insights store."),
    base_run: str = typer.Option(..., "--base-run", help="Baseline run id."),
    candidate_run: str = typer.Option(..., "--candidate-run", help="Candidate run id."),
    metric: list[str] = typer.Option([], "--metric", help="Restrict comparison to this metric (repeatable)."),
    lower_is_better: list[str] = typer.Option([], "--lower-is-better", help="Metric where a lower value is better (repeatable)."),
    output_path: str = typer.Option("", "--output-path", "--output-uri", help="Optional S3 URI to write the comparison JSON."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Compare a metric set between two run ids; flag regressed/improved."""
    if service:
        params: dict[str, Any] = {"input_uri": input_path, "base_run": base_run, "candidate_run": candidate_run}
        if metric:
            params["metric_names"] = list(metric)
        if lower_is_better:
            params["lower_is_better"] = list(lower_is_better)
        result = request_json("GET", resolve_endpoint(endpoint), "/compare", params=params, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.insights import compare

        result = compare(
            input_uri=input_path,
            base_run=base_run,
            candidate_run=candidate_run,
            metric_names=list(metric),
            lower_is_better=list(lower_is_better),
        ).model_dump(mode="json")
    if output_path.strip():
        from npa.workbench.insights.storage import uri_join, write_json_uri

        report_uri = uri_join(output_path, "comparison.json")
        write_json_uri(report_uri, result)
        result["report_uri"] = report_uri
    summary = result.get("summary", {})
    emit(result, output=output, text=f"improved: {summary.get('improved')}\nregressed: {summary.get('regressed')}\nunchanged: {summary.get('unchanged')}")


@app.command("dashboard")
def dashboard_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of the insights store."),
    output_path: str = typer.Option("", "--output-path", "--output-uri", help="Optional S3 URI to write a static HTML report."),
    workflow: str = typer.Option("", "--workflow", help="Restrict to a workflow name."),
    group_by: str = typer.Option("metric_name", "--group-by", help="metric_name|tool|stage|workflow."),
    latest_run: str = typer.Option("", "--latest-run", help="Explicit latest run id for the rollup."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Build a dashboard rollup + optional static HTML report."""
    if service:
        params = {"input_uri": input_path, "output_path": output_path, "workflow": workflow, "group_by": group_by, "latest_run": latest_run}
        params = {k: v for k, v in params.items() if v not in ("", None)}
        result = request_json("GET", resolve_endpoint(endpoint), "/dashboard", params=params, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.insights import dashboard

        result = dashboard(
            input_uri=input_path,
            output_path=output_path,
            workflow=workflow,
            group_by=group_by,
            latest_run=latest_run,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"total_records: {result.get('total_records')}\nruns: {len(result.get('runs', []))}\nhtml_uri: {result.get('html_uri')}")


@app.command("status")
def status_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI prefix of the insights store."),
    run_id: str = typer.Option("", "--run-id", help="Optional run id to roll up."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Report store totals and (optionally) a per-run rollup."""
    if service:
        params = {"input_uri": input_path, "run_id": run_id}
        params = {k: v for k, v in params.items() if v not in ("", None)}
        result = request_json("GET", resolve_endpoint(endpoint), "/status", params=params, token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.insights.service import status_for_store

        result = status_for_store(input_path, run_id)
    emit(result, output=output)


@app.command("system-info")
def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show insights runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.insights.service import system_info_payload

        result = system_info_payload()
    emit(result, output=output)


@app.command("list")
def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Insights service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List service-tracked insights stores."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/list", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.insights.service import STORES

        result = {"stores": list(STORES.values())}
    emit(result, output=output, text="\n".join(str(s.get("store_uri")) for s in result.get("stores", [])) or "No stores found.")


def resolve_endpoint(endpoint: str) -> str:
    resolved = endpoint.strip() or os.environ.get(ENDPOINT_ENV, "")
    if not resolved:
        fail("--endpoint is required")
    if not resolved.startswith(("http://", "https://")):
        fail("--endpoint must be an http:// or https:// URL")
    return resolved.rstrip("/")


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{endpoint}{path}", headers=headers, json=payload, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fail(f"Insights request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach insights endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("Insights endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("Insights endpoint returned an unexpected response")
    return data
