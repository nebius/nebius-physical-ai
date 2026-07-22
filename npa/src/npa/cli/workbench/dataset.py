"""Typer CLI for `npa workbench dataset`."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import httpx
import typer

from npa.workbench.dataset.schemas import (
    DEFAULT_COMPLETENESS_MIN,
    DEFAULT_MAX_CORRUPTION_RATE,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_TOKEN_ENV,
    DEFAULT_VERSION,
)

app = typer.Typer(
    name="dataset",
    help="Dataset-of-record: ingest, validate, curate, and query production sensor data.",
    no_args_is_help=True,
)

ENDPOINT_ENV = "NPA_DATASET_ENDPOINT"


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


@app.command("ingest")
def ingest_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI of raw sensor records to ingest."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix for the versioned dataset manifest."),
    dataset_id: str = typer.Option(..., "--dataset-id", help="Dataset-of-record identifier."),
    version: str = typer.Option(DEFAULT_VERSION, "--version", help="Dataset version label."),
    modality: list[str] = typer.Option([], "--modality", help="Allowed sensor modality (repeatable). Empty allows any."),
    source: str = typer.Option("", "--source", help="Source lineage label for the raw data."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into lineage."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Ingest raw sensor data into a versioned dataset-of-record manifest."""
    payload = {
        "input_uri": input_path,
        "output_uri": output_path,
        "dataset_id": dataset_id,
        "version": version,
        "sensor_schema": {"modalities": list(modality)},
        "source": source,
        "workflow_run": workflow_run,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/ingest", payload=payload, token_env=token_env, timeout=120.0)
    else:
        from npa.sdk.workbench.dataset import ingest

        result = ingest(
            input_uri=input_path,
            output_uri=output_path,
            dataset_id=dataset_id,
            version=version,
            sensor_schema={"modalities": list(modality)},
            source=source,
            workflow_run=workflow_run,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"dataset_id: {result.get('dataset_id')}\nversion: {result.get('version')}\nrecord_count: {result.get('record_count')}\nmanifest_uri: {result.get('manifest_uri')}")


@app.command("validate")
def validate_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI of a dataset manifest to validate."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix for the validation report."),
    completeness_min: float = typer.Option(DEFAULT_COMPLETENESS_MIN, "--completeness-min", help="Minimum mean completeness required to pass."),
    max_corruption_rate: float = typer.Option(DEFAULT_MAX_CORRUPTION_RATE, "--max-corruption-rate", help="Maximum tolerated corruption rate."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into lineage."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Validate a dataset manifest against schema + quality thresholds."""
    payload = {
        "input_uri": input_path,
        "output_uri": output_path,
        "completeness_min": completeness_min,
        "max_corruption_rate": max_corruption_rate,
        "workflow_run": workflow_run,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/validate", payload=payload, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.dataset import validate

        result = validate(
            input_uri=input_path,
            output_uri=output_path,
            completeness_min=completeness_min,
            max_corruption_rate=max_corruption_rate,
            workflow_run=workflow_run,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"passed: {result.get('passed')}\nrecord_count: {result.get('record_count')}\nreport_uri: {result.get('report_uri')}")


@app.command("curate")
def curate_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI of the parent dataset manifest."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix for the curated dataset version."),
    event: str = typer.Option("", "--event", help="Event of interest to filter on."),
    location: str = typer.Option("", "--location", help="Location to filter on."),
    quality_metric: str = typer.Option("completeness", "--quality-metric", help="Quality metric name for --min-quality."),
    min_quality: float = typer.Option(-1.0, "--min-quality", help="Minimum quality metric value; negative disables."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into lineage."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Slice a dataset version by event/location/quality with lineage."""
    resolved_min = None if min_quality < 0 else min_quality
    payload = {
        "input_uri": input_path,
        "output_uri": output_path,
        "event": event,
        "location": location,
        "quality_metric": quality_metric,
        "min_quality": resolved_min,
        "workflow_run": workflow_run,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/curate", payload=payload, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.dataset import curate

        result = curate(
            input_uri=input_path,
            output_uri=output_path,
            event=event,
            location=location,
            quality_metric=quality_metric,
            min_quality=resolved_min,
            workflow_run=workflow_run,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"version: {result.get('version')}\nrecord_count: {result.get('record_count')}\nmanifest_uri: {result.get('manifest_uri')}")


@app.command("query")
def query_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI of a dataset manifest to query."),
    event: str = typer.Option("", "--event", help="Event of interest to filter on."),
    location: str = typer.Option("", "--location", help="Location to filter on."),
    modality: str = typer.Option("", "--modality", help="Sensor modality to filter on."),
    quality_metric: str = typer.Option("completeness", "--quality-metric", help="Quality metric name for --min-quality."),
    min_quality: float = typer.Option(-1.0, "--min-quality", help="Minimum quality metric value; negative disables."),
    limit: int = typer.Option(DEFAULT_QUERY_LIMIT, "--limit", help="Maximum records to return."),
    lancedb_endpoint: str = typer.Option("", "--lancedb-endpoint", help="LanceDB service endpoint backing the query index."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Query dataset records by event/location/quality facets."""
    resolved_min = None if min_quality < 0 else min_quality
    if service:
        params = {
            "input_uri": input_path,
            "event": event,
            "location": location,
            "modality": modality,
            "quality_metric": quality_metric,
            "min_quality": resolved_min,
            "limit": limit,
            "lancedb_endpoint": lancedb_endpoint,
        }
        params = {k: v for k, v in params.items() if v not in ("", None)}
        result = request_json("GET", resolve_endpoint(endpoint), "/query", params=params, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.dataset import query

        result = query(
            input_uri=input_path,
            event=event,
            location=location,
            modality=modality,
            quality_metric=quality_metric,
            min_quality=resolved_min,
            limit=limit,
            lancedb_endpoint=lancedb_endpoint,
        ).model_dump(mode="json")
    emit(result, output=output, text=f"backend: {result.get('backend')}\ncount: {result.get('count')}")


@app.command("status")
def status_cmd(
    dataset_id: str = typer.Option(..., "--dataset-id", help="Dataset-of-record identifier."),
    version: str = typer.Option(..., "--version", help="Dataset version label."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Fetch a registered dataset version status."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/status", params={"dataset_id": dataset_id, "version": version}, token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.dataset.service import status_for_version

        try:
            result = status_for_version(dataset_id, version)
        except Exception as exc:  # HTTPException or KeyError when not registered locally.
            fail(str(getattr(exc, "detail", exc)))
            return
    emit(result, output=output)


@app.command("system-info")
def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show dataset-of-record runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.dataset.service import system_info_payload

        result = system_info_payload()
    emit(result, output=output)


@app.command("list")
def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Dataset service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List service-managed dataset versions."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/list", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.dataset.service import DATASETS

        result = {"datasets": list(DATASETS.values())}
    emit(result, output=output, text="\n".join(f"{d['dataset_id']}@{d['version']}" for d in result.get("datasets", [])) or "No datasets found.")


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
        fail(f"Dataset request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach dataset endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("Dataset endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("Dataset endpoint returned an unexpected response")
    return data
