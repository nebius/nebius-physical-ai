"""Append-only metric + lineage store and non-invasive run ingestion.

The store is an append-only index on S3 (JSONL under a configurable prefix).
``record_metrics`` writes explicit emissions; ``ingest_run`` scans an existing
run prefix for manifest/report schemas already produced by other tools and
extracts their metrics + provenance without modifying the emitting tools.
"""

from __future__ import annotations

import time
from typing import Any

from .integrations import index_metrics_in_lancedb
from .schemas import (
    EDGES_OBJECT,
    METRIC_RECORD_SCHEMA,
    RECORDS_OBJECT,
    IngestedArtifact,
    IngestRunRequest,
    IngestRunResponse,
    LineageEdge,
    LineageRef,
    MetricRecord,
    RecordRequest,
    RecordResponse,
)
from .storage import (
    append_jsonl_uri,
    list_json_uris,
    read_json_uri,
    read_jsonl_uri,
    uri_join,
)

DATASET_MANIFEST_SCHEMA = "npa.dataset.manifest.v1"
DATASET_VALIDATION_SCHEMA = "npa.dataset.validation_report.v1"
SCENARIO_ADVERSARIAL_SCHEMA = "npa.scenario_gen.adversarial_set.v1"


class InsightsStoreError(RuntimeError):
    """Raised when recording or ingesting into the store fails."""


def records_uri(store_uri: str) -> str:
    return uri_join(store_uri, RECORDS_OBJECT)


def edges_uri(store_uri: str) -> str:
    return uri_join(store_uri, EDGES_OBJECT)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_records(store_uri: str) -> list[dict[str, Any]]:
    return read_jsonl_uri(records_uri(store_uri))


def read_edges(store_uri: str) -> list[dict[str, Any]]:
    return read_jsonl_uri(edges_uri(store_uri))


def _persist(
    store_uri: str,
    records: list[MetricRecord],
    edges: list[LineageEdge],
    *,
    lancedb_endpoint: str = "",
) -> tuple[int, int]:
    """Append records + edges to the store and return the new totals."""
    rec_rows: list[dict[str, Any]] = []
    for record in records:
        row = record.model_dump(mode="json")
        row["schema"] = METRIC_RECORD_SCHEMA
        if not row.get("timestamp"):
            row["timestamp"] = _now()
        rec_rows.append(row)
    edge_rows = [edge.model_dump(mode="json") for edge in edges]

    total_records = append_jsonl_uri(records_uri(store_uri), rec_rows) if rec_rows else len(read_records(store_uri))
    total_edges = append_jsonl_uri(edges_uri(store_uri), edge_rows) if edge_rows else len(read_edges(store_uri))

    if rec_rows:
        index_metrics_in_lancedb(
            rec_rows,
            lancedb_endpoint=lancedb_endpoint,
            table="insights_metrics",
        )
    return total_records, total_edges


def _load_record_payload(payload: Any) -> tuple[list[MetricRecord], list[LineageEdge]]:
    """Parse a records/edges JSON document into validated models."""
    if isinstance(payload, list):
        rows, edge_rows = payload, []
    elif isinstance(payload, dict):
        rows = payload.get("records", [])
        edge_rows = payload.get("edges", [])
    else:
        raise InsightsStoreError("record input must be a JSON object or list of records")
    if not isinstance(rows, list) or not isinstance(edge_rows, list):
        raise InsightsStoreError("record input 'records'/'edges' must be lists")
    records = [MetricRecord.model_validate(row) for row in rows]
    edges = [LineageEdge.model_validate(row) for row in edge_rows]
    return records, edges


def record_metrics(request: RecordRequest) -> RecordResponse:
    """Record explicit metric emissions + lineage edges into the store."""
    records = list(request.records)
    edges = list(request.edges)
    if request.input_uri.strip():
        try:
            payload = read_json_uri(request.input_uri)
        except FileNotFoundError as exc:
            raise InsightsStoreError(f"record input not found: {request.input_uri}") from exc
        except Exception as exc:  # noqa: BLE001
            raise InsightsStoreError(f"cannot read record input {request.input_uri}: {exc}") from exc
        loaded_records, loaded_edges = _load_record_payload(payload)
        records.extend(loaded_records)
        edges.extend(loaded_edges)

    if not records and not edges:
        raise InsightsStoreError("record requires at least one metric record or lineage edge")

    if request.workflow_run:
        for record in records:
            if not record.run_id:
                record.run_id = request.workflow_run

    total_records, total_edges = _persist(
        request.output_uri,
        records,
        edges,
        lancedb_endpoint=request.lancedb_endpoint,
    )
    return RecordResponse(
        store_uri=request.output_uri,
        records_uri=records_uri(request.output_uri),
        edges_uri=edges_uri(request.output_uri),
        recorded_count=len(records),
        edge_count=len(edges),
        total_records=total_records,
        total_edges=total_edges,
    )


def ingest_run(request: IngestRunRequest) -> IngestRunResponse:
    """Scan an S3 run prefix for known manifests and extract metrics + lineage."""
    uris = list_json_uris(request.input_uri)
    if not uris:
        raise InsightsStoreError(f"no JSON artifacts found under run prefix: {request.input_uri}")

    all_records: list[MetricRecord] = []
    all_edges: list[LineageEdge] = []
    ingested: list[IngestedArtifact] = []
    scanned = 0

    for uri in uris:
        scanned += 1
        try:
            payload = read_json_uri(uri)
        except Exception:  # noqa: BLE001 - skip unreadable/non-object artifacts.
            continue
        if not isinstance(payload, dict):
            continue
        records, edges, schema_id = _extract(
            payload,
            source_uri=uri,
            workflow=request.workflow,
            workflow_run=request.workflow_run,
        )
        if schema_id is None:
            continue
        all_records.extend(records)
        all_edges.extend(edges)
        ingested.append(
            IngestedArtifact(uri=uri, schema_id=schema_id, records=len(records), edges=len(edges))
        )

    if not ingested:
        raise InsightsStoreError(
            f"no known manifest/report schemas found under run prefix: {request.input_uri}"
        )

    total_records, total_edges = _persist(
        request.output_uri,
        all_records,
        all_edges,
        lancedb_endpoint=request.lancedb_endpoint,
    )
    return IngestRunResponse(
        store_uri=request.output_uri,
        records_uri=records_uri(request.output_uri),
        edges_uri=edges_uri(request.output_uri),
        scanned=scanned,
        ingested=ingested,
        recorded_count=len(all_records),
        edge_count=len(all_edges),
        total_records=total_records,
        total_edges=total_edges,
    )


def _metric(
    *,
    run_id: str,
    workflow: str,
    tool: str,
    stage: str,
    name: str,
    value: float,
    unit: str = "",
    labels: dict[str, str] | None = None,
    lineage: LineageRef | None = None,
    artifact_uri: str = "",
    artifact_version: str = "",
) -> MetricRecord:
    return MetricRecord(
        run_id=run_id or "unknown",
        metric_name=name,
        value=float(value),
        workflow=workflow,
        tool=tool,
        stage=stage,
        unit=unit,
        labels=labels or {},
        lineage=lineage or LineageRef(),
        artifact_uri=artifact_uri,
        artifact_version=artifact_version,
    )


def _extract(
    payload: dict[str, Any],
    *,
    source_uri: str,
    workflow: str,
    workflow_run: str,
) -> tuple[list[MetricRecord], list[LineageEdge], str | None]:
    """Route a discovered artifact to the matching extractor."""
    schema_id = str(payload.get("schema", ""))
    if schema_id == DATASET_MANIFEST_SCHEMA:
        return (*_extract_dataset_manifest(payload, source_uri, workflow, workflow_run), schema_id)
    if schema_id == DATASET_VALIDATION_SCHEMA:
        return (*_extract_validation_report(payload, source_uri, workflow, workflow_run), schema_id)
    if schema_id == SCENARIO_ADVERSARIAL_SCHEMA:
        return (*_extract_adversarial_set(payload, source_uri, workflow, workflow_run), schema_id)
    if "decision" in payload and not schema_id:
        return (*_extract_decision(payload, source_uri, workflow, workflow_run), "decision")
    return [], [], None


def _run_id(payload: dict[str, Any], workflow_run: str) -> str:
    lineage = payload.get("lineage") or {}
    return (
        workflow_run
        or str(payload.get("run_id") or "")
        or str(lineage.get("workflow_run") or "")
        or "unknown"
    )


def _extract_dataset_manifest(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    stats = payload.get("quality_stats") or {}
    lineage_meta = payload.get("lineage") or {}
    dataset_id = str(payload.get("dataset_id", ""))
    version = str(payload.get("version", ""))
    dataset_version = f"{dataset_id}@{version}" if dataset_id else version
    record_count = int(payload.get("record_count", stats.get("record_count", 0)) or 0)
    corrupt = int(stats.get("corrupt_count", 0) or 0)
    stage = "curate" if payload.get("parent_version") else "ingest"
    run_id = _run_id(payload, workflow_run)
    input_uris = [str(u) for u in lineage_meta.get("input_uris", []) if u]
    ref = LineageRef(
        input_uris=input_uris,
        dataset_version=dataset_version,
        parent_uri=str(payload.get("parent_dataset_id", "") or lineage_meta.get("parent_dataset_id", "")),
        parent_version=str(payload.get("parent_version", "") or lineage_meta.get("parent_version", "")),
    )
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="record_count", value=record_count, unit="records", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="mean_completeness", value=float(stats.get("mean_completeness", 0.0) or 0.0), lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="corrupt_count", value=corrupt, unit="records", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="corruption_rate", value=round(corrupt / record_count, 4) if record_count else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage=stage, name="modality_count", value=len(stats.get("modalities", []) or []), unit="modalities", lineage=ref, artifact_uri=source_uri, artifact_version=dataset_version),
    ]
    edges = [
        LineageEdge(from_uri=input_uri, to_uri=source_uri, to_version=dataset_version, relation="produced_from", run_id=run_id)
        for input_uri in input_uris
    ]
    return metrics, edges


def _extract_validation_report(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    stats = payload.get("quality_stats") or {}
    source_manifest = str(payload.get("source_manifest_uri", ""))
    run_id = _run_id(payload, workflow_run)
    ref = LineageRef(input_uris=[source_manifest] if source_manifest else [])
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="validation_passed", value=1.0 if payload.get("passed") else 0.0, lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="corruption_rate", value=float(payload.get("corruption_rate", 0.0) or 0.0), lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="record_count", value=int(payload.get("record_count", stats.get("record_count", 0)) or 0), unit="records", lineage=ref, artifact_uri=source_uri),
        _metric(run_id=run_id, workflow=workflow, tool="dataset", stage="validate", name="failed_check_count", value=len(payload.get("failed_checks", []) or []), lineage=ref, artifact_uri=source_uri),
    ]
    edges: list[LineageEdge] = []
    if source_manifest:
        edges.append(LineageEdge(from_uri=source_manifest, to_uri=source_uri, relation="evaluated_on", run_id=run_id))
    return metrics, edges


def _extract_adversarial_set(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    lineage_meta = payload.get("lineage") or {}
    scenarios = payload.get("scenarios") or []
    severities = [float(s.get("severity", 0.0) or 0.0) for s in scenarios]
    diversities = [float(s.get("diversity", 0.0) or 0.0) for s in scenarios]
    run_id = _run_id(payload, workflow_run)
    policy_uri = str(lineage_meta.get("policy_uri", ""))
    base_config_uri = str(lineage_meta.get("base_config_uri", ""))
    input_uris = [u for u in (policy_uri, base_config_uri) if u]
    ref = LineageRef(input_uris=input_uris, checkpoint_uri=policy_uri)
    metrics = [
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="scenario_count", value=int(payload.get("scenario_count", len(scenarios)) or 0), unit="scenarios", lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="top_severity", value=max(severities) if severities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="mean_severity", value=round(sum(severities) / len(severities), 4) if severities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
        _metric(run_id=run_id, workflow=workflow, tool="scenario_gen", stage="generate", name="mean_diversity", value=round(sum(diversities) / len(diversities), 4) if diversities else 0.0, lineage=ref, artifact_uri=source_uri, artifact_version=run_id),
    ]
    edges = [
        LineageEdge(from_uri=input_uri, to_uri=source_uri, to_version=run_id, relation="produced_from", run_id=run_id)
        for input_uri in input_uris
    ]
    return metrics, edges


def _extract_decision(
    payload: dict[str, Any], source_uri: str, workflow: str, workflow_run: str
) -> tuple[list[MetricRecord], list[LineageEdge]]:
    from npa.orchestration.npa_workflow.decisions import normalize_decision

    raw = str(payload.get("decision", ""))
    decision = normalize_decision(raw)
    promoted = 1.0 if decision == "promote_checkpoint" else 0.0
    run_id = _run_id(payload, workflow_run)
    metric = _metric(
        run_id=run_id,
        workflow=workflow,
        tool="workflow",
        stage="gate",
        name="gate_promote",
        value=promoted,
        labels={"decision": decision},
        artifact_uri=source_uri,
    )
    return [metric], []
