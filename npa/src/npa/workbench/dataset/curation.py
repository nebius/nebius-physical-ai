"""Curate (slice) and query a dataset-of-record by event/location/quality."""

from __future__ import annotations

from typing import Any

from .ingestion import compute_manifest_sha256, manifest_uri
from .integrations import query_lancedb
from .schemas import (
    MANIFEST_SCHEMA,
    CurateRequest,
    CurateResponse,
    Lineage,
    QueryRequest,
    QueryResponse,
    QualityStats,
)
from .storage import read_json_uri, write_json_uri


class DatasetCurateError(RuntimeError):
    """Raised when curating or querying a dataset fails."""


def _record_metric(record: dict[str, Any], metric: str) -> float:
    if metric == "completeness":
        return float(record.get("completeness", 0.0))
    quality = record.get("quality") or {}
    return float(quality.get(metric, 0.0))


def _filter_records(
    records: list[dict[str, Any]],
    *,
    event: str = "",
    location: str = "",
    modality: str = "",
    quality_metric: str = "completeness",
    min_quality: float | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        if event and record.get("event") != event:
            continue
        if location and record.get("location") != location:
            continue
        if modality and record.get("modality") != modality:
            continue
        if min_quality is not None and _record_metric(record, quality_metric) < min_quality:
            continue
        filtered.append(record)
    return filtered


def _read_manifest(input_uri: str) -> dict[str, Any]:
    try:
        manifest = read_json_uri(input_uri)
    except FileNotFoundError as exc:
        raise DatasetCurateError(f"dataset manifest not found: {input_uri}") from exc
    except Exception as exc:
        raise DatasetCurateError(f"cannot read dataset manifest {input_uri}: {exc}") from exc
    records = manifest.get("records")
    if not isinstance(records, list):
        raise DatasetCurateError("dataset manifest has no records")
    return manifest


def curate_dataset(request: CurateRequest) -> CurateResponse:
    """Filter a parent version and register a lineage-tracked child version."""
    manifest = _read_manifest(request.input_uri)
    parent_dataset_id = str(manifest.get("dataset_id", ""))
    parent_version = str(manifest.get("version", ""))

    filter_predicate: dict[str, Any] = {
        "event": request.event,
        "location": request.location,
        "quality_metric": request.quality_metric,
        "min_quality": request.min_quality,
    }
    filtered = _filter_records(
        manifest["records"],
        event=request.event,
        location=request.location,
        quality_metric=request.quality_metric,
        min_quality=request.min_quality,
    )
    if not filtered:
        raise DatasetCurateError("curation filter selected zero records")

    manifest_sha = compute_manifest_sha256(
        "curate",
        {"parent": request.input_uri, "filter": filter_predicate, "count": len(filtered)},
    )
    child_version = f"{parent_version or 'v1'}.curated-{manifest_sha[:8]}"

    per_modality: dict[str, int] = {}
    events: set[str] = set()
    locations: set[str] = set()
    completeness_sum = 0.0
    corrupt = 0
    for record in filtered:
        modality = str(record.get("modality", ""))
        per_modality[modality] = per_modality.get(modality, 0) + 1
        if record.get("event"):
            events.add(str(record["event"]))
        if record.get("location"):
            locations.add(str(record["location"]))
        completeness_sum += float(record.get("completeness", 0.0))
        if float((record.get("quality") or {}).get("corruption", 0.0)) > 0.5 or not record.get("uri"):
            corrupt += 1
    quality_stats = QualityStats(
        record_count=len(filtered),
        modalities=sorted(per_modality),
        events=sorted(events),
        locations=sorted(locations),
        mean_completeness=round(completeness_sum / len(filtered), 4),
        corrupt_count=corrupt,
        per_modality_counts=per_modality,
    )

    lineage = Lineage(
        workflow_run=request.workflow_run,
        input_uris=[request.input_uri],
        source=str(manifest.get("lineage", {}).get("source", "")),
        parent_dataset_id=parent_dataset_id,
        parent_version=parent_version,
        filter_predicate=filter_predicate,
        produced_by="workbench.dataset.curate",
    )
    target_uri = manifest_uri(request.output_uri, parent_dataset_id or "dataset", child_version)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": parent_dataset_id,
        "version": child_version,
        "parent_dataset_id": parent_dataset_id,
        "parent_version": parent_version,
        "filter_predicate": filter_predicate,
        "record_count": len(filtered),
        "modalities": quality_stats.modalities,
        "manifest_sha256": manifest_sha,
        "lineage": lineage.model_dump(mode="json"),
        "quality_stats": quality_stats.model_dump(mode="json"),
        "records": filtered,
    }
    write_json_uri(target_uri, payload)

    return CurateResponse(
        dataset_id=parent_dataset_id,
        version=child_version,
        status="completed",
        manifest_uri=target_uri,
        record_count=len(filtered),
        parent_dataset_id=parent_dataset_id,
        parent_version=parent_version,
        filter_predicate=filter_predicate,
        manifest_sha256=manifest_sha,
        lineage=lineage,
    )


def query_dataset(request: QueryRequest) -> QueryResponse:
    """Query records by event/location/quality facets (LanceDB or manifest)."""
    filter_predicate: dict[str, Any] = {
        "event": request.event,
        "location": request.location,
        "modality": request.modality,
        "quality_metric": request.quality_metric,
        "min_quality": request.min_quality,
    }
    if request.lancedb_endpoint.strip():
        records = query_lancedb(
            lancedb_endpoint=request.lancedb_endpoint,
            filter_predicate=filter_predicate,
            limit=request.limit,
        )
        return QueryResponse(
            backend="lancedb",
            count=len(records),
            records=records,
            filter_predicate=filter_predicate,
        )

    manifest = _read_manifest(request.input_uri)
    filtered = _filter_records(
        manifest["records"],
        event=request.event,
        location=request.location,
        modality=request.modality,
        quality_metric=request.quality_metric,
        min_quality=request.min_quality,
    )
    limited = filtered[: request.limit]
    return QueryResponse(
        backend="manifest",
        count=len(limited),
        records=limited,
        filter_predicate=filter_predicate,
    )
