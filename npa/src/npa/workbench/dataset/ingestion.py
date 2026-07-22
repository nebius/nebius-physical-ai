"""Ingest raw sensor data into a versioned dataset-of-record manifest."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .integrations import fiftyone_handoff, index_in_lancedb
from .schemas import (
    COMPLETENESS_FIELDS,
    MANIFEST_SCHEMA,
    IngestRequest,
    IngestResponse,
    Lineage,
    QualityStats,
    SensorRecord,
    SensorSchema,
)
from .storage import read_json_uri, uri_join, write_json_uri


class DatasetIngestError(RuntimeError):
    """Raised when dataset ingestion fails."""


def compute_manifest_sha256(kind: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic manifest hash for inputs and records."""
    digest = hashlib.sha256()
    digest.update(kind.encode("utf-8"))
    digest.update(b"\n")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def manifest_uri(output_uri: str, dataset_id: str, version: str) -> str:
    return uri_join(output_uri, dataset_id, version, "manifest.json")


def load_raw_records(input_uri: str) -> list[dict[str, Any]]:
    """Read raw sensor records from an input manifest URI."""
    try:
        payload = read_json_uri(input_uri)
    except FileNotFoundError as exc:
        raise DatasetIngestError(f"raw sensor data not found: {input_uri}") from exc
    except Exception as exc:
        raise DatasetIngestError(f"cannot read raw sensor data {input_uri}: {exc}") from exc
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise DatasetIngestError("raw sensor data has no records")
    return records


def _record_completeness(raw: dict[str, Any]) -> float:
    present = sum(1 for field in COMPLETENESS_FIELDS if raw.get(field))
    return round(present / len(COMPLETENESS_FIELDS), 4)


def _is_corrupt(raw: dict[str, Any], quality: dict[str, float]) -> bool:
    if not raw.get("uri"):
        return True
    return float(quality.get("corruption", 0.0)) > 0.5


def normalize_records(
    raw_records: list[dict[str, Any]],
    schema: SensorSchema,
) -> tuple[list[SensorRecord], int]:
    """Validate raw records against the sensor schema and canonicalize them."""
    normalized: list[SensorRecord] = []
    corrupt = 0
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise DatasetIngestError(f"record {index} is not a mapping")
        for field in schema.required_fields:
            if not raw.get(field):
                raise DatasetIngestError(f"record {index} missing required field {field!r}")
        modality = str(raw["modality"])
        if schema.modalities and modality not in schema.modalities:
            raise DatasetIngestError(
                f"record {index} modality {modality!r} not in declared modalities {schema.modalities}"
            )
        record_id = str(raw["record_id"])
        if record_id in seen:
            raise DatasetIngestError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        quality = {str(k): float(v) for k, v in (raw.get("quality") or {}).items()}
        if _is_corrupt(raw, quality):
            corrupt += 1
        normalized.append(
            SensorRecord(
                record_id=record_id,
                modality=modality,
                uri=str(raw["uri"]),
                event=str(raw.get("event", "")),
                location=str(raw.get("location", "")),
                timestamp=str(raw.get("timestamp", "")),
                quality=quality,
                has_embedding=bool(raw.get("embedding")),
                completeness=_record_completeness(raw),
            )
        )
    return normalized, corrupt


def compute_quality_stats(records: list[SensorRecord], corrupt_count: int) -> QualityStats:
    per_modality: dict[str, int] = {}
    for record in records:
        per_modality[record.modality] = per_modality.get(record.modality, 0) + 1
    mean_completeness = (
        round(sum(record.completeness for record in records) / len(records), 4) if records else 0.0
    )
    return QualityStats(
        record_count=len(records),
        modalities=sorted(per_modality),
        events=sorted({record.event for record in records if record.event}),
        locations=sorted({record.location for record in records if record.location}),
        mean_completeness=mean_completeness,
        corrupt_count=corrupt_count,
        per_modality_counts=per_modality,
    )


def build_manifest_payload(
    *,
    dataset_id: str,
    version: str,
    records: list[SensorRecord],
    quality_stats: QualityStats,
    lineage: Lineage,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "version": version,
        "record_count": len(records),
        "modalities": quality_stats.modalities,
        "manifest_sha256": manifest_sha256,
        "lineage": lineage.model_dump(mode="json"),
        "quality_stats": quality_stats.model_dump(mode="json"),
        "records": [record.model_dump(mode="json") for record in records],
    }


def ingest_dataset(
    request: IngestRequest,
    *,
    lancedb_endpoint: str = "",
    lance_table: str = "",
    lance_uri: str = "",
    fiftyone_endpoint: str = "",
) -> IngestResponse:
    """Ingest raw sensor data and register a versioned dataset manifest."""
    raw_records = load_raw_records(request.input_uri)
    records, corrupt = normalize_records(raw_records, request.sensor_schema)
    quality_stats = compute_quality_stats(records, corrupt)

    lineage = Lineage(
        workflow_run=request.workflow_run,
        input_uris=[request.input_uri],
        source=request.source,
    )
    manifest = compute_manifest_sha256(
        "ingest",
        {
            "dataset_id": request.dataset_id,
            "version": request.version,
            "records": [record.model_dump(mode="json") for record in records],
        },
    )
    target_uri = manifest_uri(request.output_uri, request.dataset_id, request.version)
    payload = build_manifest_payload(
        dataset_id=request.dataset_id,
        version=request.version,
        records=records,
        quality_stats=quality_stats,
        lineage=lineage,
        manifest_sha256=manifest,
    )
    index = index_in_lancedb(
        payload["records"],
        lancedb_endpoint=lancedb_endpoint,
        table=lance_table or request.dataset_id,
        lance_uri=lance_uri,
    )
    payload["index"] = index
    write_json_uri(target_uri, payload)

    fiftyone_handoff(
        fiftyone_endpoint=fiftyone_endpoint,
        manifest_uri=target_uri,
        dataset_id=request.dataset_id,
        version=request.version,
    )

    return IngestResponse(
        dataset_id=request.dataset_id,
        version=request.version,
        status="completed",
        manifest_uri=target_uri,
        record_count=len(records),
        modalities=quality_stats.modalities,
        quality_stats=quality_stats,
        manifest_sha256=manifest,
        lineage=lineage,
    )
