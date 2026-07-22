"""Schema + quality-metric validation of a dataset-of-record manifest."""

from __future__ import annotations

from typing import Any

from .ingestion import compute_manifest_sha256
from .schemas import (
    VALIDATION_REPORT_SCHEMA,
    QualityStats,
    ValidateRequest,
    ValidateResponse,
)
from .storage import read_json_uri, uri_join, write_json_uri


class DatasetValidationError(RuntimeError):
    """Raised when a dataset manifest cannot be validated."""


def validation_report_uri(output_uri: str) -> str:
    return uri_join(output_uri, "validation_report.json")


def validate_manifest(request: ValidateRequest) -> ValidateResponse:
    """Run schema + quality-metric checks and emit a validation report."""
    try:
        manifest = read_json_uri(request.input_uri)
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"dataset manifest not found: {request.input_uri}") from exc
    except Exception as exc:
        raise DatasetValidationError(f"cannot read dataset manifest {request.input_uri}: {exc}") from exc

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise DatasetValidationError("dataset manifest has no records")

    stats = manifest.get("quality_stats") or {}
    quality_stats = QualityStats.model_validate(stats) if stats else QualityStats(record_count=len(records))

    failed_checks: list[str] = []

    # Schema check: every record carries the canonical required fields.
    missing = [
        str(record.get("record_id", f"#{index}"))
        for index, record in enumerate(records)
        if not (record.get("record_id") and record.get("modality") and record.get("uri"))
    ]
    if missing:
        failed_checks.append(f"schema: {len(missing)} record(s) missing required fields")

    # Completeness check.
    mean_completeness = float(quality_stats.mean_completeness)
    if mean_completeness < request.completeness_min:
        failed_checks.append(
            f"completeness: mean {mean_completeness} < min {request.completeness_min}"
        )

    # Corruption check.
    record_count = len(records)
    corruption_rate = round(quality_stats.corrupt_count / record_count, 4) if record_count else 0.0
    if corruption_rate > request.max_corruption_rate:
        failed_checks.append(
            f"corruption: rate {corruption_rate} > max {request.max_corruption_rate}"
        )

    passed = not failed_checks
    manifest_sha = compute_manifest_sha256(
        "validate",
        {"input_uri": request.input_uri, "record_count": record_count, "failed_checks": failed_checks},
    )
    report_uri = validation_report_uri(request.output_uri)
    report: dict[str, Any] = {
        "schema": VALIDATION_REPORT_SCHEMA,
        "manifest_sha256": manifest_sha,
        "source_manifest_uri": request.input_uri,
        "source_dataset_id": manifest.get("dataset_id", ""),
        "source_version": manifest.get("version", ""),
        "lineage": {
            "workflow_run": request.workflow_run,
            "input_uris": [request.input_uri],
            "produced_by": "workbench.dataset.validate",
        },
        "passed": passed,
        "record_count": record_count,
        "corruption_rate": corruption_rate,
        "thresholds": {
            "completeness_min": request.completeness_min,
            "max_corruption_rate": request.max_corruption_rate,
        },
        "failed_checks": failed_checks,
        "quality_stats": quality_stats.model_dump(mode="json"),
    }
    write_json_uri(report_uri, report)

    return ValidateResponse(
        report_uri=report_uri,
        passed=passed,
        record_count=record_count,
        failed_checks=failed_checks,
        quality_stats=quality_stats,
        manifest_sha256=manifest_sha,
    )
