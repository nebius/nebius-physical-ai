"""Schemas for the dataset-of-record workbench service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MANIFEST_SCHEMA = "npa.dataset.manifest.v1"
VALIDATION_REPORT_SCHEMA = "npa.dataset.validation_report.v1"

DEFAULT_PORT = 8792
DEFAULT_TOKEN_ENV = "DATASET_TOKEN"
DEFAULT_VERSION = "v1"
DEFAULT_COMPLETENESS_MIN = 0.5
DEFAULT_MAX_CORRUPTION_RATE = 0.1
DEFAULT_QUERY_LIMIT = 100

CANONICAL_REQUIRED_FIELDS = ("record_id", "modality", "uri")
# Informative fields that count toward per-record completeness.
COMPLETENESS_FIELDS = ("event", "location", "timestamp", "quality", "embedding")

RunStatus = Literal["completed", "failed", "rejected"]


class SensorSchema(BaseModel):
    """Declared sensor schema an ingest validates raw records against."""

    model_config = ConfigDict(extra="forbid")

    modalities: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=lambda: list(CANONICAL_REQUIRED_FIELDS))
    quality_thresholds: dict[str, float] = Field(default_factory=dict)


class Lineage(BaseModel):
    """Provenance threaded through every dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    workflow_run: str = ""
    input_uris: list[str] = Field(default_factory=list)
    source: str = ""
    parent_dataset_id: str = ""
    parent_version: str = ""
    filter_predicate: dict[str, Any] = Field(default_factory=dict)
    produced_by: str = "workbench.dataset"


class SensorRecord(BaseModel):
    """A canonical, normalized sensor record."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    modality: str
    uri: str
    event: str = ""
    location: str = ""
    timestamp: str = ""
    quality: dict[str, float] = Field(default_factory=dict)
    has_embedding: bool = False
    completeness: float = 0.0


class IngestRequest(BaseModel):
    """Request body for ingesting + registering a dataset-of-record version."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    dataset_id: str = Field(..., min_length=1)
    version: str = DEFAULT_VERSION
    sensor_schema: SensorSchema = Field(default_factory=SensorSchema)
    source: str = ""
    workflow_run: str = ""

    @field_validator("input_uri", "output_uri", "dataset_id", "version")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class QualityStats(BaseModel):
    """Aggregate quality statistics recorded in a dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    record_count: int = 0
    modalities: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    mean_completeness: float = 0.0
    corrupt_count: int = 0
    per_modality_counts: dict[str, int] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response returned by the ingest endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    version: str
    status: RunStatus
    manifest_uri: str
    manifest_schema: str = MANIFEST_SCHEMA
    record_count: int
    modalities: list[str] = Field(default_factory=list)
    quality_stats: QualityStats = Field(default_factory=QualityStats)
    manifest_sha256: str = ""
    lineage: Lineage = Field(default_factory=Lineage)


class ValidateRequest(BaseModel):
    """Request body for schema + quality-metric validation of a manifest."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    completeness_min: float = Field(DEFAULT_COMPLETENESS_MIN, ge=0.0, le=1.0)
    max_corruption_rate: float = Field(DEFAULT_MAX_CORRUPTION_RATE, ge=0.0, le=1.0)
    workflow_run: str = ""

    @field_validator("input_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class ValidateResponse(BaseModel):
    """Response returned by the validate endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    report_uri: str
    report_schema: str = VALIDATION_REPORT_SCHEMA
    passed: bool
    record_count: int
    failed_checks: list[str] = Field(default_factory=list)
    quality_stats: QualityStats = Field(default_factory=QualityStats)
    manifest_sha256: str = ""


class CurateRequest(BaseModel):
    """Request body for slicing a dataset version by event/location/quality."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    event: str = ""
    location: str = ""
    quality_metric: str = "completeness"
    min_quality: float | None = None
    workflow_run: str = ""

    @field_validator("input_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class CurateResponse(BaseModel):
    """Response returned by the curate endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    version: str
    status: RunStatus
    manifest_uri: str
    manifest_schema: str = MANIFEST_SCHEMA
    record_count: int
    parent_dataset_id: str
    parent_version: str
    filter_predicate: dict[str, Any] = Field(default_factory=dict)
    manifest_sha256: str = ""
    lineage: Lineage = Field(default_factory=Lineage)


class QueryRequest(BaseModel):
    """Request body for querying records by event/location/quality facets."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    event: str = ""
    location: str = ""
    modality: str = ""
    quality_metric: str = "completeness"
    min_quality: float | None = None
    limit: int = Field(DEFAULT_QUERY_LIMIT, ge=1)
    lancedb_endpoint: str = ""

    @field_validator("input_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class QueryResponse(BaseModel):
    """Response returned by the query endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    count: int
    records: list[dict[str, Any]] = Field(default_factory=list)
    filter_predicate: dict[str, Any] = Field(default_factory=dict)


class DatasetListResponse(BaseModel):
    """List response for service-managed dataset versions."""

    datasets: list[dict[str, Any]] = Field(default_factory=list)
