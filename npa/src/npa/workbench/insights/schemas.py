"""Schemas for the insights (lineage + metrics backbone) workbench service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

METRIC_RECORD_SCHEMA = "npa.insights.metric_record.v1"
LINEAGE_EDGE_SCHEMA = "npa.insights.lineage_edge.v1"
COMPARISON_SCHEMA = "npa.insights.comparison.v1"
DASHBOARD_SCHEMA = "npa.insights.dashboard.v1"

DEFAULT_PORT = 8793
DEFAULT_TOKEN_ENV = "INSIGHTS_TOKEN"
DEFAULT_QUERY_LIMIT = 100

# Append-only store layout under a configurable prefix on S3 (JSONL fallback).
RECORDS_OBJECT = "records.jsonl"
EDGES_OBJECT = "edges.jsonl"

# Relations threaded through the provenance graph.
LineageRelation = Literal["produced_from", "derived_from", "evaluated_on"]

# Substrings that mark a metric as "lower is better" for comparison direction.
LOWER_IS_BETTER_HINTS = (
    "corrupt",
    "failure",
    "fail_",
    "error",
    "loss",
    "latency",
    "violation",
    "collision",
    "regression",
    "rejected",
)


class LineageRef(BaseModel):
    """Provenance references carried alongside a metric emission."""

    model_config = ConfigDict(extra="forbid")

    input_uris: list[str] = Field(default_factory=list)
    dataset_version: str = ""
    checkpoint_uri: str = ""
    parent_uri: str = ""
    parent_version: str = ""


class MetricRecord(BaseModel):
    """A single metric emission keyed by run id + lineage refs.

    Validated against ``npa.insights.metric_record.v1``.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1)
    metric_name: str = Field(..., min_length=1)
    value: float
    workflow: str = ""
    tool: str = ""
    stage: str = ""
    unit: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    lineage: LineageRef = Field(default_factory=LineageRef)
    artifact_uri: str = ""
    artifact_version: str = ""
    timestamp: str = ""

    @field_validator("run_id", "metric_name")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class LineageEdge(BaseModel):
    """A provenance edge from one artifact/version to another.

    Validated against ``npa.insights.lineage_edge.v1``. ``from``/``to`` are
    modeled as ``from_uri``/``to_uri`` to avoid the Python keyword clash.
    """

    model_config = ConfigDict(extra="forbid")

    from_uri: str = Field(..., min_length=1)
    to_uri: str = Field(..., min_length=1)
    relation: LineageRelation = "produced_from"
    from_version: str = ""
    to_version: str = ""
    run_id: str = ""

    @field_validator("from_uri", "to_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class RecordRequest(BaseModel):
    """Request body for recording metric emissions + lineage edges."""

    model_config = ConfigDict(extra="forbid")

    output_uri: str = Field(..., min_length=1)
    input_uri: str = ""
    records: list[MetricRecord] = Field(default_factory=list)
    edges: list[LineageEdge] = Field(default_factory=list)
    workflow_run: str = ""
    lancedb_endpoint: str = ""

    @field_validator("output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class RecordResponse(BaseModel):
    """Response returned by the record/ingest endpoints and SDK."""

    model_config = ConfigDict(extra="forbid")

    store_uri: str
    records_uri: str
    edges_uri: str
    recorded_count: int
    edge_count: int
    total_records: int
    total_edges: int
    metric_record_schema: str = METRIC_RECORD_SCHEMA
    lineage_edge_schema: str = LINEAGE_EDGE_SCHEMA


class IngestRunRequest(BaseModel):
    """Request body for non-invasive ingestion of an S3 run prefix."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    workflow: str = ""
    workflow_run: str = ""
    lancedb_endpoint: str = ""

    @field_validator("input_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class IngestedArtifact(BaseModel):
    """One artifact discovered and ingested from a run prefix."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    schema_id: str
    records: int
    edges: int


class IngestRunResponse(BaseModel):
    """Response returned by the ingest-run endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    store_uri: str
    records_uri: str
    edges_uri: str
    scanned: int
    ingested: list[IngestedArtifact] = Field(default_factory=list)
    recorded_count: int
    edge_count: int
    total_records: int
    total_edges: int


class LineageRequest(BaseModel):
    """Request for traversing the provenance graph of an artifact/version."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    uri: str = Field(..., min_length=1)
    version: str = ""
    direction: Literal["both", "ancestors", "descendants"] = "both"
    depth: int = Field(-1, ge=-1)

    @field_validator("input_uri", "uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class LineageNode(BaseModel):
    """A node in the reconstructed provenance graph."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    version: str = ""


class LineageResponse(BaseModel):
    """Response returned by the lineage endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    root: LineageNode
    ancestors: list[dict[str, Any]] = Field(default_factory=list)
    descendants: list[dict[str, Any]] = Field(default_factory=list)
    nodes: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Request for querying metric records by facet."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    workflow: str = ""
    run_id: str = ""
    tool: str = ""
    stage: str = ""
    dataset_version: str = ""
    model_version: str = ""
    metric_name: str = ""
    time_start: str = ""
    time_end: str = ""
    threshold_metric: str = ""
    threshold_op: Literal["", "gt", "ge", "lt", "le", "eq"] = ""
    threshold_value: float | None = None
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
    facets: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    """Request for cross-run / cross-stack metric comparison."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    base_run: str = Field(..., min_length=1)
    candidate_run: str = Field(..., min_length=1)
    metric_names: list[str] = Field(default_factory=list)
    lower_is_better: list[str] = Field(default_factory=list)

    @field_validator("input_uri", "base_run", "candidate_run")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class MetricDelta(BaseModel):
    """Per-metric delta emitted in a comparison."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    base_value: float
    candidate_value: float
    delta: float
    pct_change: float | None = None
    lower_is_better: bool = False
    status: Literal["improved", "regressed", "unchanged"] = "unchanged"


class CompareResponse(BaseModel):
    """Response returned by the compare endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    comparison_schema: str = COMPARISON_SCHEMA
    base_run: str
    candidate_run: str
    metrics: list[MetricDelta] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    report_uri: str = ""


class DashboardRequest(BaseModel):
    """Request for a dashboard rollup + optional static HTML report."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_path: str = ""
    workflow: str = ""
    group_by: Literal["metric_name", "tool", "stage", "workflow"] = "metric_name"
    latest_run: str = ""

    @field_validator("input_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class DashboardGroup(BaseModel):
    """A grouped metric rollup shown on the dashboard."""

    model_config = ConfigDict(extra="forbid")

    key: str
    count: int
    latest_value: float
    min: float
    max: float
    mean: float


class DashboardResponse(BaseModel):
    """Response returned by the dashboard endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    dashboard_schema: str = DASHBOARD_SCHEMA
    store_uri: str
    generated_at: str
    total_records: int
    runs: list[str] = Field(default_factory=list)
    latest_run: str = ""
    group_by: str = "metric_name"
    groups: list[DashboardGroup] = Field(default_factory=list)
    latest_rollup: dict[str, float] = Field(default_factory=dict)
    html_uri: str = ""


class InsightsListResponse(BaseModel):
    """List response for service-tracked stores/runs."""

    stores: list[dict[str, Any]] = Field(default_factory=list)
