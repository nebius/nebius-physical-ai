"""Typed public and durable schemas for the Antioch Workbench integration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROJECT_SCHEMA = "npa.antioch.project.v1"
STATE_SCHEMA = "npa.antioch.operation.v1"
ARTIFACT_MANIFEST_SCHEMA = "npa.antioch.artifacts.v1"
EPISODE_SCHEMA = "npa.antioch.episode.v1"
COMPLETION_SCHEMA = "npa.antioch.completion.v1"
DATASET_PROVENANCE_SCHEMA = "npa.antioch.lerobot_provenance.v1"
DEFAULT_PORT = 8789
DEFAULT_TOKEN_ENV = "ANTIOCH_WORKBENCH_TOKEN"

OperationStatus = Literal[
    "claimed",
    "submitted",
    "queued",
    "running",
    "collecting",
    "completed",
    "failed",
    "cancelled",
]
RemoteKind = Literal["scenario", "suite"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SubmitRequest(BaseModel):
    """Idempotently submit an immutable Antioch project to the remote queue."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(
        ..., description="S3 prefix containing project-manifest.json and its archive"
    )
    output_path: str = Field(..., description="Run-scoped S3 output prefix")
    workflow_run: str = Field(..., min_length=1, max_length=128)
    state_id: str = Field(..., min_length=1, max_length=128)
    robot_type: str = Field(..., min_length=1, max_length=128)
    task: str = Field(..., min_length=1, max_length=512)
    suite: str = ""
    scenario: str = ""
    scenario_case: str = ""
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    expected_cli_version: str = "0.3.63"

    @field_validator("input_path", "output_path")
    @classmethod
    def _s3_path(cls, value: str) -> str:
        resolved = value.strip().rstrip("/")
        if not resolved.startswith("s3://") or resolved.count("/") < 3:
            raise ValueError("must be a non-empty s3:// URI")
        return resolved

    @field_validator(
        "workflow_run",
        "state_id",
        "robot_type",
        "task",
        "suite",
        "scenario",
        "scenario_case",
    )
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _one_selection(self) -> "SubmitRequest":
        if bool(self.suite) == bool(self.scenario):
            raise ValueError("exactly one of suite or scenario is required")
        if self.scenario_case and not self.scenario:
            raise ValueError("scenario_case requires scenario")
        if self.parameters and not self.scenario:
            raise ValueError("parameters are supported only for a single scenario")
        return self


class ResumeRequest(BaseModel):
    """Reconnect to, or rerun, a previously claimed Antioch operation."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    workflow_run: str
    state_id: str
    rerun_terminal: bool = False

    @field_validator("output_path")
    @classmethod
    def _output_s3(cls, value: str) -> str:
        resolved = value.strip().rstrip("/")
        if not resolved.startswith("s3://"):
            raise ValueError("output_path must be an s3:// URI")
        return resolved


class CollectRequest(ResumeRequest):
    """Collect and normalize the completed remote operation."""

    require_policy_dataset: bool = True


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uri: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str = "application/octet-stream"
    scenario_run_id: str = ""


class OperationRecord(BaseModel):
    """Secret-free durable record stored outside the adapter container."""

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal[STATE_SCHEMA] = STATE_SCHEMA
    idempotency_key: str
    request_sha256: str
    workflow_run: str
    state_id: str
    robot_type: str = ""
    task: str = ""
    input_path: str
    output_path: str
    input_sha256: str = ""
    derived_project_id: str
    remote_kind: RemoteKind
    selection: str
    remote_id: str = ""
    invocation_id: str = ""
    submission_owner: str = ""
    submission_lease_expires_at: str = ""
    collection_owner: str = ""
    collection_lease_expires_at: str = ""
    collection_phase: str = ""
    remote_phase: str = ""
    remote_outcome: str = ""
    status: OperationStatus = "claimed"
    retryable: bool = False
    error_type: str = ""
    error_message: str = ""
    artifact_manifest_uri: str = ""
    dataset_uri: str = ""
    completion_uri: str = ""
    terms_name: str = ""
    terms_url: str = ""
    terms_version: str = ""
    terms_scope: str = ""
    terms_accepted: bool = False
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    revision: int = Field(default=1, ge=1)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    cli_installed: bool
    authenticated: bool
    cli_version: str = ""
    environment: str = ""
    detail: str = ""


class SystemInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    python_version: str
    platform: str
    cpu_only: bool = True
    cli_version: str = ""
    runtime_cache: str = ""
    proprietary_payload_baked: bool = False


class OperationListResponse(BaseModel):
    operations: list[OperationRecord] = Field(default_factory=list)


class EpisodeProvenance(BaseModel):
    """Per-episode facts required before Antioch data can train a policy."""

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal[EPISODE_SCHEMA] = EPISODE_SCHEMA
    scenario: str
    case: str = ""
    seed: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    engine_version: str
    sdk_version: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assets_sha256: dict[str, str] = Field(default_factory=dict)
    observation_schema: list[str] = Field(min_length=1)
    action_schema: list[str] = Field(min_length=1)
    fps: int = Field(gt=0)


class ProjectArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "project.tar.gz"
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("name")
    @classmethod
    def _archive_name(cls, value: str) -> str:
        name = value.strip()
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("archive name must be one safe basename")
        return name


class ProjectManifest(BaseModel):
    """Immutable project package staged at the standard S3 input path."""

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal[PROJECT_SCHEMA] = PROJECT_SCHEMA
    archive: ProjectArchive
    source_name: str
    source_revision: str
    source_license: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_hashes: dict[str, str] = Field(default_factory=dict)

    @field_validator("asset_hashes")
    @classmethod
    def _asset_digests(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not name.strip() or not re_full_sha256(digest):
                raise ValueError(
                    "asset_hashes must map non-empty names to lowercase SHA-256 digests"
                )
        return value


def re_full_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)
