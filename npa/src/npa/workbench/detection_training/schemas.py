"""Schemas for the detection-training workbench service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_LANCE_URI = "s3://YOUR_S3_BUCKET/lancedb/bdd100k/"
DEFAULT_NUM_CLASSES = 10
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_PORT = 8790
DEFAULT_TOKEN_ENV = "DETECTION_TRAINING_TOKEN"
RunStatus = Literal["queued", "running", "completed", "failed"]


class TrainRequest(BaseModel):
    """Request body for starting a detector fine-tuning run."""

    model_config = ConfigDict(extra="forbid")

    view: str = Field(..., min_length=1)
    lance_uri: str = DEFAULT_LANCE_URI
    output_uri: str = Field(..., min_length=1)
    num_classes: int = Field(DEFAULT_NUM_CLASSES, ge=2)
    epochs: int = Field(DEFAULT_EPOCHS, ge=1)
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, gt=0.0)
    validation_filter_sql: str | None = None

    @field_validator("view", "lance_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved

    @field_validator("validation_filter_sql")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        resolved = value.strip()
        return resolved or None


class TrainResponse(BaseModel):
    """Response returned by the train endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    checkpoint_uri_pattern: str
    metrics_uri: str
    total_epochs: int
    manifest_sha256: str


class EvalRequest(BaseModel):
    """Request body for evaluating a detector checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_uri: str = Field(..., min_length=1)
    eval_view: str = Field(..., min_length=1)
    lance_uri: str = DEFAULT_LANCE_URI
    output_uri: str = Field(..., min_length=1)

    @field_validator("checkpoint_uri", "eval_view", "lance_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class EvalResponse(BaseModel):
    """Response returned by the eval endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    mAP: float
    mAP_50: float
    mAP_75: float
    per_category_AP: dict[str, float] = Field(default_factory=dict)
    eval_run_id: str
    manifest_sha256: str


class StatusResponse(BaseModel):
    """Status snapshot for a training run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    epochs_completed: int = 0
    total_epochs: int = 0
    checkpoint_uri_pattern: str = ""
    metrics_uri: str = ""
    manifest_sha256: str = ""
    last_metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RunListResponse(BaseModel):
    """List response for service-managed run records."""

    runs: list[StatusResponse] = Field(default_factory=list)

