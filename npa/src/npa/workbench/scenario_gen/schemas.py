"""Schemas for the adversarial scenario-generation workbench service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ADVERSARIAL_SET_SCHEMA = "npa.scenario_gen.adversarial_set.v1"
RANKED_SET_SCHEMA = "npa.scenario_gen.ranked_set.v1"

DEFAULT_PORT = 8791
DEFAULT_TOKEN_ENV = "SCENARIO_GEN_TOKEN"
DEFAULT_NUM_SCENARIOS = 8
DEFAULT_ADVERSARY_STEPS = 200_000
DEFAULT_TASK = "Isaac-Cartpole-v0"
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEED = 0
DEFAULT_TOP_K = 4
DEFAULT_SEVERITY_WEIGHT = 0.7
DEFAULT_DIVERSITY_WEIGHT = 0.3

RunStatus = Literal["queued", "running", "completed", "failed"]


class Lineage(BaseModel):
    """Provenance threaded through every scenario-generation manifest."""

    model_config = ConfigDict(extra="forbid")

    workflow_run: str = ""
    input_uris: list[str] = Field(default_factory=list)
    policy_uri: str = ""
    base_config_uri: str = ""
    task_name: str = ""
    produced_by: str = "workbench.scenario_gen"


class ScenarioRecord(BaseModel):
    """A single generated adversarial scenario and its predicted severity."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    seed: int = 0
    perturbation: dict[str, float] = Field(default_factory=dict)
    failure_score: float = 0.0
    severity: float = 0.0
    diversity: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    config_uri: str = ""


class GenerateRequest(BaseModel):
    """Request body for training an adversary and mining hard scenarios."""

    model_config = ConfigDict(extra="forbid")

    policy_uri: str = Field(..., min_length=1)
    base_config_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    task_name: str = DEFAULT_TASK
    num_scenarios: int = Field(DEFAULT_NUM_SCENARIOS, ge=1, le=1024)
    adversary_steps: int = Field(DEFAULT_ADVERSARY_STEPS, ge=1)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, gt=0.0)
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=1)
    seed: int = Field(DEFAULT_SEED, ge=0)
    workflow_run: str = ""
    visualize: bool = True

    @field_validator("policy_uri", "base_config_uri", "output_uri", "task_name")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class GenerateResponse(BaseModel):
    """Response returned by the generate endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    manifest_uri: str
    scenario_count: int
    adversarial_set_schema: str = ADVERSARIAL_SET_SCHEMA
    top_severity: float = 0.0
    manifest_sha256: str = ""
    viz_uri: str = ""
    lineage: Lineage = Field(default_factory=Lineage)


class RankRequest(BaseModel):
    """Request body for ranking a generated adversarial scenario set."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1)
    severity_weight: float = Field(DEFAULT_SEVERITY_WEIGHT, ge=0.0)
    diversity_weight: float = Field(DEFAULT_DIVERSITY_WEIGHT, ge=0.0)
    workflow_run: str = ""

    @field_validator("input_uri", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved


class RankResponse(BaseModel):
    """Response returned by the rank endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    ranked_manifest_uri: str
    ranked_count: int
    ranked_set_schema: str = RANKED_SET_SCHEMA
    top_scenarios: list[dict[str, Any]] = Field(default_factory=list)
    manifest_sha256: str = ""


class StatusResponse(BaseModel):
    """Status snapshot for a scenario-generation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    kind: str = "generate"
    scenario_count: int = 0
    manifest_uri: str = ""
    top_severity: float = 0.0
    manifest_sha256: str = ""
    error: str | None = None


class RunListResponse(BaseModel):
    """List response for service-managed run records."""

    runs: list[StatusResponse] = Field(default_factory=list)
