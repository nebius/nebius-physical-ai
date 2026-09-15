"""Schemas shared by the OpenArm CLI, SDK, and service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from npa.cli.path_contract import validate_read_path, validate_write_path

DEFAULT_PORT = 8792
DEFAULT_TOKEN_ENV = "OPENARM_TOKEN"
DEFAULT_ISAAC_TASK = "Isaac-Reach-OpenArm-v0"
DEFAULT_STEPS = 500

Simulator = Literal["mujoco", "isaac-lab"]
IsaacMode = Literal["rollout", "train"]
RunStatus = Literal["queued", "running", "completed", "failed"]


class OpenArmRunRequest(BaseModel):
    """One real OpenArm simulator workload."""

    model_config = ConfigDict(extra="forbid")

    simulator: Simulator
    output_uri: str = Field(..., min_length=1)
    steps: int = Field(DEFAULT_STEPS, ge=1)
    seed: int = Field(17, ge=0)
    render: bool = False
    task: str = Field(DEFAULT_ISAAC_TASK, min_length=1)
    num_envs: int = Field(64, ge=1)
    isaac_mode: IsaacMode = "rollout"
    max_iterations: int = Field(1, ge=1)

    @field_validator("output_uri")
    @classmethod
    def _output_is_s3(cls, value: str) -> str:
        return validate_write_path(
            value, tool="OpenArm run", option="--output-path", required=True
        )

    @field_validator("task")
    @classmethod
    def _supported_task(cls, value: str) -> str:
        task = value.strip()
        if not task.startswith("Isaac-") or "OpenArm" not in task:
            raise ValueError("task must be an upstream Isaac-*OpenArm* environment id")
        return task


class OpenArmRunResponse(BaseModel):
    """Accepted run response."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    simulator: Simulator
    output_uri: str
    manifest_sha256: str


class OpenArmStatusResponse(BaseModel):
    """Current or terminal run state."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    simulator: Simulator
    output_uri: str
    error: str | None = None
    result: dict[str, Any] | None = None


class OpenArmRunListResponse(BaseModel):
    """Collection of service run states."""

    model_config = ConfigDict(extra="forbid")

    runs: list[OpenArmStatusResponse]


class OpenArmSystemInfo(BaseModel):
    """Runtime identity without initializing either simulator."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    python: str
    platform: str
    mujoco_version: str = ""
    openarm_mujoco_version: str = ""
    openarm_mujoco_commit: str = ""
    openarm_isaac_commit: str = ""
    isaac_sim_version: str = ""
    isaac_lab_version: str = ""
    isaac_provisioning: str = "runtime-fetch"


class OpenArmQualificationRequest(BaseModel):
    """Artifact roots consumed and produced by the qualification gate."""

    model_config = ConfigDict(extra="forbid")

    input_uri: str = Field(..., min_length=1)
    output_uri: str = Field(..., min_length=1)

    @field_validator("input_uri")
    @classmethod
    def _input_is_s3(cls, value: str) -> str:
        return validate_read_path(
            value,
            tool="OpenArm qualification",
            option="--input-path",
            allow_hf=False,
            required=True,
        )

    @field_validator("output_uri")
    @classmethod
    def _qualification_output_is_s3(cls, value: str) -> str:
        return validate_write_path(
            value,
            tool="OpenArm qualification",
            option="--output-path",
            required=True,
        )


class OpenArmQualificationResponse(BaseModel):
    """Validated evidence summary for the complete dual-simulator workflow."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(alias="schema")
    status: Literal["completed"]
    input_uri: str
    output_uri: str
    artifacts: list[dict[str, Any]]
