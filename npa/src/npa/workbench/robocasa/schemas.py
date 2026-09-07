"""Schemas for the RoboCasa workbench service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from npa.cli.path_contract import validate_write_path

DEFAULT_PORT = 8791
DEFAULT_TOKEN_ENV = "ROBOCASA_TOKEN"
DEFAULT_ENV_ID = "robocasa/PickPlaceCounterToCabinet"
DEFAULT_ITERATIONS = 1
DEFAULT_NUM_ENVS = 1
DEFAULT_TIMEOUT_SECONDS = 3600

RunStatus = Literal["queued", "running", "completed", "failed"]


class RoboCasaRunRequest(BaseModel):
    """Request body for a RoboCasa capability run."""

    model_config = ConfigDict(extra="forbid")

    env_id: str = Field(DEFAULT_ENV_ID, min_length=1)
    capability: str = Field("kitchen_random_rollout", min_length=1)
    output_uri: str = Field(..., min_length=1)
    iterations: int = Field(DEFAULT_ITERATIONS, ge=1)
    num_envs: int = Field(DEFAULT_NUM_ENVS, ge=1)
    timeout_seconds: int = Field(DEFAULT_TIMEOUT_SECONDS, ge=1)
    download_assets: bool = True
    seed: int | None = Field(None, ge=0)
    checkpoint_uri: str = ""
    train_env_ids: str = ""
    heldout_env_ids: str = ""

    @field_validator("env_id", "capability", "output_uri")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        resolved = value.strip()
        if not resolved:
            raise ValueError("value must not be empty")
        return resolved

    @field_validator("capability")
    @classmethod
    def _validate_capability(cls, value: str) -> str:
        supported = {
            "kitchen_task_registration",
            "kitchen_asset_availability",
            "kitchen_egl_env_reset",
            "kitchen_random_rollout",
            "kitchen_trajectory_export",
            "kitchen_policy_eval",
        }
        if value not in supported:
            raise ValueError(f"unsupported robocasa capability: {value}")
        return value

    @field_validator("output_uri")
    @classmethod
    def _validate_output_path(cls, value: str) -> str:
        return validate_write_path(
            value,
            tool="RoboCasa run",
            option="--output-path",
            required=True,
        )


class RoboCasaRunResponse(BaseModel):
    """Response returned by the run endpoint and SDK."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    env_id: str
    capability: str
    output_uri: str
    manifest_sha256: str


class RoboCasaStatusResponse(BaseModel):
    """Status of a RoboCasa run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    capability: str
    env_id: str
    output_uri: str
    error: str | None = None
    result: dict[str, Any] | None = None


class RoboCasaRunListResponse(BaseModel):
    """List of RoboCasa runs."""

    model_config = ConfigDict(extra="forbid")

    runs: list[RoboCasaStatusResponse]


class RoboCasaSystemInfo(BaseModel):
    """System information for the RoboCasa service."""

    model_config = ConfigDict(extra="forbid")

    status: str
    python: str
    platform: str
    robocasa_version: str = ""
    robosuite_version: str = ""
    mujoco_version: str = ""
    gymnasium_version: str = ""
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str = ""
    registered_env_count: int = 0
    assets_root_exists: bool = False


__all__ = [
    "DEFAULT_ENV_ID",
    "DEFAULT_ITERATIONS",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_PORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TOKEN_ENV",
    "RoboCasaRunListResponse",
    "RoboCasaRunRequest",
    "RoboCasaRunResponse",
    "RoboCasaStatusResponse",
    "RoboCasaSystemInfo",
]
