"""Validate portable MJLab requests without importing the simulator."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from npa.workbench.storage_scope import authorize_uri

MJLAB_VERSION = "1.6.0"
DEFAULT_TASK = "Mjlab-Velocity-Flat-Unitree-G1"


class MjlabError(RuntimeError):
    """Signal a failed MJLab operation without a synthetic fallback.

    Args:
        message: Human-readable failure reason.
    Returns:
        An exception instance.
    Raises:
        None.
    """


class TaskRequest(BaseModel):
    """Common task configuration.

    Args:
        task: Registered upstream task ID, never a Python import path.
        input_path: Optional S3 motion NPZ for tracking tasks.
        output_path: S3 prefix for artifacts.
        num_envs: Environment count; None preserves upstream training defaults.
        seed: Reproducible environment and policy seed.
    Returns:
        A validated request.
    Raises:
        ValueError: Invalid fields, motion format, or handoff URI.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task: str = Field(default=DEFAULT_TASK, pattern=r"^Mjlab-[A-Za-z0-9-]+$")
    input_path: str | None = None
    output_path: str
    num_envs: int | None = Field(default=None, gt=0)
    seed: int = Field(default=42, ge=0)

    @field_validator("input_path", "output_path")
    @classmethod
    def _s3_path(cls, value):
        if value is not None:
            target = authorize_uri(value, operation="MJLab artifact")
            if target.kind != "s3" or not target.key:
                raise ValueError("MJLab handoffs require s3://bucket/key paths")
        return value

    @model_validator(mode="after")
    def _motion_input(self):
        tracking = self.task.startswith("Mjlab-Tracking-")
        if tracking and not self.input_path:
            raise ValueError(
                "tracking tasks require --input-path to a motion.npz object"
            )
        if self.input_path and (not tracking or not self.input_path.endswith(".npz")):
            raise ValueError(
                "--input-path is an MJLab motion NPZ for tracking tasks only"
            )
        return self


class TrainRequest(TaskRequest):
    """Train or resume an upstream RSL-RL policy.

    Args:
        iterations: Learning iterations; None keeps the upstream task default.
        checkpoint: Optional native RSL-RL checkpoint S3 object to resume.
        gpu_count: GPUs within the current node, handled by upstream torchrunx.
        learning_rate: Optional PPO learning-rate override.
    Returns:
        Validated training settings.
    Raises:
        ValueError: Invalid settings or checkpoint path.
    """

    iterations: int | None = Field(default=None, gt=0)
    checkpoint: str | None = None
    gpu_count: int = Field(default=1, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint_path(cls, value):
        return cls._s3_path(value)


class ExportRequest(TaskRequest):
    """Export a native MJLab checkpoint to ONNX.

    Args:
        checkpoint: Native RSL-RL checkpoint S3 object matching the task.
        device: Execution device; CPU is useful for local inspection.
    Returns:
        Validated export settings.
    Raises:
        ValueError: Invalid checkpoint URI or device.
    """

    checkpoint: str
    device: Literal["cuda:0", "cpu"] = "cuda:0"

    @field_validator("checkpoint")
    @classmethod
    def _checkpoint_path(cls, value):
        return cls._s3_path(value)


class EvalRequest(ExportRequest):
    """Measure complete episodes without a supplied or synthetic score.

    Args:
        episodes: Number of completed episodes to measure.
        success_threshold: Required survival fraction, not task success.
        video: Record the first environment's first episode as MP4.
    Returns:
        Validated evaluation settings.
    Raises:
        ValueError: Invalid count or threshold.
    """

    num_envs: int = Field(default=1, gt=0)
    episodes: int = Field(default=8, gt=0)
    success_threshold: float = Field(default=0.75, ge=0, le=1)
    video: bool = False
