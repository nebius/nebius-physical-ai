"""Authenticated HTTP request schemas for flex-pi."""

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class InferenceBody(BaseModel):
    """Operator-bounded inference controls exposed over HTTP."""

    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(
        default="run", max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    num_inference_steps: int = Field(default=4, ge=1)
    seed: int = 42
    torch_compile: bool = False
    expected_gpu: str = ""
    run_id: str = ""
    dry_run: bool = False


class TrainingBody(BaseModel):
    """Bounded execution controls for the fixed public training contract."""

    model_config = ConfigDict(extra="forbid")
    output_path: str = Field(
        default="train", max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    mode: Literal["train", "profile", "profile-resume"] = "train"
    normalization_path: str = ""
    normalization_sha256: str = ""
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    optimizer: Literal["default", "foreach", "fused"] = "default"
    memory_fill: Literal["on", "off"] = "on"
    activation_checkpointing: Literal["on", "off"] = "on"
    microbatch_per_rank: Literal[1, 3] = 1
    dry_run: bool = False


__all__ = ["InferenceBody", "TrainingBody"]
