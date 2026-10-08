"""Strict requests for Marble world acquisition, GPU processing, and HTML reports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AcquireRequest(BaseModel):
    """Describe one world source and its durable destination.

    Args: Fields describing the source, prompt, and S3 destination.
    Returns: A validated acquisition request.
    Raises: ValueError for unsupported sources or invalid fields.
    """

    model_config = ConfigDict(extra="forbid")
    output_path: str
    source: Literal["generate", "sample-hobbit"] = "generate"
    prompt: str = Field(min_length=1)
    model: str = "marble-1.1"
    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


class RunRequest(BaseModel):
    """Identify a world or result bundle and the next stage's destination.

    Args: S3 paths, run identity, and camera dataset dimensions.
    Returns: A validated stage request.
    Raises: ValueError for invalid dimensions or identity.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_path: str
    output_path: str
    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    frames: int = Field(default=120, ge=1)
    width: int = Field(default=960, ge=32)
    height: int = Field(default=540, ge=32)


class PalletBenchmarkRequest(RunRequest):
    """Configure a matched-training-budget pallet detection experiment.

    Args: Capture/snapshot/output prefixes, run ID, seed, and training settings.
    Returns: A validated benchmark request.
    Raises: ValueError for invalid training parameters.
    """

    dataset_path: str
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=0.005, gt=0)
    seed: int = Field(default=42, ge=0, le=4294967295)
