"""HTTP schemas for Alpamayo 2 Super."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InferenceBody(BaseModel):
    """Inference controls; deployment paths and model code are operator-owned."""

    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(
        default="run", max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    sample_index: int = Field(default=0, ge=0)
    diffusion_steps: int = Field(default=10, ge=1)
    seed: int = 42
    figure_style: Literal["blog", "compact"] = "blog"
    require_camera_projection: bool = True
    run_id: str = ""
    dry_run: bool = False


__all__ = ["InferenceBody"]
