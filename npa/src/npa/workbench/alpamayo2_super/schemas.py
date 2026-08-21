"""HTTP schemas for Alpamayo 2 Super."""

from pydantic import BaseModel, ConfigDict

from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)


class InferenceBody(BaseModel):
    """Validated HTTP inference body."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    dataset_revision: str = DEFAULT_DATASET_REVISION
    manifest: str = DEFAULT_MANIFEST
    sample_index: int = 0
    diffusion_steps: int = 10
    seed: int = 42
    figure_style: str = "blog"
    require_camera_projection: bool = True
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


__all__ = ["InferenceBody"]
