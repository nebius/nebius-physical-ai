"""Authenticated HTTP request schemas for flex-pi."""

from pydantic import BaseModel, ConfigDict, Field


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


__all__ = ["InferenceBody"]
