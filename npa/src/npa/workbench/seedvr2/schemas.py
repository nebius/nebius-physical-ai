"""Define immutable contracts for SeedVR2 video-restoration operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SOURCE_REPOSITORY = "https://github.com/ByteDance-Seed/SeedVR"
SOURCE_REVISION = "e4de8c24441a67e1b7df56abea10645059bb1185"
MODEL_REPOSITORY = "ByteDance-Seed/SeedVR2-3B"
MODEL_REVISION = "37255ff8cccfb01071b87f635a5948ca8d53117c"
RESULT_SCHEMA = "npa.workbench.seedvr2.result.v1"
PROBE_SCHEMA = "npa.workbench.seedvr2.probe.v1"
VERIFICATION_SCHEMA = "npa.workbench.seedvr2.verification.v1"
REVIEW_SCHEMA = "npa.workbench.seedvr2.review.v1"

MODEL_FILES: dict[str, tuple[int, str]] = {
    "ema_vae.pth": (
        1_002_691_902,
        "c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac",
    ),
    "neg_emb.pt": (
        656_540,
        "6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581",
    ),
    "pos_emb.pt": (
        595_100,
        "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
    ),
    "seedvr2_ema_3b.pth": (
        13_566_090_228,
        "6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b",
    ),
}


class VideoArtifactRequest(BaseModel):
    """Describe one request-scoped S3 video operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: str
    output_path: str
    run_id: str = Field(min_length=1)


class RestoreRequest(VideoArtifactRequest):
    """Describe one deterministic official SeedVR2-3B inference."""

    probe_path: str = ""
    output_height: int = Field(default=480, ge=16, le=4096)
    output_width: int = Field(default=640, ge=16, le=4096)
    seed: int = 666
    dry_run: bool = False

    @field_validator("output_height", "output_width")
    @classmethod
    def require_divisible_dimension(cls, value: int) -> int:
        """Require dimensions that upstream will not silently crop."""

        if value % 16:
            raise ValueError("SeedVR2 output dimensions must be divisible by 16")
        return value

    @model_validator(mode="after")
    def require_supported_pixel_budget(self) -> RestoreRequest:
        """Reject requests above the reviewed single-H100 target area."""

        if self.output_height * self.output_width > 1920 * 1080:
            raise ValueError("SeedVR2 output area must not exceed 1920x1080")
        return self


__all__ = [
    "MODEL_FILES",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "PROBE_SCHEMA",
    "RESULT_SCHEMA",
    "REVIEW_SCHEMA",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "VERIFICATION_SCHEMA",
    "RestoreRequest",
    "VideoArtifactRequest",
]
