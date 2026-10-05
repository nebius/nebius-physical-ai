"""Validate embodiment and training settings for standalone FLUX Action fine-tuning."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SOURCE_REVISION = "e2dd1d8dbc5977b54315d61f7548c63c043d6d4f"
BASE_REPOSITORY = "black-forest-labs/flux-3-action-base"
BASE_REVISION = "eb267865d35e49e4066bde4936237f8d9f15a68c"


class Channel(BaseModel):
    """Name and physical unit of one ordered control or measured-state channel.

    Args: name and unit identify the recorded quantity.
    Returns: A validated channel.
    Raises: ValueError for empty descriptions.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)


class RobotContract(BaseModel):
    """Declare the embodiment without inheriting another robot's control semantics.

    Args: Ordered channels, camera mapping, rate, and representation settings.
    Returns: A validated embodiment contract.
    Raises: ValueError for incompatible widths, cameras, or dimensions.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    embodiment: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    actions: list[Channel] = Field(min_length=1)
    states: list[Channel] = Field(min_length=1)
    cameras: dict[str, str] = Field(min_length=1)
    fps: int = Field(gt=0, strict=True)
    camera_layout: Literal["single", "side_by_side", "grid"]
    canvas_hw: tuple[int, int] = (512, 512)
    state_key: str = "observation.state"
    action_key: str = "action"
    action_parameterization: Literal["absolute", "joint_delta"] = "absolute"
    absolute_action_dims: list[int] = Field(default_factory=list)
    gripper_flip_dims: list[int] = Field(default_factory=list)
    n_action_steps: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def check_contract(self) -> RobotContract:
        """Check channel and camera consistency.

        Args: None.
        Returns: This contract.
        Raises: ValueError for incompatible settings.
        """
        width = len(self.actions)
        if len(self.states) != width:
            raise ValueError(
                "FLUX Action currently requires equal state and action widths"
            )
        for channels in (self.actions, self.states):
            if len({channel.name for channel in channels}) != width:
                raise ValueError("channel names must be unique within each vector")
        _check_cameras(self)
        for dims in (self.absolute_action_dims, self.gripper_flip_dims):
            if len(set(dims)) != len(dims) or any(d < 0 or d >= width for d in dims):
                raise ValueError(
                    "channel indices must be unique and within vector width"
                )
        if self.action_parameterization == "absolute" and self.absolute_action_dims:
            raise ValueError("absolute_action_dims applies only to joint_delta")
        if self.action_parameterization == "joint_delta" and not set(
            self.gripper_flip_dims
        ).issubset(self.absolute_action_dims):
            raise ValueError("flipped grippers must stay absolute under joint_delta")
        return self


def _check_cameras(robot: RobotContract) -> None:
    import re

    expected = {"single": 1, "side_by_side": 2}.get(robot.camera_layout)
    if expected and len(robot.cameras) != expected:
        raise ValueError("camera count does not match camera_layout")
    if len(set(robot.cameras.values())) != len(robot.cameras):
        raise ValueError("each camera must map to a distinct video feature")
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in robot.cameras):
        raise ValueError(
            "camera stream names must be identifiers without images. prefix"
        )
    if any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", value) or value == "gray"
        for value in robot.cameras.values()
    ):
        raise ValueError("camera mappings must reference real video features")
    if any(value <= 0 or value % 32 for value in robot.canvas_hw):
        raise ValueError("canvas dimensions must be positive multiples of 32")


class TrainingSettings(BaseModel):
    """Expose bounded configuration fields without arbitrary Python dataset plugins.

    Args: Optimizer schedule, batches, and checkpoint settings.
    Returns: Validated training settings.
    Raises: ValueError for inconsistent schedules.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    steps: int = Field(gt=0)
    checkpoint_every: int = Field(gt=0)
    frozen_steps: int = Field(ge=0)
    trunk_warmup_steps: int = Field(gt=0)
    heads_warmup_steps: int = Field(gt=0)
    windows_per_rank: int = Field(default=1, gt=0)
    grad_accumulation: int = Field(default=1, gt=0)
    num_workers: int = Field(default=1, gt=0)
    visits_per_epoch: int = Field(default=1, gt=0)
    seed: int = 42
    optimizer_lr: float = Field(default=5e-5, gt=0)
    optimizer_lr_heads_multiplier: float = Field(default=5.0, gt=0)
    caption_dropout: float = Field(default=0.0, ge=0, le=1)
    param_dtype: Literal["float32", "bfloat16"] = "float32"
    ema_sigma_rels: tuple[Literal[0.1, 0.05], ...] = (0.1, 0.05)

    @model_validator(mode="after")
    def check_schedule(self) -> TrainingSettings:
        """Require the requested run to reach trunk training.

        Args: None.
        Returns: This schedule.
        Raises: ValueError if the trunk remains frozen throughout.
        """
        if len(set(self.ema_sigma_rels)) != len(self.ema_sigma_rels):
            raise ValueError("ema_sigma_rels must not contain duplicates")
        if self.steps <= self.frozen_steps + 1:
            raise ValueError("steps must exceed frozen_steps + 1 to train the trunk")
        return self


class InferenceSettings(BaseModel):
    """Explicit sampling settings carried by the exported policy.

    These diagnostic defaults do not claim a robot-specific quality benchmark.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sampler: Literal["euler", "cosmos_unipc"] = "euler"
    num_inference_steps: int = Field(default=4, ge=1)
    sampler_shift: float = Field(default=1.0, gt=0)
    guidance_scale: float = 1.0
    guidance_scale_action: float = 1.0
    inference_seed: int = 0


class Recipe(BaseModel):
    """A robot contract and an explicit optimizer schedule.

    Args: robot and training settings; optional export profile.
    Returns: A validated fine-tuning recipe.
    Raises: ValueError for unknown fields or invalid nested settings.
    """

    model_config = ConfigDict(extra="forbid")
    robot: RobotContract
    training: TrainingSettings
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    val_episodes: int = Field(default=1, ge=0)
    export_profile: Literal["model", "ema_0p10", "ema_0p05"] = "ema_0p10"

    @model_validator(mode="after")
    def check_export_profile(self) -> Recipe:
        """Require the requested EMA to be present in the training schedule."""

        sigma = {"ema_0p10": 0.1, "ema_0p05": 0.05}.get(self.export_profile)
        if sigma is not None and sigma not in self.training.ema_sigma_rels:
            raise ValueError("export_profile requires its EMA in ema_sigma_rels")
        return self


class FinetuneRequest(BaseModel):
    """S3 handoffs for one finite, single-node distributed training job.

    Args: Input dataset, recipe object, output prefix, and GPU process count.
    Returns: A validated request.
    Raises: ValueError for non-S3 paths or overlapping inputs and outputs.
    """

    model_config = ConfigDict(extra="forbid")
    input_path: str
    recipe_uri: str
    output_path: str
    processes: int = Field(default=8, gt=0)

    @field_validator("input_path", "recipe_uri", "output_path")
    @classmethod
    def check_uri(cls, value: str) -> str:
        """Validate an S3 handoff.

        Args: value is an object or directory URI.
        Returns: The URI without trailing slashes.
        Raises: ValueError for invalid or noncanonical URIs.
        """
        from npa.cli.path_contract import validate_read_path

        value = validate_read_path(value, tool="flux-action", allow_hf=False)
        parsed = urlparse(value)
        if (
            parsed.query
            or parsed.fragment
            or any(
                part in ("", ".", "..") for part in parsed.path.strip("/").split("/")
            )
        ):
            raise ValueError(
                "S3 paths must be canonical and contain no query or fragment"
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def check_overlap(self) -> FinetuneRequest:
        """Prevent output writes into a source tree.

        Args: None.
        Returns: This request.
        Raises: ValueError for overlapping source and destination prefixes.
        """
        for source in (self.input_path, self.recipe_uri):
            if source == self.output_path or source.startswith(self.output_path + "/"):
                raise ValueError("output_path must not contain an input")
            if self.output_path.startswith(source + "/"):
                raise ValueError("output_path must not be inside an input")
        return self
