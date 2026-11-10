"""Declarative, no-code input contract for ordinary LeRobot v2 datasets.

The GR00T workflow has a separate explicit trusted-code path for operators who
need arbitrary Python transforms.  This module covers the common case where a
LeRobot v2 tree already has a camera, vector state/action columns, task indices,
and an observed-outcome feedback sidecar.  A single JSON document binds the
source tree, its labels, field mapping, action semantics, and native model
identity.  It never imports code named by a dataset or configuration file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


STANDARD_PROFILE = "lerobot-v2-single-wrist-v1"
SCHEMA_VERSION = "npa.groot-standard-lerobot/v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DatasetSource(_Strict):
    uri: str = Field(min_length=1)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: Literal["private-object-store", "archive"]
    source_identifier: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    privacy_classification: Literal["public", "private", "restricted"]
    redistributable_media: bool
    label_provenance: str = Field(min_length=1)


class FeedbackSource(_Strict):
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["outcome", "progress", "hybrid"]
    label_protocol: str = Field(min_length=1)
    episode_index_field: str = Field(min_length=1)
    split_field: str = Field(min_length=1)
    group_field: str = Field(min_length=1)
    outcome_field: str = Field(min_length=1)
    collection_kind_field: str = Field(min_length=1)
    tags_field: str | None = None
    progress_field: str | None = None
    progress_source: Literal["measured", "annotation_derived"] | None = None
    progress_contract: dict[str, str] | None = None


class FieldMapping(_Strict):
    camera_key: str = Field(min_length=1)
    state_key: str = Field(min_length=1)
    action_key: str = Field(min_length=1)
    instruction_key: str = Field(min_length=1)
    state_dimension: int = Field(ge=1, le=64)
    action_dimension: int = Field(ge=1, le=32)


class ActionSemantics(_Strict):
    kind: Literal["joint_position", "cartesian_relative"]
    frame: str = Field(min_length=1)
    translation_unit: str = Field(min_length=1)
    rotation_representation: str = Field(min_length=1)
    rotation_unit: str = Field(min_length=1)
    gripper_fields: list[str]
    transform_identity: str = Field(min_length=1)


class Split(_Strict):
    train_episode_count: int = Field(ge=1)
    eval_episode_count: int = Field(ge=1)
    grouping: Literal["episode", "collection"] = "collection"
    strategy: Literal["adapter"] = "adapter"


class Evidence(_Strict):
    episode_index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    frame_count: int = Field(ge=1)
    fps: int = Field(ge=1, le=120)


class ModelIdentity(_Strict):
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    groot_repository: str = Field(min_length=1)
    groot_revision: str = Field(min_length=1)


class StandardLeRobotConfig(_Strict):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    profile: Literal[STANDARD_PROFILE] = STANDARD_PROFILE
    dataset: DatasetSource
    feedback: FeedbackSource
    mapping: FieldMapping
    action: ActionSemantics
    split: Split
    evidence: Evidence
    model: ModelIdentity

    @model_validator(mode="after")
    def _contract_is_consistent(self) -> "StandardLeRobotConfig":
        if self.feedback.mode in {"progress", "hybrid"}:
            if not self.feedback.progress_field or not self.feedback.progress_source:
                raise ValueError("progress feedback requires field and source")
            contract = self.feedback.progress_contract or {}
            if not {"units", "direction", "alignment"}.issubset(contract):
                raise ValueError("progress feedback requires units, direction, and alignment")
        return self


def load_standard_config(path: Path) -> StandardLeRobotConfig:
    """Parse one exact declarative config without accepting executable metadata."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("standard_input_config_uri must contain one JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError("standard_input_config_uri must contain one JSON object")
    return StandardLeRobotConfig.model_validate(payload)


class StandardSingleWristConfig:
    """Shipped GR00T profile for the fixed logical names emitted by the adapter."""

    video_keys = ["video.wrist"]
    state_keys = ["state.robot"]
    action_keys = ["action.robot"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
        from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
        from gr00t.model.transforms import GR00TTransform

        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(apply_to=self.state_keys, normalization_modes={key: "min_max" for key in self.state_keys}),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(apply_to=self.action_keys, normalization_modes={key: "min_max" for key in self.action_keys}),
                ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys, action_concat_order=self.action_keys),
                GR00TTransform(state_horizon=len(self.observation_indices), action_horizon=len(self.action_indices), max_state_dim=64, max_action_dim=32),
            ]
        )
