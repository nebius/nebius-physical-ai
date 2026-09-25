"""Define strict versioned contracts for navigation failure and evaluation evidence."""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
_Seed = Annotated[int, Field(ge=0)]


def _object_uri(value: str) -> str:
    if any(ord(character) < 33 for character in value):
        raise ValueError("S3 URI contains whitespace or control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", parsed.netloc)
    ):
        raise ValueError("expected an S3 object URI without credentials or query")
    parts = parsed.path.removeprefix("/").split("/")
    if any(not p or p in {".", ".."} for p in parts):
        raise ValueError("S3 object key must not contain empty or traversal segments")
    if not re.fullmatch(r"[A-Za-z0-9_./=-]+", parsed.path):
        raise ValueError("S3 object key contains unsafe characters")
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Artifact(_Contract):
    uri: str
    sha256: _Digest

    _uri = field_validator("uri")(_object_uri)


class _Capture(_Contract):
    scenario_id: _Name
    group_id: _Name
    asset: _Artifact


class _HeldOut(_Capture):
    seeds: Annotated[list[_Seed], Field(min_length=1)]

    @field_validator("seeds")
    @classmethod
    def _unique_seeds(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("duplicate held-out seeds")
        return value


class _Policy(_Contract):
    policy_id: _Name
    checkpoint: _Artifact


class _Adapter(_Contract):
    entrypoint: Annotated[
        str, Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    ]
    source_sha256: _Digest
    runtime_image: Annotated[
        str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:\-]*@sha256:[0-9a-f]{64}$")
    ]


class _Metric(_Contract):
    name: _Name
    direction: Literal["higher", "lower"]
    minimum_improvement: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    maximum_regression: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    minimum: Annotated[float, Field(allow_inf_nan=False)]
    maximum: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def _bounds(self):
        if self.minimum >= self.maximum:
            raise ValueError("metric minimum must be less than maximum")
        return self


class _Bundle(_Contract):
    schema_version: Literal["npa.field-failure.bundle.v1"]
    task: Literal["navigation"]
    baseline: _Policy
    baseline_training_groups: list[_Name]
    captures: Annotated[list[_Capture], Field(min_length=1)]
    held_out: Annotated[list[_HeldOut], Field(min_length=1)]
    protocol: _Artifact
    adapters: dict[Literal["reconstruct", "train", "evaluate"], _Adapter]
    metrics: Annotated[list[_Metric], Field(min_length=1)]
    primary_metric: _Name

    @model_validator(mode="after")
    def _separation(self):
        for field in ("scenario_id", "group_id"):
            training = {getattr(item, field) for item in self.captures}
            held = {getattr(item, field) for item in self.held_out}
            if training & held:
                raise ValueError(f"training/held-out overlap: {field}")
        if set(self.baseline_training_groups) & {s.group_id for s in self.held_out}:
            raise ValueError("baseline training overlaps held-out groups")
        for field in ("uri", "sha256"):
            if {getattr(s.asset, field) for s in self.captures} & {
                getattr(s.asset, field) for s in self.held_out
            }:
                raise ValueError("training/held-out asset overlap")
        _unique(self.captures, "scenario_id")
        _unique(self.held_out, "scenario_id")
        _unique(self.metrics, "name")
        if self.primary_metric not in {m.name for m in self.metrics}:
            raise ValueError("primary metric is not defined")
        if set(self.adapters) != {"reconstruct", "train", "evaluate"}:
            raise ValueError("all three sealed adapter identities are required")
        return self


class _Scene(_Contract):
    scenario_id: _Name
    group_id: _Name
    capture_sha256: _Digest
    asset: _Artifact


class _StageRecord(_Contract):
    bundle_sha256: _Digest
    adapter: _Adapter
    attempt_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    inputs_sha256: dict[str, _Digest]


class _Reconstruction(_StageRecord):
    schema_version: Literal["npa.field-failure.reconstruction.v1"]
    scenes: Annotated[list[_Scene], Field(min_length=1)]
    evidence: _Artifact


class _Training(_StageRecord):
    schema_version: Literal["npa.field-failure.training.v1"]
    reconstruction_sha256: _Digest
    initial_checkpoint_sha256: _Digest
    candidate: _Policy
    training_scenario_ids: list[_Name]
    training_group_ids: list[_Name]
    training_scene_sha256: list[_Digest]
    evidence: _Artifact


class _Measurements(_Contract):
    scenario_id: _Name
    scene_sha256: _Digest
    seed: _Seed
    status: Literal["completed"]
    termination: Literal["success", "failure", "timeout"]
    steps: Annotated[int, Field(gt=0)]
    metrics: dict[str, float]

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, value):
        if not value or any(not math.isfinite(v) for v in value.values()):
            raise ValueError("episode metrics must be nonempty and finite")
        return value


class _Episode(_Measurements):
    evidence: _Artifact


class _EpisodeEvidence(_Measurements):
    schema_version: Literal["npa.field-failure.episode.v1"]
    bundle_sha256: _Digest
    checkpoint_sha256: _Digest
    protocol_sha256: _Digest
    trajectory: _Artifact


class _Evaluation(_StageRecord):
    schema_version: Literal["npa.field-failure.evaluation.v1"]
    protocol_sha256: _Digest
    policy: _Policy
    episodes: Annotated[list[_Episode], Field(min_length=1)]


def _unique(items, field):
    values = [getattr(item, field) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field}")
