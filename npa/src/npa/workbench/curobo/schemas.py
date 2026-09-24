"""Strict, portable cuRobo V2 input contracts; no executable configuration."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SOURCE_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"
DATASET_REVISION = "81e3d1d605de84100d8ab880b43096aba221a48b"
Mode = Literal["kinematic", "dynamics"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Pose(StrictModel):
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @field_validator("quaternion_wxyz")
    @classmethod
    def unit_quaternion(cls, value):
        if not math.isclose(sum(x * x for x in value), 1.0, abs_tol=1e-4):
            raise ValueError("quaternion must have unit norm in wxyz order")
        return value


class Cuboid(StrictModel):
    dims: tuple[float, float, float]
    pose: tuple[float, float, float, float, float, float, float]

    @model_validator(mode="after")
    def geometry(self):
        if any(x <= 0 for x in self.dims):
            raise ValueError("cuboid dimensions must be positive")
        Pose(position_xyz=self.pose[:3], quaternion_wxyz=self.pose[3:])
        return self


class PlanningProblem(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    start: tuple[float, float, float, float, float, float, float]
    goal_pose: Pose
    cuboids: dict[str, Cuboid] = Field(default_factory=dict)

    @field_validator("cuboids")
    @classmethod
    def names(cls, value):
        import re

        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in value):
            raise ValueError("obstacle names must be simple identifiers")
        return value


class PlanManifest(StrictModel):
    schema_version: Literal["npa.curobo.plan.v1"] = "npa.curobo.plan.v1"
    robot: Literal["franka.yml"] = "franka.yml"
    problems: list[PlanningProblem] = Field(min_length=1)

    @field_validator("problems")
    @classmethod
    def unique_ids(cls, value):
        if len({p.id for p in value}) != len(value):
            raise ValueError("problem ids must be unique")
        return value


class BenchmarkManifest(StrictModel):
    schema_version: Literal["npa.curobo.benchmark.v1"] = "npa.curobo.benchmark.v1"
    modes: list[Mode] = Field(
        default_factory=lambda: ["kinematic", "dynamics"], min_length=1
    )
    dataset_revision: Literal[DATASET_REVISION] = DATASET_REVISION

    @field_validator("modes")
    @classmethod
    def unique_modes(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("benchmark modes must be unique")
        return value


class RunRequest(StrictModel):
    input_path: str
    output_path: str
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PrepareRequest(StrictModel):
    output_path: str
    mode: Literal["both", "kinematic", "dynamics"] = "both"
