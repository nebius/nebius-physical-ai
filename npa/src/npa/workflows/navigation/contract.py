"""Validate sealed Isaac navigation inputs without importing a simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Case(BaseModel):
    """Describe one deterministic world-frame navigation reset.

    Args:
        **data: Identifier, seed, root position, heading and goal in SI units.
    Returns:
        Validated reset case.
    Raises:
        ValueError: Fields are missing, nonfinite or invalid.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]+$")]
    seed: Annotated[int, Field(ge=0)]
    position_m: Annotated[list[Number], Field(min_length=3, max_length=3)]
    heading_rad: Number
    goal_m: Annotated[list[Number], Field(min_length=2, max_length=2)]


class Probe(BaseModel):
    """Describe free-space and obstacle-contact positive controls.

    Args:
        **data: Reset cases, peer parking pose, and native action sequence.
    Returns:
        Validated behavioral probe input.
    Raises:
        ValueError: Required probe data is invalid.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    free: Case
    obstacle: Case
    parked: Case
    actions: Annotated[list[list[Number]], Field(min_length=2)]
    tolerance: Annotated[float, Field(gt=0, le=0.001, allow_inf_nan=False)] = 1e-5


class Camera(BaseModel):
    """Require explicit RGB/depth dimensions and measured camera tolerances.

    Args:
        **data: Resolution, tolerances and positive-control translation.
    Returns:
        Validated all-robot-hidden camera contract.
    Raises:
        ValueError: Parameters are missing or nonfinite.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    camera_visibility: Literal["all_robot_geometry_hidden"]
    height: Annotated[int, Field(gt=1)]
    width: Annotated[int, Field(gt=1)]
    rgb_tolerance: Annotated[float, Field(ge=0, le=0.05, allow_inf_nan=False)]
    depth_tolerance_m: Annotated[float, Field(ge=0, le=0.01, allow_inf_nan=False)]
    minimum_changed_fraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    translation_m: Annotated[list[Number], Field(min_length=3, max_length=3)]
    peer_in_view: Case


class InitialCheckpoint(BaseModel):
    """Bind native training initialization to exact checkpoint bytes.

    Args:
        **data: Bundle-relative checkpoint file and its SHA-256.
    Returns:
        Validated optional starting checkpoint.
    Raises:
        ValueError: The path or digest is invalid.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    file: str
    sha256: Digest

    @model_validator(mode="after")
    def _path(self):
        path = PurePosixPath(self.file)
        if path.is_absolute() or ".." in path.parts or str(path) != self.file:
            raise ValueError("initial checkpoint must use a normalized relative path")
        if path.suffix != ".pt" or path.name == "policy.pt":
            raise ValueError(
                "initial checkpoint must be a .pt file other than policy.pt"
            )
        return self


class Recipe(BaseModel):
    """Bind a BYOF task, scene, training recipe and held-out reset set.

    Args:
        **data: Operator-owned integration and experiment fields.
    Returns:
        Validated recipe with an explicit supported perception mode.
    Raises:
        ValueError: Inputs violate isolation, provenance or split contracts.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["npa.navigation.recipe.v1"]
    task: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]
    adapter_module: Annotated[str, Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")]
    adapter_sha256: Digest
    image: Annotated[str, Field(pattern=r"^[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}$")]
    scene_file: str
    scene_sha256: Digest
    scene_prim: Annotated[str, Field(pattern=r"^/World/[A-Za-z0-9_/]+$")]
    sensor_mode: Literal["state", "static_raycast", "rgbd"]
    camera: Camera | None = None
    num_envs: Annotated[int, Field(ge=2)]
    iterations: Annotated[int, Field(gt=0)]
    episode_steps: Annotated[int, Field(gt=0)]
    goal_tolerance_m: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    minimum_success_rate: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    train_cases: Annotated[list[Case], Field(min_length=2)]
    eval_cases: Annotated[list[Case], Field(min_length=2)]
    probe: Probe
    initial_checkpoint: InitialCheckpoint | None = None
    source_bundle_sha256: Digest | None = None
    reference_controller_sha256: Digest | None = None

    @model_validator(mode="after")
    def _reference_controller(self):
        builtin = self.adapter_module == "npa.workflows.navigation.reference"
        if builtin != (self.reference_controller_sha256 is not None):
            raise ValueError(
                "only the built-in reference requires reference_controller_sha256"
            )
        return self

    @model_validator(mode="after")
    def _check_contract(self):
        if (self.sensor_mode == "rgbd") != (self.camera is not None):
            raise ValueError(
                "rgbd requires the all_robot_geometry_hidden camera contract"
            )
        if self.camera and not any(self.camera.translation_m):
            raise ValueError("camera positive-control translation cannot be zero")
        path = PurePosixPath(self.scene_file)
        if path.is_absolute() or ".." in path.parts or str(path) != self.scene_file:
            raise ValueError("scene_file must be a normalized bundle-relative path")
        if path.suffix.lower() != ".usdz":
            raise ValueError("scene_file must be a self-contained USDZ scene")
        cases = self.train_cases + self.eval_cases
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("train/eval case identifiers must be unique and disjoint")
        if {c.seed for c in self.train_cases} & {c.seed for c in self.eval_cases}:
            raise ValueError("train/eval seeds must be disjoint")
        resets = [
            json.dumps(c.model_dump(exclude={"id", "seed"}), sort_keys=True)
            for c in cases
        ]
        if len(set(resets)) != len(resets):
            raise ValueError("train/eval physical reset inputs must be disjoint")
        if len(self.eval_cases) != self.num_envs:
            raise ValueError("eval_cases must supply exactly num_envs held-out resets")
        widths = {len(action) for action in self.probe.actions}
        if len(widths) != 1 or 0 in widths:
            raise ValueError("probe actions must have one nonempty native action width")
        return self


def read_recipe(root: Path) -> Recipe:
    """Read a strict recipe and verify the scene file within its sealed bundle.

    Args:
        root: Materialized input directory containing recipe.json.
    Returns:
        Recipe with an existing scene of the requested hash.
    Raises:
        ValueError: Recipe, path containment or scene digest is invalid.
        OSError: Required input is missing or unreadable.
    """
    recipe = Recipe.model_validate_json((root / "recipe.json").read_text())
    scene = root / recipe.scene_file
    if scene.is_symlink() or not scene.resolve().is_relative_to(root.resolve()):
        raise ValueError("scene file escapes input bundle")
    if hashlib.sha256(scene.read_bytes()).hexdigest() != recipe.scene_sha256:
        raise ValueError("scene SHA-256 mismatch")
    if recipe.initial_checkpoint:
        checkpoint = root / recipe.initial_checkpoint.file
        if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError("initial checkpoint escapes input bundle")
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != recipe.initial_checkpoint.sha256
        ):
            raise ValueError("initial checkpoint SHA-256 mismatch")
    return recipe


def finite_array(value, shape: tuple[int, ...], name: str):
    """Convert a measurement to a finite NumPy array with an exact shape.

    Args:
        value: Numeric array, including detached native tensor conversions.
        shape: Required dimensions.
        name: Measurement name for errors.
    Returns:
        Finite floating-point array.
    Raises:
        ValueError: Shape or numeric contents are invalid.
    """
    import numpy as np

    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array.copy()
