"""Strict, portable Open3D input contracts; no executable configuration.

Every parameter here maps onto a documented Open3D argument, and the names are
Open3D's own (`voxel_down_sample`, `compute_fpfh_feature`,
`registration_ransac_based_on_feature_matching`, `registration_icp`,
`global_optimization`). The radius/distance *factors* are the multiples of
`voxel_size` the upstream Global-registration and Multiway-registration
tutorials use, kept as factors so one `voxel_size` stays the single scale knob.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The pinned `open3d` wheel this contract was reviewed against.
SOURCE_VERSION = "0.20.0"
#: Upstream `isl-org/open3d_downloads` release backing `open3d.data` fragments.
DEMO_DATA_RELEASE = "20220301-data"
#: `open3d.data` class whose fragments the staged demo path reads.
DEMO_DATA_CLASS = "DemoICPPointClouds"

IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
SHA256 = r"^[0-9a-f]{64}$"
#: Point-cloud containers Open3D's `read_point_cloud` reads without a guess.
POINT_CLOUD_SUFFIXES = (".pcd", ".ply")

IcpEstimation = Literal["point_to_plane", "point_to_point"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Fragment(StrictModel):
    """One input scan, identified by content rather than by position."""

    id: str = Field(pattern=IDENTIFIER)
    uri: str
    sha256: str = Field(pattern=SHA256)
    bytes: int = Field(gt=0)

    @field_validator("uri")
    @classmethod
    def readable_suffix(cls, value: str) -> str:
        if not value.lower().endswith(POINT_CLOUD_SUFFIXES):
            raise ValueError(
                "fragment uri must end in " + " or ".join(POINT_CLOUD_SUFFIXES)
            )
        return value


class RegistrationManifest(StrictModel):
    """The recipe `register` and `multiway` consume, and `prepare` writes."""

    schema_version: Literal["npa.open3d.registration.v1"] = "npa.open3d.registration.v1"
    fragments: list[Fragment] = Field(min_length=2)
    voxel_size: float = Field(gt=0.0, le=10.0)
    icp_estimation: IcpEstimation = "point_to_plane"
    # Upstream tutorial factors. `distance` also bounds RANSAC correspondences.
    normal_radius_factor: float = Field(default=2.0, gt=0.0, le=100.0)
    feature_radius_factor: float = Field(default=5.0, gt=0.0, le=100.0)
    # Upstream keeps two correspondence thresholds: a loose one for RANSAC on the
    # downsampled clouds, and a tight one for ICP on the full-resolution clouds.
    distance_factor: float = Field(default=1.5, gt=0.0, le=100.0)
    icp_distance_factor: float = Field(default=0.4, gt=0.0, le=100.0)
    #: `open3d.utility.random.seed` value, so a rerun repeats the same RANSAC.
    random_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    ransac_max_iteration: int = Field(default=100_000, ge=1_000, le=10_000_000)
    ransac_confidence: float = Field(default=0.999, gt=0.0, lt=1.0)
    #: Provenance only when `prepare --stage-demo-fragments` sourced the scans.
    demo_data_release: str | None = None

    @field_validator("fragments")
    @classmethod
    def unique_ids(cls, value: list[Fragment]) -> list[Fragment]:
        if len({fragment.id for fragment in value}) != len(value):
            raise ValueError("fragment ids must be unique")
        if len({fragment.uri for fragment in value}) != len(value):
            raise ValueError("fragment uris must be unique")
        return value

    @field_validator("demo_data_release")
    @classmethod
    def known_release(cls, value: str | None) -> str | None:
        if value is not None and value != DEMO_DATA_RELEASE:
            raise ValueError("demo data release does not match the reviewed contract")
        return value


class StageDemoRequest(StrictModel):
    """Publish the upstream `open3d.data` scans as this run's input fragments."""

    output_path: str
    voxel_size: float = Field(default=0.05, gt=0.0, le=10.0)
    icp_estimation: IcpEstimation = "point_to_plane"


class PrepareRequest(StageDemoRequest):
    """Index operator scans that already live under an S3 prefix."""

    input_path: str


class RunRequest(StrictModel):
    input_path: str
    output_path: str
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ReconstructRequest(RunRequest):
    #: Poisson octree depth. Upstream's surface-reconstruction tutorial uses 9;
    #: the ceiling keeps one stage from asking for a 2**14 lattice.
    poisson_depth: int = Field(default=9, ge=4, le=12)
    #: Drop the lowest-density vertices Poisson extrapolates past the samples,
    #: as the upstream tutorial does with its density quantile crop.
    density_quantile: float = Field(default=0.02, ge=0.0, lt=0.5)
    #: Discard surface farther than this many `voxel_size` units from any observed
    #: sample. The default of 1.0 is the sampling geometry, not a tuned number:
    #: the cloud is voxel-downsampled at `voxel_size`, so a surface point that
    #: interpolates between neighbouring samples sits at most about half a voxel
    #: diagonal from one; past a full voxel there is nothing left to interpolate
    #: from and Poisson is extrapolating. Pass 0.0 to publish the closed surface
    #: unchanged, which is the right choice when the scan is already complete.
    support_distance_factor: float = Field(default=1.0, ge=0.0, le=100.0)


def fragment_id_for(uri: str) -> str:
    """Derive a stable, contract-legal fragment id from a URI basename."""

    stem = uri.rstrip("/").rsplit("/", 1)[-1]
    for suffix in POINT_CLOUD_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", stem).lstrip("_-")
    if not cleaned:
        raise ValueError(f"cannot derive a fragment id from {uri!r}")
    return cleaned
