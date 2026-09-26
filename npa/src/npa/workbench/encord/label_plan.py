"""Validate portable, content-bound video bounding-box prelabels."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BoxAnnotation(_StrictModel):
    """One zero-based video frame and normalized bounding box.

    Args:
        frame: Zero-based frame number; coordinates are fractions of the image.
    Returns:
        A validated annotation.
    Raises:
        ValueError: The box extends outside the image.
    """

    frame: int = Field(ge=0, strict=True)
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside_image(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Bounding box extends outside the image")
        return self


class BoxTrack(_StrictModel):
    """One persistent object with explicit per-frame prelabels.

    Args:
        track_id: Unique within the video; class_name selects an ontology object.
        boxes: Explicit annotations without implicit interpolation.
    Returns:
        A validated track.
    Raises:
        ValueError: A frame is repeated.
    """

    track_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    boxes: list[BoxAnnotation] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_frames(self):
        if len({box.frame for box in self.boxes}) != len(self.boxes):
            raise ValueError("A track cannot repeat a frame")
        return self


class VideoLabels(_StrictModel):
    """Bind tracks to exact video bytes and decoded dimensions.

    Args:
        source_uri: Exact S3 object URI from the push receipt.
        source_sha256: SHA-256 of the annotated video bytes.
        width, height, frame_count: Decoded video geometry.
        tracks: Nonempty object tracks.
    Returns:
        Validated video annotations.
    Raises:
        ValueError: Track identities repeat or frames exceed the video length.
    """

    source_uri: str = Field(pattern=r"^s3://[^/]+/.+")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    frame_count: int = Field(gt=0, strict=True)
    tracks: list[BoxTrack] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_tracks(self):
        if len({track.track_id for track in self.tracks}) != len(self.tracks):
            raise ValueError("Track identities must be unique within a video")
        if any(box.frame >= self.frame_count for t in self.tracks for box in t.boxes):
            raise ValueError("Annotation frame exceeds the video length")
        return self


class LabelPlan(_StrictModel):
    """Portable prelabels whose provenance remains visible to reviewers.

    Args:
        schema_version: The supported plan schema.
        provenance: How the prelabels were produced; no review is implied.
        videos: Exact video identities and their annotations.
    Returns:
        A validated plan.
    Raises:
        ValueError: Source identities repeat.
    """

    schema_version: Literal["npa.encord.label_plan.v1"]
    provenance: str = Field(min_length=1)
    videos: list[VideoLabels] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_sources(self):
        if len({video.source_uri for video in self.videos}) != len(self.videos):
            raise ValueError("Video source identities must be unique")
        return self
