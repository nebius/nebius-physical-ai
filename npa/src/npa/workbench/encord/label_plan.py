"""Validate content-bound video tracks and temporal scene prelabels."""

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


class Point(_StrictModel):
    """A normalized polygon vertex.

    Args:
        x, y: Fractions of image dimensions, including the image boundary.
    Returns:
        A finite vertex inside the image.
    Raises:
        ValueError: Coordinates fall outside [0, 1].
    """

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class PolygonAnnotation(_StrictModel):
    """One simple polygon on one video frame, without holes.

    Args:
        frame: Zero-based frame number.
        points: Ordered vertices; the closing edge is implicit.
    Returns:
        A validated polygon annotation.
    Raises:
        ValueError: Vertices repeat, edges intersect, or area is zero.
    """

    frame: int = Field(ge=0, strict=True)
    points: list[Point] = Field(min_length=3)

    @model_validator(mode="after")
    def _simple_polygon(self):
        points = [(point.x, point.y) for point in self.points]
        if len(set(points)) != len(points):
            raise ValueError("Polygon vertices must be distinct")
        edges = list(zip(points, points[1:] + points[:1]))
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)
        if abs(area) < 1e-12:
            raise ValueError("Polygon area must be nonzero")
        for i, edge in enumerate(edges):
            for j in range(i + 2, len(edges)):
                if i == 0 and j == len(edges) - 1:
                    continue
                if _intersect(*edge, *edges[j]):
                    raise ValueError("Polygon edges must not intersect")
        return self


def _intersect(a, b, c, d):
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    if any(
        max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k])
        for k in (0, 1)
    ):
        return False
    return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0


class _Track(_StrictModel):
    track_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_frames(self):
        if len({annotation.frame for annotation in self.annotations}) != len(
            self.annotations
        ):
            raise ValueError("A track cannot repeat a frame")
        return self


class BoxTrack(_Track):
    """One persistent bounding-box object with explicit frame annotations.

    Args:
        track_id, class_name: Video-local identity and ontology object name.
        boxes: Annotations without implicit interpolation.
    Returns:
        A validated box track; omitted shape defaults to bounding_box.
    Raises:
        ValueError: A frame is repeated.
    """

    shape: Literal["bounding_box"] = "bounding_box"
    boxes: list[BoxAnnotation] = Field(min_length=1)

    @property
    def annotations(self):
        """Expose explicit per-frame boxes.

        Args:
            None.
        Returns:
            Validated bounding-box annotations.
        Raises:
            None.
        """
        return self.boxes


class PolygonTrack(_Track):
    """One persistent polygon object with explicit frame annotations.

    Args:
        track_id, class_name: Video-local identity and ontology object name.
        shape: Must be polygon; polygons contains explicit frame annotations.
    Returns:
        A validated polygon track.
    Raises:
        ValueError: A frame is repeated or polygon geometry is invalid.
    """

    shape: Literal["polygon"]
    polygons: list[PolygonAnnotation] = Field(min_length=1)

    @property
    def annotations(self):
        """Expose explicit per-frame polygons.

        Args:
            None.
        Returns:
            Validated polygon annotations.
        Raises:
            None.
        """
        return self.polygons


class SceneSegment(_StrictModel):
    """A scene classification value over an inclusive frame interval.

    Args:
        start_frame, end_frame: Zero-based inclusive bounds.
        value: Ontology option name.
    Returns:
        A validated temporal classification segment.
    Raises:
        ValueError: The interval is reversed.
    """

    start_frame: int = Field(ge=0, strict=True)
    end_frame: int = Field(ge=0, strict=True)
    value: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_frame < self.start_frame:
            raise ValueError("Classification interval is reversed")
        return self


class SceneClassification(_StrictModel):
    """One radio classification with nonoverlapping temporal segments.

    Args:
        name: Ontology classification name.
        segments: Explicit intervals; gaps remain unclassified.
    Returns:
        A validated classification.
    Raises:
        ValueError: Segments overlap.
    """

    name: str = Field(min_length=1)
    segments: list[SceneSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def _nonoverlapping(self):
        ordered = sorted(self.segments, key=lambda s: s.start_frame)
        if any(a.end_frame >= b.start_frame for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Classification segments must not overlap")
        return self


class VideoLabels(_StrictModel):
    """Bind tracks and scene classifications to exact video bytes.

    Args:
        source_uri, source_sha256: Exact S3 identity and SHA-256 from push.
        width, height, frame_count: Decoded video geometry.
        tracks: Object tracks; classifications defaults to an empty list.
    Returns:
        Validated video annotations.
    Raises:
        ValueError: Identities repeat or annotations exceed the video length.
    """

    source_uri: str = Field(pattern=r"^s3://[^/]+/.+")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    frame_count: int = Field(gt=0, strict=True)
    tracks: list[BoxTrack | PolygonTrack] = Field(min_length=1)
    classifications: list[SceneClassification] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_tracks(self):
        if len({track.track_id for track in self.tracks}) != len(self.tracks):
            raise ValueError("Track identities must be unique within a video")
        if any(
            annotation.frame >= self.frame_count
            for track in self.tracks
            for annotation in track.annotations
        ):
            raise ValueError("Annotation frame exceeds the video length")
        if len({classification.name for classification in self.classifications}) != len(
            self.classifications
        ):
            raise ValueError("Classification names must be unique within a video")
        if any(
            segment.end_frame >= self.frame_count
            for classification in self.classifications
            for segment in classification.segments
        ):
            raise ValueError("Classification frame exceeds the video length")
        return self


class LabelPlan(_StrictModel):
    """Portable prelabels whose provenance remains visible to reviewers.

    Args:
        schema_version: v1 boxes or v2 mixed tracks and classifications.
        provenance: How the prelabels were produced; no review is implied.
        videos: Exact video identities and their annotations.
    Returns:
        A validated plan.
    Raises:
        ValueError: Identities, ontology shapes, or schema features conflict.
    """

    schema_version: Literal["npa.encord.label_plan.v1", "npa.encord.label_plan.v2"]
    provenance: str = Field(min_length=1)
    videos: list[VideoLabels] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_ontology(self):
        if len({video.source_uri for video in self.videos}) != len(self.videos):
            raise ValueError("Video source identities must be unique")
        shapes = {}
        for video in self.videos:
            for track in video.tracks:
                if shapes.setdefault(track.class_name, track.shape) != track.shape:
                    raise ValueError(
                        "An ontology class must use one shape across videos"
                    )
            if self.schema_version.endswith(".v1") and (
                video.classifications
                or any(track.shape != "bounding_box" for track in video.tracks)
            ):
                raise ValueError("Polygons and classifications require label_plan.v2")
        if set(shapes) & {
            classification.name
            for video in self.videos
            for classification in video.classifications
        }:
            raise ValueError("Object and classification names must be distinct")
        return self
