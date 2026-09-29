"""Compare persisted Encord tracks and temporal classifications with their import."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from npa.workbench.encord.label_plan import BoxAnnotation, PolygonAnnotation
from npa.workbench.encord.schemas import EncordToolError


@dataclass
class VerifiedLabels:
    """Exported annotations that passed exact identity and geometry checks.

    Args:
        objects: Frame-indexed (class name, annotation, track ID) tuples.
        scenes: Frame-indexed scene classification strings.
        classification_segments: Number of verified temporal segments.
    Returns:
        Labels ready for rendering from the export.
    Raises:
        None.
    """

    objects: dict = field(default_factory=dict)
    scenes: dict = field(default_factory=dict)
    classification_segments: int = 0


def _verify_export(video, identity, exported):
    if (
        exported.get("data_hash") != identity["data_hash"]
        or exported.get("label_hash") != identity["label_hash"]
    ):
        raise EncordToolError("Exported labels belong to a different data row")
    units = list(exported.get("data_units", {}).values())
    if len(units) != 1 or exported.get("data_type") != "video":
        raise EncordToolError("Expected one video data unit in the Encord export")
    expected = _expected_objects(video, identity)
    actual = _exported_objects(units[0])
    if set(actual) != set(expected):
        raise EncordToolError(
            "Exported label frames or object identities differ from import"
        )
    result = VerifiedLabels()
    for key, (name, annotation, track_id) in expected.items():
        actual_name, actual_annotation = actual[key]
        if actual_name != name or not _same_geometry(annotation, actual_annotation):
            raise EncordToolError(
                "Exported label classes or coordinates differ from import"
            )
        result.objects.setdefault(key[0], []).append(
            (actual_name, actual_annotation, track_id)
        )
    result.scenes, result.classification_segments = _classification_frames(
        video, identity, exported
    )
    return result


def _expected_objects(video, identity):
    objects = {track["track_id"]: track["object_hash"] for track in identity["tracks"]}
    if (
        len(objects) != len(identity["tracks"])
        or set(objects) != {track.track_id for track in video.tracks}
        or len(set(objects.values())) != len(objects)
    ):
        raise EncordToolError(
            "Label receipt has missing or duplicate object identities"
        )
    return {
        (annotation.frame, objects[track.track_id]): (
            track.class_name,
            annotation,
            track.track_id,
        )
        for track in video.tracks
        for annotation in track.annotations
    }


def _exported_objects(unit):
    actual = {}
    for frame_text, labels in unit.get("labels", {}).items():
        frame = int(frame_text)
        if labels.get("classifications"):
            raise EncordToolError(
                "Unexpected frame-local classifications in label export"
            )
        for obj in labels.get("objects", []):
            if obj.get("isDeleted") or obj.get("manualAnnotation") is True:
                raise EncordToolError(
                    "Deleted or manually changed object in label export"
                )
            key = (frame, obj["objectHash"])
            if key in actual:
                raise EncordToolError("Duplicate exported object at the same frame")
            actual[key] = (obj["name"], _exported_geometry(frame, obj))
    return actual


def _exported_geometry(frame, obj):
    if obj.get("shape") == "bounding_box":
        coords = obj["boundingBox"]
        return BoxAnnotation(
            frame=frame,
            x=coords["x"],
            y=coords["y"],
            width=coords["w"],
            height=coords["h"],
        )
    if obj.get("shape") != "polygon":
        raise EncordToolError("Unsupported object shape in label export")
    vertices = obj["polygon"]
    if set(vertices) != {str(i) for i in range(len(vertices))}:
        raise EncordToolError("Polygon vertex indices are not consecutive")
    polygon = PolygonAnnotation(
        frame=frame, points=[vertices[str(i)] for i in range(len(vertices))]
    )
    if "polygons" in obj:
        rings = obj["polygons"]
        flat = [v for p in polygon.points for v in (p.x, p.y)]
        if (
            len(rings) != 1
            or len(rings[0]) != 1
            or not _same_numbers(flat, rings[0][0])
        ):
            raise EncordToolError("Polygon export contains changed rings or holes")
    return polygon


def _same_numbers(expected, actual):
    return len(expected) == len(actual) and all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(expected, actual)
    )


def _same_geometry(expected, actual):
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, BoxAnnotation):
        return _same_numbers(
            [getattr(expected, k) for k in ("x", "y", "width", "height")],
            [getattr(actual, k) for k in ("x", "y", "width", "height")],
        )
    return _same_numbers(
        [v for p in expected.points for v in (p.x, p.y)],
        [v for p in actual.points for v in (p.x, p.y)],
    )


def _classification_frames(video, identity, exported):
    expected = {
        (c.name, s.start_frame, s.end_frame): s.value
        for c in video.classifications
        for s in c.segments
    }
    saved = identity.get("classifications", [])
    saved_keys = {(s["name"], s["start_frame"], s["end_frame"]) for s in saved}
    actual = exported.get("classification_answers", {})
    if (
        saved_keys != set(expected)
        or len(saved_keys) != len(saved)
        or len(actual) != len(saved)
    ):
        raise EncordToolError("Classification segments differ from import")
    if {s["classification_hash"] for s in saved} != set(actual):
        raise EncordToolError("Exported classification identities differ from import")
    frames = {}
    for segment in saved:
        key = (segment["name"], segment["start_frame"], segment["end_frame"])
        answer = actual[segment["classification_hash"]]
        text = _verified_scene(segment, answer, expected[key])
        for frame in range(segment["start_frame"], segment["end_frame"] + 1):
            frames.setdefault(frame, []).append(text)
    return frames, len(saved)


def _verified_scene(segment, answer, value):
    if (
        answer.get("classificationHash") != segment["classification_hash"]
        or answer.get("featureHash") != segment["feature_hash"]
        or answer.get("range") != [[segment["start_frame"], segment["end_frame"]]]
        or answer.get("spaces")
        or answer.get("manualAnnotation") is True
    ):
        raise EncordToolError("Exported classification identity or frame range changed")
    attributes = answer.get("classifications", [])
    if (
        _answer_values(attributes) != _answer_values(segment["answers"])
        or len(attributes) != 1
    ):
        raise EncordToolError("Exported classification answers differ from import")
    attribute = attributes[0]
    options = attribute.get("answers", [])
    if (
        attribute.get("name") != segment["name"]
        or len(options) != 1
        or options[0].get("name") != value
    ):
        raise EncordToolError(
            "Exported classification value differs from the label plan"
        )
    return f"{attribute['name']}: {options[0]['name']}"


def _answer_values(attributes):
    # Encord's SDK export can reset this nested metadata flag on reload.
    # Identity, ontology feature hashes, names, and option values must still match.
    return [
        {key: value for key, value in attribute.items() if key != "manualAnnotation"}
        for attribute in attributes
    ]
