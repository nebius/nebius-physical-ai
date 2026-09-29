"""Content identity, remote export equality, and real MP4 label rendering."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from npa.workbench.encord.label_import import (
    _validate_sources,
    _ontology_structure,
    _populate_row,
)
from npa.workbench.encord.label_plan import LabelPlan, PolygonAnnotation
from npa.workbench.encord.label_render import (
    _encode_overlay,
    _render_one,
    _verified_plan,
    _verify_export,
)
from npa.workbench.encord.schemas import EncordToolError
from npa.workbench.encord.storage import json_bytes


def label_payload():
    return {
        "schema_version": "npa.encord.label_plan.v1",
        "provenance": "Synthetic test annotations, not reviewed",
        "videos": [
            {
                "source_uri": "s3://test-bucket/input/clip.mp4",
                "source_sha256": "a" * 64,
                "width": 64,
                "height": 64,
                "frame_count": 2,
                "tracks": [
                    {
                        "track_id": "bottle-1",
                        "class_name": "bottle",
                        "boxes": [
                            {
                                "frame": 0,
                                "x": 0.25,
                                "y": 0.5,
                                "width": 0.25,
                                "height": 0.25,
                            },
                            {
                                "frame": 1,
                                "x": 0.5,
                                "y": 0.5,
                                "width": 0.25,
                                "height": 0.25,
                            },
                        ],
                    }
                ],
            }
        ],
    }


def label_export():
    return {
        "data_hash": "data-1",
        "label_hash": "label-1",
        "data_type": "video",
        "data_units": {
            "data-1": {
                "labels": {
                    str(frame): {
                        "objects": [
                            {
                                "name": "bottle",
                                "shape": "bounding_box",
                                "objectHash": "object-1",
                                "boundingBox": {"x": x, "y": 0.5, "w": 0.25, "h": 0.25},
                            }
                        ]
                    }
                    for frame, x in [(0, 0.25), (1, 0.5)]
                }
            }
        },
    }


def identity():
    return {
        "data_hash": "data-1",
        "label_hash": "label-1",
        "tracks": [{"track_id": "bottle-1", "object_hash": "object-1"}],
    }


@pytest.mark.parametrize(
    "change", ["nan", "outside", "negative", "duplicate", "late", "fractional"]
)
def test_label_plan_rejects_invalid_coordinates_and_frames(change):
    payload = label_payload()
    boxes = payload["videos"][0]["tracks"][0]["boxes"]
    if change == "nan":
        boxes[0]["x"] = float("nan")
    if change == "outside":
        boxes[0]["width"] = 0.9
    if change == "negative":
        boxes[0]["frame"] = -1
    if change == "duplicate":
        boxes.append(copy.deepcopy(boxes[0]))
    if change == "late":
        boxes[0]["frame"] = 2
    if change == "fractional":
        boxes[0]["frame"] = 0.5
    with pytest.raises(ValidationError):
        LabelPlan.model_validate(payload)


@pytest.mark.parametrize("change", ["checksum", "uri", "incomplete", "no_dataset"])
def test_import_rejects_wrong_media_before_remote_mutation(change):
    plan = LabelPlan.model_validate(label_payload())
    item = SimpleNamespace(
        source_uri=plan.videos[0].source_uri,
        outcome="successful",
        source_checksum_kind="sha256",
        source_checksum="a" * 64,
    )
    push = SimpleNamespace(status="completed", dataset_hash="dataset-1", items=[item])
    if change == "checksum":
        item.source_checksum = "b" * 64
    if change == "uri":
        item.source_uri = "s3://test-bucket/other/clip.mp4"
    if change == "incomplete":
        push.status = "failed"
    if change == "no_dataset":
        push.dataset_hash = ""
    with pytest.raises(EncordToolError):
        _validate_sources(plan, push)


def test_failed_ontology_checkpoint_stops_before_project_creation(monkeypatch):
    from npa.workbench.encord import label_import

    monkeypatch.setattr(label_import, "_ontology_structure", lambda _: object())
    client = Mock()
    client.create_ontology.return_value.ontology_hash = "ontology-1"
    checkpoint = Mock(side_effect=OSError("storage unavailable"))
    receipt = {"status": "running"}
    with pytest.raises(OSError, match="storage unavailable"):
        label_import._import_project(
            client,
            SimpleNamespace(provenance="test"),
            None,
            "new-project",
            receipt,
            checkpoint,
        )
    client.create_project.assert_not_called()
    assert receipt["ontology_hash"] == "ontology-1"
    assert receipt["status"] == "failed"


@pytest.mark.parametrize("kind", ["sha256", "s3_checksum_sha256", "etag_opaque"])
def test_import_accepts_full_object_sha256_and_never_opaque_etags(kind):
    plan = LabelPlan.model_validate(label_payload())
    item = SimpleNamespace(
        source_uri=plan.videos[0].source_uri,
        outcome="successful",
        source_checksum_kind=kind,
        source_checksum="a" * 64,
    )
    push = SimpleNamespace(status="completed", dataset_hash="dataset-1", items=[item])
    if kind == "etag_opaque":
        with pytest.raises(EncordToolError, match="SHA-256"):
            _validate_sources(plan, push)
    else:
        _validate_sources(plan, push)


def test_failed_label_save_stops_before_next_video_and_records_intent(monkeypatch):
    from npa.workbench.encord import label_import

    row_one, row_two = Mock(), Mock()
    row_one.backing_item_uuid, row_two.backing_item_uuid = "item-1", "item-2"
    row_one.save.side_effect = RuntimeError("provider unavailable")
    project = Mock(list_label_rows_v2=Mock(return_value=[row_one, row_two]))
    videos = [SimpleNamespace(source_uri=f"s3://test-bucket/{i}.mp4") for i in (1, 2)]
    push = SimpleNamespace(
        items=[
            SimpleNamespace(source_uri=v.source_uri, item_uuid=f"item-{i}")
            for i, v in enumerate(videos, 1)
        ]
    )
    monkeypatch.setattr(label_import, "_validate_geometry", lambda *args: None)
    monkeypatch.setattr(
        label_import, "_populate_row", lambda *args: {"status": "saving"}
    )
    receipt, checkpoint = {"items": []}, Mock()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        label_import._save_project_rows(
            project, SimpleNamespace(videos=videos), push, receipt, checkpoint
        )
    assert receipt["items"] == [{"status": "saving"}]
    checkpoint.assert_called_once()
    row_two.initialise_labels.assert_not_called()


def test_export_is_bound_to_exact_object_track_and_frame_coordinates():
    video = LabelPlan.model_validate(label_payload()).videos[0]
    frames = _verify_export(video, identity(), label_export())
    assert list(frames.objects) == [0, 1]
    assert frames.objects[1][0][1].x == 0.5


@pytest.mark.parametrize(
    "change", ["row", "missing", "extra", "class", "box", "object", "duplicate"]
)
def test_export_mismatch_fails_instead_of_rendering_original_prelabels(change):
    video = LabelPlan.model_validate(label_payload()).videos[0]
    exported = label_export()
    labels = exported["data_units"]["data-1"]["labels"]
    obj = labels["0"]["objects"][0]
    if change == "row":
        exported["data_hash"] = "wrong-data"
    if change == "missing":
        del labels["1"]
    if change == "extra":
        labels["2"] = copy.deepcopy(labels["0"])
    if change == "class":
        obj["name"] = "basket"
    if change == "box":
        obj["boundingBox"]["x"] = 0.1
    if change == "object":
        obj["objectHash"] = "another-object"
    if change == "duplicate":
        labels["0"]["objects"].append(copy.deepcopy(obj))
    with pytest.raises(EncordToolError):
        _verify_export(video, identity(), exported)


@pytest.mark.parametrize("change", ["plan", "project", "verification"])
def test_render_rejects_unrelated_or_modified_evidence(change):
    payload = label_payload()
    store = Mock(read_json=Mock(return_value=payload))
    manifest = SimpleNamespace(
        status="completed",
        source_kind="project",
        source_id="project-1",
        label_export="initialize",
        manifest_uri="s3://test-bucket/pull.json",
    )
    receipt = {
        "schema_version": "npa.encord.label_receipt.v1",
        "status": "completed",
        "project_hash": "project-1",
        "plan_uri": "s3://test-bucket/plan.json",
        "plan_sha256": hashlib.sha256(json_bytes(payload)).hexdigest(),
    }
    report = SimpleNamespace(passed=True, manifest_uri=manifest.manifest_uri)
    if change == "plan":
        payload["provenance"] = "different provenance"
    if change == "project":
        manifest.source_id = "another-project"
    if change == "verification":
        report.manifest_uri = "s3://test-bucket/other-pull.json"
    with pytest.raises(EncordToolError):
        _verified_plan(store, manifest, receipt, report)


def test_renderer_refuses_media_that_changed_after_verification(tmp_path, monkeypatch):
    from npa.workbench.encord import label_render

    gateway = Mock(download_to_file=Mock(return_value=SimpleNamespace(sha256="b" * 64)))
    monkeypatch.setattr(label_render, "S3ObjectStorageGateway", lambda _: gateway)
    video = LabelPlan.model_validate(label_payload()).videos[0]
    with pytest.raises(EncordToolError, match="SHA-256"):
        _render_one(
            None,
            tmp_path,
            video,
            SimpleNamespace(destination_uri="s3://test-bucket/video.mp4"),
            {},
        )
    assert not (tmp_path / "annotated.mp4").exists()


def _write_video(path: Path):
    import av
    from PIL import Image

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for _ in range(2):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), "gray"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_renderer_encodes_moving_exported_boxes_and_decodes_every_frame(tmp_path):
    import av

    source, output = tmp_path / "source.mp4", tmp_path / "overlay.mp4"
    _write_video(source)
    video = LabelPlan.model_validate(label_payload()).videos[0]
    frames = _verify_export(video, identity(), label_export())
    assert _encode_overlay(source, output, video, frames) == 2
    with av.open(str(output)) as container:
        decoded = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    assert len(decoded) == 2
    for frame, left in zip(decoded, (16, 32)):
        red, green, blue = frame[40, left + 1].astype(int)
        assert max(red, green, blue) - min(red, green, blue) > 90
    video.frame_count = 3
    with pytest.raises(EncordToolError, match="frame count"):
        _encode_overlay(source, tmp_path / "bad.mp4", video, frames)


def _plan():
    payload = label_payload()
    payload["schema_version"] = "npa.encord.label_plan.v2"
    video = payload["videos"][0]
    second = copy.deepcopy(video["tracks"][0])
    second["track_id"] = "bottle-2"
    for box in second["boxes"]:
        box["y"] = 0.1
    video["tracks"].extend(
        [
            second,
            {
                "track_id": "basket-1",
                "class_name": "basket",
                "shape": "polygon",
                "polygons": [
                    {
                        "frame": 0,
                        "points": [
                            {"x": 0.05, "y": 0.5},
                            {"x": 0.2, "y": 0.5},
                            {"x": 0.1, "y": 0.9},
                        ],
                    }
                ],
            },
        ]
    )
    video["classifications"] = [
        {
            "name": "phase",
            "segments": [
                {"start_frame": 0, "end_frame": 0, "value": "approach"},
                {"start_frame": 1, "end_frame": 1, "value": "contact"},
            ],
        }
    ]
    return payload


def _sdk_export(payload=None):
    pytest.importorskip("encord")
    from encord.objects.frames import ranges_to_list

    plan = LabelPlan.model_validate(payload or _plan())
    objects, classifications = [], []
    row = SimpleNamespace(
        ontology_structure=_ontology_structure(plan),
        backing_item_uuid="test-item",
        data_hash="test-data",
        label_hash="test-label",
        add_object_instance=objects.append,
        add_classification_instance=classifications.append,
    )
    saved = _populate_row(row, plan.videos[0])
    labels = _sdk_objects(objects)
    answers = {
        c.classification_hash: {
            "classificationHash": c.classification_hash,
            "featureHash": c.feature_hash,
            "range": ranges_to_list(c.range_list),
            "manualAnnotation": c.manual_annotation,
            "classifications": [a.to_encord_dict() for a in c.get_all_static_answers()],
        }
        for c in classifications
    }
    exported = {
        "data_hash": row.data_hash,
        "label_hash": row.label_hash,
        "data_type": "video",
        "data_units": {row.data_hash: {"labels": labels}},
        "classification_answers": answers,
    }
    return plan.videos[0], saved, exported


def _sdk_objects(objects):
    from encord.objects.coordinates import PolygonCoordinates, PolygonCoordsToDict

    labels = {}
    for instance in objects:
        for frame in instance.get_annotation_frames():
            coords = instance.get_annotation(frame).coordinates
            obj = {
                "name": instance.ontology_item.name,
                "objectHash": instance.object_hash,
                "shape": instance.ontology_item.shape.value,
            }
            if isinstance(coords, PolygonCoordinates):
                obj.update(
                    polygon=coords.to_dict(),
                    polygons=coords.to_dict(PolygonCoordsToDict.multiple_polygons),
                )
            else:
                obj["boundingBox"] = coords.to_dict()
            labels.setdefault(str(frame), {"objects": []})["objects"].append(obj)
    return labels


def test_real_sdk_keeps_same_class_instances_polygons_and_temporal_values_distinct(
    tmp_path,
):
    import av

    video, identity, exported = _sdk_export()
    labels = _verify_export(video, identity, exported)
    assert len({t["object_hash"] for t in identity["tracks"]}) == 3
    assert len(labels.objects[0]) == 3
    assert isinstance(labels.objects[0][2][1], PolygonAnnotation)
    assert labels.scenes == {0: ["phase: approach"], 1: ["phase: contact"]}
    assert labels.classification_segments == 2
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    _write_video(source)
    assert _encode_overlay(source, output, video, labels) == 2
    with av.open(str(output)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 2


@pytest.mark.parametrize(
    "change",
    [
        "vertex",
        "extra_ring",
        "classification_value",
        "classification_range",
        "classification_identity",
        "missing_classification",
        "extra_classification",
        "duplicate_track",
    ],
)
def test_changed_exported_mixed_labels_fail_before_rendering(change):
    video, identity, exported = _sdk_export()
    polygon = exported["data_units"]["test-data"]["labels"]["0"]["objects"][2]
    answer = next(iter(exported["classification_answers"].values()))
    if change == "vertex":
        polygon["polygon"]["0"]["x"] += 0.01
    elif change == "extra_ring":
        polygon["polygons"][0].append([0.1, 0.1, 0.2, 0.1, 0.15, 0.2])
    elif change == "classification_value":
        answer["classifications"][0]["answers"][0]["name"] = "wrong"
    elif change == "classification_range":
        answer["range"] = [[0, 1]]
    elif change == "classification_identity":
        answer["featureHash"] = "wrong-feature"
    elif change == "missing_classification":
        exported["classification_answers"].pop(
            next(iter(exported["classification_answers"]))
        )
    elif change == "extra_classification":
        exported["classification_answers"]["extra"] = copy.deepcopy(answer)
    else:
        identity["tracks"][1]["object_hash"] = identity["tracks"][0]["object_hash"]
    with pytest.raises(EncordToolError):
        _verify_export(video, identity, exported)


@pytest.mark.parametrize(
    "change",
    [
        "outside",
        "duplicate_vertex",
        "zero_area",
        "self_intersection",
        "late_polygon",
        "overlap",
        "reversed",
        "late_classification",
        "v1",
        "shape_conflict",
    ],
)
def test_invalid_mixed_plan_is_rejected_before_remote_mutations(change):
    payload = _plan()
    video = payload["videos"][0]
    polygon = video["tracks"][2]["polygons"][0]
    segments = video["classifications"][0]["segments"]
    if change == "outside":
        polygon["points"][0]["x"] = -0.1
    elif change == "duplicate_vertex":
        polygon["points"].append(polygon["points"][0])
    elif change == "zero_area":
        for p in polygon["points"]:
            p["y"] = 0.5
    elif change == "self_intersection":
        polygon["points"] = [
            {"x": x, "y": y} for x, y in [(0, 0), (0.9, 0.9), (0, 0.8), (0.7, 0)]
        ]
    elif change == "late_polygon":
        polygon["frame"] = 2
    elif change == "overlap":
        segments[0]["end_frame"] = 1
    elif change == "reversed":
        segments[1]["end_frame"] = 0
    elif change == "late_classification":
        segments[1]["end_frame"] = 2
    elif change == "v1":
        payload["schema_version"] = "npa.encord.label_plan.v1"
    else:
        video["tracks"][2]["class_name"] = "bottle"
    with pytest.raises(ValidationError):
        LabelPlan.model_validate(payload)


def test_multiple_videos_share_ontology_and_allow_different_scene_options():
    pytest.importorskip("encord")
    payload = _plan()
    second = copy.deepcopy(payload["videos"][0])
    second["source_uri"] = "s3://test-bucket/input/second.mp4"
    second["classifications"][0]["segments"][1]["value"] = "occluded"
    payload["videos"].append(second)
    plan = LabelPlan.model_validate(payload)
    from encord.objects import Classification, Option

    ontology = _ontology_structure(plan)
    phase = ontology.get_child_by_title(title="phase", type_=Classification)
    assert phase.get_child_by_title(title="occluded", type_=Option).label == "occluded"


def test_sdk_export_manual_answer_metadata_does_not_imply_human_review():
    video, identity, exported = _sdk_export()
    for answer in exported["classification_answers"].values():
        answer["classifications"][0]["manualAnnotation"] = True
    labels = _verify_export(video, identity, exported)
    assert labels.classification_segments == 2
    assert labels.scenes[0] == ["phase: approach"]
