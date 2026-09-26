"""Content identity, remote export equality, and real MP4 label rendering."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from npa.workbench.encord.label_import import _validate_sources
from npa.workbench.encord.label_plan import LabelPlan
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
    assert list(frames) == [0, 1]
    assert frames[1][0][1].x == 0.5


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
        assert green > red + 50 and green > blue + 30
    video.frame_count = 3
    with pytest.raises(EncordToolError, match="frame count"):
        _encode_overlay(source, tmp_path / "bad.mp4", video, frames)
