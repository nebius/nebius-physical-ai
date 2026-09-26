"""Verify Encord-exported object tracks and render them over verified media."""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workbench.encord.label_plan import BoxAnnotation, LabelPlan
from npa.workbench.encord.schemas import EncordToolError, PullManifest, RoundtripReport
from npa.workbench.encord.storage import (
    ConditionalArtifactStore,
    S3ObjectStorageGateway,
    json_bytes,
)


def render_labels(
    *,
    input_path: str,
    label_receipt_uri: str,
    verification_uri: str,
    output_path: str,
    storage_client: Any = None,
) -> dict:
    """Render only persisted Encord labels after exact media verification.

    Args:
        input_path: Completed project pull manifest in S3.
        label_receipt_uri: Import receipt binding the plan and Encord objects.
        verification_uri: Successful media roundtrip report in S3.
        output_path: New S3 prefix for silent annotated MP4s and demo.json.
        storage_client: Optional injected storage client.
    Returns:
        Evidence with frame, track, annotation, and output checksum counts.
    Raises:
        EncordToolError: Exported labels or media differ from the imported plan.
        ArtifactConflict: Output evidence already exists.
    """
    storage = storage_client or StorageClient.from_environment()
    store = ConditionalArtifactStore(storage)
    manifest = PullManifest.model_validate(store.read_json(input_path))
    receipt = dict(store.read_json(label_receipt_uri))
    report = RoundtripReport.model_validate(store.read_json(verification_uri))
    plan = _verified_plan(store, manifest, receipt, report)
    exports = _verified_exports(store, manifest, receipt, plan)
    return _render_outputs(storage, store, plan, exports, output_path)


def _verified_plan(store, manifest, receipt, report):
    if receipt.get("schema_version") != "npa.encord.label_receipt.v1":
        raise EncordToolError("Unsupported label receipt schema")
    if receipt["status"] != "completed" or not report.passed:
        raise EncordToolError(
            "Completed label import and media verification are required"
        )
    if (
        manifest.status != "completed"
        or manifest.source_kind != "project"
        or manifest.source_id != receipt["project_hash"]
        or manifest.label_export != "initialize"
    ):
        raise EncordToolError("Pull must export labels from the imported project")
    if report.manifest_uri != manifest.manifest_uri:
        raise EncordToolError("Verification report belongs to a different pull")
    payload = store.read_json(receipt["plan_uri"])
    if hashlib.sha256(json_bytes(payload)).hexdigest() != receipt["plan_sha256"]:
        raise EncordToolError("Label plan changed after import")
    return LabelPlan.model_validate(payload)


def _verified_exports(store, manifest, receipt, plan):
    media = {item.item_uuid: item for item in manifest.items}
    labels = {item.item_uuid: item for item in manifest.label_artifacts}
    saved = {item["source_uri"]: item for item in receipt["items"]}
    expected = {video.source_uri for video in plan.videos}
    if set(saved) != expected or len(saved) != len(receipt["items"]):
        raise EncordToolError("Label receipt has missing or duplicate sources")
    result = []
    for video in plan.videos:
        identity = saved[video.source_uri]
        item = media.get(identity["item_uuid"])
        artifact = labels.get(identity["item_uuid"])
        if item is None or artifact is None or artifact.outcome != "successful":
            raise EncordToolError("Missing exported media or label artifact")
        exported = store.read_json(artifact.artifact_uri)
        frames = _verify_export(video, identity, exported)
        result.append((item, frames))
    return result


def _verify_export(video, identity, exported):
    if (
        exported.get("data_hash") != identity["data_hash"]
        or exported.get("label_hash") != identity["label_hash"]
    ):
        raise EncordToolError("Exported labels belong to a different data row")
    units = list(exported.get("data_units", {}).values())
    if len(units) != 1 or exported.get("data_type") != "video":
        raise EncordToolError("Expected one video data unit in the Encord export")
    objects = {track["track_id"]: track["object_hash"] for track in identity["tracks"]}
    expected = {}
    for track in video.tracks:
        for box in track.boxes:
            expected[(box.frame, objects[track.track_id])] = (track.class_name, box)
    actual, frames = _exported_boxes(units[0])
    if set(actual) != set(expected):
        raise EncordToolError(
            "Exported label frames or object identities differ from import"
        )
    for key, (name, box) in expected.items():
        actual_name, actual_box = actual[key]
        if actual_name != name or not _same_box(box, actual_box):
            raise EncordToolError(
                "Exported label classes or coordinates differ from import"
            )
    return frames


def _exported_boxes(unit):
    actual, frames = {}, {}
    for frame_text, frame_labels in unit.get("labels", {}).items():
        frame = int(frame_text)
        for obj in frame_labels.get("objects", []):
            if obj.get("shape") != "bounding_box" or obj.get("isDeleted"):
                raise EncordToolError("Unsupported or deleted object in label export")
            coords = obj["boundingBox"]
            box = BoxAnnotation(
                frame=frame,
                x=coords["x"],
                y=coords["y"],
                width=coords["w"],
                height=coords["h"],
            )
            key = (frame, obj["objectHash"])
            if key in actual:
                raise EncordToolError("Duplicate exported object at the same frame")
            actual[key] = (obj["name"], box)
            frames.setdefault(frame, []).append((obj["name"], box))
    return actual, frames


def _same_box(expected, actual):
    return all(
        math.isclose(getattr(expected, name), getattr(actual, name), abs_tol=1e-6)
        for name in ("x", "y", "width", "height")
    )


def _render_outputs(storage, store, plan, exports, prefix):
    output_uri = prefix.rstrip("/") + "/demo.json"
    result = {
        "schema_version": "npa.encord.label_demo.v1",
        "status": "running",
        "label_source": "encord_project_export",
        "review_status": "unreviewed_prelabels",
        "provenance": plan.provenance,
        "videos": [],
    }
    version = store.create_json(output_uri, result)
    try:
        with tempfile.TemporaryDirectory(prefix="npa-encord-render-") as directory:
            for index, (video, (item, frames)) in enumerate(zip(plan.videos, exports)):
                info = _render_one(storage, Path(directory), video, item, frames)
                info["uri"] = prefix.rstrip("/") + f"/annotated-{index + 1}.mp4"
                storage.upload_file(str(Path(directory) / "annotated.mp4"), info["uri"])
                result["videos"].append(info)
                version = store.replace_json(output_uri, result, version)
        result["status"] = "completed"
        store.replace_json(output_uri, result, version)
    except Exception as exc:  # noqa: BLE001 - retain partial artifact evidence
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        store.replace_json(output_uri, result, version)
        raise
    return result


def _render_one(storage, directory, video, item, frames):
    source = directory / "source.mp4"
    digest = S3ObjectStorageGateway(storage).download_to_file(
        item.destination_uri, source
    )
    if digest.sha256 != video.source_sha256:
        raise EncordToolError(
            "Returned video does not match the annotated source SHA-256"
        )
    output = directory / "annotated.mp4"
    count = _encode_overlay(source, output, video, frames)
    return {
        "source_sha256": digest.sha256,
        "frames": count,
        "tracks": len(video.tracks),
        "box_annotations": sum(len(v) for v in frames.values()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }


def _encode_overlay(source, output, video, frames):
    import av

    count = 0
    with av.open(str(source)) as decoded, av.open(str(output), "w") as encoded:
        stream = encoded.add_stream(
            "libx264", rate=decoded.streams.video[0].average_rate
        )
        stream.width, stream.height = (
            video.width + video.width % 2,
            video.height + video.height % 2,
        )
        stream.pix_fmt = "yuv420p"
        for frame in decoded.decode(video=0):
            if (frame.width, frame.height) != (video.width, video.height):
                raise EncordToolError(
                    "Decoded video geometry differs from the label plan"
                )
            image = _draw_frame(frame.to_image(), frames.get(count, []), count)
            painted = av.VideoFrame.from_image(image)
            painted.pts, painted.time_base = frame.pts, frame.time_base
            for packet in stream.encode(painted):
                encoded.mux(packet)
            count += 1
        for packet in stream.encode():
            encoded.mux(packet)
    if count != video.frame_count:
        raise EncordToolError("Decoded video frame count differs from the label plan")
    with av.open(str(output)) as check:
        if sum(1 for _ in check.decode(video=0)) != count:
            raise EncordToolError(
                "Annotated MP4 failed decoded frame-count verification"
            )
    return count


def _draw_frame(image, annotations, frame):
    from PIL import Image, ImageDraw

    width, height = image.size
    canvas = Image.new("RGB", (width + width % 2, height + height % 2))
    canvas.paste(image)
    draw = ImageDraw.Draw(canvas)
    for name, box in annotations:
        left, top = box.x * width, box.y * height
        draw.rectangle(
            (left, top, left + box.width * width, top + box.height * height),
            outline="#00ff80",
            width=3,
        )
        draw.text(
            (left + 3, max(24, top - 15)),
            name,
            fill="#00ff80",
            stroke_width=1,
            stroke_fill="black",
        )
    draw.rectangle((0, 0, width, 21), fill="black")
    draw.text(
        (6, 4),
        f"Encord exported prelabels | frame {frame} | review pending",
        fill="white",
    )
    return canvas
