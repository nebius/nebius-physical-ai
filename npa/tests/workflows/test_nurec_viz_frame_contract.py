"""Preserve source image meaning and frame identities in NuRec review recordings."""

import io
from collections import defaultdict
from pathlib import Path

import pytest
from npa.viz.recordings import load_recording

from npa.workflows import data_factory_viz as viz


def _write_frames(root: Path) -> dict:
    from PIL import Image

    sources = {}
    for modality_index, modality in enumerate(
        ("pred_rgb", "pred_distance", "pred_opacity")
    ):
        for camera_index, frames in enumerate(
            ((3, 9, 18, 29, 43, 60, 79), (107, 113, 127, 149))
        ):
            entity = f"/reconstruction/val/{modality}/sensor_{camera_index}"
            for ordinal, frame in enumerate(frames):
                path = root / entity.lstrip("/") / f"{frame:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                color = (
                    17 + 53 * modality_index,
                    31 + 71 * camera_index,
                    19 + 23 * ordinal,
                )
                Image.new("RGB", (32, 24), color).save(path)
                sources[entity, frame] = path
    for frame in (5, 12, 31, 57, 103):
        path = root / "novel_views/front" / f"{frame:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (frame, 42, 173)).save(path)
        sources["/novel_view/front", frame] = path
    return sources


def _review_pixels(path: Path) -> bytes:
    from PIL import Image

    with Image.open(path) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="JPEG", quality=75)
    with Image.open(io.BytesIO(buffer.getvalue())) as review:
        return review.convert("RGB").tobytes()


def _decoded_images(recording: Path) -> dict:
    from PIL import Image

    observed = {}
    for chunk in load_recording(recording).chunks():
        entity = str(chunk.entity_path)
        if not entity.startswith(("/reconstruction/", "/novel_view/")):
            continue
        batch = chunk.to_record_batch()
        assert {"frame", "EncodedImage:blob"} <= set(batch.schema.names)
        for frame, blobs in zip(
            batch.column("frame").to_pylist(),
            batch.column("EncodedImage:blob").to_pylist(),
            strict=True,
        ):
            key = entity, frame
            assert key not in observed, (
                "A modality/camera/frame must have exactly one image"
            )
            assert len(blobs) == 1
            with Image.open(io.BytesIO(bytes(blobs[0]))) as image:
                assert image.size == (32, 24)
                observed[key] = image.convert("RGB").tobytes()
    return observed


def _assert_selection(sources: dict, observed: dict, cap: int) -> None:
    expected_groups, actual_groups = defaultdict(set), defaultdict(set)
    for entity, frame in sources:
        expected_groups[entity].add(frame)
    for entity, frame in observed:
        actual_groups[entity].add(frame)
    assert set(actual_groups) == set(expected_groups)
    for entity, frames in expected_groups.items():
        selected = actual_groups[entity]
        expected_count = len(frames) if cap <= 0 else min(len(frames), cap)
        assert len(selected) == expected_count
        assert selected <= frames
        assert min(frames) in selected
        if expected_count > 1:
            assert max(frames) in selected
        if cap <= 0 or len(frames) <= cap:
            assert selected == frames


@pytest.mark.parametrize("cap", [24, 3, 2, 1, 0, -1])
def test_nurec_recording_preserves_modality_camera_and_original_frame(
    tmp_path, monkeypatch, cap
):
    pytest.importorskip("rerun")
    pytest.importorskip("PIL")
    monkeypatch.setattr(viz, "RRD_MAX_FRAME_DIM", 0)
    monkeypatch.setattr(viz, "RRD_JPEG_QUALITY", 75)
    monkeypatch.setattr(viz, "RRD_MAX_FRAMES_PER_ENTITY", cap)
    run = tmp_path / "synthetic-review"
    sources = _write_frames(run)
    output = tmp_path / "recording.rrd"

    result = viz.build_run_rrd(str(run), str(output), app_id="neural-reconstruction")

    assert result["status"] == "completed"
    decoded = _decoded_images(output)
    _assert_selection(sources, decoded, cap)
    for identity, pixels in decoded.items():
        assert pixels == _review_pixels(sources[identity])
    recording = load_recording(output)
    assert recording.application_id() == "neural-reconstruction"
    assert recording.recording_id() == run.name
