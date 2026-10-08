"""Exercise source-border detection and exact generated-scene preservation."""

import json
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.cosmos_evaluator.appearance_fidelity import _iter_rgb_frames
from npa.workflows import paidf_cosmos3 as workflow
from npa.workflows.paidf_cosmos3_media import probe_video, video_sha256
from npa.workflows.video_padding_detection import detect_source_padding
from npa.workflows.video_padding_preservation import (
    preserve_source_padding,
    resolve_source_content_region,
)
from test_cosmos_evaluator_padding import _write_video
from test_cosmos_evaluator_padding import media_tools as media_tools
from test_cosmos_evaluator_padding import padded as padded


@pytest.mark.parametrize("bounds", [[8, 0, 72, 48], [0, 6, 80, 42], [8, 6, 72, 42]])
def test_detects_embedded_pillarbox_letterbox_and_windowbox(tmp_path, bounds):
    left, top, right, bottom = bounds
    frames = np.zeros((12, 48, 80, 3), np.uint8)
    frames[:, top:bottom, left:right] = 120
    frames[:, 16:28, 24:36] = 0  # real dark object inside the scene
    source = _write_video(tmp_path / "source.mkv", frames)
    record = detect_source_padding(source)
    assert record["bounds"] == bounds
    assert record["padding_detection"]["status"] == "detected"
    assert record["padding_detection"]["decoded_frames"] == 12


@pytest.mark.parametrize(
    "kind",
    ["black-scene", "one-edge", "moving-edge", "fade", "dim-scene", "textured-edge"],
)
def test_ambiguous_dark_scene_content_is_not_treated_as_padding(tmp_path, kind):
    frames = np.full((12, 48, 80, 3), 100, np.uint8)
    if kind == "black-scene":
        frames[:] = 0
    elif kind == "one-edge":
        frames[:, :, :8] = 0
    elif kind == "moving-edge":
        frames[:6, :, :8] = frames[:6, :, 72:] = 0
    elif kind == "fade":
        frames[:6] = 0
    elif kind == "dim-scene":
        frames[:] = 18
        frames[:, :, :8] = frames[:, :, 72:] = 0
    elif kind == "textured-edge":
        frames[:, :, :8] = frames[:, :, 72:] = 0
        frames[:, 10:20, :8] = frames[:, 10:20, 72:] = 25
    source = _write_video(tmp_path / "source.mkv", frames)
    assert detect_source_padding(source)["bounds"] == [0, 0, 80, 48]


def test_full_clip_analysis_does_not_hide_a_late_entering_object(tmp_path):
    frames = np.full((101, 48, 80, 3), 120, np.uint8)
    frames[:, :, :8] = frames[:, :, 72:] = 0
    frames[-1, 10:20, :8] = 255
    source = _write_video(tmp_path / "source.mkv", frames)
    assert detect_source_padding(source)["bounds"] == [0, 0, 80, 48]


def test_detects_embedded_bars_inside_known_normalization_padding(tmp_path):
    frames = np.zeros((12, 48, 80, 3), np.uint8)
    frames[:, 6:42, 8:72] = 120
    source = _write_video(tmp_path / "source.mkv", frames)
    record = detect_source_padding(source, normalization_bounds=[0, 6, 80, 42])
    assert record["bounds"] == [8, 6, 72, 42]
    assert resolve_source_content_region(source, record) == record


def test_preparation_detects_bars_already_embedded_in_source(padded, tmp_path):
    from npa.workflows.paidf_cosmos3_media import prepare_reference

    source, _, _, _ = padded
    timeline = prepare_reference(source, tmp_path / "prepared.mp4")
    region = timeline["source_content_region"]
    assert region["padding_detection"]["status"] == "detected"
    assert region["bounds"][0] > region["normalization_bounds"][0]
    assert region["bounds"][2] < region["normalization_bounds"][2]


@pytest.mark.parametrize(
    "change", ["one-sided", "missing-evidence", "incomplete", "threshold"]
)
def test_detected_padding_rejects_invalid_evidence(padded, change):
    source, _, _, _ = padded
    record = detect_source_padding(source)
    if change == "one-sided":
        record["bounds"] = record["padding_detection"]["bounds"] = [8, 0, 80, 48]
    elif change == "missing-evidence":
        del record["padding_detection"]
    elif change == "incomplete":
        record["padding_detection"]["decoded_frames"] -= 1
    else:
        record["padding_detection"]["black_level"] = 100
    with pytest.raises(ValueError):
        resolve_source_content_region(source, record)


def test_preservation_rejects_truncated_model_output(padded, tmp_path):
    source, _, _, frames = padded
    truncated = _write_video(tmp_path / "truncated.mkv", frames[:-1])
    with pytest.raises(ValueError, match="aligned complete videos"):
        preserve_source_padding(
            source, truncated, tmp_path / "out.mp4", detect_source_padding(source)
        )


def test_preserves_source_borders_and_every_generated_scene_pixel(padded, tmp_path):
    source, raw, _, _ = padded
    record = resolve_source_content_region(source)
    output = tmp_path / "preserved.mp4"
    raw_hash = video_sha256(raw)
    receipt = preserve_source_padding(source, raw, output, record)
    assert receipt["scene_pixels_unchanged"] and receipt["padding_matches_source"]
    assert receipt["raw_model_sha256"] == video_sha256(raw) == raw_hash
    assert receipt["published_sha256"] == video_sha256(output)
    source_frames = np.stack(list(_iter_rgb_frames(source, 48, 80)))
    raw_frames = np.stack(list(_iter_rgb_frames(raw, 48, 80)))
    actual = np.stack(list(_iter_rgb_frames(output, 48, 80)))
    assert np.array_equal(actual[:, :, :8], source_frames[:, :, :8])
    assert np.array_equal(actual[:, :, 72:], source_frames[:, :, 72:])
    assert np.array_equal(actual[:, :, 8:72], raw_frames[:, :, 8:72])
    assert probe_video(output)["decoded_frames"] == 12


def test_generated_changes_to_black_scene_objects_are_retained(padded, tmp_path):
    source, _, _, frames = padded
    frames[:, 4:28, 12:30] = (180, 90, 30)
    raw = _write_video(tmp_path / "modified.mkv", frames)
    output = tmp_path / "preserved.mp4"
    preserve_source_padding(source, raw, output, resolve_source_content_region(source))
    actual = np.stack(list(_iter_rgb_frames(output, 48, 80)))
    assert np.array_equal(actual[:, 4:28, 12:30], frames[:, 4:28, 12:30])


def test_scoring_uses_same_scene_with_zero_corrected_border_error(padded, tmp_path):
    from npa.workbench.cosmos_evaluator.spatial_evidence import prepare_spatial_evidence

    source, raw, _, _ = padded
    record = resolve_source_content_region(source)
    published = tmp_path / "preserved.mp4"
    preserve_source_padding(source, raw, published, record)
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    _, raw_scene, _ = prepare_spatial_evidence(record, source, raw, before)
    _, fixed_scene, corrected = prepare_spatial_evidence(
        record, source, published, after
    )
    expected = np.stack(list(_iter_rgb_frames(raw_scene, 48, 64)))
    actual = np.stack(list(_iter_rgb_frames(fixed_scene, 48, 64)))
    assert np.array_equal(actual, expected)
    assert corrected["padding"]["changed_pixel_fraction"] == 0
    assert corrected["padding"]["mean_absolute_rgb_error"] == 0


def test_no_padding_requires_no_output_reencoding(tmp_path):
    frames = np.full((12, 48, 80, 3), 100, np.uint8)
    source = _write_video(tmp_path / "source.mkv", frames)
    output = tmp_path / "preserved.mp4"
    assert (
        preserve_source_padding(source, source, output, detect_source_padding(source))
        is None
    )
    assert not output.exists()


def test_preservation_rejects_stale_hashes_and_overwrites(padded, tmp_path):
    source, raw, _, _ = padded
    record = resolve_source_content_region(source)
    with pytest.raises(ValueError, match="new output"):
        preserve_source_padding(source, raw, raw, record)
    record["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        preserve_source_padding(source, raw, tmp_path / "new.mp4", record)


class _Storage:
    def __init__(self):
        self.objects = {}

    def upload_file(self, source, uri):
        self.objects[uri] = Path(source).read_bytes()
        return uri


def _metadata(source):
    return {
        "prompt": "change the work surface",
        "model": "Cosmos3-Nano",
        "seed": 1,
        "guidance": 5,
        "steps": 35,
        "guardrails": True,
        "attempt": 0,
        "input_provenance_uri": "s3://example-bucket/input/provenance.json",
        "source_content_region": resolve_source_content_region(source),
    }


def test_publication_preserves_raw_output_and_publishes_corrected_video(padded):
    source, raw, _, _ = padded
    storage = _Storage()
    root = "s3://example-bucket/output/"
    manifest = workflow._publish_variant(
        result={"output_path": str(raw)},
        output_uri=root,
        clip="variant-0000",
        variables={"lighting": "daylight"},
        metadata=_metadata(source),
        storage=storage,
        source_video=source,
    )
    base = root + "variant-0000/"
    metadata = json.loads(storage.objects[base + "metadata.json"])
    assert storage.objects[base + "raw_model_video.mp4"] == raw.read_bytes()
    assert storage.objects[base + "augmented_video.mp4"] != raw.read_bytes()
    assert (
        metadata["padding_preservation"]["raw_model_video_uri"]
        == base + "raw_model_video.mp4"
    )
    assert (
        metadata["published_video_sha256"]
        == metadata["padding_preservation"]["published_sha256"]
    )
    assert manifest["padding_preservation"] == metadata["padding_preservation"]
    assert manifest["video_bytes"] == len(storage.objects[base + "augmented_video.mp4"])
    assert metadata["motion_preservation"] is None


def test_encoding_failure_retains_raw_and_does_not_publish_uncorrected_video(
    padded, monkeypatch
):
    from npa.workflows import video_padding_preservation as preservation

    source, raw, _, _ = padded
    storage = _Storage()

    def fail(*args):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(preservation, "preserve_source_padding", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        workflow._publish_variant(
            result={"output_path": str(raw)},
            output_uri="s3://example-bucket/output/",
            clip="variant-0000",
            variables={},
            metadata=_metadata(source),
            storage=storage,
            source_video=source,
        )
    assert any(key.endswith("raw_model_video.mp4") for key in storage.objects)
    assert not any(key.endswith("augmented_video.mp4") for key in storage.objects)
