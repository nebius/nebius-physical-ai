"""Prove artificial padding cannot contaminate scene-quality evidence."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.cosmos_evaluator import evaluate as evaluator
from npa.workbench.cosmos_evaluator import appearance_fidelity as appearance
from npa.workbench.cosmos_evaluator import temporal_consistency as temporal
from npa.workbench.cosmos_evaluator.attribute_verification import (
    AttributeVerificationResult,
)
from npa.workbench.cosmos_evaluator.hallucination import check_hallucination
from npa.workbench.cosmos_evaluator.spatial_evidence import (
    prepare_spatial_evidence,
    remap_regions,
)
from npa.workflows.paidf_cosmos3_media import prepare_reference, probe_video
from npa.workflows.video_content_region import (
    content_region_record,
    crop_content,
    validate_content_region,
)


@pytest.fixture(autouse=True)
def media_tools():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires real ffmpeg and ffprobe")


def _write_video(path, pixels):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "80x48",
            "-framerate",
            "24",
            "-i",
            "-",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-f",
            "matroska",
            str(path),
        ],
        input=pixels.tobytes(),
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def padded(tmp_path):
    pixels = np.zeros((12, 48, 80, 3), dtype=np.uint8)
    pixels[:, :, 8:72] = 100
    # A genuinely black object inside the scene must remain in the crop.
    pixels[:, 4:28, 12:30] = 0
    for index, frame in enumerate(pixels):
        frame[30:40, 35 + index : 43 + index] = 220
    source = _write_video(tmp_path / "source.mkv", pixels)
    changed = pixels.copy()
    changed[::2, :, :8] = (255, 0, 0)
    changed[1::2, :, 72:] = (0, 255, 255)
    video = _write_video(tmp_path / "changed.mkv", changed)
    record = content_region_record(probe_video(source), [8, 0, 72, 48])
    return source, video, record, pixels


def _scores(source, video):
    arguments = dict(clip_id="test", original_video=source, augmented_video=video)
    return {
        "hallucination": check_hallucination(**arguments, prefer_upstream=False).score,
        "temporal": temporal.check_temporal_consistency(**arguments).score,
        "appearance": appearance.check_appearance_fidelity(**arguments).score,
    }


def test_padding_changes_do_not_affect_scene_metrics(padded, tmp_path):
    source, video, record, _ = padded
    full_scores = _scores(source, video)
    source_scene, video_scene, evidence = prepare_spatial_evidence(
        record, source, video, tmp_path
    )
    assert _scores(source_scene, video_scene) == _scores(source, source)
    assert full_scores["hallucination"] < 1
    assert full_scores["temporal"] < 1
    assert evidence["padding"]["changed_pixel_fraction"] == 0.5
    assert evidence["padding"]["mean_absolute_rgb_error"] == 63.75
    assert evidence["padding"]["included_in_scene_score"] is False
    assert evidence["excluded_pixel_fraction"] == pytest.approx(0.2)
    measured = probe_video(video_scene)
    assert measured["decoded_frames"] == 12
    assert (measured["width"], measured["height"]) == (64, 48)


def test_real_black_objects_and_scene_defects_still_count(padded, tmp_path):
    source, _, record, pixels = padded
    pixels[::2, 4:28, 12:30] = 255
    video = _write_video(tmp_path / "scene-defect.mkv", pixels)
    source_scene, video_scene, evidence = prepare_spatial_evidence(
        record, source, video, tmp_path
    )
    scores = _scores(source_scene, video_scene)
    assert scores["hallucination"] < 0.75
    assert scores["temporal"] < 0.75
    assert scores["appearance"] < 0.75
    assert evidence["padding"]["changed_pixel_fraction"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"origin": "black-pixel-detection"},
        {"bounds": [8, 0, 76, 48]},
        {"bounds": [8, 2, 72, 46]},
        {"bounds": [8, 0, 100, 48]},
        {"bounds": [True, 0, 72, 48]},
        {"bounds": "8,0,72,48"},
        {"canvas": [82, 48]},
        {"source_sha256": "stale"},
        {"source_sha256": None},
    ],
)
def test_invalid_or_stale_provenance_fails_closed(padded, tmp_path, change):
    source, video, record, _ = padded
    with pytest.raises(ValueError):
        prepare_spatial_evidence({**record, **change}, source, video, tmp_path)


def test_content_provenance_requires_complete_corresponding_media(padded, tmp_path):
    source, _, record, pixels = padded
    short = _write_video(tmp_path / "short.mkv", pixels[:-1])
    with pytest.raises(ValueError, match="aligned complete"):
        prepare_spatial_evidence(record, source, short, tmp_path)
    with pytest.raises(ValueError, match="both complete"):
        prepare_spatial_evidence(record, source, None, tmp_path)


def test_evidence_crops_are_reproducible_and_pixel_exact(padded, tmp_path):
    source, _, record, pixels = padded
    first, second = tmp_path / "first.mkv", tmp_path / "second.mkv"
    assert crop_content(source, first, record["bounds"]) == crop_content(
        source, second, record["bounds"]
    )
    decoded = list(appearance._iter_rgb_frames(first, 48, 64))
    assert np.array_equal(np.stack(decoded), pixels[:, :, 8:72])


def test_legacy_and_full_canvas_evidence_remain_uncropped(padded, tmp_path):
    source, video, record, _ = padded
    for provenance in (None, {**record, "bounds": [0, 0, 80, 48]}):
        paths = prepare_spatial_evidence(provenance, source, video, tmp_path)
        assert paths[:2] == (source, video)
        assert paths[2]["mode"] == "full-frame"
    assert not list(tmp_path.glob("*-scene.mkv"))


@pytest.mark.parametrize("record", [None, {}, {"schema": "unknown"}])
def test_recorded_invalid_provenance_blocks_grading(
    padded, tmp_path, monkeypatch, record
):
    source, video, _, _ = padded
    variant = tmp_path / "augment" / "variant-0000"
    variant.mkdir(parents=True)
    shutil.copy2(video, variant / "augmented_video.mp4")
    (variant / "metadata.json").write_text(
        json.dumps(
            {
                "input_conditioned": True,
                "source_content_region": record,
                "variables": {"lighting": "daylight"},
            }
        )
    )

    def unexpected_call(**kwargs):
        pytest.fail("Invalid spatial evidence must stop scoring before VLM calls")

    monkeypatch.setattr(evaluator, "verify_attributes", unexpected_call)
    result = evaluator.evaluate_run(
        augment_uri=str(variant.parent),
        original_video=str(source),
        output_uri=str(tmp_path / "grade"),
        storage=object(),
    )
    assert result.status == "degraded" and not result.passed
    assert result.score == 0.0
    assert result.clips[0].spatial_evidence["status"] == "failed"


def test_explicit_regions_keep_their_original_scene_coordinates(padded, tmp_path):
    source, video, record, _ = padded
    _, _, evidence = prepare_spatial_evidence(record, source, video, tmp_path)
    assert remap_regions("", evidence, temporal.parse_regions) == ""
    regions = json.dumps([{"id": "left-half", "bounds": [0, 0, 0.5, 1]}])
    result = json.loads(remap_regions(regions, evidence, temporal.parse_regions))
    assert result == [{"id": "left-half", "bounds": [0, 0, 0.5, 1]}]
    with pytest.raises(ValueError, match="only padding"):
        remap_regions("[[0,0,0.1,1]]", evidence, appearance.parse_regions)


def test_run_uses_scene_evidence_for_vlm_and_all_pixel_checks(
    padded, tmp_path, monkeypatch
):
    source, video, record, _ = padded
    variant = tmp_path / "augment" / "variant-0000"
    variant.mkdir(parents=True)
    shutil.copy2(video, variant / "augmented_video.mp4")
    metadata = {
        "variables": {"lighting": "daylight"},
        "input_conditioned": True,
        "source_content_region": record,
    }
    (variant / "metadata.json").write_text(json.dumps(metadata))
    calls = []

    def verify(**kwargs):
        calls.append(kwargs)
        for key in ("video", "reference_video"):
            assert probe_video(Path(kwargs[key]))["width"] == 64
        return AttributeVerificationResult("test", True, 1, 1, 0, 1.0, "llm", "vlm")

    monkeypatch.setattr(evaluator, "verify_attributes", verify)
    result = evaluator.evaluate_run(
        augment_uri=str(variant.parent),
        original_video=str(source),
        output_uri=str(tmp_path / "grade"),
        storage=object(),
    )
    assert len(calls) == 1
    assert result.passed and result.score == 1.0
    clip = result.clips[0]
    assert clip.temporal_consistency["score"] == 1.0
    assert clip.appearance_fidelity["score"] == 1.0
    assert clip.spatial_evidence["padding"]["changed_pixel_fraction"] == 0.5
    assert clip.spatial_evidence["generated_sha256"] == probe_video(video)["sha256"]


@pytest.mark.parametrize(
    "size,expected",
    [
        ("640x480", [96, 0, 736, 480]),
        ("1280x720", [0, 6, 832, 474]),
        ("832x480", [0, 0, 832, 480]),
        ("400x480", [216, 0, 616, 480]),
        ("398x480", [216, 0, 614, 480]),
    ],
)
def test_preparation_records_actual_scaled_geometry(tmp_path, size, expected):
    source, output = tmp_path / "original.mp4", tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=white:s={size}:r=24",
            "-frames:v",
            "6",
            str(source),
        ],
        check=True,
    )
    timeline = prepare_reference(source, output)
    record = timeline["source_content_region"]
    assert record["bounds"] == expected
    assert record["source_sha256"] == timeline["prepared"]["sha256"]
    validate_content_region(record, output, output)


def test_preparation_measures_display_rotation(tmp_path):
    raw, source, output = [
        tmp_path / name for name in ("raw.mp4", "rotated.mp4", "source.mp4")
    ]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=white:s=640x480:r=24",
            "-frames:v",
            "6",
            str(raw),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(raw),
            "-c",
            "copy",
            str(source),
        ],
        check=True,
    )
    timeline = prepare_reference(source, output)
    assert timeline["source_content_region"]["bounds"] == [236, 0, 596, 480]
    validate_content_region(timeline["source_content_region"], output, output)


def test_input_provenance_publishes_preparation_rectangle(padded, tmp_path):
    from npa.workflows.paidf_cosmos3 import prepare_input

    source, _, _, _ = padded
    output = tmp_path / "input"
    result = prepare_input(
        "video",
        str(source),
        "",
        0,
        "",
        str(output),
        str(output / "provenance.json"),
        "test",
        conditioning_fps=24,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    timeline = json.loads((output / "timeline.json").read_text())
    assert result["source_content_region"] == provenance["source_content_region"]
    assert provenance["source_content_region"] == timeline["source_content_region"]
    assert provenance["source_content_region"]["source_sha256"] == provenance["sha256"]
