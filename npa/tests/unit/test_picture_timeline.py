"""Check real picture timing, dissolve pixels, cache reuse and delivery preservation."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from npa.studio_renderer import picture_timeline as timeline


@pytest.fixture
def clips(tmp_path):
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        pytest.skip("Picture integration checks require FFmpeg and ffprobe")
    result = []
    for index, (color, frames) in enumerate((("red", 33), ("blue", 33), ("green", 30))):
        path = tmp_path / f"source {index}'s.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:s=64x36:r=30",
            "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ], check=True)
        result.append(path)
    return [timeline.PictureShot(result[0], 30, tail_frames=3),
            timeline.PictureShot(result[1], 30, head_frames=3),
            timeline.PictureShot(result[2], 30)]


def _frames(path):
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-pix_fmt", "rgb24",
        "-f", "rawvideo", "pipe:1",
    ])
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64, 3).astype(float)


def test_centered_dissolve_preserves_duration_and_clear_cut(clips, tmp_path):
    output = tmp_path / "picture.mp4"
    report = timeline.assemble_picture(clips, output, cache_dir=tmp_path / "cache")
    pixels = _frames(output).mean(axis=(1, 2))
    assert pixels.shape == (90, 3)
    assert report["frames"] == 90 and report["dissolves"] == 1
    assert report["segments_rendered"] == 4
    assert np.max(np.abs(pixels[:28] - pixels[0])) < 2
    assert np.max(np.abs(pixels[32:60] - pixels[59])) < 2
    assert np.max(np.abs(pixels[60:] - pixels[60])) < 2
    assert np.all(np.diff(pixels[27:33, 0]) < 0)
    assert np.all(np.diff(pixels[27:33, 2]) > 0)
    assert pixels[60, 1] > 100 and pixels[59, 2] > 200
    assert report["sha256"] == timeline._hash(output)


def test_picture_reuses_verified_segments_and_repairs_corruption(clips, tmp_path):
    output, cache = tmp_path / "picture.mp4", tmp_path / "cache"
    first = timeline.assemble_picture(clips, output, cache_dir=cache)
    second = timeline.assemble_picture(clips, output, cache_dir=cache)
    assert second["segments_reused"] == 4 and second["segments_rendered"] == 0
    assert second["sha256"] == first["sha256"]
    next(cache.glob("picture/*/segment.mp4")).write_bytes(b"interrupted write")
    repaired = timeline.assemble_picture(clips, output, cache_dir=cache)
    assert repaired["segments_reused"] == 3 and repaired["segments_rendered"] == 1
    assert repaired["sha256"] == first["sha256"]


def test_failed_encoding_preserves_previous_delivery(clips, tmp_path, monkeypatch):
    output = tmp_path / "picture.mp4"
    output.write_bytes(b"previous delivery")

    def fail(*args):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(timeline, "_encode_segment", fail)
    with pytest.raises(subprocess.CalledProcessError):
        timeline.assemble_picture(clips, output, cache_dir=tmp_path / "cache")
    assert output.read_bytes() == b"previous delivery"


def test_mid_assembly_source_edit_preserves_previous_delivery(clips, tmp_path, monkeypatch):
    output = tmp_path / "picture.mp4"
    output.write_bytes(b"previous delivery")
    concatenate = timeline._concatenate

    def edit_source(*args):
        concatenate(*args)
        clips[0].path.write_bytes(b"concurrent edit")

    monkeypatch.setattr(timeline, "_concatenate", edit_source)
    with pytest.raises(ValueError, match="changed during"):
        timeline.assemble_picture(clips, output, cache_dir=tmp_path / "cache")
    assert output.read_bytes() == b"previous delivery"


@pytest.mark.parametrize("shots, fps, message", [
    ([], 30, "needs shots"),
    ([timeline.PictureShot(Path("unused"), 30)], True, "integer fps"),
    ([timeline.PictureShot(Path("unused"), 30, head_frames=1)], 30, "first head"),
    ([timeline.PictureShot(Path("unused"), 30, tail_frames=1)], 30, "last tail"),
    ([timeline.PictureShot(Path("unused"), 30.0)], 30, "integers"),
    ([timeline.PictureShot(Path("unused"), 0)], 30, "clear body"),
    ([timeline.PictureShot(Path("a"), 30, tail_frames=3),
      timeline.PictureShot(Path("b"), 30, head_frames=2)], 30, "must match"),
    ([timeline.PictureShot(Path("a"), 2, tail_frames=2),
      timeline.PictureShot(Path("b"), 30, head_frames=2)], 30, "clear body"),
])
def test_invalid_edits_fail_before_reading_media(shots, fps, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        timeline.assemble_picture(shots, tmp_path / "out.mp4", cache_dir=tmp_path / "cache", fps=fps)


def test_missing_handle_frames_are_rejected(clips, tmp_path):
    wrong = [timeline.PictureShot(clips[0].path, 31, tail_frames=3), *clips[1:]]
    with pytest.raises(ValueError, match="frame count"):
        timeline.assemble_picture(wrong, tmp_path / "out.mp4", cache_dir=tmp_path / "cache")


def test_output_cannot_replace_source(clips, tmp_path):
    with pytest.raises(ValueError, match="overwrite an input"):
        timeline.assemble_picture(clips, clips[0].path, cache_dir=tmp_path / "cache")


def test_cut_only_single_shot_uses_all_frames(clips, tmp_path):
    output = tmp_path / "picture.mp4"
    report = timeline.assemble_picture([clips[2]], output, cache_dir=tmp_path / "cache")
    assert report["frames"] == 30 and report["dissolves"] == 0
    assert _frames(output).shape[0] == 30
    video = timeline._probe(output)
    assert video["color_range"] == "tv"
    assert all(video[key] == "bt709" for key in ("color_space", "color_transfer", "color_primaries"))
    assert json.loads(next((tmp_path / "cache").glob("picture/*/cache-receipt.json")).read_text())["files"]


def test_middle_shot_can_have_both_handles_without_timeline_drift(clips, tmp_path):
    shots = [clips[0], timeline.PictureShot(clips[1].path, 27, head_frames=3, tail_frames=3),
             timeline.PictureShot(clips[2].path, 27, head_frames=3)]
    output = tmp_path / "picture.mp4"
    report = timeline.assemble_picture(shots, output, cache_dir=tmp_path / "cache")
    pixels = _frames(output).mean(axis=(1, 2))
    assert pixels.shape[0] == 84 and report["dissolves"] == 2
    assert pixels[33, 2] > 200 and pixels[54, 2] > 200
    assert pixels[59, 1] > 100 and pixels[-1, 1] > 100
    assert np.all(np.diff(pixels[54:60, 2]) < 0)


def test_frame_rate_mismatch_is_rejected(clips, tmp_path):
    with pytest.raises(ValueError, match="constant 24 fps"):
        timeline.assemble_picture(clips, tmp_path / "out.mp4", cache_dir=tmp_path / "cache", fps=24)
