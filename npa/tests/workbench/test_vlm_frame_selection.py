"""Verify sampler indices, exact submitted pixels, and source provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
from dataclasses import asdict

import numpy as np
from PIL import Image
import pytest

from npa.workbench import vlm_eval


def test_live_sampler_writer_serializes_public_result(tmp_path) -> None:
    from tests.e2e.test_vlm_frame_selection_live import _write_live_result

    result = vlm_eval.evaluate_stub(
        input_path="synthetic-control",
        output_path=str(tmp_path / "result.json"),
        score=0.0,
    )
    _write_live_result(result)
    import json

    assert json.loads(Path(result.result_uri).read_text()) == asdict(result)


@pytest.mark.parametrize(
    ("count", "maximum", "expected"),
    [
        (250, 4, [0, 112, 225, 249]),
        (7, 4, [0, 2, 5, 6]),
        (6, 3, [0, 4, 5]),
        (5, 3, [0, 3, 4]),
        (3, 2, [0, 2]),
        (454, 4, [0, 204, 408, 453]),
        (221, 4, [0, 98, 198, 220]),
    ],
)
def test_keyframe_frozen_vectors(count, maximum, expected) -> None:
    assert (
        vlm_eval._selected_indices(
            count, frame_selection="keyframes", max_frames=maximum
        )
        == expected
    )


def _legacy_sequence(count: int, maximum: int) -> list[int]:
    selected = min(count, maximum)
    if selected == 0:
        return []
    if selected == 1:
        return [count - 1]
    return sorted(
        {round(index * (count - 1) / (selected - 1)) for index in range(selected)}
    )


def test_complete_known_count_property_domain() -> None:
    for count in range(1001):
        for maximum in range(1, 129):
            sequence = vlm_eval._selected_indices(
                count, frame_selection="sequence", max_frames=maximum
            )
            assert sequence == _legacy_sequence(count, maximum)
            keyframes = vlm_eval._selected_indices(
                count, frame_selection="keyframes", max_frames=maximum
            )
            assert len(keyframes) == min(count, maximum)
            assert keyframes == sorted(set(keyframes))
            assert all(0 <= index < count for index in keyframes)
            if len(keyframes) >= 2:
                assert keyframes[0] == 0 and keyframes[-1] == count - 1
            if count <= maximum:
                assert keyframes == list(range(count))


def _write_distinct_frames(source: Path) -> None:
    source.mkdir()
    for index in range(6):
        Image.new("RGB", (16, 16), (index * 30, 0, 0)).save(
            source / f"frame-{index:03d}.png"
        )


@pytest.mark.parametrize("kind", ["image", "numpy"])
def test_public_path_retains_exact_source_pixels(tmp_path, kind) -> None:
    source = tmp_path / "frames"
    _write_distinct_frames(source)
    if kind == "numpy":
        array = np.stack(
            [np.asarray(Image.open(path)) for path in sorted(source.glob("*.png"))]
        )
        source = tmp_path / "episode.npy"
        np.save(source, array)
    selected = {}
    for strategy in ("sequence", "keyframes"):
        selected[strategy] = vlm_eval.select_rollout_frames(
            source, frame_selection=strategy, max_frames=3
        )
    assert [f.source_index for f in selected["sequence"]] == [0, 2, 5]
    assert [f.source_index for f in selected["keyframes"]] == [0, 4, 5]
    for frames in selected.values():
        assert all(f.source_count == 6 for f in frames)
        for frame in frames:
            evidence = vlm_eval._frame_evidence(frame)
            assert evidence.sha256 == hashlib.sha256(frame.data).hexdigest()
            assert evidence.source_index == frame.source_index
    assert selected["sequence"][0].data == selected["keyframes"][0].data
    assert selected["sequence"][-1].data == selected["keyframes"][-1].data


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="requires real video decoder",
)
def test_real_video_retains_source_timestamps(tmp_path) -> None:
    source = tmp_path / "frames"
    _write_distinct_frames(source)
    video = tmp_path / "rollout.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            "2",
            "-i",
            str(source / "frame-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    for strategy, indices in [("sequence", [0, 2, 5]), ("keyframes", [0, 4, 5])]:
        frames = vlm_eval.select_rollout_frames(
            video, frame_selection=strategy, max_frames=3
        )
        assert [f.source_index for f in frames] == indices
        assert [f.source_timestamp_s for f in frames] == pytest.approx(
            [index / 2 for index in indices]
        )
        assert all(f.source_count == 6 for f in frames)


@pytest.mark.parametrize("strategy", ["final", "sequence", "keyframes"])
def test_unknown_count_fallback_never_invents_provenance(
    monkeypatch, tmp_path, strategy
) -> None:
    source = tmp_path / "unknown.mp4"
    source.write_bytes(b"synthetic-video-placeholder")
    monkeypatch.setattr(vlm_eval, "_video_frame_count", lambda _path: None)
    monkeypatch.setattr(vlm_eval.shutil, "which", lambda _name: "/synthetic/tool")
    calls = []

    def extract(_source, directory, **kwargs):
        calls.append(kwargs)
        for index in range(1 if strategy == "final" else 2):
            Image.new("RGB", (8, 8), "gray").save(directory / f"frame-{index:03d}.png")
        return []

    monkeypatch.setattr(vlm_eval, "_extract_final_video_frame", extract)
    monkeypatch.setattr(vlm_eval, "_extract_video_sample", extract)
    frames = vlm_eval.select_rollout_frames(
        source, frame_selection=strategy, max_frames=2
    )
    manifest = vlm_eval._sampling_manifest(
        frames, frame_selection=strategy, max_frames=2
    )
    assert calls == ([{}] if strategy == "final" else [{"max_frames": 2}])
    assert len(frames) == (1 if strategy == "final" else 2)
    assert all(
        f.source_index is None
        and f.source_timestamp_s is None
        and f.source_count is None
        for f in frames
    )
    assert manifest["coverage_complete"] is False
    assert manifest["timestamps_complete"] is False
