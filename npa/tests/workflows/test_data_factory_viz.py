"""Unit tests for the Physical AI Data Factory Rerun recording builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.data_factory_viz import DataFactoryVizError, build_run_rrd


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_build_run_rrd_from_local_run(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    run = tmp_path / "df-run"
    # Input frames for two clips.
    _write_png(run / "input" / "video_0_frame_01.png", (10, 20, 30))
    _write_png(run / "input" / "video_0_frame_02.png", (40, 50, 60))
    _write_png(run / "input" / "video_1_frame_01.png", (70, 80, 90))
    # One augmented clip with metadata.
    aug = run / "cosmos_augmented" / "video_0_aug0"
    _write_png(aug / "frame_01.png", (11, 22, 33))
    (aug / "metadata.json").write_text('{"variables": {"weather": "rainy", "time_of_day": "night"}}')

    out = tmp_path / "reports" / "sim2real.rrd"
    result = build_run_rrd(str(run), str(out))

    assert result["status"] == "completed"
    assert result["frames_logged"] == 4
    assert result["run_id"] == "df-run"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_build_run_rrd_requires_rrd_output(tmp_path: Path) -> None:
    with pytest.raises(DataFactoryVizError):
        build_run_rrd(str(tmp_path), str(tmp_path / "out.json"))


def test_build_run_rrd_errors_when_no_frames(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(DataFactoryVizError):
        build_run_rrd(str(empty), str(tmp_path / "reports" / "sim2real.rrd"))
