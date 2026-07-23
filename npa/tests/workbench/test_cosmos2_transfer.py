"""Unit tests for the non-stub Cosmos2 transfer reference augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from npa.workbench.cosmos.transfer import reference_augment_frames


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def test_reference_augment_produces_real_image_frames(tmp_path: Path) -> None:
    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (200, 40, 40))
    _write_png(src / "frame_001.png", (40, 200, 40))

    result = reference_augment_frames(
        str(src), str(out), run_id="unit", variants_per_frame=2
    )

    # 2 sources x 2 variants = 4 real augmented frames, plus an index manifest.
    assert result["frame_count"] == 4
    assert result["source_frame_count"] == 2
    frames = sorted(out.glob("frame-*.png"))
    assert len(frames) == 4
    for frame in frames:
        # Each output is a real, openable image (not a JSON descriptor).
        with Image.open(frame) as img:
            assert img.size == (64, 48)
    index = json.loads((out / "index.json").read_text())
    assert index["frame_count"] == 4
    assert {f["perturbation"] for f in index["frames"]} <= {
        "lighting",
        "contrast",
        "color",
        "blur",
    }


def test_reference_augment_actually_transforms_pixels(tmp_path: Path) -> None:
    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (128, 128, 128))

    reference_augment_frames(str(src), str(out), run_id="unit", variants_per_frame=1)

    original = Image.open(src / "frame_000.png").convert("RGB").tobytes()
    augmented = Image.open(next(out.glob("frame-*.png"))).convert("RGB").tobytes()
    assert original != augmented, "reference augmentation must change pixels, not copy"


def test_reference_augment_without_sources_raises(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(RuntimeError, match="no source images"):
        reference_augment_frames(str(src), str(tmp_path / "out"), run_id="unit")


def test_cosmos2_transfer_cli_default_emits_real_frames(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (10, 20, 30))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            str(src),
            "--output-uri",
            str(out),
            "--run-id",
            "cli-unit",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Not a descriptor stub: it ran a real reference augmentation with frames.
    assert payload["status"] == "executed_reference"
    assert payload["mode"] == "reference_augment"
    assert payload["frame_count"] >= 1
    assert list(out.glob("frame-*.png"))
