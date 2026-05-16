from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.cli.main import app
from npa.cli.viz.lerobot import (
    BackendName,
    LayoutName,
    VizRenderResult,
    viz_lerobot_deprecated,
)


runner = CliRunner()


def _write_g1_raw_dataset(root: Path, *, frames: int = 3) -> Path:
    raw = root / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
    state[:, 0] = np.sin(t * np.pi) * 0.10
    state[:, 6] = np.sin(t * np.pi * 2.0) * 0.25
    state[:, 18] = np.cos(t * np.pi * 2.0) * 0.20
    np.save(episode / "state.npy", state)
    np.save(episode / "actions.npy", state + 0.05)
    return raw


def _write_lerobot_dataset(root: Path) -> Path:
    return convert(
        _write_g1_raw_dataset(root),
        root / "lerobot",
        fps=3,
        task="Isaac-Velocity-Flat-G1-v0",
    )


def test_viz_lerobot_help_marks_deprecated() -> None:
    result = runner.invoke(app, ["viz", "lerobot", "--help"])

    assert result.exit_code == 0
    assert "DEPRECATED" in result.output
    assert "npa convert lerobot-to-mp4" in result.output


def test_deprecation_warning_emitted(tmp_path: Path, mocker) -> None:
    output = tmp_path / "out.mp4"
    mocker.patch(
        "npa.cli.viz.lerobot.render_lerobot",
        return_value=VizRenderResult(
            local_path=output,
            saved_to=str(output),
            duration_s=1.0,
            resolution=(160, 120),
            fps=3,
            frame_count=3,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        viz_lerobot_deprecated(
            input_path=str(tmp_path / "dataset"),
            output_path=str(output),
            backend=BackendName.matplotlib,
            predictions_path="",
            layout=LayoutName.single,
            duration=1.0,
            resolution="160x120",
            fps=3,
            title="",
        )

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("npa convert lerobot-to-mp4" in str(w.message) for w in caught)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for MP4 validation",
)
def test_forwarding_works(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path)
    output = tmp_path / "out.mp4"

    with pytest.warns(DeprecationWarning, match="npa convert lerobot-to-mp4"):
        viz_lerobot_deprecated(
            input_path=str(dataset),
            output_path=str(output),
            backend=BackendName.matplotlib,
            predictions_path="",
            layout=LayoutName.single,
            duration=1.0,
            resolution="160x120",
            fps=3,
            title="",
        )

    assert output.exists()
    assert output.stat().st_size > 0
