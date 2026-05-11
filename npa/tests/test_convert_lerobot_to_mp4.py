from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, convert
from npa.adapter.lerobot.render import (
    LeRobotMP4RenderError,
    LeRobotMP4RenderResult,
    render_lerobot_to_mp4,
)
from npa.cli.main import app
from npa.cli.viz.backends import matplotlib as matplotlib_backend
from npa.viz.lerobot import REAL_G1_ACTION_DIM


runner = CliRunner()


def _write_g1_raw_dataset(root: Path, *, frames: int = 4) -> Path:
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


def _write_lerobot_dataset(root: Path, *, frames: int = 4, fps: int = 4) -> Path:
    return convert(
        _write_g1_raw_dataset(root, frames=frames),
        root / "lerobot",
        fps=fps,
        task="Isaac-Velocity-Flat-G1-v0",
    )


def _write_predictions(root: Path, *, frames: int = 2) -> Path:
    predictions = root / "predictions"
    predictions.mkdir()
    actions = np.zeros((frames, REAL_G1_ACTION_DIM), dtype=np.float32)
    actions[:, 32] = 0.5
    np.savez_compressed(predictions / "predicted_actions.npz", trajectory_0=actions)
    return predictions


def _write_tiny_mp4(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _assert_valid_mp4(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    subprocess.run(["ffprobe", "-v", "error", str(path)], check=True)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for MP4 validation",
)


def test_matplotlib_renderer_produces_valid_mp4(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=3, fps=3)
    output = tmp_path / "matplotlib.mp4"

    rendered = render_lerobot_to_mp4(
        dataset,
        output,
        renderer="matplotlib",
        duration=1.0,
        resolution="160x120",
        fps=3,
    )

    assert rendered == output
    _assert_valid_mp4(output)


def test_rerun_renderer_produces_valid_mp4_with_backend_dispatch(
    tmp_path: Path,
    mocker,
) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=2, fps=2)
    output = tmp_path / "rerun.mp4"

    def render(*args, **kwargs) -> None:
        _write_tiny_mp4(args[3])

    get_backend = mocker.patch(
        "npa.cli.viz.backends.get_backend",
        return_value=SimpleNamespace(render=render),
    )

    rendered = render_lerobot_to_mp4(
        dataset,
        output,
        renderer="rerun",
        duration=1.0,
        resolution="160x120",
        fps=2,
    )

    assert rendered == output
    get_backend.assert_called_once_with("rerun")
    _assert_valid_mp4(output)


def test_predictions_overlay_keeps_short_prediction_window(
    tmp_path: Path, mocker
) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=4, fps=4)
    predictions = _write_predictions(tmp_path, frames=1)
    output = tmp_path / "overlay.mp4"

    def render(*args, **kwargs) -> None:
        skeleton_data, predictions_data, layout, output_path = args[:4]
        assert layout == "overlay"
        assert skeleton_data.shape[0] == 4
        assert predictions_data is not None
        assert predictions_data.shape[0] == 1
        assert matplotlib_backend.PREDICTION_COLOR == "#ff8800"
        _write_tiny_mp4(output_path)

    mocker.patch(
        "npa.cli.viz.backends.get_backend", return_value=SimpleNamespace(render=render)
    )

    render_lerobot_to_mp4(
        dataset,
        output,
        renderer="matplotlib",
        duration=1.0,
        predictions_path=predictions,
        layout="overlay",
        resolution="160x120",
        fps=4,
    )

    _assert_valid_mp4(output)


def test_convert_lerobot_to_mp4_renderer_flag_dispatches(
    tmp_path: Path, mocker
) -> None:
    output = tmp_path / "cli.mp4"
    render = mocker.patch(
        "npa.cli.convert.lerobot_to_mp4.render_lerobot_to_mp4_result",
        return_value=LeRobotMP4RenderResult(
            local_path=output,
            saved_to=str(output),
            duration_s=5.0,
            resolution=(640, 360),
            fps=12,
            frame_count=60,
        ),
    )

    result = runner.invoke(
        app,
        [
            "convert",
            "lerobot-to-mp4",
            "--input-path",
            str(tmp_path / "dataset"),
            "--renderer",
            "rerun",
            "--output-path",
            str(output),
            "--duration",
            "5",
            "--resolution",
            "640x360",
            "--fps",
            "12",
        ],
    )

    assert result.exit_code == 0
    render.assert_called_once_with(
        input_path=str(tmp_path / "dataset"),
        output_path=str(output),
        renderer="rerun",
        predictions_path=None,
        layout="single",
        duration=5.0,
        resolution="640x360",
        fps=12,
        title=None,
    )


def test_output_path_is_honored(tmp_path: Path, mocker) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=2, fps=2)
    output = tmp_path / "nested" / "honored.mp4"

    def render(*args, **kwargs) -> None:
        _write_tiny_mp4(args[3])

    mocker.patch(
        "npa.cli.viz.backends.get_backend", return_value=SimpleNamespace(render=render)
    )

    rendered = render_lerobot_to_mp4(
        dataset, output, renderer="matplotlib", duration=1.0, fps=2
    )

    assert rendered == output
    _assert_valid_mp4(output)


def test_backend_errors_propagate_without_wrapping(tmp_path: Path, mocker) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=2, fps=2)

    def render(*args, **kwargs) -> None:
        raise RuntimeError("backend exploded")

    mocker.patch(
        "npa.cli.viz.backends.get_backend", return_value=SimpleNamespace(render=render)
    )

    with pytest.raises(RuntimeError, match="backend exploded"):
        render_lerobot_to_mp4(
            dataset,
            tmp_path / "out.mp4",
            renderer="matplotlib",
            duration=1.0,
            fps=2,
        )


def test_missing_input_raises_render_error(tmp_path: Path) -> None:
    with pytest.raises(LeRobotMP4RenderError, match="Input path does not exist"):
        render_lerobot_to_mp4(
            tmp_path / "missing",
            tmp_path / "out.mp4",
            renderer="matplotlib",
        )
