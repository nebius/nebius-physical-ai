from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import av
import matplotlib
import numpy as np
import pytest

from npa.viz.backends import matplotlib as backend


matplotlib.use("Agg", force=True)
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


@pytest.fixture
def caller_figure():
    existing = set(plt.get_fignums())
    fig = plt.figure()
    yield fig
    for number in set(plt.get_fignums()) - existing:
        plt.close(number)


def _render(output_path: Path) -> None:
    skeleton = np.array(
        [[[0, 0, 0], [0, 0, 1]], [[0, 0, 0], [0.5, 0, 1]], [[0, 0, 0], [1, 0, 1]]],
        dtype=np.float32,
    )
    backend.render(skeleton, None, "single", output_path, (320, 240), 3, 1.0, "Synthetic motion", [(0, 1)])


@pytest.mark.parametrize("failure_stage", ["layout", "frame", "save"])
def test_render_failure_closes_only_its_figure(tmp_path, monkeypatch, caller_figure, failure_stage):
    expected_figures = set(plt.get_fignums())
    error = RuntimeError(f"{failure_stage} failed")

    def fail(*args, **kwargs):
        raise error

    def save(self, *args, **kwargs):
        self._func(0)

    if failure_stage == "layout":
        monkeypatch.setattr(backend, "_build_layout", fail)
    elif failure_stage == "frame":
        monkeypatch.setattr(backend, "_update_skeleton", fail)
        monkeypatch.setattr(animation.Animation, "save", save)
    else:
        monkeypatch.setattr(animation.Animation, "save", fail)

    with pytest.raises(RuntimeError) as excinfo:
        _render(tmp_path / "failed.mp4")

    assert excinfo.value is error
    assert set(plt.get_fignums()) == expected_figures
    assert plt.figure(caller_figure.number) is caller_figure


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_repeated_ffmpeg_failures_do_not_accumulate_figures(tmp_path, caller_figure):
    output = tmp_path / "directory.mp4"
    output.mkdir()
    expected_figures = set(plt.get_fignums())
    observed_figures = []

    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            _render(output)
        assert excinfo.value.returncode != 0
        assert excinfo.value.cmd[-1] == str(output)
        assert "Is a directory" in excinfo.value.stderr
        observed_figures.append(set(plt.get_fignums()))

    assert observed_figures == [expected_figures, expected_figures]
    assert plt.figure(caller_figure.number) is caller_figure


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
def test_successful_mp4_is_decodable_and_preserves_caller_figure(tmp_path, caller_figure):
    output = tmp_path / "motion.mp4"
    expected_figures = set(plt.get_fignums())

    _render(output)

    assert set(plt.get_fignums()) == expected_figures
    assert plt.figure(caller_figure.number) is caller_figure
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (320, 240)
        assert stream.average_rate == 3
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx([0, 1 / 3, 2 / 3])
    images = [frame.to_ndarray(format="rgb24") for frame in frames]
    assert all(image.std() > 0 for image in images)
    assert not np.array_equal(images[0], images[-1])
