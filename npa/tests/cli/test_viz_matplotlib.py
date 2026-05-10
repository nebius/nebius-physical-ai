from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM
from npa.cli.viz.backends import matplotlib as matplotlib_backend
from npa.viz.lerobot import G1_JOINT_CONNECTIONS, g1_state_vectors_to_skeleton


pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering anything:UserWarning"
)


def _skeleton(frames: int = 4) -> np.ndarray:
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state[:, 0] = np.sin(t * np.pi)
    state[:, 6] = np.sin(t * np.pi * 2.0) * 0.3
    state[:, 18] = np.cos(t * np.pi * 2.0) * 0.2
    return g1_state_vectors_to_skeleton(state)


@pytest.mark.parametrize("layout", ["single", "side-by-side", "overlay"])
def test_matplotlib_backend_saves_animation_for_layouts(tmp_path: Path, mocker, layout: str) -> None:
    skeleton = _skeleton()
    predictions = skeleton + np.array([0.03, 0.0, 0.0], dtype=np.float32) if layout != "single" else None
    save = mocker.patch("matplotlib.animation.Animation.save")

    matplotlib_backend.render(
        skeleton,
        predictions,
        layout,
        tmp_path / f"{layout}.mp4",
        (320, 180),
        4,
        1.0,
        "test render",
        G1_JOINT_CONNECTIONS,
    )

    save.assert_called_once()


def test_matplotlib_backend_requires_predictions_for_overlay(tmp_path: Path) -> None:
    with pytest.raises(matplotlib_backend.MatplotlibRenderError, match="predictions_data is required"):
        matplotlib_backend.render(
            _skeleton(),
            None,
            "overlay",
            tmp_path / "overlay.mp4",
            (320, 180),
            4,
            1.0,
            "test render",
            G1_JOINT_CONNECTIONS,
        )
