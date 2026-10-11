"""Reject video scaling, timing drift and incomplete recorded episodes."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from npa.workbench.mjlab import worker
from npa.workbench.mjlab.rollout import _video_writer
from npa.workbench.mjlab.schemas import EvalRequest, MjlabError


def _request(**settings):
    return EvalRequest(
        checkpoint="s3://fixture/checkpoint.pt",
        output_path="s3://fixture/eval",
        video=True,
        **settings,
    )


@pytest.mark.parametrize(
    "settings", [{"video_width": 1919}, {"video_height": 1079}, {"video_width": 0}]
)
def test_h264_dimensions_rejected_before_execution(settings):
    with pytest.raises(ValueError):
        _request(**settings)


def test_native_capture_configuration_preserves_task_camera_and_horizon(monkeypatch):
    viewer = SimpleNamespace(width=320, height=240, body_name="torso_link")
    playing_config = SimpleNamespace(viewer=viewer, scene=SimpleNamespace(num_envs=1))
    training = SimpleNamespace(episode_length_s=20, terminations={"timeout": object()})
    monkeypatch.setattr(
        worker,
        "_configs",
        lambda request, inputs, play=False: (
            playing_config if play else training,
            None,
        ),
    )
    cfg, _ = worker._policy_configs(_request(), {})
    assert (cfg.viewer.width, cfg.viewer.height) == (1920, 1080)
    assert cfg.viewer.body_name == "torso_link"
    assert cfg.episode_length_s == 20 and cfg.terminations == training.terminations


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"height": 1088}, "dimensions"),
        ({"fps": 25}, "timing"),
        ({"frames": 999}, "frame count"),
    ],
)
def test_encoded_video_must_match_native_episode(monkeypatch, change, reason):
    video = {"width": 1920, "height": 1080, "fps": 50, "frames": 1000}
    monkeypatch.setattr(worker, "_inspect_video", lambda path: {**video, **change})
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02))
    report = {"episodes": [{"environment": 0, "episode": 0, "length": 1000}]}
    with pytest.raises(MjlabError, match=reason):
        worker._verify_video("unused", _request(), env, report)


def test_encoder_does_not_round_1080_up_to_a_macroblock(tmp_path):
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02))
    with ExitStack() as stack:
        writer = _video_writer(stack, env, tmp_path)
        for _ in range(3):
            writer.append_data(numpy.zeros((1080, 1920, 3), dtype=numpy.uint8))
    video = worker._inspect_video(tmp_path / "rollout.mp4")
    assert (video["width"], video["height"], video["fps"], video["frames"]) == (
        1920,
        1080,
        50,
        3,
    )
