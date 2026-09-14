"""Verify simulator video captures terminal state before the base environment resets."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from npa.workbench.isaac_arena import simulator_video


class _RecorderTerm:
    def __init__(self, cfg, env):
        self._env = env


class _Wrapper:
    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env

    def step(self, action):
        return self.env.step(action)


@pytest.fixture
def simulator_modules(monkeypatch):
    managers = ModuleType("isaaclab.managers.recorder_manager")
    managers.RecorderTerm = _RecorderTerm
    managers.RecorderTermCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(
        sys.modules, "isaaclab.managers", ModuleType("isaaclab.managers")
    )
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    gym = ModuleType("gymnasium")
    gym.Wrapper = _Wrapper
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setattr(
        simulator_video, "_step_tensor", lambda step, env: np.array([[step]])
    )


class _AutoResetEnvironment:
    def __init__(self, directory: Path, *, warmup_renders: int = 0):
        self.cfg = SimpleNamespace(
            recorders=SimpleNamespace(dataset_export_dir_path=str(directory))
        )
        self.num_envs = 1
        self.device = "cpu"
        self.state = 0
        self.renders = 0
        self.action_steps = []
        self.warmup_renders = warmup_renders
        self.sim = SimpleNamespace(is_stopped=lambda: False)
        annotator = SimpleNamespace(get_data=self._annotator_data)
        self.video_recorder = SimpleNamespace(
            _capture=SimpleNamespace(_rgb_annotator=annotator)
        )
        simulator_video.configure_video_capture(self.cfg)
        recorder_cfg = self.cfg.recorders.npa_video
        self.recorder = recorder_cfg.class_type(recorder_cfg, self)

    def _pixels(self):
        return np.full((24, 32, 3), 40 + self.state * 40, dtype=np.uint8)

    def _annotator_data(self):
        return self._pixels() if self.renders >= self.warmup_renders else np.array([])

    def render(self):
        self.renders += 1
        return (
            self._pixels()
            if self.renders >= self.warmup_renders
            else np.zeros((24, 32, 3), dtype=np.uint8)
        )

    def reset(self):
        self.recorder.record_pre_reset([0])
        self.state = 0
        self.recorder.record_post_reset([0])

    def step(self, action):
        self.state += action
        key, value = self.recorder.record_post_step()
        assert key == "npa_video/action_step"
        self.action_steps.append(int(value[0, 0]))
        terminal = self.state >= 3
        if terminal:
            self.recorder.record_pre_reset([0])
            self.state = 0
            self.recorder.record_post_reset([0])
        return self.state, terminal


def test_video_records_terminal_frame_before_autoreset(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    frames = []
    for _ in range(3):
        wrapper.step(1)
        frames.append(wrapper.render())
    assert env.state == 0
    assert [int(frame[0, 0, 0]) for frame in frames] == [80, 120, 160]
    assert env.action_steps == [1, 2, 3]
    assert env.renders == 4
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    assert evidence["captured_action_steps"] == 3
    assert evidence["capture_phase"] == "recorder_post_step_before_autoreset"
    assert evidence["initial"]["action_step"] == 0
    assert evidence["terminals"][0]["action_step"] == 3
    assert evidence["terminals"][0]["source_frame_index"] == 2
    final = np.asarray(Image.open(tmp_path / evidence["terminals"][0]["path"]))
    np.testing.assert_array_equal(final, frames[-1])
    assert (
        evidence["terminals"][0]["decoded_rgb_sha256"]
        == hashlib.sha256(final.tobytes()).hexdigest()
    )


def test_initial_renderer_warmup_adds_no_action_or_video_frames(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, warmup_renders=3)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    assert env.renders == 3
    assert env.action_steps == []
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    initial = np.asarray(Image.open(tmp_path / evidence["initial"]["path"]))
    assert int(initial[0, 0, 0]) == 40
    assert evidence["captured_action_steps"] == 0
    with pytest.raises(RuntimeError, match="before its first action"):
        wrapper.render()
    wrapper.step(1)
    assert int(wrapper.render()[0, 0, 0]) == 80
    assert env.action_steps == [1]


def test_outer_video_reads_copy_without_rerendering(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    wrapper.step(1)
    rendered = wrapper.render()
    rendered[:] = 0
    assert int(wrapper.render()[0, 0, 0]) == 80
    assert env.renders == 2


def test_capture_requires_real_renderer_frame(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path)
    env.render = lambda: None
    with pytest.raises(RuntimeError, match="no RGB uint8 frame"):
        env.reset()
    assert not (tmp_path / "simulator-initial.png").exists()


def test_capture_setup_preserves_existing_metric_configuration(
    simulator_modules, tmp_path: Path
) -> None:
    success_recorder = object()
    cfg = SimpleNamespace(
        recorders=SimpleNamespace(
            success=success_recorder, dataset_export_dir_path=str(tmp_path)
        )
    )
    simulator_video.configure_video_capture(cfg)
    assert cfg.recorders.success is success_recorder
    assert (
        cfg.recorders.npa_video.class_type.__mro__[1]
        is simulator_video._VideoCaptureMethods
    )
    with pytest.raises(RuntimeError, match="requires the task's metric recorder"):
        simulator_video.configure_video_capture(SimpleNamespace(recorders=None))
