"""Capture real Arena frames before automatic reset and retain their state binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _annotator_ready(env: Any) -> bool:
    capture = getattr(getattr(env, "video_recorder", None), "_capture", None)
    annotator = getattr(capture, "_rgb_annotator", None)
    if annotator is None:
        return True
    data = annotator.get_data()
    if isinstance(data, dict):
        data = data.get("data", [])
    return np.asarray(data).size > 0


def _capture_frame(env: Any) -> np.ndarray:
    while True:
        frame = np.asarray(env.render())
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError("Arena simulator renderer returned no RGB uint8 frame")
        if _annotator_ready(env):
            return frame.copy()
        if env.sim.is_stopped():
            raise RuntimeError(
                "Arena simulator stopped before its RGB annotator was ready"
            )


def _frame_record(
    frame: np.ndarray, directory: Path, name: str, step: int
) -> dict[str, Any]:
    from PIL import Image

    path = directory / name
    Image.fromarray(frame).save(path, format="PNG")
    return {
        "path": name,
        "action_step": step,
        "source_frame_index": step - 1 if step else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "width": frame.shape[1],
        "height": frame.shape[0],
    }


def _step_tensor(step: int, env: Any) -> Any:
    import torch

    return torch.tensor([[step]], dtype=torch.int64, device=env.device)


class _VideoCaptureMethods:
    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise RuntimeError("Arena simulator video capture requires one environment")
        self._directory = Path(env.cfg.recorders.dataset_export_dir_path)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._evidence = {
            "schema": "npa.isaac-arena.video-capture.v1",
            "capture_phase": "recorder_post_step_before_autoreset",
            "initial": None,
            "terminals": [],
            "captured_action_steps": 0,
        }
        env._npa_video_frame = None

    def _write_evidence(self) -> None:
        path = self._directory / "simulator-video-evidence.json"
        path.write_text(json.dumps(self._evidence, indent=2) + "\n", encoding="utf-8")

    def record_post_reset(self, env_ids: Any) -> tuple[str | None, Any]:
        if self._step or self._evidence["initial"] is not None:
            return None, None
        frame = _capture_frame(self._env)
        self._evidence["initial"] = _frame_record(
            frame, self._directory, "simulator-initial.png", 0
        )
        self._write_evidence()
        return "npa_video/initial_action_step", _step_tensor(0, self._env)

    def record_post_step(self) -> tuple[str, Any]:
        self._step += 1
        self._env._npa_video_frame = _capture_frame(self._env)
        self._evidence["captured_action_steps"] = self._step
        self._write_evidence()
        return "npa_video/action_step", _step_tensor(self._step, self._env)

    def record_pre_reset(self, env_ids: Any) -> tuple[str | None, Any]:
        if not self._step:
            return None, None
        index = len(self._evidence["terminals"])
        terminal = _frame_record(
            self._env._npa_video_frame,
            self._directory,
            f"simulator-episode-{index}-terminal.png",
            self._step,
        )
        self._evidence["terminals"].append(terminal)
        self._write_evidence()
        return "npa_video/terminal_action_step", _step_tensor(self._step, self._env)


def _capture_recorder_type() -> type:
    from isaaclab.managers.recorder_manager import RecorderTerm

    class _SimulatorVideoRecorder(_VideoCaptureMethods, RecorderTerm):
        pass

    return _SimulatorVideoRecorder


def configure_video_capture(env_cfg: Any) -> None:
    """Add a pre-reset frame recorder after Isaac Sim has started.

    Args:
        env_cfg: Arena manager configuration with its real metric recorder.
    Returns:
        None; adds a recorder term to the existing configuration.
    Raises:
        RuntimeError: No metric recorder exists for the environment.
    """
    from isaaclab.managers.recorder_manager import RecorderTermCfg

    if env_cfg.recorders is None:
        raise RuntimeError("Arena video capture requires the task's metric recorder")
    env_cfg.recorders.npa_video = RecorderTermCfg(class_type=_capture_recorder_type())


def cached_frame_env(env: Any) -> Any:
    """Wrap Gym video capture to return the real frame from before automatic reset.

    Args:
        env: Base Arena environment with configure_video_capture installed.
    Returns:
        A Gym wrapper whose render method returns the cached post-action frame.
    Raises:
        RuntimeError: A frame is requested before the first action capture.
    """
    import gymnasium as gym

    class _CachedFrameEnv(gym.Wrapper):
        def render(self) -> np.ndarray:
            frame = self.unwrapped._npa_video_frame
            if frame is None:
                raise RuntimeError(
                    "Arena video requested a frame before its first action"
                )
            return frame.copy()

    return _CachedFrameEnv(env)
