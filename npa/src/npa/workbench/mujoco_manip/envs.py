"""MuJoCo environments for contact-rich manipulation tasks.

:class:`MujocoManipEnv` exposes the minimal interface the evaluation harness
needs — ``reset(seed)``, ``step(action)``, ``action_space`` /
``observation_space`` shapes, and ``info["success"]`` — over real MuJoCo
simulation.  The ``mujoco`` import stays inside the constructor so importing
this module (and the CLI surface above it) works on machines without MuJoCo
installed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from npa.workbench.mujoco_manip.scenes import (
    PEG_INSERTION,
    REACH,
    SCREW_DRIVING,
    TASKS,
    TASK_SPECS,
    _f,
    _xyz,
    build_scene,
)

FRAME_SKIP = 5
#: Per-control-step target deltas for [dx, dy, dz, dyaw].
ACTION_SCALE = np.array([0.02, 0.02, 0.02, 0.15], dtype=np.float64)
#: Indices of the actuated dofs (wx, wy, wz, wyaw) in qpos/qvel.
ACTUATED = (0, 1, 2, 3)


class MujocoManipError(RuntimeError):
    """Raised when a MuJoCo manipulation invariant is not met."""


class MujocoManipEnv:
    """A MuJoCo manipulation task with a harness-compatible interface."""

    def __init__(self, task: str, *, max_steps: int = 500) -> None:
        if task not in TASKS:
            raise MujocoManipError(
                f"unknown mujoco_manip task {task!r}; known tasks: {TASKS}"
            )
        if max_steps < 1:
            raise MujocoManipError("max_steps must be >= 1")
        try:
            import mujoco
        except ImportError as exc:
            raise MujocoManipError(
                "the mujoco package is required to run mujoco_manip tasks "
                "(pip install mujoco); the environment interface stays "
                "importable without it."
            ) from exc

        self._mujoco = mujoco
        self.task = task
        self.spec = TASK_SPECS[task]
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_string(build_scene(task))
        self.data = mujoco.MjData(self.model)
        self._renderer: Any = None

        self._tip_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )
        if self._tip_site < 0:
            raise MujocoManipError(f"task {task!r} scene has no 'tip' site")

        nq = int(self.model.nq)
        self._obs_dim = 2 * nq + 6
        self._target = np.zeros(4, dtype=np.float64)
        self._steps = 0
        # The wrist reference pose is baked into the scene XML; the slide/hinge
        # joints start at zero displacement from it.
        self._start_qpos = np.zeros(nq, dtype=np.float64)
        self._ctrl_low = np.array([-0.3, -0.3, -0.28, -np.inf])
        self._ctrl_high = np.array([0.3, 0.3, 0.06, np.inf])

    # -- harness interface -------------------------------------------------
    @property
    def action_space(self) -> Mapping[str, Any]:
        return {
            "shape": (4,),
            "low": -np.ones(4, dtype=np.float64),
            "high": np.ones(4, dtype=np.float64),
            "names": ("dx", "dy", "dz", "dyaw"),
        }

    @property
    def observation_space(self) -> Mapping[str, Any]:
        return {"shape": (self._obs_dim,), "dtype": "float32"}

    def reset(self, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else 0)
        self._mujoco.mj_resetData(self.model, self.data)
        qpos = self._start_qpos.copy()
        # Small seeded start-pose jitter so episodes differ by seed.
        qpos[0] += float(rng.uniform(-0.005, 0.005))
        qpos[1] += float(rng.uniform(-0.005, 0.005))
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self._target = qpos[list(ACTUATED)].copy()
        self.data.ctrl[:] = self._target
        self._mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._obs()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.shape != (4,):
            raise MujocoManipError(
                f"expected a 4-D action in [-1, 1], got shape {a.shape}"
            )
        current = self.data.qpos[list(ACTUATED)]
        self._target = np.clip(
            current + a * ACTION_SCALE, self._ctrl_low, self._ctrl_high
        )
        self.data.ctrl[:] = self._target
        for _ in range(FRAME_SKIP):
            self._mujoco.mj_step(self.model, self.data)
        self._steps += 1
        success = self._success()
        done = bool(success) or self._steps >= self.max_steps
        obs = self._obs()
        reward = self._reward(success)
        info = {
            "success": bool(success),
            "steps": self._steps,
            "tip_pos": self._tip_pos().tolist(),
            "tilt": float(self._tilt()),
        }
        return obs, reward, done, info

    def render_rgb(
        self, width: int = 256, height: int = 256
    ) -> np.ndarray | None:
        """Render an RGB frame; None when headless rendering is unavailable."""
        try:
            if self._renderer is None:
                self._renderer = self._mujoco.Renderer(
                    self.model, width, height
                )
            self._renderer.update_scene(self.data)
            frame: np.ndarray = self._renderer.render()
            return frame
        except Exception:
            return None

    # -- task internals ----------------------------------------------------
    def _tip_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self._tip_site], dtype=np.float64)

    def _tilt(self) -> float:
        """Tilt of the tool away from vertical, in radians (0 if rigid)."""
        if self.task != SCREW_DRIVING:
            return 0.0
        tx = float(self.data.qpos[4])
        ty = float(self.data.qpos[5])
        return float(np.hypot(tx, ty))

    def _success(self) -> bool:
        tip = self._tip_pos()
        if self.task == PEG_INSERTION:
            depth = _f(self.spec, "plate_top_z") - tip[2]
            radial = float(np.hypot(tip[0], tip[1]))
            return bool(
                depth >= _f(self.spec, "insert_depth")
                and radial <= _f(self.spec, "radial_tol")
            )
        if self.task == SCREW_DRIVING:
            depth = _f(self.spec, "socket_top_z") - tip[2]
            radial = float(np.hypot(tip[0], tip[1]))
            return bool(
                depth >= _f(self.spec, "drive_depth")
                and radial <= _f(self.spec, "radial_tol")
                and self._tilt() <= _f(self.spec, "tilt_tol")
            )
        # REACH
        goal = np.array(_xyz(self.spec, "goal"))
        return bool(np.linalg.norm(tip - goal) <= _f(self.spec, "reach_tol"))

    def _reward(self, success: bool) -> float:
        tip = self._tip_pos()
        goal = np.array(_xyz(self.spec, "goal"))
        reward = -float(np.linalg.norm(tip - goal))
        if self.task == SCREW_DRIVING:
            reward -= self._tilt()
        if success:
            reward += 10.0
        return reward

    def _obs(self) -> np.ndarray:
        qpos = np.array(self.data.qpos, dtype=np.float64)
        qvel = np.array(self.data.qvel, dtype=np.float64)
        tip = self._tip_pos()
        goal = np.array(_xyz(self.spec, "goal"))
        return np.concatenate([qpos, qvel, tip, goal]).astype(np.float32)


def make_env(task: str, *, max_steps: int = 500) -> MujocoManipEnv:
    """Build the MuJoCo environment for *task* (imports mujoco here)."""
    return MujocoManipEnv(task, max_steps=max_steps)


__all__ = [
    "FRAME_SKIP",
    "MujocoManipEnv",
    "MujocoManipError",
    "make_env",
    "REACH",
    "PEG_INSERTION",
    "SCREW_DRIVING",
    "TASKS",
]
