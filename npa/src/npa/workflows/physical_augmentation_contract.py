"""Define reproducible physics cases, executed-action semantics, and lift acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
ACTION_NAMES = ["tcp_x_m", "tcp_y_m", "tcp_z_m", "qx", "qy", "qz", "qw", "gripper"]
CONDITIONS = {
    "nominal": {"offset_xy_m": [0.0, 0.0], "mass_kg": 0.1, "friction": 1.0},
    "displaced": {"offset_xy_m": [0.04, 0.08], "mass_kg": 0.1, "friction": 1.0},
    "heavy": {"offset_xy_m": [0.0, 0.0], "mass_kg": 0.2, "friction": 1.0},
    "slippery": {"offset_xy_m": [0.0, 0.0], "mass_kg": 0.1, "friction": 0.25},
}


def make_recipe(run_id: str, seed: int, episodes: int, steps: int) -> dict:
    """Seal the public task before collecting any outcomes.

    Args:
        run_id: Operator's run identifier.
        seed: Nonnegative reset seed shared across physical conditions.
        episodes: Attempts per condition, including failed attempts.
        steps: Control steps after which an unfinished attempt fails.
    Returns:
        JSON-compatible experiment contract.
    Raises:
        ValueError: The requested experiment has invalid counts or identity.
    """
    from npa.workflows.franka_rl_validity import validity_contract

    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or any(type(value) is not int for value in (seed, episodes, steps))
        or seed < 0
        or episodes < 1
        or steps < 1
    ):
        raise ValueError("Run identity, nonnegative seed, and positive counts required")
    return {
        "schema": "npa.physical-augmentation.recipe.v1",
        "run_id": run_id,
        "task": TASK,
        "seed": seed,
        "episodes_per_condition": episodes,
        "episode_steps": steps,
        "conditions": deepcopy(CONDITIONS),
        "simulation_validity": validity_contract(),
        **_semantics(),
    }


def _semantics() -> dict:
    return {
        "controller": "measured-state-cartesian-lift-v8",
        "gripper_servo": {"native_velocity_fraction": 0.25, "effort_limit_n": 20.0},
        "physics_solver": {
            "type": "TGS",
            "external_forces_every_iteration": True,
            "position_iterations": 16,
            "velocity_iterations": 1,
            "physics_dt_s": 0.0025,
            "control_dt_s": 0.02,
        },
        "presentation": {"scene": "studio-table-v2", "width": 1280, "height": 720},
        "tcp_contract": {"body": "panda_hand", "offset_m": [0.0, 0.0, 0.107]},
        "action_names": list(ACTION_NAMES),
        "action_semantics": "absolute TCP pose in robot root frame, xyzw; gripper +1 open/-1 close",
        "alignment": "rgb[t], state[t], action[t], next_state[t]; timestamps are simulation time",
        "success": {
            "lift_m": 0.10,
            "max_speed_m_s": 0.03,
            "hold_steps": 30,
            "max_tcp_distance_m": 0.06,
            "max_xy_drift_m": 0.04,
        },
        "physical_robot_tested": False,
    }


def read_recipe(path: Path) -> dict:
    """Validate the complete public reference contract before native execution.

    Args:
        path: Sealed recipe file, materialized through the stage checksum contract.
    Returns:
        Validated settings with fixed condition identifiers and action semantics.
    Raises:
        ValueError: The artifact changes task, physics, acceptance, or path identifiers.
        OSError: The file is unavailable.
    """
    recipe = json.loads(path.read_text())
    try:
        expected = make_recipe(
            recipe["run_id"],
            recipe["seed"],
            recipe["episodes_per_condition"],
            recipe["episode_steps"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid physical augmentation recipe") from error
    if recipe != expected:
        raise ValueError(
            "Physical augmentation recipe differs from the supported contract"
        )
    return recipe


@dataclass
class LiftController:
    """Generate fresh, speed-bounded Cartesian actions from measured scene state.

    Args:
        dt: Simulation control interval in seconds.
    Returns:
        Stateful controller; no learned policy or world action model is loaded.
    Raises:
        ValueError: An observation is nonfinite or has the wrong shape.
    """

    dt: float
    phase: int = 0
    dwell: int = 0
    grasp: np.ndarray | None = None
    commanded_position: np.ndarray | None = None
    commanded_quaternion: np.ndarray | None = None

    def action(
        self, tcp: np.ndarray, cube: np.ndarray, quaternion: np.ndarray
    ) -> np.ndarray:
        """Return the next commanded pose and gripper setting.

        Args:
            tcp: Measured tool position in robot root coordinates.
            cube: Measured object position in the same coordinates.
            quaternion: Measured tool orientation in XYZW order.
        Returns:
            Eight-dimensional absolute IK action.
        Raises:
            ValueError: Observations or controller timing are invalid.
        """
        if (
            tcp.shape != (3,)
            or cube.shape != (3,)
            or quaternion.shape != (4,)
            or self.dt <= 0
            or not np.isfinite([*tcp, *cube, *quaternion, self.dt]).all()
            or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4)
        ):
            raise ValueError("Invalid controller observation")
        _, aligned = _downward_command(quaternion, self.dt)
        target = (self.grasp if self.grasp is not None else cube).copy()
        target[2] += (0.12, 0.0, 0.0, 0.18)[self.phase]
        delta = target - tcp
        close = np.linalg.norm(delta) < 0.008 and aligned
        self.dwell = self.dwell + 1 if close else 0
        commanded_phase = self.phase
        if self.phase < 3 and self.dwell * self.dt >= (0.15, 0.15, 0.6)[self.phase]:
            self.phase += 1
            self.dwell = 0
            if self.phase == 2:
                self.grasp = cube.copy()
        position, orientation = self._pose_command(tcp, target, quaternion)
        return np.asarray(
            [*position, *orientation, 1.0 if commanded_phase < 2 else -1.0],
            dtype=np.float32,
        )

    def _pose_command(self, tcp, target, quaternion) -> tuple[np.ndarray, np.ndarray]:
        # Integrate bounded setpoints; resetting to measured pose on every call
        # makes actuator tracking error erase most of each intended movement.
        if self.commanded_position is None:
            self.commanded_position = tcp.copy()
            self.commanded_quaternion = quaternion.copy()
        delta = target - self.commanded_position
        self.commanded_position += delta * min(
            1.0, 0.15 * self.dt / max(np.linalg.norm(delta), 1e-9)
        )
        self.commanded_quaternion, _ = _downward_command(
            self.commanded_quaternion, self.dt
        )
        return self.commanded_position, self.commanded_quaternion


def _downward_command(quaternion: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
    """Rotate toward a downward grasp at at most 0.8 radians per second."""
    current = quaternion / np.linalg.norm(quaternion)
    target = np.array([1.0, 0.0, 0.0, 0.0])
    if np.dot(current, target) < 0:
        target = -target
    angle = float(np.arccos(np.clip(np.dot(current, target), -1, 1)))
    if angle < 1e-8:
        return target, True
    fraction = min(1.0, 0.8 * dt / (2 * angle))
    command = (
        np.sin((1 - fraction) * angle) * current + np.sin(fraction * angle) * target
    ) / np.sin(angle)
    return command, 2 * angle < 0.04


def accepted_steps(
    arrays: dict, initial_cube: np.ndarray, criteria: dict
) -> np.ndarray:
    """Recompute success from measured post-action object and tool telemetry.

    Args:
        arrays: Aligned actions, object poses/velocities, and tool poses.
        initial_cube: Settled initial object XYZ.
        criteria: Sealed geometric and stability criteria.
    Returns:
        Boolean acceptance predicate per transition.
    Raises:
        KeyError: Required measured channels or criteria are absent.
    """
    cube, tcp = arrays["next_object"], arrays["next_tcp"]
    return (
        (cube[:, 2] - initial_cube[2] >= criteria["lift_m"])
        & (np.linalg.norm(arrays["next_velocity"], axis=1) <= criteria["max_speed_m_s"])
        & (np.linalg.norm(cube - tcp, axis=1) <= criteria["max_tcp_distance_m"])
        & (
            np.linalg.norm(cube[:, :2] - initial_cube[:2], axis=1)
            <= criteria["max_xy_drift_m"]
        )
        & (arrays["actions"][:, -1] == -1.0)
    )


def longest_hold(mask: np.ndarray) -> int:
    """Count the longest uninterrupted sequence of accepted measured states.

    Args:
        mask: One-dimensional per-transition success predicate.
    Returns:
        Maximum consecutive count, including zero for an empty mask.
    Raises:
        None.
    """
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
