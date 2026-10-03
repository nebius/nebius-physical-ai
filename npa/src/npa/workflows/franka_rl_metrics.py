"""Measure sustained Franka object-goal tracking and rank validation checkpoints."""

from __future__ import annotations

import math

import numpy as np


class PlacementMetrics:
    """Accumulate a reset-bounded lift-and-hold success event.

    Args:
        count: Number of independent simulator environments.
        recipe: Sealed geometric and stability thresholds.
    Returns:
        None.
    Raises:
        ValueError: An observation contains invalid geometry.
    """

    def __init__(self, count: int, recipe: dict) -> None:
        self.recipe = recipe
        self.active = np.ones(count, dtype=bool)
        self.streak = np.zeros(count, dtype=np.int64)
        self.longest = np.zeros(count, dtype=np.int64)
        self.lifted = np.zeros(count, dtype=bool)
        self.closest = np.full(count, np.inf)
        self.success = np.zeros(count, dtype=bool)
        self.steps = np.zeros(count, dtype=np.int64)
        self.domain_exit = np.zeros(count, dtype=bool)

    def update(self, distance, speed, height, done, domain_exit=None) -> None:
        """Count only physical observations before an automatic reset.

        Args:
            distance: Object-goal distance in metres per environment.
            speed: Object speed in metres/second per environment.
            height: Object height above the environment origin in metres.
            done: Automatic-reset flags for this step.
            domain_exit: Optional pre-reset task-domain departure flags.
        Returns:
            None.
        Raises:
            ValueError: Geometry is nonfinite, negative, or has a wrong shape.
        """
        distance, speed, height = [
            np.asarray(x, dtype=float) for x in (distance, speed, height)
        ]
        if any(
            x.shape != self.active.shape or not np.isfinite(x).all()
            for x in (distance, speed, height)
        ):
            raise ValueError(
                "Franka geometry must be finite and match the environment batch"
            )
        if (distance < 0).any() or (speed < 0).any():
            raise ValueError("Distance and speed cannot be negative")
        done = np.asarray(done, dtype=bool)
        if done.shape != self.active.shape:
            raise ValueError("Reset flags must match the environment batch")
        self._domain_failures(domain_exit)
        self.active &= ~done
        stable = (
            (distance < self.recipe["success_distance_m"])
            & (speed < self.recipe["maximum_object_speed_m_s"])
            & (height > self.recipe["minimum_object_height_m"])
        )
        self.streak = np.where(
            self.active, np.where(stable, self.streak + 1, 0), self.streak
        )
        self.longest = np.maximum(self.longest, self.streak)
        self.closest = np.where(
            self.active, np.minimum(self.closest, distance), self.closest
        )
        self.lifted |= self.active & (height > self.recipe["minimum_object_height_m"])
        self.success |= self.active & (self.streak >= self.recipe["stable_steps"])
        self.steps += self.active

    def _domain_failures(self, flags) -> None:
        if flags is None:
            return
        flags = np.asarray(flags, dtype=bool)
        if flags.shape != self.active.shape:
            raise ValueError("Task-domain flags must match the environment batch")
        self.domain_exit |= self.active & flags
        self.active &= ~self.domain_exit
        self.success &= ~self.domain_exit
        self.lifted &= ~self.domain_exit


def rank_checkpoint(rows: list[dict]) -> tuple[float, float, int]:
    """Rank a checkpoint using validation success, distance, then earlier iteration.

    Args:
        rows: All validation episodes for one exact checkpoint.
    Returns:
        Lexicographic ranking key with higher values preferred.
    Raises:
        ValueError: Rows are empty, mixed, nonfinite, or include test episodes.
    """
    if not rows or any(row["split"] != "validation" for row in rows):
        raise ValueError("Checkpoint selection requires validation episodes only")
    identities = {(row["checkpoint_sha256"], row["iteration"]) for row in rows}
    if len(identities) != 1 or len({row["env_index"] for row in rows}) != len(rows):
        raise ValueError("Validation checkpoint identities or episode indices disagree")
    distances = [float(row["closest_distance_m"]) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in distances):
        raise ValueError("Validation distances must be finite and nonnegative")
    return (
        sum(row["success"] for row in rows) / len(rows),
        -sum(distances) / len(rows),
        -int(rows[0]["iteration"]),
    )
