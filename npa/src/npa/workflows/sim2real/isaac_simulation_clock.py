"""Track physical elapsed time independently of camera playback cadence."""

from __future__ import annotations

import math


class SimulationClock:
    """Track completed environment steps from the start of one rollout.

    Args:
        step_seconds: Isaac's environment step duration, including decimation.
    Returns:
        A clock starting immediately before the first rollout step.
    Raises:
        ValueError: The environment step duration is not finite and positive.
    """

    def __init__(self, step_seconds: float) -> None:
        try:
            duration = float(step_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError("Isaac step_dt must be finite and positive") from error
        if (
            isinstance(step_seconds, bool)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError("Isaac step_dt must be finite and positive")
        self.step_seconds = duration
        self.completed_steps = 0

    def advance(self) -> None:
        """Record one successfully completed env.step call.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        self.completed_steps += 1

    def sample(self) -> dict[str, float | int | str]:
        """Describe the current physical time without advancing the simulation.

        Args:
            None.
        Returns:
            Physical time, its origin, step duration, and completed step count.
        Raises:
            None.
        """

        return {
            "timestamp_seconds": self.completed_steps * self.step_seconds,
            "timestamp_timebase": "simulation_elapsed_since_rollout_start",
            "simulation_step_seconds": self.step_seconds,
            "completed_simulation_steps": self.completed_steps,
        }
