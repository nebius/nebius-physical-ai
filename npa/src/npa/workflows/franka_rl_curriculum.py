"""Advance training distributions from completed training episodes, never evaluation scores."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class TrainingCurriculum:
    """Track outcome-driven difficulty using disjoint windows of completed episodes.

    Args:
        minimum_episodes: Completed episodes required to assess a difficulty.
        success_threshold: Fraction of successful episodes required to advance.
        initial_fraction: Initial fraction of the full training distribution.
        increment: Difficulty increase after a successful window.
    Returns:
        A curriculum with an auditable history and no evaluation inputs.
    Raises:
        ValueError: A setting is outside its valid range.
    """

    minimum_episodes: int
    success_threshold: float
    initial_fraction: float = 0.25
    increment: float = 0.15
    fraction: float = field(init=False)
    episodes: int = field(default=0, init=False)
    successes: int = field(default=0, init=False)
    total_episodes: int = field(default=0, init=False)
    history: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        values = (self.success_threshold, self.initial_fraction, self.increment)
        if (
            type(self.minimum_episodes) is not int
            or self.minimum_episodes < 1
            or not all(math.isfinite(value) and 0 < value <= 1 for value in values)
        ):
            raise ValueError("Invalid training curriculum settings")
        self.fraction = self.initial_fraction

    def observe(self, successes: int, episodes: int) -> None:
        """Accumulate episodes generated at the current difficulty and assess a window.

        Args:
            successes: Episodes satisfying the sealed stable-goal predicate.
            episodes: Completed episodes at the current difficulty.
        Returns:
            None.
        Raises:
            ValueError: Counts are invalid.
        """
        if (
            type(successes) is not int
            or type(episodes) is not int
            or not 0 <= successes <= episodes
        ):
            raise ValueError("Invalid completed training episode counts")
        self.episodes += episodes
        self.successes += successes
        self.total_episodes += episodes
        if self.episodes < self.minimum_episodes:
            return
        rate = self.successes / self.episodes
        previous = self.fraction
        if rate >= self.success_threshold:
            self.fraction = min(1.0, round(self.fraction + self.increment, 10))
        self.history.append(
            {
                "fraction": previous,
                "next_fraction": self.fraction,
                "episodes": self.episodes,
                "successes": self.successes,
                "success_rate": rate,
                "total_episodes": self.total_episodes,
            }
        )
        self.episodes = 0
        self.successes = 0


def training_ranges(
    fraction: float,
    reset_ranges: dict,
    goal_ranges: dict,
    easy_goal_height: tuple[float, float],
) -> tuple[dict, dict]:
    """Interpolate training resets and goals while preserving the exact full distribution.

    Args:
        fraction: Curriculum difficulty in (0, 1].
        reset_ranges: Original object reset offsets.
        goal_ranges: Original commanded object position ranges.
        easy_goal_height: Initial vertical goal range, already above the lift threshold.
    Returns:
        Independent object-reset and goal-position range dictionaries.
    Raises:
        ValueError: Difficulty is invalid.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Training difficulty must be in (0, 1]")
    resets = dict(reset_ranges)
    goals = dict(goal_ranges)
    for axis in ("x", "y"):
        resets[axis] = tuple(value * fraction for value in reset_ranges[axis])
        low, high = goal_ranges["pos_" + axis]
        middle, radius = (low + high) / 2, (high - low) * fraction / 2
        goals["pos_" + axis] = (middle - radius, middle + radius)
    goals["pos_z"] = tuple(
        easy + fraction * (full - easy)
        for easy, full in zip(easy_goal_height, goal_ranges["pos_z"])
    )
    if fraction == 1:
        return dict(reset_ranges), dict(goal_ranges)
    return resets, goals
