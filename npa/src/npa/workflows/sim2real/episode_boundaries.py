"""Preserve simulator resets between sparse observations and exclude ambiguous credit."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

EPISODE_BOUNDARY_SCHEMA = "npa.sim2real.episode_boundary.v1"


class EpisodeBoundaries:
    """Track independent environment generations after the rollout's initial reset.

    Args:
        count: Number of parallel simulator environments.
    Returns:
        None.
    Raises:
        ValueError: The environment count is invalid.
    """

    def __init__(self, count: int):
        if type(count) is not int or count < 1:
            raise ValueError("episode tracking requires a positive environment count")
        self.generations = [0] * count
        self._pending: list[list[dict[str, int]]] = [[] for _ in range(count)]
        self._current = [False] * count
        self._step = -1

    def advance(self, done: list[bool], sim_step: int) -> None:
        """Record every step, including resets between sampled actions.

        Args:
            done: Per-environment termination or truncation flags.
            sim_step: Consecutive zero-based simulator step.
        Returns:
            None.
        Raises:
            ValueError: The step or environment coverage is inconsistent.
        """
        if (type(sim_step) is not int or sim_step != self._step + 1
                or len(done) != len(self.generations)
                or any(type(value) is not bool for value in done)):
            raise ValueError("episode tracking requires consecutive complete steps")
        self._step = sim_step
        self._current = list(done)
        for index, reset in enumerate(done):
            if not reset:
                continue
            generation = self.generations[index]
            self.generations[index] += 1
            self._pending[index].append({
                "sim_step": sim_step, "terminated_episode_id": generation,
                "next_episode_id": generation + 1,
            })

    def sample(self, index: int) -> dict[str, Any]:
        """Snapshot boundary evidence without discarding other environments' events.

        Args:
            index: Simulator environment index.
        Returns:
            Detached metadata for the sampled action and returned state.
        Raises:
            IndexError: The environment index is out of range.
        """
        return {
            "schema": EPISODE_BOUNDARY_SCHEMA,
            "simulator_episode_id": self.generations[index],
            "action_episode_id": self.generations[index] - int(self._current[index]),
            "reset_events": deepcopy(self._pending[index]),
            "reset_on_current_step": self._current[index],
            "action_outcome_valid": not self._current[index],
            "temporal_credit_valid": not self._pending[index],
        }

    def sampled(self) -> None:
        """Clear pending events after all environments' sampled rows are archived.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self._pending = [[] for _ in self.generations]

    def frame_episode(self, index: int) -> int | None:
        """Identify a rendered episode only when autoreset cannot leave stale pixels.

        Args:
            index: Simulator environment index.
        Returns:
            The episode generation, or None on the current reset step.
        Raises:
            IndexError: The environment index is out of range.
        """
        return None if self._current[index] else self.generations[index]


def _index(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_events(boundary: dict[str, Any], sim_step: int) -> None:
    events = boundary.get("reset_events")
    if not isinstance(events, list):
        raise ValueError("episode boundary requires reset events")
    last_step = -1
    generation = boundary["simulator_episode_id"] - len(events)
    for event in events:
        if (not isinstance(event, dict) or set(event) != {
                "sim_step", "terminated_episode_id", "next_episode_id"}
                or any(not _index(value) for value in event.values())
                or not last_step < event["sim_step"] <= sim_step
                or event["terminated_episode_id"] != generation
                or event["next_episode_id"] != generation + 1):
            raise ValueError("episode boundary reset events are not contiguous")
        last_step = event["sim_step"]
        generation += 1
    current = bool(events and events[-1]["sim_step"] == sim_step)
    expected = {
        "reset_on_current_step": current, "action_outcome_valid": not current,
        "temporal_credit_valid": not events,
    }
    if any(type(boundary.get(key)) is not bool or boundary[key] != value
           for key, value in expected.items()):
        raise ValueError("episode boundary validity contradicts reset events")
    if boundary["action_episode_id"] != boundary["simulator_episode_id"] - int(current):
        raise ValueError("episode boundary action belongs to a different generation")


def validate_episode_boundary(row: dict[str, Any]) -> dict[str, Any]:
    """Reject missing, malformed, or contradictory action boundary evidence.

    Args:
        row: An action or evaluation row with its native simulator step.
    Returns:
        The validated boundary mapping.
    Raises:
        ValueError: Boundary evidence is missing or inconsistent.
    """
    if not isinstance(row, dict):
        raise ValueError("episode boundary requires an action object")
    boundary = row.get("episode_boundary")
    if (not isinstance(boundary, dict)
            or set(boundary) != {
                "schema", "simulator_episode_id", "action_episode_id", "reset_events",
                "reset_on_current_step", "action_outcome_valid", "temporal_credit_valid"}
            or boundary.get("schema") != EPISODE_BOUNDARY_SCHEMA
            or not _index(row.get("sim_step"))
            or not _index(boundary.get("simulator_episode_id"))
            or not _index(boundary.get("action_episode_id"))):
        raise ValueError("episode boundary requires native step and generation identifiers")
    _validate_events(boundary, row["sim_step"])
    return boundary


def validate_episode_sequence(rows: list[dict[str, Any]]) -> None:
    """Prove all sampled intervals account for their generation transitions.

    Args:
        rows: Complete action sequence; input order is immaterial.
    Returns:
        None.
    Raises:
        ValueError: Events are lost, repeated, or assigned outside their interval.
    """
    for row in rows:
        validate_episode_boundary(row)
    previous_step, generation = -1, 0
    for row in sorted(rows, key=lambda item: item["sim_step"]):
        boundary = row["episode_boundary"]
        events = boundary["reset_events"]
        if (row["sim_step"] <= previous_step
                or boundary["simulator_episode_id"] != generation + len(events)
                or any(event["sim_step"] <= previous_step for event in events)):
            raise ValueError("episode boundary disagrees with the sampled interval")
        previous_step = row["sim_step"]
        generation = boundary["simulator_episode_id"]


def temporal_credit_valid(row: dict[str, Any]) -> bool:
    """Honor explicit boundary evidence while retaining legacy signal readability.

    Args:
        row: A temporal signal or an archived legacy evaluation row.
    Returns:
        Whether the row is eligible for training credit.
    Raises:
        ValueError: Present boundary evidence is malformed.
    """
    if "episode_boundary" not in row:
        return True
    return validate_episode_boundary(row)["temporal_credit_valid"]
