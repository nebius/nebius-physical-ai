"""Build deterministic matched samples around expert gripper transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


ACTION_DIMENSION = 23
GRIPPER_COORDINATES = {"left": 14, "right": 22}
OPEN_COMMAND = 0.0
CLOSED_COMMAND = 1.0


@dataclass(frozen=True, order=True)
class SampleKey:
    """Identify one action-chunk start without changing source cadence.

    Args:
        task_id: Released BEHAVIOR task identifier.
        episode_id: Source episode identifier within the task.
        frame_index: Source frame at which the action chunk starts.
    Returns:
        None.
    Raises:
        None.
    """

    task_id: int
    episode_id: int
    frame_index: int


@dataclass(frozen=True, order=True)
class GripperTransition:
    """Describe one unambiguous persistent expert gripper transition.

    Args:
        side: ``left`` or ``right``.
        command: ``open`` or ``close`` for the new persistent command.
        frame_index: First frame carrying the new command.
    Returns:
        None.
    Raises:
        None.
    """

    side: str
    command: str
    frame_index: int


@dataclass(frozen=True)
class EpisodeActions:
    """Pair one episode identity with its original-cadence expert actions.

    Args:
        task_id: Released BEHAVIOR task identifier.
        episode_id: Source episode identifier within the task.
        actions: Finite float action rows shaped ``(frames, 23)``.
    Returns:
        None.
    Raises:
        None.
    """

    task_id: int
    episode_id: int
    actions: np.ndarray


@dataclass(frozen=True)
class MatchedSamples:
    """Hold matched control and boundary-weighted candidate sample keys.

    Args:
        control: Uniform control draws in shared final-permutation order.
        candidate: Candidate draws with identical anchor-task samples.
        candidate_strata: Candidate stratum for every candidate sample.
    Returns:
        None.
    Raises:
        None.
    """

    control: tuple[SampleKey, ...]
    candidate: tuple[SampleKey, ...]
    candidate_strata: tuple[str, ...]


@dataclass(frozen=True)
class BoundarySamplingPlan:
    """Freeze all choices that define matched boundary sampling.

    Args:
        task_quotas: Exact total sample quota for every task.
        fit_membership: Exact TRAIN-fit episode IDs for every task.
        calibration_membership: Exact held-out TRAIN-calibration IDs for every task.
        focal_task_id: Only task whose candidate sampling changes.
        focal_uniform_quota: Focal samples shared with the control.
        close_quota: Focal samples containing one persistent close.
        open_quota: Focal samples containing one persistent open.
        seed: Nonnegative NumPy PCG64 seed.
        action_horizon: Consecutive source actions in one sample.
        persistence: Stable commands required around a transition.
    Returns:
        None.
    Raises:
        None.
    """

    task_quotas: Mapping[int, int]
    fit_membership: Mapping[int, Sequence[int]]
    calibration_membership: Mapping[int, Sequence[int]]
    focal_task_id: int
    focal_uniform_quota: int
    close_quota: int
    open_quota: int
    seed: int
    action_horizon: int
    persistence: int = 3

    def __post_init__(self) -> None:
        """Copy quotas and memberships into immutable containers.

        Args:
            None.
        Returns:
            None.
        Raises:
            TypeError: The quota input cannot be converted to a dictionary.
        """
        object.__setattr__(
            self, "task_quotas", MappingProxyType(dict(self.task_quotas))
        )
        for name in ("fit_membership", "calibration_membership"):
            membership = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        task_id: tuple(episode_ids)
                        for task_id, episode_ids in membership.items()
                    }
                ),
            )


def persistent_gripper_transitions(
    actions: np.ndarray, *, persistence: int = 3
) -> tuple[GripperTransition, ...]:
    """Index isolated binary gripper changes with stable commands on both sides.

    Args:
        actions: Original-cadence expert action rows shaped ``(frames, 23)``.
        persistence: Equal commands required immediately before and after a change.
    Returns:
        Unambiguous persistent transitions in frame order.
    Raises:
        ValueError: The action schema, values, or persistence differ.
    """
    values = _validated_actions(actions)
    if type(persistence) is not int or persistence < 1:
        raise ValueError("persistence must be a positive integer")
    grippers = values[:, tuple(GRIPPER_COORDINATES.values())]
    changes = grippers[1:] != grippers[:-1]
    transitions: list[GripperTransition] = []
    for frame in range(persistence, len(values) - persistence + 1):
        changed_sides = np.flatnonzero(changes[frame - 1])
        if len(changed_sides) != 1:
            continue
        side_index = int(changed_sides[0])
        before = grippers[frame - persistence : frame, side_index]
        after = grippers[frame : frame + persistence, side_index]
        if np.all(before == before[0]) and np.all(after == after[0]):
            side = tuple(GRIPPER_COORDINATES)[side_index]
            command = "close" if after[0] == CLOSED_COMMAND else "open"
            transitions.append(GripperTransition(side, command, frame))
    return tuple(transitions)


def construct_matched_samples(
    episodes: Sequence[EpisodeActions],
    *,
    plan: BoundarySamplingPlan,
) -> MatchedSamples:
    """Construct matched uniform control and boundary-weighted candidate draws.

    Args:
        episodes: Source episodes containing original-cadence expert actions.
        plan: Predeclared task quotas, strata, geometry, and random seed.
    Returns:
        Exact matched control and candidate sample sequences.
    Raises:
        ValueError: Inputs, quotas, action schema, or boundary pools differ.
    """
    _validate_plan(plan)
    episode_rows = _validated_episodes(episodes, plan=plan)
    uniform, boundaries = _sample_pools(
        episode_rows,
        action_horizon=plan.action_horizon,
        persistence=plan.persistence,
    )
    return _construct_sequences(plan, uniform, boundaries)


def _validate_plan(plan: BoundarySamplingPlan) -> None:
    _validate_sampling_inputs(
        plan.task_quotas, plan.focal_task_id, plan.seed, plan.action_horizon
    )
    for name, quota in (
        ("focal uniform", plan.focal_uniform_quota),
        ("close boundary", plan.close_quota),
        ("open boundary", plan.open_quota),
    ):
        if type(quota) is not int or quota < 0:
            raise ValueError(f"{name} quota must be a nonnegative integer")
    total = plan.focal_uniform_quota + plan.close_quota + plan.open_quota
    if total != plan.task_quotas[plan.focal_task_id]:
        raise ValueError("focal candidate strata must equal its task quota")
    _validate_memberships(plan)


def _construct_sequences(
    plan: BoundarySamplingPlan,
    uniform: Mapping[int, Sequence[SampleKey]],
    boundaries: Mapping[tuple[int, str], Sequence[SampleKey]],
) -> MatchedSamples:
    generator = np.random.Generator(np.random.PCG64(plan.seed))
    control: list[SampleKey] = []
    candidate: list[SampleKey] = []
    strata: list[str] = []
    for task_id in sorted(plan.task_quotas):
        quota = plan.task_quotas[task_id]
        shared = _draw(uniform.get(task_id, ()), quota, generator, "uniform")
        control.extend(shared)
        if task_id != plan.focal_task_id:
            candidate.extend(shared)
            strata.extend(["uniform"] * quota)
            continue
        _extend_focal_candidate(candidate, strata, shared, plan, boundaries, generator)
    order = generator.permutation(len(control))
    return MatchedSamples(
        tuple(control[int(index)] for index in order),
        tuple(candidate[int(index)] for index in order),
        tuple(strata[int(index)] for index in order),
    )


def _extend_focal_candidate(
    candidate: list[SampleKey],
    strata: list[str],
    shared: Sequence[SampleKey],
    plan: BoundarySamplingPlan,
    boundaries: Mapping[tuple[int, str], Sequence[SampleKey]],
    generator: np.random.Generator,
) -> None:
    candidate.extend(shared[: plan.focal_uniform_quota])
    strata.extend(["uniform"] * plan.focal_uniform_quota)
    for command, quota in (("close", plan.close_quota), ("open", plan.open_quota)):
        pool = boundaries.get((plan.focal_task_id, command), ())
        candidate.extend(_draw(pool, quota, generator, f"{command} boundary"))
        strata.extend([command] * quota)


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != ACTION_DIMENSION:
        raise ValueError("expert actions must have shape (frames, 23)")
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("expert actions must contain finite floating-point values")
    grippers = values[:, tuple(GRIPPER_COORDINATES.values())]
    if not np.all((grippers == OPEN_COMMAND) | (grippers == CLOSED_COMMAND)):
        raise ValueError("expert gripper commands must be binary 0.0 or 1.0")
    return values


def _validate_sampling_inputs(
    task_quotas: Mapping[int, int], focal_task_id: int, seed: int, horizon: int
) -> None:
    if not task_quotas or focal_task_id not in task_quotas:
        raise ValueError("task quotas must include the focal task")
    if any(
        type(key) is not int or type(value) is not int or value < 1
        for key, value in task_quotas.items()
    ):
        raise ValueError("task quotas must map integer task IDs to positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("action_horizon must be a positive integer")


def _validated_episodes(
    episodes: Sequence[EpisodeActions], *, plan: BoundarySamplingPlan
) -> tuple[EpisodeActions, ...]:
    identities: set[tuple[int, int]] = set()
    result = []
    for episode in episodes:
        if type(episode.task_id) is not int or type(episode.episode_id) is not int:
            raise ValueError("episode task and episode IDs must be integers")
        identity = (episode.task_id, episode.episode_id)
        if identity in identities:
            raise ValueError("episode identities must be unique")
        identities.add(identity)
        _validated_actions(episode.actions)
        result.append(episode)
    fit = {
        (task_id, episode_id)
        for task_id, episode_ids in plan.fit_membership.items()
        for episode_id in episode_ids
    }
    if identities != fit:
        raise ValueError(
            "supplied episode identities differ from frozen fit membership"
        )
    return tuple(sorted(result, key=lambda item: (item.task_id, item.episode_id)))


def _validate_memberships(plan: BoundarySamplingPlan) -> None:
    tasks = set(plan.task_quotas)
    if set(plan.fit_membership) != tasks or set(plan.calibration_membership) != tasks:
        raise ValueError("fit and calibration memberships must cover the quota tasks")
    for task_id in sorted(tasks):
        fit = plan.fit_membership[task_id]
        calibration = plan.calibration_membership[task_id]
        if not fit or not calibration:
            raise ValueError("fit and calibration memberships must be nonempty")
        if any(type(item) is not int or item < 0 for item in (*fit, *calibration)):
            raise ValueError("episode memberships must contain nonnegative integers")
        if len(set(fit)) != len(fit) or len(set(calibration)) != len(calibration):
            raise ValueError("episode memberships must contain unique IDs")
        if set(fit) & set(calibration):
            raise ValueError("fit and calibration memberships must not overlap")


def _sample_pools(
    episodes: Sequence[EpisodeActions], *, action_horizon: int, persistence: int
) -> tuple[
    dict[int, tuple[SampleKey, ...]], dict[tuple[int, str], tuple[SampleKey, ...]]
]:
    uniform: dict[int, list[SampleKey]] = {}
    boundaries: dict[tuple[int, str], list[SampleKey]] = {}
    for episode in episodes:
        actions = _validated_actions(episode.actions)
        transitions = persistent_gripper_transitions(actions, persistence=persistence)
        grippers = actions[:, tuple(GRIPPER_COORDINATES.values())]
        raw_change_frames = tuple(
            int(frame) + 1
            for frame in np.flatnonzero(np.any(grippers[1:] != grippers[:-1], axis=1))
        )
        for start in range(max(0, len(actions) - action_horizon + 1)):
            key = SampleKey(episode.task_id, episode.episode_id, start)
            uniform.setdefault(episode.task_id, []).append(key)
            inside = [
                item
                for item in transitions
                if start <= item.frame_index < start + action_horizon
            ]
            raw_inside = [
                frame
                for frame in raw_change_frames
                if start <= frame < start + action_horizon
            ]
            if len(inside) == 1 and raw_inside == [inside[0].frame_index]:
                boundary = (episode.task_id, inside[0].command)
                boundaries.setdefault(boundary, []).append(key)
    frozen_uniform = {key: tuple(value) for key, value in uniform.items()}
    frozen_boundaries = {key: tuple(value) for key, value in boundaries.items()}
    return frozen_uniform, frozen_boundaries


def _draw(
    pool: Sequence[SampleKey], count: int, generator: np.random.Generator, name: str
) -> list[SampleKey]:
    if type(count) is not int or count < 0:
        raise ValueError(f"{name} quota must be a nonnegative integer")
    if count and not pool:
        raise ValueError(f"{name} sample pool is empty")
    result: list[SampleKey] = []
    while len(result) < count:
        order = generator.permutation(len(pool))
        remaining = count - len(result)
        result.extend(pool[int(index)] for index in order[:remaining])
    return result
