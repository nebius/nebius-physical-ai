"""Derive task-1 semantic events deterministically from authorized TRAIN traces."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

from .schema import (
    BoundaryRequirement,
    CanState,
    FailureBoundary,
    Phase,
    SemanticEvent,
    SemanticLabel,
    TrainingTraceFrame,
)

SPLIT_DOMAIN = "npa-task1-semantic-monitor-v1:"


def deterministic_episode_split(
    episodes: Sequence[int], *, calibration_count: int
) -> dict[str, tuple[int, ...]]:
    """Create an order-independent episode-level fit/calibration split.

    Args:
        episodes: Exact authorized TRAIN episode identifiers.
        calibration_count: Number of hash-ranked calibration episodes.
    Returns:
        Sorted fit and calibration memberships.
    Raises:
        ValueError: Episode identities or calibration size are invalid.
    """
    if not episodes or any(type(item) is not int or item < 0 for item in episodes):
        raise ValueError("TRAIN episode identifiers must be nonnegative integers")
    if len(set(episodes)) != len(episodes):
        raise ValueError("TRAIN episode identifiers must be unique")
    if type(calibration_count) is not int or not 0 < calibration_count < len(episodes):
        raise ValueError("calibration count must divide a nonempty TRAIN membership")
    ranked = sorted(episodes, key=_episode_rank)
    calibration = tuple(sorted(ranked[:calibration_count]))
    fit = tuple(sorted(ranked[calibration_count:]))
    return {"fit": fit, "calibration": calibration}


def collect_episode_labels(
    records: Iterable[Mapping[str, object]],
    *,
    authorized_episodes: frozenset[int],
    boundary_requirements: Sequence[BoundaryRequirement],
    persistence_frames: int = 2,
) -> tuple[SemanticLabel, ...]:
    """Collect labels for one chronological task-1 TRAIN episode.

    Args:
        records: Offline trace mappings with privileged TRAIN predicates.
        authorized_episodes: Exact allowed task-1 TRAIN episode membership.
        boundary_requirements: Immutable stage-to-event promotion rules.
        persistence_frames: Consecutive frames required for grasp and placement.
    Returns:
        One semantic label per input frame.
    Raises:
        ValueError: Records, ordering, persistence, or stage rules are invalid.
        TypeError: A trace record has the wrong container type.
    """
    if type(persistence_frames) is not int or persistence_frames < 2:
        raise ValueError("persistence requires at least two consecutive frames")
    requirements = _requirements(boundary_requirements)
    frames = tuple(
        TrainingTraceFrame.from_mapping(row, authorized_episodes=authorized_episodes)
        for row in records
    )
    _validate_episode(frames)
    state = _EpisodeState()
    labels = []
    for frame in frames:
        labels.append(state.observe(frame, requirements, persistence_frames))
    return tuple(labels)


class _EpisodeState:
    def __init__(self) -> None:
        self.grasp_streak = [0, 0, 0]
        self.inside_streak = [0, 0, 0]
        self.established = [False, False, False]
        self.placed = [False, False, False]

    def observe(
        self,
        frame: TrainingTraceFrame,
        requirements: Mapping[int, BoundaryRequirement],
        persistence: int,
    ) -> SemanticLabel:
        active = frame.can_states[frame.target_can_slot]
        observations = {
            row.slot: self._update(row, persistence) for row in frame.can_states
        }
        event, failure = observations[active.slot]
        phase = self._phase(active)
        return SemanticLabel(
            episode_id=frame.episode_id,
            frame_index=frame.frame_index,
            observation_sha256=frame.observation_sha256,
            can_slot=active.slot,
            phase=phase,
            event=event,
            grasp_established=self.established[active.slot],
            grasp_retained=self.established[active.slot] and active.grasped,
            inside_target=active.inside_target,
            completed_can_count=sum(self.placed),
            failure_boundary=failure,
            native_stage_before=frame.native_stage_before,
            native_stage_proposal=frame.native_stage_proposal,
            promotion_label=self._promotion_label(frame, requirements),
        )

    def _promotion_label(
        self,
        frame: TrainingTraceFrame,
        requirements: Mapping[int, BoundaryRequirement],
    ) -> str:
        if frame.native_stage_proposal != frame.native_stage_before + 1:
            return "hold"
        requirement = requirements.get(frame.native_stage_before)
        if requirement is None or requirement.can_slot != frame.target_can_slot:
            return "hold"
        if not self._boundary_satisfied(requirement):
            return "hold"
        return "accept"

    def _boundary_satisfied(self, requirement: BoundaryRequirement) -> bool:
        slot = requirement.can_slot
        if requirement.event == "placement_confirmed":
            return self.placed[slot]
        return self.established[slot] and self.grasp_streak[slot] > 0

    def _update(
        self, active: CanState, persistence: int
    ) -> tuple[SemanticEvent, FailureBoundary]:
        slot = active.slot
        was_established = self.established[slot]
        was_placed = self.placed[slot]
        self.grasp_streak[slot] = self.grasp_streak[slot] + 1 if active.grasped else 0
        placement_signal = active.inside_target and not active.grasped
        self.inside_streak[slot] = (
            self.inside_streak[slot] + 1 if placement_signal else 0
        )
        if was_placed and (not active.inside_target or active.grasped):
            self.placed[slot] = False
        event: SemanticEvent = "none"
        if not was_established and self.grasp_streak[slot] >= persistence:
            self.established[slot] = True
            event = "grasp_established"
        elif was_established and active.grasped:
            event = "grasp_retained"
        if not was_placed and self.inside_streak[slot] >= persistence:
            self.placed[slot] = True
            event = "placement_confirmed"
        failure = _failure(active, was_established)
        if not active.grasped:
            self.established[slot] = False
        return event, failure

    def _phase(self, active: CanState) -> Phase:
        if self.placed[active.slot]:
            return "placed"
        if self.established[active.slot] and active.grasped:
            return "transport"
        if self.established[active.slot] and active.inside_target:
            return "release"
        if active.grasped:
            return "grasped"
        if active.near:
            return "approach"
        return "seek"


def _failure(active: CanState, was_established: bool) -> FailureBoundary:
    if active.release_attempted and not active.inside_target:
        return "missed_placement"
    if was_established and not active.grasped and not active.inside_target:
        return "dropped"
    if active.grasp_attempted and not active.grasped:
        return "missed_grasp"
    return "none"


def _requirements(
    values: Sequence[BoundaryRequirement],
) -> dict[int, BoundaryRequirement]:
    if not values:
        raise ValueError("at least one boundary requirement is required")
    result = {}
    for value in values:
        if type(value.stage) is not int or value.stage < 0:
            raise ValueError("boundary stage must be a nonnegative integer")
        if type(value.can_slot) is not int or value.can_slot not in {0, 1, 2}:
            raise ValueError("boundary can slot differs")
        if value.event not in {
            "grasp_established",
            "grasp_retained",
            "placement_confirmed",
        }:
            raise ValueError("boundary event differs")
        if value.stage in result:
            raise ValueError("boundary stages must be unique")
        result[value.stage] = value
    return result


def _validate_episode(frames: tuple[TrainingTraceFrame, ...]) -> None:
    if not frames:
        raise ValueError("TRAIN episode trace is empty")
    episodes = {frame.episode_id for frame in frames}
    if len(episodes) != 1:
        raise ValueError("collector accepts exactly one TRAIN episode")
    indices = [frame.frame_index for frame in frames]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("TRAIN episode frames must be unique and chronological")
    if any(after != before + 1 for before, after in zip(indices, indices[1:])):
        raise ValueError("persistence requires consecutive TRAIN frames")


def _episode_rank(episode: int) -> tuple[str, int]:
    digest = hashlib.sha256(f"{SPLIT_DOMAIN}{episode}".encode()).hexdigest()
    return digest, episode
