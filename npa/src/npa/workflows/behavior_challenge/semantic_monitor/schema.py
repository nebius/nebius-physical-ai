"""Define immutable task-1 TRAIN trace and semantic-label records."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any, Literal

TASK_ID = 1
TASK_NAME = "picking_up_trash"
CAN_SLOTS = frozenset({0, 1, 2})
SHA256 = re.compile(r"[0-9a-f]{64}")
TRACE_KEYS = frozenset(
    {
        "schema",
        "split",
        "source",
        "task_id",
        "task_name",
        "episode_id",
        "frame_index",
        "observation_sha256",
        "native_stage_before",
        "native_stage_proposal",
        "target_can_slot",
        "can_states",
    }
)
CAN_STATE_KEYS = frozenset(
    {
        "slot",
        "near",
        "grasped",
        "inside_target",
        "grasp_attempted",
        "release_attempted",
    }
)

Phase = Literal["seek", "approach", "grasped", "transport", "release", "placed"]
FailureBoundary = Literal["none", "missed_grasp", "dropped", "missed_placement"]
SemanticEvent = Literal[
    "none", "grasp_established", "grasp_retained", "placement_confirmed"
]


@dataclasses.dataclass(frozen=True)
class CanState:
    """Hold privileged state used only by the offline TRAIN labeler.

    Args:
        slot: Stable zero-based can identity.
        near: Whether the robot is within the frozen approach predicate.
        grasped: Whether the simulator reports a grasp/contact.
        inside_target: Whether the can satisfies the BDDL inside predicate.
        grasp_attempted: Whether a grasp attempt completed at this frame.
        release_attempted: Whether a release attempt completed at this frame.
    Returns:
        None.
    Raises:
        None.
    """

    slot: int
    near: bool
    grasped: bool
    inside_target: bool
    grasp_attempted: bool
    release_attempted: bool


@dataclasses.dataclass(frozen=True)
class TrainingTraceFrame:
    """Represent one authorized task-1 TRAIN frame for offline labeling.

    Args:
        episode_id: TRAIN episode identifier.
        frame_index: Zero-based frame index.
        observation_sha256: Digest of the allowed observation payload.
        native_stage_before: Stage before native voting.
        native_stage_proposal: Stage proposed by native voting.
        target_can_slot: Can currently named by the TRAIN plan.
        can_states: Privileged state for exactly three cans.
    Returns:
        None.
    Raises:
        None.
    """

    episode_id: int
    frame_index: int
    observation_sha256: str
    native_stage_before: int
    native_stage_proposal: int
    target_can_slot: int
    can_states: tuple[CanState, ...]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, authorized_episodes: frozenset[int]
    ) -> TrainingTraceFrame:
        """Validate and load an authorized TRAIN trace frame.

        Args:
            value: Candidate trace record.
            authorized_episodes: Exact task-1 TRAIN membership.
        Returns:
            Validated immutable frame.
        Raises:
            TypeError: The record or nested can rows are not mappings.
            ValueError: Provenance, task identity, membership, or fields differ.
        """
        if not isinstance(value, Mapping):
            raise TypeError("TRAIN trace frame must be a mapping")
        if frozenset(value) != TRACE_KEYS:
            raise ValueError("TRAIN trace fields differ")
        _validate_provenance(value)
        episode_id = _integer(value, "episode_id", minimum=0)
        if episode_id not in authorized_episodes:
            raise ValueError("episode is outside authorized TRAIN membership")
        can_states = _can_states(value["can_states"])
        return cls(
            episode_id=episode_id,
            frame_index=_integer(value, "frame_index", minimum=0),
            observation_sha256=_digest(value["observation_sha256"]),
            native_stage_before=_integer(value, "native_stage_before", minimum=0),
            native_stage_proposal=_integer(value, "native_stage_proposal", minimum=0),
            target_can_slot=_slot(value["target_can_slot"]),
            can_states=can_states,
        )


@dataclasses.dataclass(frozen=True)
class BoundaryRequirement:
    """Bind one native stage to a privileged TRAIN semantic boundary.

    Args:
        stage: Native stage whose promotion is being judged.
        can_slot: Can whose event completes the stage.
        event: Required derived semantic event.
    Returns:
        None.
    Raises:
        None.
    """

    stage: int
    can_slot: int
    event: SemanticEvent


@dataclasses.dataclass(frozen=True)
class SemanticLabel:
    """Store one deterministic offline semantic label.

    Args:
        episode_id: TRAIN episode identifier.
        frame_index: Frame within the episode.
        observation_sha256: Digest binding the label to allowed observations.
        can_slot: Can described by the phase and event.
        phase: Derived manipulation phase.
        event: Derived persistent event at this frame.
        grasp_established: Whether a persistent grasp is currently established.
        grasp_retained: Whether an established grasp remains active.
        inside_target: Current privileged inside predicate.
        completed_can_count: Number of persistently placed cans.
        failure_boundary: First local failure type at this frame.
        native_stage_before: Native stage before voting.
        native_stage_proposal: Native proposed stage.
        promotion_label: Whether the proposal should be held or accepted.
    Returns:
        None.
    Raises:
        None.
    """

    episode_id: int
    frame_index: int
    observation_sha256: str
    can_slot: int
    phase: Phase
    event: SemanticEvent
    grasp_established: bool
    grasp_retained: bool
    inside_target: bool
    completed_can_count: int
    failure_boundary: FailureBoundary
    native_stage_before: int
    native_stage_proposal: int
    promotion_label: Literal["hold", "accept"]


def _validate_provenance(value: Mapping[str, Any]) -> None:
    if value["schema"] != "npa.behavior.task1-semantic-trace.v1":
        raise ValueError("TRAIN trace schema differs")
    if value["split"] != "train":
        raise ValueError("semantic labels accept TRAIN split only")
    source = value["source"]
    if not isinstance(source, str) or not source:
        raise ValueError("TRAIN trace source is missing")
    tokens = re.split(r"[^a-z0-9]+", source.lower())
    forbidden = {"dev", "report", "holdout", "validation", "evaluation"}
    if tokens[0] != "train" or any(token in tokens for token in forbidden):
        raise ValueError("TRAIN trace source names a forbidden split")
    if (
        type(value["task_id"]) is not int
        or value["task_id"] != TASK_ID
        or value["task_name"] != TASK_NAME
    ):
        raise ValueError("semantic labels support task 1 only")


def _can_states(value: Any) -> tuple[CanState, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("TRAIN trace requires exactly three can states")
    rows = tuple(_can_state(row) for row in value)
    if {row.slot for row in rows} != CAN_SLOTS:
        raise ValueError("TRAIN trace can slots differ")
    return tuple(sorted(rows, key=lambda row: row.slot))


def _can_state(value: Any) -> CanState:
    if not isinstance(value, Mapping):
        raise TypeError("can state must be a mapping")
    if frozenset(value) != CAN_STATE_KEYS:
        raise ValueError("privileged can-state fields differ")
    booleans = {key: value[key] for key in CAN_STATE_KEYS - {"slot"}}
    if any(type(item) is not bool for item in booleans.values()):
        raise ValueError("privileged can-state predicates must be booleans")
    return CanState(slot=_slot(value["slot"]), **booleans)


def _slot(value: Any) -> int:
    if type(value) is not int or value not in CAN_SLOTS:
        raise ValueError("can slot must be 0, 1, or 2")
    return value


def _integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    item = value[key]
    if type(item) is not int or item < minimum:
        raise ValueError(f"{key} must be an integer at least {minimum}")
    return item


def _digest(value: Any) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError("observation SHA-256 is malformed")
    return value
