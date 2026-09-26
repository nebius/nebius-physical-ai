"""Apply the frozen transition-refresh execution variant to selected RLC."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
NATIVE_EXECUTION = "native"
TRANSITION_REFRESH = "transition-refresh"
EXECUTION_VARIANTS = (NATIVE_EXECUTION, TRANSITION_REFRESH)
LEFT_GRIPPER_SLICE = slice(24, 26)
RIGHT_GRIPPER_SLICE = slice(49, 51)
GRIPPER_MAX_WIDTH = 0.1
TRANSITION_REFRESH_PROVENANCE = MappingProxyType(
    {
        "experiment_source_sha256": (
            "7efdb10f245a380addc84b9d58a922cdaa0572a76bab01980d4b77fb5dab6c32"
        ),
        "experiment_config_sha256": (
            "a4b5ce6278f57a1aaad04549b0cceabf6fc274a5b8fdad53a13d688bb45c05dd"
        ),
        "config_schema": "npa.behavior.rlc-test-time-variant.v1",
        "config_name": "transition_refresh",
    }
)


@dataclasses.dataclass(frozen=True)
class TransitionRefreshConfig:
    """Hold the immutable transition-refresh execution settings.

    Args:
        actions_to_execute: Predicted actions considered by native execution.
        actions_to_keep: Predicted actions retained for rolling inpainting.
        execute_in_n_steps: Simulator steps between predictions.
        history_len: Native stage-prediction history length.
        votes_to_promote: Native stage votes required for promotion.
        gripper_open_threshold: Normalized width above which a gripper is open.
        gripper_closed_threshold: Normalized width below which a gripper is closed.
        gripper_stable_observations: Matching observations required for a change.
    Returns:
        None.
    Raises:
        None.
    """

    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    gripper_open_threshold: float = 0.8
    gripper_closed_threshold: float = -0.8
    gripper_stable_observations: int = 2


CONFIG = TransitionRefreshConfig()


def configure_selected_execution(policy: Any, variant: str) -> Any:
    """Apply an opt-in selected-RLC execution variant.

    Args:
        policy: Newly constructed pinned ``B1KPolicyWrapper``.
        variant: ``native`` or ``transition-refresh``.
    Returns:
        The unchanged native policy or a transition-refresh wrapper.
    Raises:
        ValueError: The variant is unknown or the policy has episode history.
    """

    if variant == NATIVE_EXECUTION:
        return policy
    if variant != TRANSITION_REFRESH:
        raise ValueError("Unsupported selected-RLC execution variant")
    _verify_pristine(policy)
    policy.config = dataclasses.replace(
        policy.config,
        actions_to_execute=CONFIG.actions_to_execute,
        actions_to_keep=CONFIG.actions_to_keep,
        execute_in_n_steps=CONFIG.execute_in_n_steps,
        history_len=CONFIG.history_len,
        votes_to_promote=CONFIG.votes_to_promote,
    )
    policy.prediction_history = deque(maxlen=CONFIG.history_len)
    return RefreshingPolicy(policy)


def _verify_pristine(policy: Any) -> None:
    if policy.last_actions is not None or policy.action_index != 0:
        raise ValueError("Transition refresh requires a pristine upstream wrapper")
    if policy.next_initial_actions is not None or policy.prediction_history:
        raise ValueError("Transition refresh cannot inherit episode history")


class GripperTransitionDetector:
    """Detect stable open/closed changes from permitted proprioception.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear cross-observation classifications.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        self._stable = [None, None]
        self._candidate = [None, None]
        self._streak = [0, 0]

    def observe(self, proprioception: np.ndarray) -> tuple[str, ...]:
        """Return newly stable gripper changes.

        Args:
            proprioception: Finite 61-element official robot proprioception.
        Returns:
            Labels such as ``left:open->closed``.
        Raises:
            ValueError: The proprioception vector is invalid.
        """

        state = np.asarray(proprioception)
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 61-element BEHAVIOR 2026 proprioception")
        events = []
        for index, classification in enumerate(self._classify_grippers(state)):
            event = self._update_one(index, classification)
            if event is not None:
                events.append(event)
        return tuple(events)

    def _classify_grippers(self, state: np.ndarray) -> tuple[str | None, ...]:
        widths = (
            float(state[LEFT_GRIPPER_SLICE].sum()),
            float(state[RIGHT_GRIPPER_SLICE].sum()),
        )
        normalized = (2.0 * width / GRIPPER_MAX_WIDTH - 1.0 for width in widths)
        return tuple(self._classify(value) for value in normalized)

    def _classify(self, value: float) -> str | None:
        if value > CONFIG.gripper_open_threshold:
            return "open"
        if value < CONFIG.gripper_closed_threshold:
            return "closed"
        return None

    def _update_one(self, index: int, classification: str | None) -> str | None:
        if classification is None:
            self._candidate[index] = None
            self._streak[index] = 0
            return None
        if classification != self._candidate[index]:
            self._candidate[index] = classification
            self._streak[index] = 1
        else:
            self._streak[index] += 1
        if self._streak[index] < CONFIG.gripper_stable_observations:
            return None
        previous = self._stable[index]
        self._stable[index] = classification
        if previous is None or previous == classification:
            return None
        side = "left" if index == 0 else "right"
        return f"{side}:{previous}->{classification}"


class RefreshingPolicy:
    """Refresh queued actions at gripper and accepted-stage changes.

    Args:
        policy: Pristine native selected-RLC wrapper.
    Returns:
        None.
    Raises:
        None.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.detector = GripperTransitionDetector()
        self._next_episode_ordinal = 0
        self._active_episode_ordinal: int | None = None
        self._flushed_episode_ordinals: set[int] = set()
        self._telemetry: dict[str, Any] = {}
        self._reset_telemetry()

    def reset(self) -> None:
        """Flush telemetry and reset upstream episode state.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        self.finalize_telemetry()
        self.policy.reset()
        self.detector.reset()
        self._reset_telemetry()

    def act(self, observation: Mapping[str, Any]) -> Any:
        """Return one native action and refresh stale plans at transitions.

        Args:
            observation: Adapter-filtered RGB and local proprioception fields.
        Returns:
            The action produced by the pinned upstream wrapper.
        Raises:
            KeyError: The proprioception field is absent.
            ValueError: The proprioception vector is invalid.
        """

        self._begin_episode()
        self._telemetry["observations"] += 1
        events = self.detector.observe(observation["robot_r1::proprio"])
        if events:
            self._clear_queue("gripper_transition", {"transitions": events})
        stage_before = int(self.policy.current_stage)
        action = self.policy.act(observation)
        stage_after = int(self.policy.current_stage)
        if stage_after != stage_before:
            details = {"from_stage": stage_before, "to_stage": stage_after}
            self._clear_queue("stage_transition", details)
        return action

    def telemetry(self) -> Mapping[str, Any]:
        """Return a read-only episode telemetry snapshot.

        Args:
            None.
        Returns:
            Counts derived from policy state and permitted proprioception.
        Raises:
            None.
        """

        snapshot = dict(self._telemetry)
        snapshot["events"] = tuple(dict(event) for event in snapshot["events"])
        return MappingProxyType(snapshot)

    def finalize_telemetry(self) -> None:
        """Emit the current nonempty episode once.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        ordinal = self._active_episode_ordinal
        if not self._telemetry.get("observations", 0) or ordinal is None:
            return
        if ordinal in self._flushed_episode_ordinals:
            return
        summary = {
            key: value for key, value in self._telemetry.items() if key != "events"
        }
        LOGGER.info("RLC_VARIANT_SUMMARY %s", json.dumps(summary, sort_keys=True))
        self._flushed_episode_ordinals.add(ordinal)

    def _begin_episode(self) -> None:
        if self._active_episode_ordinal is not None:
            return
        ordinal = self._next_episode_ordinal
        self._next_episode_ordinal += 1
        self._active_episode_ordinal = ordinal
        self._telemetry["episode_ordinal"] = ordinal

    def _reset_telemetry(self) -> None:
        self._active_episode_ordinal = None
        self._telemetry = {
            "variant": TRANSITION_REFRESH_PROVENANCE["config_name"],
            "observations": 0,
            "replans_requested": 0,
            "gripper_transitions": 0,
            "stage_transitions": 0,
            "events": [],
        }

    def _clear_queue(self, reason: str, details: Mapping[str, Any]) -> None:
        self.policy.last_actions = None
        self.policy.action_index = 0
        self.policy.next_initial_actions = None
        self._telemetry["replans_requested"] += 1
        self._telemetry[f"{reason}s"] += 1
        event = {"reason": reason, "observation": self._telemetry["observations"]}
        event["episode_ordinal"] = self._active_episode_ordinal
        event.update(details)
        self._telemetry["events"].append(event)
        LOGGER.info("RLC_VARIANT %s", json.dumps(event, sort_keys=True))
