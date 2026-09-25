"""Specify a read-only stage-promotion monitor boundary for future integration."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import numpy as np

CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
RGB_KEYS = frozenset(f"robot_r1::robot_r1:{camera}:Camera:0::rgb" for camera in CAMERAS)
DEPTH_KEYS = frozenset(
    f"robot_r1::robot_r1:{camera}:Camera:0::depth_linear" for camera in CAMERAS
)
PROPRIO_KEY = "robot_r1::proprio"
ALLOWED_KEYS = RGB_KEYS | DEPTH_KEYS | {PROPRIO_KEY}
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclasses.dataclass(frozen=True)
class PromotionRequest:
    """Carry one environment's unbatched history and native stage proposal.

    Args:
        observations: Oldest-to-newest onboard observation history.
        current_stage: Native stage before the proposed transition.
        proposed_stage: Stage proposed by the native voter.
    Returns:
        None.
    Raises:
        None.
    """

    observations: tuple[Mapping[str, Any], ...]
    current_stage: int
    proposed_stage: int

    def validated(self, *, maximum_history: int = 8) -> PromotionRequest:
        """Return this request after checking the evaluation boundary.

        Args:
            maximum_history: Maximum permitted observation history length.
        Returns:
            The same immutable request.
        Raises:
            ValueError: Stages, history length, or observations are invalid.
        """
        if not 0 < len(self.observations) <= maximum_history:
            raise ValueError("monitor observation history length differs")
        if type(self.current_stage) is not int or self.current_stage < 0:
            raise ValueError("current stage must be a nonnegative integer")
        if type(self.proposed_stage) is not int or self.proposed_stage not in {
            self.current_stage,
            self.current_stage + 1,
        }:
            raise ValueError("monitor accepts only hold or next-stage proposals")
        for observation in self.observations:
            validate_evaluation_observation(observation)
        return self


@dataclasses.dataclass(frozen=True)
class PromotionDecision:
    """Return a proposed hold/accept decision without mutating policy state.

    Args:
        action: Proposed stage action.
        completion_probability: Model completion score in the probability range.
        retention_probability: Model grasp-retention score in that range.
        artifact_sha256: Exact immutable monitor artifact identity.
    Returns:
        None.
    Raises:
        None.
    """

    action: Literal["hold", "accept"]
    completion_probability: float
    retention_probability: float
    artifact_sha256: str

    def validated(self) -> PromotionDecision:
        """Return this decision after validating values and provenance.

        Args:
            None.
        Returns:
            The same immutable decision.
        Raises:
            ValueError: The action, probabilities, or artifact identity is invalid.
        """
        if self.action not in {"hold", "accept"}:
            raise ValueError("monitor decision action differs")
        probabilities = (self.completion_probability, self.retention_probability)
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities
        ):
            raise ValueError("monitor probabilities must be finite and within [0, 1]")
        if SHA256.fullmatch(self.artifact_sha256) is None:
            raise ValueError("monitor artifact SHA-256 is malformed")
        return self


class PromotionMonitor(Protocol):
    """Describe the future inference interface; it has no policy mutation API.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    def decide(self, request: PromotionRequest) -> PromotionDecision:
        """Propose whether the caller should accept a native promotion.

        Args:
            request: Validated allowed-input history and native proposal.
        Returns:
            Read-only promotion recommendation with artifact provenance.
        Raises:
            ValueError: The request or loaded artifact is invalid.
        """
        ...


def validate_evaluation_observation(observation: Mapping[str, Any]) -> None:
    """Reject privileged, batched, or malformed fields at the monitor boundary.

    Args:
        observation: Candidate RGB, depth, and proprioception mapping.
    Returns:
        None.
    Raises:
        TypeError: The observation is not a mapping.
        ValueError: Keys, arrays, shapes, dtypes, or values are invalid.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("monitor observation must be a mapping")
    keys = frozenset(observation)
    if PROPRIO_KEY not in keys or not RGB_KEYS.issubset(keys):
        raise ValueError("monitor observation lacks required onboard inputs")
    if not keys.issubset(ALLOWED_KEYS):
        raise ValueError("monitor observation contains privileged or unknown inputs")
    _validate_proprio(observation[PROPRIO_KEY])
    for key in RGB_KEYS:
        _validate_rgb(key, observation[key])
    for key in keys & DEPTH_KEYS:
        _validate_depth(key, observation[key])


def _validate_proprio(value: Any) -> None:
    state = np.asarray(value)
    if state.shape != (61,) or not np.issubdtype(state.dtype, np.number):
        raise ValueError("monitor requires numeric 61-element proprioception")
    if not np.isfinite(state).all():
        raise ValueError("monitor proprioception must be finite")


def _validate_rgb(key: str, value: Any) -> None:
    image = np.asarray(value)
    camera = next(camera for camera in CAMERAS if camera in key)
    size = 720 if camera == "zed_link" else 480
    if image.shape not in {(size, size, 3), (size, size, 4)}:
        raise ValueError("monitor RGB shape differs from official cameras")
    if image.dtype != np.uint8:
        raise ValueError("monitor RGB must be uint8")


def _validate_depth(key: str, value: Any) -> None:
    depth = np.asarray(value)
    camera = next(camera for camera in CAMERAS if camera in key)
    size = 720 if camera == "zed_link" else 480
    if depth.shape not in {(size, size), (size, size, 1)}:
        raise ValueError("monitor depth shape differs from official cameras")
    if not np.issubdtype(depth.dtype, np.floating) or not np.isfinite(depth).all():
        raise ValueError("monitor depth must contain finite floating-point values")
