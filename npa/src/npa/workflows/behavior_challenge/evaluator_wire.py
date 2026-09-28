"""Adapt versioned BEHAVIOR evaluator messages to singleton policy servers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__:
    from .evaluator_versions import UPSTREAM_COMMITS, require_supported_upstream
else:
    from evaluator_versions import UPSTREAM_COMMITS, require_supported_upstream

PROPRIOCEPTION = "robot_r1::proprio"
CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")
RGB_KEYS = tuple(f"robot_r1::robot_r1:{camera}:Camera:0::rgb" for camera in CAMERAS)
POLICY_OBSERVATIONS = (PROPRIOCEPTION, *RGB_KEYS)
ACTION_CHUNK_REQUEST = "__action_chunk_size__"


def _array(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an array leaf") from error
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} is not a numeric array leaf")
    return array


def _legacy_observations(message: dict) -> dict[str, np.ndarray]:
    selected = {}
    for key in POLICY_OBSERVATIONS:
        array = _array(message[key], key)
        expected_rank = 1 if key == PROPRIOCEPTION else 3
        if array.ndim != expected_rank:
            raise ValueError("Legacy evaluator observation is not unbatched")
        selected[key] = array
    return selected


def _singleton_observations(message: dict) -> dict[str, np.ndarray]:
    arrays = {key: _array(value, key) for key, value in message.items()}
    if any(array.ndim == 0 or array.shape[0] != 1 for array in arrays.values()):
        raise ValueError("Current evaluator requires one consistent batch row")
    return {key: arrays[key][0] for key in POLICY_OBSERVATIONS}


@dataclass(frozen=True)
class EvaluatorWire:
    """Bind one websocket connection to an exact supported evaluator revision.

    Args:
        upstream_commit: Exact official evaluator Git commit for this connection.
    Raises:
        ValueError: The evaluator revision is unsupported.
    """

    upstream_commit: str

    def __post_init__(self) -> None:
        require_supported_upstream(self.upstream_commit)

    def is_reset(self, message: object) -> bool:
        """Recognize only the exact evaluator reset message.

        Args:
            message: Decoded websocket payload.
        Returns:
            True only for ``{"reset": True}``.
        Raises:
            ValueError: A reset-bearing payload is malformed.
        """
        if not isinstance(message, dict):
            raise ValueError("Evaluator message must be a dictionary")
        if "reset" not in message:
            return False
        if set(message) != {"reset"} or message["reset"] is not True:
            raise ValueError("Evaluator reset message differs")
        return True

    def observation_for_policy(self, message: object) -> dict[str, np.ndarray]:
        """Return only RGB and proprioception in the frozen policy wire shape.

        Args:
            message: Decoded version-bound evaluator observation.
        Returns:
            Unbatched singleton policy inputs with privileged leaves discarded.
        Raises:
            KeyError: A required policy observation is absent.
            ValueError: The message version, batch, or chunk contract differs.
        """
        if not isinstance(message, dict) or "reset" in message:
            raise ValueError("Evaluator observation message differs")
        if ACTION_CHUNK_REQUEST in message:
            raise ValueError("Action chunk requests are unsupported")
        if self.upstream_commit == UPSTREAM_COMMITS["3.9.3"]:
            return _singleton_observations(message)
        return _legacy_observations(message)

    def action_for_evaluator(self, value: object) -> np.ndarray:
        """Return one finite policy action in the versioned evaluator shape.

        Args:
            value: Validated unbatched singleton policy action.
        Returns:
            Shape ``(23,)`` for v3.9.2 or ``(1, 23)`` for v3.9.3.
        Raises:
            ValueError: The policy action is malformed or nonfinite.
        """
        action = _array(value, "Policy action")
        numeric = np.issubdtype(action.dtype, np.number)
        if action.shape != (23,) or not numeric or not np.isfinite(action).all():
            raise ValueError("Policy action must be one finite 23-vector")
        if self.upstream_commit == UPSTREAM_COMMITS["3.9.3"]:
            return action[None]
        return action
