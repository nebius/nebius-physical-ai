"""Transactional k16 controller around an injected native policy wrapper."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .extraction import ACTION_DIMENSION, CHECK_OFFSET, HORIZON, build_features


@dataclass(frozen=True)
class Proposal:
    """Hold an uncommitted native proposal and transaction evidence.

    Args:
        actions: Proposed full 32-action queue.
        transformed_state: State at the proposal boundary.
        features: Exact 1,233-feature gate row.
        rng_before: Native policy RNG state before inference.
        rng_after: Native policy RNG state after inference.
        inference_result: Native inference output mapping.
    Returns:
        None.
    Raises:
        None.
    """

    actions: np.ndarray
    transformed_state: np.ndarray
    features: np.ndarray
    rng_before: Any
    rng_after: Any
    inference_result: dict[str, Any]


class ConditionalQueueController:
    """Apply proposal, decision, and commit at the native k16 boundary.

    Args:
        wrapper: Native receding-horizon wrapper.
        normalize: Absolute-to-normalized action transform.
        decide: Feature-row refresh decision callback.
    Returns:
        None.
    Raises:
        ValueError: If wrapper semantics differ.
    """

    def __init__(
        self,
        wrapper: Any,
        normalize: Callable[[np.ndarray], np.ndarray],
        decide: Callable[[np.ndarray], bool],
    ) -> None:
        """Bind an inspected wrapper and explicit scientific callbacks.

        Args:
            wrapper: Native receding-horizon wrapper.
            normalize: Absolute-to-normalized action transform.
            decide: Feature-row refresh decision callback.
        Raises:
            ValueError: If the wrapper lacks required native semantics.
        """
        self.wrapper, self.normalize, self.decide = wrapper, normalize, decide
        self._start_state: np.ndarray | None = None
        self._emitted = 0
        self._verify_wrapper()

    def reset(self) -> None:
        """Reset upstream and local episode transaction state.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.wrapper.reset()
        self._start_state, self._emitted = None, 0

    def act(self, observation: dict[str, Any]) -> Any:
        """Emit one action and transact once at each k16 boundary.

        Args:
            observation: Native serving observation.
        Returns:
            The next native wrapper action.
        Raises:
            ValueError: If proposal inputs differ.
        """
        if not self.wrapper.action_queue:
            self._start_state = self._state(self._batch(observation))
            result = self.wrapper.act(observation)
            self._emitted = 1
            return result
        if self._emitted == CHECK_OFFSET:
            proposal = self.propose(observation)
            try:
                refresh = bool(self.decide(proposal.features.copy()))
            except Exception:
                self.wrapper.policy._rng = proposal.rng_before
                raise
            self.commit(proposal, refresh)
        result = self.wrapper.act(observation)
        self._emitted += 1
        return result

    def propose(self, observation: dict[str, Any]) -> Proposal:
        """Infer speculatively without changing the committed action queue.

        Args:
            observation: Native serving observation at k16.
        Returns:
            Uncommitted proposal including before/after RNG states.
        Raises:
            ValueError: If called outside the exact k16 boundary.
        """
        if (
            self._emitted != CHECK_OFFSET
            or len(self.wrapper.action_queue) != CHECK_OFFSET
        ):
            raise ValueError("proposal is only valid at the full k16 boundary")
        batch, rng_before = self._batch(observation), self.wrapper.policy._rng
        state = self._state(batch)
        try:
            result = self.wrapper.policy.infer(batch)
            actions = np.ascontiguousarray(result.get("actions"))
            if (
                actions.shape != (HORIZON, ACTION_DIMENSION)
                or not np.isfinite(actions).all()
            ):
                raise ValueError("native proposal representation differs")
            features = self._features(actions, state)
            return Proposal(
                actions, state, features, rng_before, self.wrapper.policy._rng, result
            )
        except Exception:
            self.wrapper.policy._rng = rng_before
            raise

    def commit(self, proposal: Proposal, refresh: bool) -> None:
        """Commit a proposal or roll its RNG mutation back exactly.

        Args:
            proposal: Proposal returned by ``propose``.
            refresh: Whether to replace the remaining native queue.
        Returns:
            None.
        Raises:
            None.
        """
        if refresh:
            self.wrapper.action_queue = deque(np.array(proposal.actions, copy=True))
            self.wrapper.last_action = copy.deepcopy(proposal.inference_result)
            self._start_state = proposal.transformed_state.copy()
            self._emitted = 0
        else:
            self.wrapper.policy._rng = proposal.rng_before

    def _features(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self._start_state is None:
            raise ValueError("controller start state is absent")
        continued = np.asarray(tuple(self.wrapper.action_queue))
        if continued.shape != (CHECK_OFFSET, ACTION_DIMENSION):
            raise ValueError("committed queue representation differs")
        cont = np.asarray(self.normalize(continued), dtype=np.float32)
        fresh = np.asarray(self.normalize(actions[:CHECK_OFFSET]), dtype=np.float32)
        return build_features(cont, fresh, self._start_state, state)

    def _batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        processed = self.wrapper.process_obs(observation)
        images = processed["observation"]
        if images.shape[-1] != 3:
            images = np.transpose(images, (0, 1, 3, 4, 2))
        batch = {
            "observation/egocentric_camera": images[0, 0],
            "observation/wrist_image_left": images[0, 1],
            "observation/wrist_image_right": images[0, 2],
            "observation/state": processed["proprio"][0],
            "prompt": self.wrapper.task_prompt,
        }
        if "observation/egocentric_depth" in processed:
            batch["observation/egocentric_depth"] = processed[
                "observation/egocentric_depth"
            ][0]
        return batch

    def _state(self, batch: dict[str, Any]) -> np.ndarray:
        transformed = self.wrapper.policy._input_transform(copy.deepcopy(batch))
        value = np.asarray(transformed.get("state"), dtype=np.float32)
        if (
            value.shape != (HORIZON,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("native transformed state differs")
        return value.copy()

    def _verify_wrapper(self) -> None:
        policy = getattr(self.wrapper, "policy", None)
        if (
            self.wrapper.control_mode != "receeding_horizon"
            or self.wrapper.max_len != HORIZON
        ):
            raise ValueError("controller requires 32-action receding horizon")
        if (
            self.wrapper.fine_grained_level != 0
            or getattr(policy, "_is_pytorch_model", None) is not False
        ):
            raise ValueError("controller requires the native JAX policy path")
        if not hasattr(policy, "_rng") or not hasattr(policy, "_input_transform"):
            raise ValueError("native policy transaction fields are absent")
