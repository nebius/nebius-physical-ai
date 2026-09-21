"""Apply opt-in execution policies to a pristine pinned RLC wrapper."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any

_NATIVE = "native"
_FINAL_STAGE_BACKTRACK = "final-stage-backtrack"
_ADAPTIVE_SHORT_CHUNK = "adaptive-short-chunk"
_ADAPTIVE_TRANSITION_REFRESH = "adaptive-short-chunk-transition-refresh"
_STOCK_VARIANTS = (
    _NATIVE,
    _FINAL_STAGE_BACKTRACK,
    _ADAPTIVE_SHORT_CHUNK,
    _ADAPTIVE_TRANSITION_REFRESH,
)
_NATIVE_EXECUTION = (26, 4, 20, 3, 2)
_NATIVE_WRAPPER_SHA256 = (
    "59711eefc2829cfee9db30d0985794dd7a12024843e3c7271fcc8364f5b5c97e"
)
_OVERLAY_WRAPPER_SHA256 = (
    "1b75af0bb28b815929ab941ffca8f00cc7adf291d8d5e4ab098c6524a4dd5453"
)
_NATIVE_UPDATE_AST_SHA256 = (
    "d7ad9b3f31acbd2a19a11dc70ba435a0a9df8da1d524b8c4265bca5031657768"
)
_FINAL_EVENT = "RLC_FINAL_STAGE_EVENT "
_FINAL_SUMMARY = "RLC_FINAL_STAGE_SUMMARY "
_HORIZON_EVENT = "RLC_HORIZON_EVENT "
_HORIZON_SUMMARY = "RLC_HORIZON_SUMMARY "
_LOGGER = logging.getLogger(__name__)
_EXPERIMENT_IDENTITIES = {
    (_FINAL_STAGE_BACKTRACK, False): (
        "bb250a21ada69ba7ef9a845d1fe5bda3b42bd0c706c83ec0b4c7477578d0d993",
        "f8b214ecda7c63acff3209b6d50f08f080d0882477821261afe1f18e4cff9b8d",
    ),
    (_FINAL_STAGE_BACKTRACK, True): (
        "59b421f0ecf8c4f82387df606b6e5decf50fcdb6d8e4c066a9235e7f23299238",
        "4a223f433e1d6165bcf8398ae770f3eb264baa9b24408ece1c1fb89c08bf6f36",
    ),
    (_ADAPTIVE_SHORT_CHUNK, False): (
        "bb250a21ada69ba7ef9a845d1fe5bda3b42bd0c706c83ec0b4c7477578d0d993",
        "3ec700a31a2bf83a9522b9e44ca54086fc356a07152536a298fcd7036c454151",
    ),
}


def _canonical_ast(value: Any) -> Any:
    if isinstance(value, ast.AST):
        fields = []
        for name in value._fields:
            child = getattr(value, name, None)
            if name == "type_params" and child == []:
                continue
            fields.append([name, _canonical_ast(child)])
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return [_canonical_ast(item) for item in value]
    return value


def native_update_ast_sha256(path: Path) -> str:
    """Hash the native stage update across Python 3.11 and 3.12.

    Args:
        path: Python source containing ``B1KPolicyWrapper``.
    Returns:
        SHA-256 of the version-neutral ``update_current_stage`` AST.
    Raises:
        OSError: The source cannot be read.
        StopIteration: The expected class or method is absent.
        SyntaxError: The source is invalid Python.
    """
    tree = ast.parse(path.read_bytes())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "B1KPolicyWrapper"
    )
    method = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_current_stage"
    )
    encoded = json.dumps(
        _canonical_ast(method), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(policy: Any) -> tuple[str, str]:
    source = Path(inspect.getsourcefile(type(policy)) or "")
    if not source.is_file():
        raise ValueError("Live native wrapper source is unavailable")
    return hashlib.sha256(source.read_bytes()).hexdigest(), native_update_ast_sha256(
        source
    )


def _verify_native_policy(policy: Any) -> None:
    source_sha256, method_sha256 = _source_identity(policy)
    if source_sha256 not in {_NATIVE_WRAPPER_SHA256, _OVERLAY_WRAPPER_SHA256}:
        raise ValueError("Live native wrapper source differs")
    if method_sha256 != _NATIVE_UPDATE_AST_SHA256:
        raise ValueError("Live native stage-update AST differs")
    execution = (
        policy.config.actions_to_execute,
        policy.config.actions_to_keep,
        policy.config.execute_in_n_steps,
        policy.config.history_len,
        policy.config.votes_to_promote,
    )
    if execution != _NATIVE_EXECUTION:
        raise ValueError("Native RLC execution defaults differ")
    if policy.last_actions is not None or policy.next_initial_actions is not None:
        raise ValueError("Execution variant requires a pristine native wrapper")
    if policy.action_index != 0 or policy.prediction_history:
        raise ValueError("Execution variant cannot inherit episode state")


def configure_execution(policy: Any, variant: str) -> Any:
    """Apply one optional execution policy while leaving native as the default.

    Args:
        policy: Newly created pinned ``B1KPolicyWrapper``.
        variant: One of the stock RLC execution variants in ``_STOCK_VARIANTS``.
    Returns:
        The unchanged native policy or the requested execution wrapper.
    Raises:
        ValueError: The variant, native source, defaults, or live state differs.
    """
    if variant == _NATIVE:
        return policy
    if variant not in _STOCK_VARIANTS:
        raise ValueError("Unsupported RLC execution variant")
    _verify_native_policy(policy)
    if variant == _FINAL_STAGE_BACKTRACK:
        return _FinalStageBacktrackPolicy(policy)
    if variant == _ADAPTIVE_TRANSITION_REFRESH:
        return _AdaptiveTransitionRefreshPolicy(policy)
    return _AdaptiveShortChunkPolicy(policy)


def execution_provenance(variant: str, *, selected: bool) -> Mapping[str, Any] | None:
    """Describe the exact private experiment that introduced a public option.

    Args:
        variant: Requested execution policy.
        selected: Whether selected step-3599 weights are served.
    Returns:
        Immutable provenance for a non-native option, or ``None`` for native.
    Raises:
        ValueError: The policy/variant combination is unsupported.
    """
    if variant == _NATIVE:
        return None
    adaptive_variants = {_ADAPTIVE_SHORT_CHUNK, _ADAPTIVE_TRANSITION_REFRESH}
    if variant in adaptive_variants and selected:
        raise ValueError("Adaptive short chunks are supported for stock weights only")
    if variant == _ADAPTIVE_TRANSITION_REFRESH:
        return MappingProxyType(
            {
                "parent_variant": _ADAPTIVE_SHORT_CHUNK,
                "intervention": "refresh_action_queue_after_accepted_stage_transition",
                "evaluation": "experimental_no_aggregate_gain_established",
            }
        )
    identity = _EXPERIMENT_IDENTITIES.get((variant, selected))
    if identity is None:
        raise ValueError("Unsupported RLC execution variant")
    source, config = identity
    return MappingProxyType(
        {
            "experiment_source_sha256": source,
            "experiment_config_sha256": config,
            "evaluation": "experimental_no_aggregate_gain_established",
        }
    )


class _EpisodeTelemetry:
    """Track ordinal-only telemetry shared by the two execution policies."""

    def _initialize_telemetry(self) -> None:
        self._next_episode_ordinal = 0
        self._active_episode_ordinal: int | None = None
        self._flushed_episode_ordinals: set[int] = set()

    def _begin_episode(self) -> None:
        if self._active_episode_ordinal is None:
            self._active_episode_ordinal = self._next_episode_ordinal
            self._next_episode_ordinal += 1
            self._telemetry["episode_ordinal"] = self._active_episode_ordinal

    def _finish_episode(self, prefix: str) -> None:
        ordinal = self._active_episode_ordinal
        if ordinal is None or not self._telemetry["observations"]:
            return
        if ordinal in self._flushed_episode_ordinals:
            return
        summary = {
            key: value for key, value in self._telemetry.items() if key != "events"
        }
        _LOGGER.info("%s%s", prefix, json.dumps(summary, sort_keys=True))
        self._flushed_episode_ordinals.add(ordinal)


class _FinalStageBacktrackPolicy(_EpisodeTelemetry):
    """Extend only the native final-stage unanimous backward-vote boundary."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self._native_update = policy.update_current_stage
        self._initialize_telemetry()
        self._reset_telemetry()
        policy.update_current_stage = MethodType(self._update_current_stage, policy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def reset(self) -> None:
        self.finalize_telemetry()
        self.policy.reset()
        self._reset_telemetry()

    def act(self, observation: Mapping[str, Any]) -> Any:
        self._begin_episode()
        self._telemetry["observations"] += 1
        return self.policy.act(observation)

    def telemetry(self) -> Mapping[str, Any]:
        """Return a read-only snapshot without evaluator case identifiers."""
        value = dict(self._telemetry)
        value["events"] = tuple(dict(event) for event in value["events"])
        return MappingProxyType(value)

    def _update_current_stage(self, native_self: Any, logits: Any) -> Any:
        before = int(native_self.current_stage)
        result = self._native_update(logits)
        if native_self.task_id is None:
            return result
        stages = self._native_update.__func__.__globals__["TASK_NUM_STAGES"]
        maximum = int(stages[native_self.task_id]) - 1
        history = tuple(int(value) for value in native_self.prediction_history)
        previous = maximum - 1
        if self._should_backtrack(
            before, int(native_self.current_stage), history, maximum
        ):
            native_self.current_stage = previous
            native_self.prediction_history.clear()
            self._record_backtrack(maximum, previous, history)
        return result

    @staticmethod
    def _should_backtrack(
        before: int, after: int, history: tuple[int, ...], maximum: int
    ) -> bool:
        previous = maximum - 1
        return (
            maximum > 0
            and before == maximum
            and after == maximum
            and history == (previous, previous, previous)
        )

    def _record_backtrack(
        self, maximum: int, previous: int, history: tuple[int, ...]
    ) -> None:
        event = {
            "episode_ordinal": self._active_episode_ordinal,
            "observation_ordinal": self._telemetry["observations"] - 1,
            "from_stage": maximum,
            "to_stage": previous,
            "history": list(history),
        }
        self._telemetry["final_stage_backtracks"] += 1
        self._telemetry["events"].append(event)
        _LOGGER.info("%s%s", _FINAL_EVENT, json.dumps(event, sort_keys=True))

    def finalize_telemetry(self) -> None:
        self._finish_episode(_FINAL_SUMMARY)

    def _reset_telemetry(self) -> None:
        self._active_episode_ordinal = None
        self._telemetry = {
            "variant": _FINAL_STAGE_BACKTRACK,
            "observations": 0,
            "final_stage_backtracks": 0,
            "events": [],
        }


class _AdaptiveShortChunkPolicy(_EpisodeTelemetry):
    """Use native execution until either of the final two stages needs a prediction."""

    _VARIANT = _ADAPTIVE_SHORT_CHUNK

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self._native_config = policy.config
        self._initialize_telemetry()
        self._reset_telemetry()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def reset(self) -> None:
        self.finalize_telemetry()
        self.policy.config = self._native_config
        self.policy.reset()
        self._reset_telemetry()

    def act(self, observation: Mapping[str, Any]) -> Any:
        self._begin_episode()
        self._telemetry["observations"] += 1
        self._apply_task_change(observation)
        action, before_stage, after_stage = self._act_once(observation)
        if after_stage != before_stage:
            self._record_transition(before_stage, after_stage)
        return action

    def _act_once(self, observation: Mapping[str, Any]) -> tuple[Any, int, int]:
        profile = self._select_profile()
        before_stage = int(self.policy.current_stage)
        before_predictions = int(self.policy.prediction_count)
        action = self.policy.act(observation)
        after_stage = int(self.policy.current_stage)
        if int(self.policy.prediction_count) == before_predictions + 1:
            self._record_prediction(profile, before_stage, after_stage)
        return action, before_stage, after_stage

    def telemetry(self) -> Mapping[str, Any]:
        """Return a read-only snapshot without evaluator case identifiers."""
        value = dict(self._telemetry)
        value["events"] = tuple(dict(event) for event in value["events"])
        return MappingProxyType(value)

    def _apply_task_change(self, observation: Mapping[str, Any]) -> None:
        if "task_id" not in observation:
            return
        task_id = int(observation["task_id"][0])
        if self.policy.task_id != task_id:
            self.policy._handle_task_change(task_id)

    def _select_profile(self) -> str:
        if not self._needs_prediction():
            return self._profile_from_config()
        maximum = self._maximum_stage()
        profile = (
            "precision" if int(self.policy.current_stage) >= maximum - 1 else "coarse"
        )
        self._set_profile(profile)
        return profile

    def _needs_prediction(self) -> bool:
        return (
            self.policy.last_actions is None
            or self.policy.action_index >= self.policy.config.execute_in_n_steps
        )

    def _maximum_stage(self) -> int:
        globals_ = self.policy.update_current_stage.__func__.__globals__
        return int(globals_["TASK_NUM_STAGES"][self.policy.task_id]) - 1

    def _set_profile(self, profile: str) -> None:
        if profile == "coarse":
            values = (26, 4, 20)
        elif profile == "precision":
            values = (10, 4, 10)
        else:
            raise ValueError("Unknown adaptive execution profile")
        self.policy.config = dataclasses.replace(
            self._native_config,
            actions_to_execute=values[0],
            actions_to_keep=values[1],
            execute_in_n_steps=values[2],
        )

    def _profile_from_config(self) -> str:
        values = (
            self.policy.config.actions_to_execute,
            self.policy.config.actions_to_keep,
            self.policy.config.execute_in_n_steps,
        )
        if values == (26, 4, 20):
            return "coarse"
        if values == (10, 4, 10):
            return "precision"
        raise ValueError("Live adaptive execution profile differs")

    def _record_prediction(self, profile: str, before: int, after: int) -> None:
        self._telemetry["predictions"] += 1
        self._telemetry[f"{profile}_predictions"] += 1
        event = {
            "kind": "prediction",
            "episode_ordinal": self._active_episode_ordinal,
            "observation_ordinal": self._telemetry["observations"] - 1,
            "profile": profile,
            "stage_before": before,
            "stage_after": after,
            "actions_to_execute": self.policy.config.actions_to_execute,
            "execute_in_n_steps": self.policy.config.execute_in_n_steps,
            "actions_to_keep": self.policy.config.actions_to_keep,
        }
        self._telemetry["events"].append(event)
        _LOGGER.info("%s%s", _HORIZON_EVENT, json.dumps(event, sort_keys=True))

    def _record_transition(self, before: int, after: int) -> None:
        self._telemetry["accepted_stage_transitions"] += 1
        event = {
            "kind": "accepted_stage_transition",
            "episode_ordinal": self._active_episode_ordinal,
            "observation_ordinal": self._telemetry["observations"] - 1,
            "from_stage": before,
            "to_stage": after,
        }
        self._telemetry["events"].append(event)
        _LOGGER.info("%s%s", _HORIZON_EVENT, json.dumps(event, sort_keys=True))

    def finalize_telemetry(self) -> None:
        self._finish_episode(_HORIZON_SUMMARY)

    def _reset_telemetry(self) -> None:
        self._active_episode_ordinal = None
        self._telemetry = {
            "variant": self._VARIANT,
            "observations": 0,
            "predictions": 0,
            "coarse_predictions": 0,
            "precision_predictions": 0,
            "accepted_stage_transitions": 0,
            "transition_queue_refreshes": 0,
            "events": [],
        }


class _AdaptiveTransitionRefreshPolicy(_AdaptiveShortChunkPolicy):
    """Resample once from a newly accepted stage after discarding its stale queue."""

    _VARIANT = _ADAPTIVE_TRANSITION_REFRESH

    def act(self, observation: Mapping[str, Any]) -> Any:
        self._begin_episode()
        self._telemetry["observations"] += 1
        self._apply_task_change(observation)
        step_before = int(self.policy.step_count)
        action, before_stage, after_stage = self._act_once(observation)
        if after_stage == before_stage:
            return action
        self._record_transition(before_stage, after_stage)
        self._refresh_transition_queue(before_stage, after_stage)
        self.policy.step_count = step_before
        return self._bounded_resample(observation)

    def _bounded_resample(self, observation: Mapping[str, Any]) -> Any:
        step_before = int(self.policy.step_count)
        action, before_stage, after_stage = self._act_once(observation)
        if after_stage == before_stage:
            return action
        self._record_transition(before_stage, after_stage)
        self._refresh_transition_queue(before_stage, after_stage)
        self.policy.step_count = step_before
        raise RuntimeError("Stage changed again during the bounded transition refresh")

    def _refresh_transition_queue(self, before: int, after: int) -> None:
        queued = self.policy.last_actions
        queue_length = len(queued) if queued is not None else 0
        discarded = max(queue_length - int(self.policy.action_index), 0)
        retained = self.policy.next_initial_actions
        retained_count = len(retained) if retained is not None else 0
        self.policy.last_actions = None
        self.policy.action_index = 0
        self.policy.next_initial_actions = None
        self._telemetry["transition_queue_refreshes"] += 1
        event = {
            "kind": "transition_queue_refresh",
            "episode_ordinal": self._active_episode_ordinal,
            "observation_ordinal": self._telemetry["observations"] - 1,
            "from_stage": before,
            "to_stage": after,
            "discarded_queued_actions": discarded,
            "discarded_inpainting_actions": retained_count,
        }
        self._telemetry["events"].append(event)
        _LOGGER.info("%s%s", _HORIZON_EVENT, json.dumps(event, sort_keys=True))
