"""Verify optional RLC execution policies against the pinned wrapper behavior."""

from __future__ import annotations

import ast
from collections import deque
import dataclasses
import hashlib
import json
import importlib.util
import inspect
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from npa.workflows.behavior_challenge import rlc_execution

TASK_NUM_STAGES = {1: 6}
# Import only the checked-in, attributed source fixture. Its method ASTs are
# verified below against the pinned upstream revision.
FIXTURE_PATH = Path(__file__).with_name("fixtures") / "behavior_native_execution.py"
SPEC = importlib.util.spec_from_file_location("behavior_native_execution", FIXTURE_PATH)
assert SPEC is not None and SPEC.loader is not None
pinned_native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pinned_native)
PINNED_WRAPPER_SOURCE = FIXTURE_PATH.read_text()
PINNED_ACT_SOURCE = inspect.getsource(pinned_native.act)
PINNED_ACT_AST_SHA256 = (
    "e480dcb417f72e0f4312273b62ef59e7fbae488f328d19cd5e486b579a585bd2"
)


@dataclasses.dataclass(frozen=True)
class NativeConfig:
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2


class B1KPolicyWrapper:
    """Small executable fixture with the pinned native stage-update method."""

    def __init__(self) -> None:
        self.config = NativeConfig()
        self.last_actions = None
        self.action_index = 0
        self.next_initial_actions = None
        self.prediction_history = deque(maxlen=3)
        self.current_stage = 0
        self.task_id = 1
        self.step_count = 0
        self.prediction_count = 0
        self.advance_to: int | None = None

    def update_current_stage(self, predicted_subtask_logits):
        """Update current stage using the pinned majority voting."""
        if self.task_id is None:
            return
        max_stage = TASK_NUM_STAGES[self.task_id] - 1
        predicted_stage = int(np.argmax(predicted_subtask_logits))
        if predicted_stage > max_stage:
            predicted_stage = max_stage
        self.prediction_history.append(predicted_stage)
        if len(self.prediction_history) == self.config.history_len:
            self._apply_stage_votes(max_stage)

    def _apply_stage_votes(self, max_stage: int) -> None:
        next_stage = self.current_stage + 1
        if next_stage > max_stage:
            return
        votes_for_next = sum(pred == next_stage for pred in self.prediction_history)
        votes_to_skip = sum(pred == next_stage + 1 for pred in self.prediction_history)
        votes_to_go_back = sum(
            pred == self.current_stage - 1 for pred in self.prediction_history
        )
        if votes_for_next >= self.config.votes_to_promote:
            self.current_stage = next_stage
            self.prediction_history.clear()
        elif votes_to_skip == self.config.history_len:
            self.current_stage = next_stage
            self.prediction_history.clear()
        elif votes_to_go_back == self.config.history_len and self.current_stage > 0:
            self.current_stage -= 1
            self.prediction_history.clear()

    def act(self, observation):
        self.prediction_count += 1
        self.last_actions = np.ones((self.config.execute_in_n_steps, 23))
        self.action_index = 1
        self.next_initial_actions = np.ones((4, 23))
        if self.advance_to is not None:
            self.current_stage = self.advance_to
        return observation

    def reset(self) -> None:
        self.last_actions = None
        self.action_index = 0
        self.next_initial_actions = None
        self.prediction_history.clear()
        self.current_stage = 0


@pytest.fixture(autouse=True)
def source_identity(monkeypatch):
    monkeypatch.setattr(
        rlc_execution,
        "_source_identity",
        lambda policy: (
            rlc_execution._OVERLAY_WRAPPER_SHA256,
            rlc_execution._NATIVE_UPDATE_AST_SHA256,
        ),
    )


def logits(stage: int) -> np.ndarray:
    values = np.zeros(15, dtype=np.float32)
    values[stage] = 1
    return values


def install_pinned_native_act(native: B1KPolicyWrapper) -> list[dict[str, object]]:
    """Install the attributed native action method with lightweight dependencies."""

    method = ast.parse(PINNED_ACT_SOURCE).body[0]
    encoded = json.dumps(
        rlc_execution._canonical_ast(method), sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == PINNED_ACT_AST_SHA256

    calls: list[dict[str, object]] = []

    class InferencePolicy:
        def infer(self, model_input, initial_actions=None):
            calls.append(
                {
                    "stage": int(model_input["subtask_state"]),
                    "initial_actions": initial_actions,
                }
            )
            return {
                "actions": np.arange(30 * 23, dtype=np.float32).reshape(30, 23),
                "subtask_logits": logits(4),
            }

    @dataclasses.dataclass(frozen=True)
    class PinnedActConfig:
        actions_to_execute: int = 26
        actions_to_keep: int = 4
        execute_in_n_steps: int = 20
        history_len: int = 3
        votes_to_promote: int = 2
        apply_eval_tricks: bool = True

    native.config = PinnedActConfig()
    native.policy = InferencePolicy()
    native.process_obs = MethodType(
        lambda _self, observation: dict(observation), native
    )
    native.prepare_batch_for_pi_behavior = MethodType(
        lambda self, batch: {**batch, "subtask_state": self.current_stage}, native
    )
    native._interpolate_actions = MethodType(
        lambda _self, actions, target_steps: actions[:target_steps].copy(), native
    )
    native._handle_task_change = MethodType(
        lambda self, task_id: setattr(self, "task_id", task_id), native
    )
    native.act = MethodType(pinned_native.act, native)
    return calls


def test_native_default_returns_same_pristine_policy():
    policy = B1KPolicyWrapper()

    assert rlc_execution.configure_execution(policy, "native") is policy
    assert policy.config == NativeConfig()


def test_final_stage_unanimous_backtrack_changes_only_final_boundary(caplog):
    native = B1KPolicyWrapper()
    native.current_stage = 5
    policy = rlc_execution.configure_execution(native, "final-stage-backtrack")
    policy.act({"task_id": np.array([1])})

    with caplog.at_level("INFO", logger=rlc_execution.__name__):
        for _ in range(3):
            native.update_current_stage(logits(4))
        policy.finalize_telemetry()

    assert native.current_stage == 4
    assert not native.prediction_history
    summary = next(
        json.loads(record.message.split("RLC_FINAL_STAGE_SUMMARY ", 1)[1])
        for record in caplog.records
        if "RLC_FINAL_STAGE_SUMMARY " in record.message
    )
    assert summary["final_stage_backtracks"] == 1


def test_final_stage_partial_vote_and_native_middle_vote_are_unchanged():
    native = B1KPolicyWrapper()
    native.current_stage = 5
    rlc_execution.configure_execution(native, "final-stage-backtrack")
    for stage in (4, 4, 3):
        native.update_current_stage(logits(stage))
    assert native.current_stage == 5

    native.current_stage = 3
    native.prediction_history.clear()
    for _ in range(3):
        native.update_current_stage(logits(2))
    assert native.current_stage == 2


def test_final_stage_preserves_native_noop_without_task_identity():
    native = B1KPolicyWrapper()
    native.task_id = None
    native.current_stage = 5
    policy = rlc_execution.configure_execution(native, "final-stage-backtrack")

    native.update_current_stage(logits(4))

    assert native.current_stage == 5
    assert not native.prediction_history
    assert policy.telemetry()["final_stage_backtracks"] == 0


def test_adaptive_short_chunk_changes_only_prediction_boundary():
    native = B1KPolicyWrapper()
    policy = rlc_execution.configure_execution(native, "adaptive-short-chunk")

    policy.act({"task_id": np.array([1])})
    assert native.config == NativeConfig()
    native.last_actions = None
    native.current_stage = 4
    native.advance_to = 5
    action = policy.act({"task_id": np.array([1])})

    assert action["task_id"].tolist() == [1]
    assert dataclasses.astuple(native.config) == (10, 4, 10, 3, 2)
    assert native.last_actions is not None
    telemetry = policy._telemetry
    assert telemetry["precision_predictions"] == 1
    assert telemetry["accepted_stage_transitions"] == 1
    assert telemetry["transition_queue_refreshes"] == 0


def test_adaptive_pinned_native_act_keeps_queue_across_profile_boundary():
    native = B1KPolicyWrapper()
    calls = install_pinned_native_act(native)
    policy = rlc_execution.configure_execution(native, "adaptive-short-chunk")
    native.current_stage = 3
    native.prediction_history.extend((4, 4))
    observation = {
        "task_id": np.array([1]),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
    }

    policy.act(observation)

    assert native.current_stage == 4
    assert native.config.execute_in_n_steps == 20
    assert len(calls) == 1
    queue = native.last_actions
    assert queue is not None and queue.shape == (20, 23)
    assert native.action_index == 1

    policy.act(observation)

    assert native.last_actions is queue
    assert native.action_index == 2
    assert native.config.execute_in_n_steps == 20
    assert len(calls) == 1

    native.action_index = 20
    policy.act(observation)

    assert len(calls) == 2
    assert native.config.execute_in_n_steps == 10
    assert native.last_actions is not queue
    assert native.last_actions.shape == (10, 23)
    telemetry = policy.telemetry()
    assert telemetry["accepted_stage_transitions"] == 1
    assert telemetry["transition_queue_refreshes"] == 0


def test_adaptive_transition_refresh_no_transition_matches_adaptive_control():
    control_native = B1KPolicyWrapper()
    control_calls = install_pinned_native_act(control_native)
    control = rlc_execution.configure_execution(
        control_native, "adaptive-short-chunk"
    )
    refresh_native = B1KPolicyWrapper()
    refresh_calls = install_pinned_native_act(refresh_native)
    refresh = rlc_execution.configure_execution(
        refresh_native, "adaptive-short-chunk-transition-refresh"
    )
    observation = {
        "task_id": np.array([1]),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
    }

    control_action = control.act(observation)
    refresh_action = refresh.act(observation)

    np.testing.assert_array_equal(control_action.value, refresh_action.value)
    assert control_calls == refresh_calls
    assert control_native.current_stage == refresh_native.current_stage == 0
    np.testing.assert_array_equal(control_native.last_actions, refresh_native.last_actions)
    assert control_native.action_index == refresh_native.action_index == 1
    assert refresh.telemetry()["transition_queue_refreshes"] == 0


def test_adaptive_transition_refresh_discards_once_and_resamples_new_stage():
    native = B1KPolicyWrapper()
    calls = install_pinned_native_act(native)
    policy = rlc_execution.configure_execution(
        native, "adaptive-short-chunk-transition-refresh"
    )
    native.current_stage = 3
    native.prediction_history.extend((4, 4))
    observation = {
        "task_id": np.array([1]),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
    }

    policy.act(observation)

    assert [call["stage"] for call in calls] == [3, 4]
    assert native.current_stage == 4
    assert native.config.execute_in_n_steps == 10
    assert native.last_actions is not None and native.last_actions.shape == (10, 23)
    assert native.action_index == 1
    assert native.step_count == 1
    telemetry = policy.telemetry()
    assert telemetry["predictions"] == 2
    assert telemetry["coarse_predictions"] == 1
    assert telemetry["precision_predictions"] == 1
    assert telemetry["accepted_stage_transitions"] == 1
    assert telemetry["transition_queue_refreshes"] == 1
    refresh_event = next(
        event for event in telemetry["events"] if event["kind"] == "transition_queue_refresh"
    )
    assert refresh_event["discarded_queued_actions"] == 19
    assert refresh_event["discarded_inpainting_actions"] == 4

    policy.act(observation)

    assert len(calls) == 2
    assert policy.telemetry()["transition_queue_refreshes"] == 1


def test_adaptive_transition_refresh_is_bounded_when_resample_transitions_again():
    native = B1KPolicyWrapper()

    def transition_every_prediction(self, observation):
        self.prediction_count += 1
        self.last_actions = np.ones((self.config.execute_in_n_steps, 23))
        self.action_index = 1
        self.next_initial_actions = np.ones((4, 23))
        self.current_stage += 1
        return observation

    native.act = MethodType(transition_every_prediction, native)
    policy = rlc_execution.configure_execution(
        native, "adaptive-short-chunk-transition-refresh"
    )

    with pytest.raises(RuntimeError, match="bounded transition refresh"):
        policy.act({"task_id": np.array([1])})

    assert native.prediction_count == 2
    assert native.last_actions is None
    assert policy.telemetry()["accepted_stage_transitions"] == 2
    assert policy.telemetry()["transition_queue_refreshes"] == 2


def test_adaptive_transition_refresh_reset_clears_episode_state():
    native = B1KPolicyWrapper()
    install_pinned_native_act(native)
    policy = rlc_execution.configure_execution(
        native, "adaptive-short-chunk-transition-refresh"
    )
    native.current_stage = 3
    native.prediction_history.extend((4, 4))
    observation = {
        "task_id": np.array([1]),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
    }
    policy.act(observation)

    policy.reset()

    assert native.config.actions_to_execute == 26
    assert native.config.execute_in_n_steps == 20
    assert native.last_actions is None
    assert native.action_index == 0
    assert native.next_initial_actions is None
    assert not native.prediction_history
    assert policy.telemetry()["observations"] == 0
    assert policy.telemetry()["transition_queue_refreshes"] == 0
    assert policy.telemetry()["variant"] == "adaptive-short-chunk-transition-refresh"


def test_reset_restores_native_execution_defaults():
    native = B1KPolicyWrapper()
    native.current_stage = 4
    policy = rlc_execution.configure_execution(native, "adaptive-short-chunk")
    policy.act({"task_id": np.array([1])})

    policy.reset()

    assert native.config == NativeConfig()


def test_actual_native_method_ast_fixture_matches_pinned_hash(tmp_path):
    fixture = tmp_path / "wrapper.py"
    fixture.write_text(PINNED_WRAPPER_SOURCE)

    actual = rlc_execution.native_update_ast_sha256(fixture)

    assert actual == rlc_execution._NATIVE_UPDATE_AST_SHA256


def test_final_backtrack_wraps_the_executable_pinned_method():
    native = pinned_native.B1KPolicyWrapper()
    native.config = NativeConfig()
    native.last_actions = None
    native.next_initial_actions = None
    native.action_index = 0
    native.prediction_history = deque(maxlen=3)
    native.current_stage = 5
    native.task_id = 1
    native.step_count = 0

    policy = rlc_execution.configure_execution(native, "final-stage-backtrack")
    for _ in range(3):
        native.update_current_stage(logits(4))

    assert native.current_stage == 4
    assert policy.telemetry()["final_stage_backtracks"] == 1


def test_provenance_distinguishes_stock_and_selected_trials():
    stock = rlc_execution.execution_provenance("final-stage-backtrack", selected=False)
    selected = rlc_execution.execution_provenance(
        "final-stage-backtrack", selected=True
    )

    assert stock is not None and selected is not None
    assert stock["experiment_config_sha256"] != selected["experiment_config_sha256"]
    assert stock["evaluation"] == "experimental_no_aggregate_gain_established"
    with pytest.raises(ValueError, match="stock weights only"):
        rlc_execution.execution_provenance("adaptive-short-chunk", selected=True)

    refresh = rlc_execution.execution_provenance(
        "adaptive-short-chunk-transition-refresh", selected=False
    )
    assert refresh is not None
    assert refresh["parent_variant"] == "adaptive-short-chunk"
    assert refresh["evaluation"] == "experimental_no_aggregate_gain_established"
