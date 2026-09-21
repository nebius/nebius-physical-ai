"""Verify optional RLC execution policies against the pinned wrapper behavior."""

from __future__ import annotations

import ast
from collections import deque
import dataclasses
import hashlib
import json
import logging
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from npa.workflows.behavior_challenge import rlc_execution

TASK_NUM_STAGES = {1: 6}
# The stage-update fixture below is from Ilia Larchenko's Apache-2.0-licensed
# behavior-1k-solution, commit ca556f74a455cef7987a2be4537b5ac85cc56dd7.
# Copyright 2025 Ilia Larchenko. It is embedded unchanged in a minimal test class;
# the surrounding fixtures and tests are additions. The license is included at
# npa/src/npa/workflows/behavior_challenge/POLICY_LICENSE.
PINNED_WRAPPER_SOURCE = '''class B1KPolicyWrapper():
    def update_current_stage(self, predicted_subtask_logits):
        """Update current stage using majority voting."""
        if self.task_id is None:
            return

        max_stage = TASK_NUM_STAGES[self.task_id] - 1
        predicted_stage = int(np.argmax(predicted_subtask_logits))

        if predicted_stage > max_stage:
            predicted_stage = max_stage

        self.prediction_history.append(predicted_stage)

        if len(self.prediction_history) == self.config.history_len:
            next_stage = self.current_stage + 1

            if next_stage <= max_stage:
                votes_for_next = sum(1 for pred in self.prediction_history if pred == next_stage)
                votes_to_skip = sum(1 for pred in self.prediction_history if pred == next_stage + 1)
                votes_to_go_back = sum(1 for pred in self.prediction_history if pred == self.current_stage - 1)

                if votes_for_next >= self.config.votes_to_promote:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⬆️  Stage advanced: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_skip == self.config.history_len:
                    old_stage = self.current_stage
                    self.current_stage = next_stage
                    self.prediction_history.clear()
                    logger.info(f"⏭️  Stage skipped: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
                elif votes_to_go_back == self.config.history_len and self.current_stage > 0:
                    old_stage = self.current_stage
                    self.current_stage -= 1
                    self.prediction_history.clear()
                    logger.info(f"⬅️  Stage went back: {old_stage} → {self.current_stage} (task {self.task_id}, step {self.step_count})")
'''

# This executable method fixture is the same Apache-2.0 B1KPolicyWrapper.act
# implementation and upstream revision cited above. Keeping the actual queue
# branch here lets the adaptive test cover native queue lifetime, rather than a
# mock that always produces a new queue.
PINNED_ACT_SOURCE = '''def act(self, obs: dict) -> torch.Tensor:
    """Main action function."""
    if 'task_id' in obs:
        new_task_id = int(obs['task_id'][0])
        self._handle_task_change(new_task_id)
    raw_state = obs['robot_r1::proprio']
    current_state = extract_state_from_proprio(raw_state)
    if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
        model_input = self.process_obs(obs)
        model_input = self.prepare_batch_for_pi_behavior(model_input)
        if self.next_initial_actions is not None and ('initial_actions' not in model_input or model_input['initial_actions'] is None):
            model_input['initial_actions'] = self.next_initial_actions
        if 'initial_actions' in model_input and model_input['initial_actions'] is not None:
            output = self.policy.infer(model_input, initial_actions=model_input['initial_actions'])
        else:
            output = self.policy.infer(model_input)
        actions = output['actions']
        if len(actions.shape) == 3:
            actions = actions[0]
        if actions.shape[1] > 23:
            actions = actions[:, :23]
        should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
        if self.config.apply_eval_tricks:
            if self.task_id is not None:
                actions_before = actions.copy()
                actions, corrected_stage = apply_correction_rules(self.task_id, self.current_stage, current_state, actions)
                if corrected_stage != self.current_stage:
                    logger.info(f'🔧 Correction rule: Stage corrected {self.current_stage} → {corrected_stage} (task {self.task_id}, step {self.step_count})')
                    self.current_stage = corrected_stage
                    self.prediction_history.clear()
                if not np.allclose(actions_before, actions, rtol=0.001):
                    max_diff = np.max(np.abs(actions_before - actions))
                    logger.info(f'🔧 Correction rule: Actions modified (max diff: {max_diff:.4f}, task {self.task_id}, stage {self.current_stage})')
            if should_compress:
                has_high_variation, mean_var, max_var = check_gripper_variation(actions, self.config.actions_to_execute)
                if has_high_variation:
                    should_compress = False
                    logger.info(f'🔧 Gripper variation: Compression disabled (mean: {mean_var:.4f}, max: {max_var:.4f})')
        actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
        execute_steps = self.config.execute_in_n_steps
        inpainting_start = actions_to_execute
        inpainting_end = inpainting_start + self.config.actions_to_keep
        if len(actions) >= inpainting_end:
            self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
        else:
            self.next_initial_actions = None
        self.last_actions = actions[:actions_to_execute].copy()
        if should_compress:
            compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
            compression_factor = actions_to_execute / execute_steps
            compressed_actions[:, :3] *= compression_factor
            self.last_actions = compressed_actions
        self.action_index = 0
        self.prediction_count += 1
        if self.prediction_count % 10 == 0:
            compression_status = f'compressed {actions_to_execute}→{execute_steps}' if should_compress else f'uncompressed ({execute_steps})'
            logger.info(f'🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | Inpainting: {self.next_initial_actions is not None}')
        if 'subtask_logits' in output:
            self.update_current_stage(output['subtask_logits'])
    if self.action_index >= len(self.last_actions):
        self.action_index = 0
    current_action = self.last_actions[self.action_index]
    self.action_index += 1
    self.step_count += 1
    if self.step_count % 100 == 0:
        logger.info(f'📊 Step {self.step_count} | Task: {self.task_id} | Stage: {self.current_stage}/{TASK_NUM_STAGES[self.task_id] - 1} | Predictions: {self.prediction_count}')
    action_tensor = torch.from_numpy(current_action).float()
    if len(action_tensor) > 23:
        action_tensor = action_tensor[:23]
    return action_tensor
'''
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

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def float(self):
            return self

        def __len__(self):
            return len(self.value)

        def __getitem__(self, item):
            return Tensor(self.value[item])

    namespace = {
        "TASK_NUM_STAGES": TASK_NUM_STAGES,
        "apply_correction_rules": lambda _task, stage, _state, actions: (
            actions,
            stage,
        ),
        "check_gripper_variation": lambda *_: (False, 0.0, 0.0),
        "extract_state_from_proprio": lambda value: value,
        "logger": logging.getLogger("pinned-native-act-fixture"),
        "np": np,
        "torch": SimpleNamespace(from_numpy=Tensor, Tensor=Tensor),
    }
    exec(PINNED_ACT_SOURCE, namespace)

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
    native.act = MethodType(namespace["act"], native)
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
    namespace = {
        "TASK_NUM_STAGES": TASK_NUM_STAGES,
        "np": np,
        "logger": logging.getLogger("pinned-wrapper-fixture"),
    }
    exec(PINNED_WRAPPER_SOURCE, namespace)
    native = namespace["B1KPolicyWrapper"]()
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
