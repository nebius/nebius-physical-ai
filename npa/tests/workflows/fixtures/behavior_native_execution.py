"""Static pinned RLC methods with lightweight test dependencies.

The stage-update and action methods are from Ilia Larchenko's
Apache-2.0-licensed behavior-1k-solution, commit
ca556f74a455cef7987a2be4537b5ac85cc56dd7. Copyright 2025 Ilia Larchenko.
The surrounding test dependencies are additions. The license is included at
npa/src/npa/workflows/behavior_challenge/POLICY_LICENSE.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

TASK_NUM_STAGES = {1: 6}
logger = logging.getLogger(__name__)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def float(self):
        return self

    def __len__(self):
        return len(self.value)

    def __getitem__(self, item):
        return Tensor(self.value[item])


torch = SimpleNamespace(from_numpy=Tensor, Tensor=Tensor)


def apply_correction_rules(_task, stage, _state, actions):
    return actions, stage


def check_gripper_variation(*_):
    return False, 0.0, 0.0


def extract_state_from_proprio(value):
    return value


# Preserve the attributed upstream methods for their pinned AST checks.
# fmt: off
class B1KPolicyWrapper():
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


def act(self, obs: dict) -> torch.Tensor:
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

# fmt: on
