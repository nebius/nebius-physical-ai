"""Apply an additive baseline penalty inside the pinned native PPO optimization step."""

# Adapted from RSL-RL v5.0.1 rsl_rl/algorithms/ppo.py (PPO.update).
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
# The full license is embedded so Python-only runtime staging retains it.
# Copyright (c) 2026, ETH Zurich
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn


def update_with_anchor(algorithm):
    """Preserve native minibatches and optimization order while adding baseline KL.

    Args:
        algorithm: Bound, supported BaselineAnchoredPPO instance.
    Returns:
        Native mean losses plus the measured baseline KL penalty.
    Raises:
        ValueError: Teacher is unbound or a distribution becomes nonfinite.
    """
    if algorithm.anchor_teacher is None:
        raise ValueError("baseline teacher must be bound after the initial snapshot")
    losses = dict(value=0.0, surrogate=0.0, entropy=0.0, baseline_kl=0.0)
    batches = algorithm.storage.mini_batch_generator(
        algorithm.num_mini_batches, algorithm.num_learning_epochs
    )
    for batch in batches:
        measured = _update_batch(algorithm, batch)
        for name, value in measured.items():
            losses[name] += value.item()
        algorithm.anchor_steps += 1
        algorithm.anchor_samples += batch.observations.batch_size[0]
    updates = algorithm.num_learning_epochs * algorithm.num_mini_batches
    algorithm.storage.clear()
    return {name: value / updates for name, value in losses.items()}


def _update_batch(algorithm, batch):
    if algorithm.normalize_advantage_per_mini_batch:
        with torch.no_grad():
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std() + 1e-8
            )
    algorithm.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
    )
    log_prob = algorithm.actor.get_output_log_prob(batch.actions)
    values = algorithm.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
    )
    distribution = algorithm.actor.output_distribution_params
    entropy = algorithm.actor.output_entropy
    _adapt_learning_rate(algorithm, batch.old_distribution_params, distribution)
    surrogate = _surrogate(algorithm, batch, log_prob)
    value_loss = _value_loss(algorithm, batch, values)
    penalty = algorithm.baseline_loss(batch.observations, distribution)
    loss = surrogate + algorithm.value_loss_coef * value_loss
    loss = loss - algorithm.entropy_coef * entropy.mean()
    loss = loss + algorithm.anchor_config.coefficient * penalty
    algorithm.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(algorithm.actor.parameters(), algorithm.max_grad_norm)
    nn.utils.clip_grad_norm_(algorithm.critic.parameters(), algorithm.max_grad_norm)
    algorithm.optimizer.step()
    return dict(
        value=value_loss,
        surrogate=surrogate,
        entropy=entropy.mean(),
        baseline_kl=penalty,
    )


def _adapt_learning_rate(algorithm, previous, current):
    if algorithm.desired_kl is None or algorithm.schedule != "adaptive":
        return
    with torch.inference_mode():
        kl_mean = torch.mean(algorithm.actor.get_kl_divergence(previous, current))
        if kl_mean > algorithm.desired_kl * 2.0:
            algorithm.learning_rate = max(1e-5, algorithm.learning_rate / 1.5)
        elif kl_mean < algorithm.desired_kl / 2.0 and kl_mean > 0.0:
            algorithm.learning_rate = min(1e-2, algorithm.learning_rate * 1.5)
        for group in algorithm.optimizer.param_groups:
            group["lr"] = algorithm.learning_rate


def _surrogate(algorithm, batch, log_prob):
    ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
    surrogate = -torch.squeeze(batch.advantages) * ratio
    clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - algorithm.clip_param, 1.0 + algorithm.clip_param
    )
    return torch.max(surrogate, clipped).mean()


def _value_loss(algorithm, batch, values):
    if not algorithm.use_clipped_value_loss:
        return (batch.returns - values).pow(2).mean()
    clipped = batch.values + (values - batch.values).clamp(
        -algorithm.clip_param, algorithm.clip_param
    )
    loss = (values - batch.returns).pow(2)
    clipped_loss = (clipped - batch.returns).pow(2)
    return torch.max(loss, clipped_loss).mean()
