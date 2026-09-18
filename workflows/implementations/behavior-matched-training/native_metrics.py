"""Compute parent stage logits and deterministic native RLC holdout losses."""

from __future__ import annotations

from collections.abc import Mapping


def predict_stage_logits(model, observation):
    """Return the unmasked 15-way native classifier logits without action denoising."""

    import jax.numpy as jnp
    from b1k.models.observation import preprocess_observation
    from openpi.models.pi0 import make_attn_mask

    observation = preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_output, _), _ = model.PaliGemma.llm(
        [prefix_tokens, None], mask=attention_mask, positions=positions
    )
    base_task_index = jnp.argmax(prefix_ar_mask) - 1
    return model.stage_pred_from_vlm(prefix_output[:, base_task_index, :])


def deterministic_batch_metrics(
    model,
    observation,
    actions,
    teacher_stages,
    *,
    seed: int,
    batch_ordinal: int,
    flow_draws: int,
) -> Mapping[str, object]:
    """Compute action and teacher-stage diagnostics with a fixed per-batch RNG."""

    import jax
    import jax.numpy as jnp

    rng = jax.random.fold_in(jax.random.key(seed), batch_ordinal)
    losses = model.compute_detailed_loss(
        rng, observation, actions, train=False, num_flow_samples=flow_draws
    )
    logits = predict_stage_logits(model, observation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    teacher_stages = jnp.asarray(teacher_stages, dtype=jnp.int32)
    stage_cross_entropy = -jnp.take_along_axis(
        log_probs, teacher_stages[:, None], axis=-1
    ).squeeze(-1)
    action_dimensions = _action_dimension_losses(losses)
    return {
        "action_loss": losses["action_loss"],
        "stage_cross_entropy": stage_cross_entropy,
        "action_dimension_losses": jnp.stack(action_dimensions, axis=-1),
        "stage_logits": logits,
    }


def _action_dimension_losses(losses):
    dimensions = [losses[f"action_loss_base_vel_{axis}"] for axis in "xyz"]
    dimensions.extend(losses[f"action_loss_trunk_{index}"] for index in range(4))
    dimensions.extend(losses[f"action_loss_left_arm_{index}"] for index in range(7))
    dimensions.append(losses["action_loss_left_gripper"])
    dimensions.extend(losses[f"action_loss_right_arm_{index}"] for index in range(7))
    dimensions.append(losses["action_loss_right_gripper"])
    return dimensions
