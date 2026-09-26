"""Measure fixed per-environment episode quotas without completion-order bias."""

from __future__ import annotations

from contextlib import ExitStack
import math

from .schemas import MjlabError


def measure_episodes(env, policy, request, outputs) -> dict:
    """Run complete episodes and optionally record the first episode.

    Args:
        env: Upstream RSL-RL vector environment with finite task horizon.
        policy: Upstream inference policy callable.
        request: Validated evaluation settings.
        outputs: Local artifact directory.
    Returns:
        Episode records, measured returns and a survival gate.
    Raises:
        MjlabError: Non-finite reward, action or incomplete episode accounting.
    """
    import torch

    quotas = [
        request.episodes // env.num_envs + (i < request.episodes % env.num_envs)
        for i in range(env.num_envs)
    ]
    counts, returns, lengths = ([0] * env.num_envs for _ in range(3))
    rows = []
    observations = env.get_observations()
    with ExitStack() as stack:
        writer = _video_writer(stack, env, outputs) if request.video else None
        while any(count < quota for count, quota in zip(counts, quotas)):
            with torch.inference_mode():
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise MjlabError("Policy emitted non-finite actions")
                if writer is not None and counts[0] == 0:
                    writer.append_data(env.unwrapped.render())
                observations, rewards, dones, _ = env.step(actions)
            _record_step(env, rewards, dones, quotas, counts, returns, lengths, rows)
            if hasattr(policy, "reset"):
                policy.reset(dones)
    return _summary(rows, request)


def _record_step(env, rewards, dones, quotas, counts, returns, lengths, rows):
    for lane in range(env.num_envs):
        if counts[lane] >= quotas[lane]:
            continue
        reward = float(rewards[lane])
        if not math.isfinite(reward):
            raise MjlabError("Simulator emitted a non-finite reward")
        returns[lane] += reward
        lengths[lane] += 1
        if bool(dones[lane]):
            failed = bool(env.unwrapped.reset_terminated[lane])
            timed_out = bool(env.unwrapped.reset_time_outs[lane])
            rows.append(
                {
                    "environment": lane,
                    "episode": counts[lane],
                    "return": returns[lane],
                    "length": lengths[lane],
                    "survived": timed_out and not failed,
                }
            )
            counts[lane] += 1
            returns[lane], lengths[lane] = 0, 0
        elif lengths[lane] > env.max_episode_length:
            raise MjlabError(
                "Environment did not terminate at its configured episode horizon"
            )


def _video_writer(stack, env, outputs):
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(outputs / "rollout.mp4"), fps=1.0 / env.unwrapped.step_dt, codec="libx264"
    )
    stack.callback(writer.close)
    return writer


def _summary(rows, request):
    if len(rows) != request.episodes:
        raise MjlabError("Measured episode count differs from the request")
    survived = sum(row["survived"] for row in rows)
    score = survived / len(rows)
    passed = score >= request.success_threshold
    return {
        "status": "passed" if passed else "needs_iteration",
        "passed": passed,
        "score": score,
        "score_definition": "fraction reaching the task horizon without termination",
        "success_threshold": request.success_threshold,
        "episodes": rows,
        "episodes_completed": len(rows),
        "mean_return": sum(r["return"] for r in rows) / len(rows),
        "mean_episode_length": sum(r["length"] for r in rows) / len(rows),
        "seed": request.seed,
    }
