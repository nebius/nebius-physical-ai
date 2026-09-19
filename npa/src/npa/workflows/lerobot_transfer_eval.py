"""Measure exact ACT checkpoints on paired PushT resets and explicit transfer shifts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, tree_hashes, write_json
from npa.workflows.lerobot_transfer_training import runtime_versions


def shift_pixels(pixels: np.ndarray, condition: str) -> np.ndarray:
    """Apply fixed photometric stress without moving scene geometry.

    Args:
        pixels: Camera image with uint8 RGB channels.
        condition: Clean, dim, warm, or delay benchmark condition.
    Returns:
        Image with the same geometry, shape, and dtype.
    Raises:
        ValueError: Condition is unsupported.
    """
    gains = {
        "clean": (1, 1, 1),
        "dim": (0.45, 0.45, 0.45),
        "warm": (1.25, 0.85, 0.6),
        "delay": (1, 1, 1),
    }
    if condition not in gains:
        raise ValueError(f"Unknown transfer condition: {condition}")
    return np.clip(pixels.astype(np.float32) * gains[condition], 0, 255).astype(
        np.uint8
    )


def make_shifted_env(condition: str):
    """Construct the native PushT environment with a reset-safe shift wrapper.

    Args:
        condition: Named evaluation condition.
    Returns:
        Gymnasium environment with native success and termination semantics.
    Raises:
        ValueError: Condition is unsupported.
        ImportError: PushT dependencies are absent.
    """
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    class ShiftedPushT(gym.Wrapper):
        task = "PushT-v0"
        task_description = "Push the T-shaped block onto the target."

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            self.previous_action = observation["agent_pos"].copy()
            return self._observation(observation), info

        def step(self, action):
            applied = self.previous_action if condition == "delay" else action
            self.previous_action = np.asarray(action).copy()
            observation, reward, terminated, truncated, info = self.env.step(applied)
            return self._observation(observation), reward, terminated, truncated, info

        def _observation(self, observation):
            return {
                **observation,
                "pixels": shift_pixels(observation["pixels"], condition),
            }

        def render(self):
            return shift_pixels(self.env.render(), condition)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
    )
    return ShiftedPushT(env)


def evaluate_pair(prepared: Path, baseline: Path, robust: Path, output: Path) -> None:
    """Evaluate both trained arms on fixed validation and untouched test seed sets.

    Args:
        prepared: Verified preparation artifact.
        baseline: Verified baseline training artifact.
        robust: Verified robust training artifact.
        output: New evaluation output directory.
    Returns:
        None.
    Raises:
        ValueError: Checkpoint provenance differs from the sealed experiment.
        RuntimeError: CUDA or native policy evaluation fails.
    """
    runtime = runtime_versions()
    recipe = json.loads((prepared / "recipe.json").read_text())
    recipe_hash = file_sha256(prepared / "recipe.json")
    records, checkpoints = [], {}
    for arm, trained in (("baseline", baseline), ("robust", robust)):
        provenance = json.loads((trained / "training.json").read_text())
        hashes = tree_hashes(trained / "checkpoint")
        if provenance["recipe_sha256"] != recipe_hash or provenance["arm"] != arm:
            raise ValueError(
                "Training provenance does not match evaluation recipe and arm"
            )
        if not hashes or hashes != provenance["checkpoint_hashes"]:
            raise ValueError("Checkpoint differs from the exact trained weights")
        checkpoints[arm] = hashes
        records.extend(_evaluate_arm(trained / "checkpoint", output, arm, recipe))
    write_json(
        output / "evaluation.json",
        {
            "schema": "npa.lerobot-transfer.evaluation.v1",
            "recipe": recipe,
            "recipe_sha256": recipe_hash,
            "checkpoint_hashes": checkpoints,
            "runtime": runtime,
            "trials": records,
            "success_definition": "native gym-pusht coverage > 0.95 before episode termination",
            "physical_robot_tested": False,
        },
    )


def _evaluate_arm(checkpoint: Path, output: Path, arm: str, recipe: dict) -> list[dict]:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint).to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    processors = (preprocessor, postprocessor)
    records = []
    for split in ("validation", "test"):
        for condition in recipe["conditions"]:
            result = _evaluate_condition(
                policy,
                processors,
                output / arm / split / condition,
                condition,
                split,
                recipe,
            )
            write_json(output / arm / split / condition / "metrics.json", result)
            records.extend(
                {**trial, "arm": arm, "split": split, "condition": condition}
                for trial in result["per_episode"]
            )
    return records


def _evaluate_condition(
    policy, processors, output: Path, condition: str, split: str, recipe: dict
) -> dict:
    import torch
    from lerobot.utils.random_utils import set_seed

    count = recipe[f"{split}_episodes"]
    start = recipe[f"{split}_seed"]
    set_seed(recipe["seed"])
    records = []
    for offset in range(0, count, recipe["eval_batch_size"]):
        seeds = list(
            range(
                start + offset, start + min(count, offset + recipe["eval_batch_size"])
            )
        )
        with torch.inference_mode():
            records.extend(
                _evaluate_batch(
                    policy,
                    processors,
                    condition,
                    seeds,
                    output if offset == 0 else None,
                )
            )
    return {"per_episode": records}


def _evaluate_batch(
    policy, processors, condition: str, seeds: list[int], output: Path | None
) -> list[dict]:
    import gymnasium as gym
    from lerobot.envs.configs import PushtEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    environment = gym.vector.SyncVectorEnv(
        [lambda: make_shifted_env(condition) for _ in seeds],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env_before, env_after = make_env_pre_post_processors(PushtEnv(), policy.config)
    frames = []
    render = (lambda env: frames.append(env.envs[0].render())) if output else None
    try:
        result = rollout(
            environment,
            policy,
            env_before,
            env_after,
            *processors,
            seeds=seeds,
            render_callback=render,
        )
        records = summarize_rollout(result, seeds)
        if output:
            _save_video(frames[: records[0]["steps"]], output / "videos/episode.mp4")
        return records
    finally:
        environment.close()


def summarize_rollout(rollout: dict, seeds: list[int]) -> list[dict]:
    """Reduce native rollout tensors through each episode's first done step only.

    Args:
        rollout: Native LeRobot reward, success, and done arrays on CPU.
        seeds: The exact simulator reset seeds in batch order.
    Returns:
        Paired trial records that exclude all post-reset actions and rewards.
    Raises:
        ValueError: Shapes differ or an episode has no termination boundary.
    """
    rewards, successes, done = (
        np.asarray(rollout[key]) for key in ("reward", "success", "done")
    )
    if (
        rewards.ndim != 2
        or rewards.shape != successes.shape
        or rewards.shape != done.shape
    ):
        raise ValueError("Native rollout arrays must have matching batch/time shapes")
    if len(seeds) != len(done) or not done.any(axis=1).all():
        raise ValueError("Every native rollout must have a first termination boundary")
    records = []
    for index, seed in enumerate(seeds):
        stop = int(done[index].argmax()) + 1
        records.append(
            {
                "seed": seed,
                "steps": stop,
                "sum_reward": float(rewards[index, :stop].sum()),
                "max_reward": float(rewards[index, :stop].max()),
                "success": bool(successes[index, :stop].any()),
            }
        )
    return records


def _save_video(frames: list[np.ndarray], output: Path) -> None:
    import av

    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
