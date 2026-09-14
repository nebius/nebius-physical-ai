"""Real upstream OpenArm Isaac Lab rollout/training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rollout", "train"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rollout(args: argparse.Namespace) -> dict[str, object]:
    import gymnasium as gym
    import numpy as np
    import openarm.tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    rewards = []
    observations = []
    started = time.time()
    try:
        env.reset(seed=args.seed)
        for _ in range(args.steps):
            action = torch.as_tensor(
                env.action_space.sample(), device=device, dtype=torch.float32
            )
            observation, reward, _terminated, _truncated, _info = env.step(action)
            rewards.append(float(torch.as_tensor(reward).mean().item()))
            policy = (
                observation.get("policy") if isinstance(observation, dict) else None
            )
            if policy is not None:
                observations.append(torch.as_tensor(policy)[0].detach().cpu().numpy())
    finally:
        env.close()
    if not rewards or not np.isfinite(rewards).all():
        raise RuntimeError("OpenArm Isaac rollout produced no finite rewards")
    trace = args.output_dir / "isaac_rollout.npz"
    np.savez_compressed(
        trace, reward=np.asarray(rewards), policy_observation=np.asarray(observations)
    )
    return {
        "schema": "npa.openarm.isaac_lab_rollout.v1",
        "status": "completed",
        "simulator": "isaac-lab",
        "mode": "rollout",
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "device": device,
        "mean_reward": float(np.mean(rewards)),
        "duration_seconds": time.time() - started,
        "artifact": {"path": trace.name, "bytes": trace.stat().st_size},
    }


def _train(args: argparse.Namespace) -> dict[str, object]:
    root = Path(os.environ.get("OPENARM_ISAAC_ROOT", "/opt/openarm/isaac"))
    script = root / "scripts/reinforcement_learning/rsl_rl/train.py"
    command = [
        sys.executable,
        str(script),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        str(args.seed),
        "--headless",
    ]
    started = time.time()
    completed = subprocess.run(
        command, cwd=args.output_dir, text=True, capture_output=True, check=False
    )
    (args.output_dir / "training_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (args.output_dir / "training_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-2000:])
    checkpoints = sorted(args.output_dir.rglob("model_*.pt"))
    if not checkpoints:
        raise RuntimeError("upstream RSL-RL training wrote no model_*.pt checkpoint")
    return {
        "schema": "npa.openarm.isaac_lab_training.v1",
        "status": "completed",
        "simulator": "isaac-lab",
        "mode": "train",
        "task": args.task,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "duration_seconds": time.time() - started,
        "checkpoints": [str(path.relative_to(args.output_dir)) for path in checkpoints],
    }


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_root = Path(os.environ.get("OPENARM_ISAAC_ROOT", "/opt/openarm/isaac"))
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(0, str(upstream_root / "source/openarm"))
    launcher = AppLauncher(args) if args.mode == "rollout" else None
    try:
        result = _rollout(args) if launcher is not None else _train(args)
        (args.output_dir / "isaac_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
