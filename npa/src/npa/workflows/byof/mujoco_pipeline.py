"""mujoco workbench workflow stage (real implementation).

Mirrors the ``newton_pipeline.py`` module pattern: a stage function backed by
an argparse entry point (``main``), a module-level error type, and JSON
stage artifacts written to URIs.

Stage ``run`` executes N episodes of a scripted policy on a MuJoCo
manipulation task (real MuJoCo stepping, programmatic MJCF scenes) and writes
a trajectories + metrics report.

Invoked as ``python3 -m npa.workflows.byof.mujoco_pipeline``.  The ``mujoco``
import stays inside the stage function so the CLI surface imports on machines
without MuJoCo installed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

SCHEMA_RUN = "npa.workbench.mujoco_manip.run.v1"
TRAJECTORY_STRIDE = 20


class MujocoPipelineError(RuntimeError):
    """Raised when a mujoco pipeline invariant is not met."""


def _write_json_uri(uri: str, payload: Mapping[str, Any]) -> None:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = Path(parsed.path if parsed.scheme == "file" else uri)
    else:
        raise MujocoPipelineError(
            f"unsupported output URI scheme {parsed.scheme!r} in {uri!r}; "
            "mujoco run writes to file:// URIs and local paths"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    *,
    task: str,
    policy: str,
    output_uri: str,
    episodes: int = 10,
    seed: int = 0,
    max_steps: int = 500,
) -> Mapping[str, Any]:
    """Run N episodes of a scripted policy; write trajectories + metrics."""
    from npa.workbench import mujoco_manip

    if not task:
        raise MujocoPipelineError("task is required")
    if not policy:
        raise MujocoPipelineError("policy is required")
    if not output_uri:
        raise MujocoPipelineError("output_uri is required")
    if episodes < 1:
        raise MujocoPipelineError("episodes must be >= 1")
    if max_steps < 1:
        raise MujocoPipelineError("max_steps must be >= 1")

    short_task = task.split(".", 1)[-1]
    policy_name = policy.split(":", 1)[1] if ":" in policy else policy

    episode_reports: list[dict[str, Any]] = []
    successes = 0
    for index in range(episodes):
        episode_seed = seed + index
        env = mujoco_manip.make_env(short_task, max_steps=max_steps)
        policy_fn = mujoco_manip.get_policy(short_task, policy_name, seed=episode_seed)
        obs = env.reset(seed=episode_seed)
        trajectory: list[list[float]] = []
        total_reward = 0.0
        success = False
        steps = 0
        for step in range(1, max_steps + 1):
            obs, reward, done, info = env.step(policy_fn(obs))
            total_reward += float(reward)
            steps = step
            if step % TRAJECTORY_STRIDE == 0 or done:
                trajectory.append([float(v) for v in obs])
            if info.get("success"):
                success = True
            if done:
                break
        successes += int(success)
        episode_reports.append(
            {
                "index": index,
                "seed": episode_seed,
                "success": success,
                "steps": steps,
                "total_reward": total_reward,
                "final_tip_pos": info.get("tip_pos"),
                "final_tilt": info.get("tilt"),
                "trajectory_stride": TRAJECTORY_STRIDE,
                "trajectory_obs": trajectory,
            }
        )

    n = len(episode_reports)
    report: dict[str, Any] = {
        "schema": SCHEMA_RUN,
        "stage": "run",
        "status": "completed",
        "inputs": {
            "task": task,
            "policy": policy,
            "episodes": episodes,
            "seed": seed,
            "max_steps": max_steps,
        },
        "summary": {
            "episodes": n,
            "successes": successes,
            "success_rate": (successes / n) if n else 0.0,
            "mean_steps": (sum(e["steps"] for e in episode_reports) / n if n else 0.0),
        },
        "episodes": episode_reports,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_uri(output_uri, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mujoco_pipeline",
        description="MuJoCo contact-rich manipulation rollouts.",
    )
    commands = parser.add_subparsers(dest="stage", required=True)

    run_cmd = commands.add_parser("run")
    run_cmd.add_argument(
        "--task",
        required=True,
        choices=("peg_insertion", "screw_driving", "reach"),
    )
    run_cmd.add_argument(
        "--policy",
        required=True,
        help="Scripted policy: 'scripted:<expert|noisy|random>'.",
    )
    run_cmd.add_argument("--episodes", type=int, default=10)
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--output-uri", required=True)
    run_cmd.add_argument("--max-steps", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "run":
        run(
            task=args.task,
            policy=args.policy,
            output_uri=args.output_uri,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        return 0
    raise MujocoPipelineError(f"unknown stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
