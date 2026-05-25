"""Isaac Lab container functional smoke checks."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


EXPECTED_ISAAC_LAB_VERSION = os.environ.get("ISAAC_LAB_VERSION", "2.3.2.post1")
TASK_NAME = os.environ.get("ISAAC_LAB_SMOKE_TASK", "Isaac-Reach-Franka-v0")
NUM_ENVS = int(os.environ.get("ISAAC_LAB_SMOKE_NUM_ENVS", "64"))
TRAIN_STEPS = int(os.environ.get("ISAAC_LAB_SMOKE_TRAIN_STEPS", "100"))
EVAL_EPISODES = int(os.environ.get("ISAAC_LAB_SMOKE_EVAL_EPISODES", "1"))
EVAL_MAX_STEPS = int(os.environ.get("ISAAC_LAB_SMOKE_EVAL_MAX_STEPS", "50"))
OUTPUT_ROOT = Path(os.environ.get("NPA_ISAAC_LAB_OUTPUT_DIR", "/workspace/isaaclab/npa-runs"))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    output_dir: Path
    checkpoint_path: Path
    eval_summary_path: Path
    simulation_app: Any | None = None
    env: Any | None = None
    train_summary: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _sample_action(env: Any, device: str) -> Any:
    import torch

    return torch.as_tensor(env.action_space.sample(), device=device, dtype=torch.float32)


def _make_env(task: str, num_envs: int, device: str) -> Any:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    return gym.make(task, cfg=env_cfg)


def check_version(state: SmokeState) -> CheckResult:
    try:
        version = metadata.version("isaaclab")
    except Exception as exc:
        return CheckResult("check Isaac Lab version", False, _format_exception(exc))
    if version != EXPECTED_ISAAC_LAB_VERSION:
        return CheckResult(
            "check Isaac Lab version",
            False,
            f"expected {EXPECTED_ISAAC_LAB_VERSION}; found {version}",
        )
    return CheckResult("check Isaac Lab version", True, f"version: {version}")


def check_launch_runtime(state: SmokeState) -> CheckResult:
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        state.simulation_app = app_launcher.app
        if state.simulation_app is None:
            return CheckResult("launch Isaac Sim runtime", False, "AppLauncher.app is None")
        return CheckResult("launch Isaac Sim runtime", True, "headless app launched")
    except Exception as exc:
        return CheckResult("launch Isaac Sim runtime", False, _format_exception(exc))


def check_train(state: SmokeState) -> CheckResult:
    if state.simulation_app is None:
        return CheckResult("run short training session", False, "skipped because runtime launch failed")
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        started = time.time()
        state.env = _make_env(TASK_NAME, NUM_ENVS, device)
        state.env.reset()
        reward_total = 0.0
        for step in range(TRAIN_STEPS):
            _, rewards, _, _, _ = state.env.step(_sample_action(state.env, device))
            reward_total += float(torch.as_tensor(rewards).mean().item())
            if (step + 1) % max(1, min(25, TRAIN_STEPS)) == 0 or (step + 1) == TRAIN_STEPS:
                print(f"ISAAC_LAB_SMOKE_TRAIN_STEP step={step + 1}/{TRAIN_STEPS}", flush=True)

        checkpoint = {
            "format": "npa_isaac_lab_random_policy_v1",
            "task": TASK_NAME,
            "policy": "action_space_sample",
            "num_envs": NUM_ENVS,
            "steps": TRAIN_STEPS,
            "device": device,
            "created_unix": round(time.time(), 3),
        }
        state.checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
        state.train_summary = {
            "status": "success",
            "task": TASK_NAME,
            "num_envs": NUM_ENVS,
            "steps": TRAIN_STEPS,
            "device": device,
            "mean_reward": reward_total / TRAIN_STEPS,
            "checkpoint_path": str(state.checkpoint_path),
            "duration_seconds": round(time.time() - started, 3),
        }
        (state.output_dir / "npa_isaac_lab_train_summary.json").write_text(json.dumps(state.train_summary, indent=2))
        return CheckResult(
            "run short training session",
            True,
            f"task={TASK_NAME}; num_envs={NUM_ENVS}; steps={TRAIN_STEPS}; device={device}",
        )
    except Exception as exc:
        return CheckResult("run short training session", False, _format_exception(exc))


def check_checkpoint(state: SmokeState) -> CheckResult:
    if not state.checkpoint_path.exists():
        return CheckResult("verify checkpoint output", False, f"missing {state.checkpoint_path}")
    try:
        checkpoint = json.loads(state.checkpoint_path.read_text())
    except Exception as exc:
        return CheckResult("verify checkpoint output", False, _format_exception(exc))
    if checkpoint.get("format") != "npa_isaac_lab_random_policy_v1":
        return CheckResult("verify checkpoint output", False, f"unexpected checkpoint: {checkpoint}")
    return CheckResult("verify checkpoint output", True, str(state.checkpoint_path))


def check_eval(state: SmokeState) -> CheckResult:
    if state.simulation_app is None:
        return CheckResult("run checkpoint eval", False, "skipped because runtime launch failed")
    if not state.checkpoint_path.exists():
        return CheckResult("run checkpoint eval", False, "skipped because checkpoint is missing")
    if state.env is None:
        return CheckResult("run checkpoint eval", False, "skipped because training env is missing")
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        started = time.time()
        episode_results = []
        for episode in range(EVAL_EPISODES):
            state.env.reset()
            reward_total = 0.0
            steps_ran = 0
            for step in range(EVAL_MAX_STEPS):
                _, rewards, terminated, truncated, _ = state.env.step(_sample_action(state.env, device))
                reward_total += float(torch.as_tensor(rewards).mean().item())
                steps_ran = step + 1
                done = bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
                if done:
                    break
            episode_results.append({"episode": episode + 1, "steps": steps_ran, "reward": reward_total})
            print(
                f"ISAAC_LAB_SMOKE_EVAL_EPISODE episode={episode + 1}/{EVAL_EPISODES} "
                f"steps={steps_ran} reward={reward_total:.6f}",
                flush=True,
            )

        mean_reward = sum(item["reward"] for item in episode_results) / EVAL_EPISODES
        checkpoint_info = json.loads(state.checkpoint_path.read_text())
        state.eval_summary = {
            "status": "success",
            "task": TASK_NAME,
            "checkpoint": str(state.checkpoint_path),
            "checkpoint_format": checkpoint_info.get("format", "unknown"),
            "num_episodes": EVAL_EPISODES,
            "max_steps_per_episode": EVAL_MAX_STEPS,
            "device": device,
            "mean_reward": mean_reward,
            "episodes": episode_results,
            "duration_seconds": round(time.time() - started, 3),
            "output_path": str(state.eval_summary_path),
        }
        state.eval_summary_path.write_text(json.dumps(state.eval_summary, indent=2))
        return CheckResult("run checkpoint eval", True, f"episodes={EVAL_EPISODES}; device={device}")
    except Exception as exc:
        return CheckResult("run checkpoint eval", False, _format_exception(exc))


def check_eval_metrics(state: SmokeState) -> CheckResult:
    if not state.eval_summary_path.exists():
        return CheckResult("verify eval metrics output", False, f"missing {state.eval_summary_path}")
    try:
        summary = json.loads(state.eval_summary_path.read_text())
    except Exception as exc:
        return CheckResult("verify eval metrics output", False, _format_exception(exc))
    if summary.get("status") != "success" or "mean_reward" not in summary:
        return CheckResult("verify eval metrics output", False, f"unexpected summary: {summary}")
    return CheckResult(
        "verify eval metrics output",
        True,
        f"{state.eval_summary_path}; mean_reward={summary['mean_reward']:.6f}",
    )


def _close_state(state: SmokeState) -> None:
    if state.env is not None:
        close = getattr(state.env, "close", None)
        if callable(close):
            close()
    if state.simulation_app is not None:
        close = getattr(state.simulation_app, "close", None)
        if callable(close):
            close()


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    run_id = uuid.uuid4().hex[:10]
    output_dir = OUTPUT_ROOT / f"npa_isaac_lab_smoke_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    state = SmokeState(
        output_dir=output_dir,
        checkpoint_path=output_dir / "npa_isaac_lab_random_policy_checkpoint.json",
        eval_summary_path=output_dir / "eval" / "npa_isaac_lab_eval_summary.json",
    )
    state.eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Smoke workspace: {state.output_dir}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_version,
        check_launch_runtime,
        check_train,
        check_checkpoint,
        check_eval,
        check_eval_metrics,
    ]
    results = []
    for check in checks:
        result = check(state)
        results.append(result)
        _print_result(result)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
