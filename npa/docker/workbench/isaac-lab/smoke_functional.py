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


EXPECTED_ISAAC_LAB_VERSION = os.environ.get("ISAAC_LAB_VERSION", "3.0.0b2.post1")
TASK_NAME = os.environ.get("ISAAC_LAB_SMOKE_TASK", "Isaac-Reach-Franka-v0")
NUM_ENVS = int(os.environ.get("ISAAC_LAB_SMOKE_NUM_ENVS", "64"))
STEP_COUNT = int(os.environ.get("ISAAC_LAB_SMOKE_STEP_COUNT", "100"))
REPLAY_STEPS = int(os.environ.get("ISAAC_LAB_SMOKE_REPLAY_STEPS", "50"))
OUTPUT_ROOT = Path(os.environ.get("NPA_ISAAC_LAB_OUTPUT_DIR", "/workspace/isaaclab/npa-runs"))

if NUM_ENVS < 1 or STEP_COUNT < 1 or REPLAY_STEPS < 1:
    raise ValueError("smoke counts must all be positive")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    output_dir: Path
    trace_path: Path
    replay_summary_path: Path
    simulation_app: Any | None = None
    env: Any | None = None
    step_summary: dict[str, Any] | None = None
    replay_summary: dict[str, Any] | None = None


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

        app_launcher = AppLauncher(visualizer="none")
        state.simulation_app = app_launcher.app
        if state.simulation_app is None:
            return CheckResult("launch Isaac Sim runtime", False, "AppLauncher.app is None")
        return CheckResult("launch Isaac Sim runtime", True, "headless app launched")
    except Exception as exc:
        return CheckResult("launch Isaac Sim runtime", False, _format_exception(exc))


def check_environment_steps(state: SmokeState) -> CheckResult:
    if state.simulation_app is None:
        return CheckResult("step vectorized environment", False, "skipped because runtime launch failed")
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        started = time.time()
        state.env = _make_env(TASK_NAME, NUM_ENVS, device)
        state.env.reset()
        reward_total = 0.0
        for step in range(STEP_COUNT):
            _, rewards, _, _, _ = state.env.step(_sample_action(state.env, device))
            reward_total += float(torch.as_tensor(rewards).mean().item())
            if (step + 1) % max(1, min(25, STEP_COUNT)) == 0 or (step + 1) == STEP_COUNT:
                print(f"ISAAC_LAB_SMOKE_ENV_STEP step={step + 1}/{STEP_COUNT}", flush=True)

        trace = {
            "format": "npa_isaac_lab_environment_step_trace_v1",
            "task": TASK_NAME,
            "action_source": "action_space_sample",
            "num_envs": NUM_ENVS,
            "steps": STEP_COUNT,
            "device": device,
            "created_unix": round(time.time(), 3),
        }
        state.trace_path.write_text(json.dumps(trace, indent=2))
        state.step_summary = {
            "status": "success",
            "task": TASK_NAME,
            "num_envs": NUM_ENVS,
            "steps": STEP_COUNT,
            "device": device,
            "mean_reward": reward_total / STEP_COUNT,
            "trace_path": str(state.trace_path),
            "duration_seconds": round(time.time() - started, 3),
        }
        (state.output_dir / "npa_isaac_lab_environment_step_summary.json").write_text(
            json.dumps(state.step_summary, indent=2)
        )
        return CheckResult(
            "step vectorized environment",
            True,
            f"task={TASK_NAME}; num_envs={NUM_ENVS}; steps={STEP_COUNT}; device={device}",
        )
    except Exception as exc:
        return CheckResult("step vectorized environment", False, _format_exception(exc))


def check_trace(state: SmokeState) -> CheckResult:
    if not state.trace_path.exists():
        return CheckResult("verify environment trace", False, f"missing {state.trace_path}")
    try:
        trace = json.loads(state.trace_path.read_text())
    except Exception as exc:
        return CheckResult("verify environment trace", False, _format_exception(exc))
    if trace.get("format") != "npa_isaac_lab_environment_step_trace_v1":
        return CheckResult("verify environment trace", False, f"unexpected trace: {trace}")
    return CheckResult("verify environment trace", True, str(state.trace_path))


def check_replay(state: SmokeState) -> CheckResult:
    if state.simulation_app is None:
        return CheckResult("replay environment steps", False, "skipped because runtime launch failed")
    if not state.trace_path.exists():
        return CheckResult("replay environment steps", False, "skipped because trace is missing")
    if state.env is None:
        return CheckResult("replay environment steps", False, "skipped because environment is missing")
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        started = time.time()
        state.env.reset()
        reward_total = 0.0
        for step in range(REPLAY_STEPS):
            _, rewards, _, _, _ = state.env.step(_sample_action(state.env, device))
            reward_total += float(torch.as_tensor(rewards).mean().item())
        state.replay_summary = {
            "status": "success",
            "task": TASK_NAME,
            "trace": str(state.trace_path),
            "steps": REPLAY_STEPS,
            "device": device,
            "mean_reward": reward_total / REPLAY_STEPS,
            "duration_seconds": round(time.time() - started, 3),
            "output_path": str(state.replay_summary_path),
        }
        state.replay_summary_path.write_text(json.dumps(state.replay_summary, indent=2))
        return CheckResult("replay environment steps", True, f"steps={REPLAY_STEPS}; device={device}")
    except Exception as exc:
        return CheckResult("replay environment steps", False, _format_exception(exc))


def check_replay_metrics(state: SmokeState) -> CheckResult:
    if not state.replay_summary_path.exists():
        return CheckResult("verify replay metrics", False, f"missing {state.replay_summary_path}")
    try:
        summary = json.loads(state.replay_summary_path.read_text())
    except Exception as exc:
        return CheckResult("verify replay metrics", False, _format_exception(exc))
    if summary.get("status") != "success" or "mean_reward" not in summary:
        return CheckResult("verify replay metrics", False, f"unexpected summary: {summary}")
    return CheckResult(
        "verify replay metrics",
        True,
        f"{state.replay_summary_path}; mean_reward={summary['mean_reward']:.6f}",
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
        trace_path=output_dir / "npa_isaac_lab_environment_step_trace.json",
        replay_summary_path=output_dir / "replay" / "npa_isaac_lab_replay_summary.json",
    )
    state.replay_summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Smoke workspace: {state.output_dir}")

    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_version,
        check_launch_runtime,
        check_environment_steps,
        check_trace,
        check_replay,
        check_replay_metrics,
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
