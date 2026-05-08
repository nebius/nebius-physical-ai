"""Standalone Isaac Lab functional smoke checks.

This script launches Isaac Sim headlessly, creates a Franka reach
manipulation task, steps it, and verifies observations are returned.

Run with:
    python -m npa.smoke.test_isaac_lab_functional
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable

from npa.smoke._versions import supported_tool_version


TASK_NAME = "Isaac-Reach-Franka-v0"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    isaaclab: Any | None = None
    simulation_app: Any | None = None
    env: Any | None = None
    observation: Any | None = None


def _package_version(module, distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
        return str(version) if version else "unknown"


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _observation_available(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value) and any(_observation_available(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and any(_observation_available(item) for item in value)

    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel()) > 0

    shape = getattr(value, "shape", None)
    if shape is not None:
        return all(int(dim) > 0 for dim in tuple(shape))

    try:
        return len(value) > 0
    except TypeError:
        return True


def _zero_action(env: Any) -> Any:
    try:
        import torch

        unwrapped = getattr(env, "unwrapped", env)
        num_envs = int(getattr(unwrapped, "num_envs", 1))
        device = getattr(unwrapped, "device", "cpu")
        action_shape = tuple(getattr(env.action_space, "shape", ()) or ())
        if action_shape[:1] != (num_envs,):
            action_shape = (num_envs, *action_shape)
        return torch.zeros(action_shape, device=device, dtype=torch.float32)
    except Exception:
        return env.action_space.sample()


def check_isaac_lab_version(state: SmokeState) -> CheckResult:
    try:
        expected = supported_tool_version("isaac-lab", __file__)
        state.isaaclab = importlib.import_module("isaaclab")
        version = _package_version(state.isaaclab, "isaaclab")
    except Exception as exc:
        return CheckResult("check isaaclab version", False, _format_exception(exc))

    if version != expected:
        return CheckResult(
            "check isaaclab version",
            False,
            f"expected version: {expected}; found: {version}",
        )
    return CheckResult("check isaaclab version", True, f"version: {version}")


def check_launch_runtime(state: SmokeState) -> CheckResult:
    if state.isaaclab is None:
        return CheckResult(
            "launch Isaac Sim runtime",
            False,
            "skipped because isaaclab import failed",
        )

    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        state.simulation_app = app_launcher.app
        if state.simulation_app is None:
            return CheckResult("launch Isaac Sim runtime", False, "AppLauncher.app is None")
        return CheckResult("launch Isaac Sim runtime", True, "headless app launched")
    except Exception as exc:
        return CheckResult("launch Isaac Sim runtime", False, _format_exception(exc))


def check_create_manipulation_env(state: SmokeState) -> CheckResult:
    if state.simulation_app is None:
        return CheckResult(
            f"create {TASK_NAME}",
            False,
            "skipped because Isaac Sim runtime launch failed",
        )

    try:
        gym = importlib.import_module("gymnasium")
        importlib.import_module("isaaclab_tasks")
        task_utils = importlib.import_module("isaaclab_tasks.utils")

        cfg = task_utils.load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
        scene_cfg = getattr(cfg, "scene", None)
        if scene_cfg is not None and hasattr(scene_cfg, "num_envs"):
            scene_cfg.num_envs = 1

        state.env = gym.make(TASK_NAME, cfg=cfg)
        reset_result = state.env.reset()
        state.observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if not _observation_available(state.observation):
            return CheckResult(f"create {TASK_NAME}", False, "reset returned no observations")
        return CheckResult(f"create {TASK_NAME}", True, "reset returned observations")
    except Exception as exc:
        return CheckResult(f"create {TASK_NAME}", False, _format_exception(exc))


def check_step_env(state: SmokeState) -> CheckResult:
    if state.env is None:
        return CheckResult("step environment 10 times", False, "skipped because env creation failed")

    try:
        for _ in range(10):
            action = _zero_action(state.env)
            step_result = state.env.step(action)
            if not isinstance(step_result, tuple) or not step_result:
                return CheckResult("step environment 10 times", False, "step returned no tuple")
            state.observation = step_result[0]

        if not _observation_available(state.observation):
            return CheckResult(
                "step environment 10 times",
                False,
                "last step returned no observations",
            )
        return CheckResult("step environment 10 times", True, "observations returned after step 10")
    except Exception as exc:
        return CheckResult("step environment 10 times", False, _format_exception(exc))


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
    state = SmokeState()
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_isaac_lab_version,
        check_launch_runtime,
        check_create_manipulation_env,
        check_step_env,
    ]
    results: list[CheckResult] = []

    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        _close_state(state)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
