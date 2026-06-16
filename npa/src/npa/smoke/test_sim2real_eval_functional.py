"""Sim2Real eval container functional golden eval (FrankaPickPlace on CUDA)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_cuda_available() -> CheckResult:
    if not torch.cuda.is_available():
        return CheckResult("cuda available", False, "torch.cuda.is_available() is False")
    return CheckResult("cuda available", True, torch.cuda.get_device_name(0))


def check_franka_pick_place_rollout() -> CheckResult:
    from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

    env = FrankaPickPlaceEnv(
        EnvConfig(
            n_envs=2,
            enable_cameras=False,
            max_episode_steps=16,
            domain_randomize=False,
            action_space="cartesian",
        )
    )
    obs = env.reset()
    if obs is None:
        return CheckResult("franka pick-place rollout", False, "reset returned None")
    for _ in range(4):
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, _reward, _done, _info = env.step(action)
        if obs is None:
            return CheckResult("franka pick-place rollout", False, "step returned None obs")
    return CheckResult(
        "franka pick-place rollout",
        True,
        f"num_envs={env.num_envs} device={env.device}",
    )


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_cuda_available,
        check_franka_pick_place_rollout,
    ]
    failed = 0
    for check in checks:
        result = check()
        state = "PASS" if result.ok else "FAIL"
        print(f"[{state}] {result.name}: {result.detail}")
        if not result.ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
