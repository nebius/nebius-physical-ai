"""Sim2Real envgen container functional golden eval (env manifest + Genesis CUDA)."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from npa.workflows.sim2real_envgen import EnvGenConfig, build_scene_spec, generate_raw_envs


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_raw_env_generation() -> CheckResult:
    config = EnvGenConfig(
        run_id="golden-smoke",
        output_uri="s3://golden-smoke/local/",
        env_count=16,
        shard_index=0,
        shard_count=1,
        seed=7,
        scene_spec=build_scene_spec(seed=7),
    )
    envs = generate_raw_envs(config)
    if len(envs) != 16:
        return CheckResult("raw env generation", False, f"expected 16 envs, got {len(envs)}")
    with tempfile.TemporaryDirectory(prefix="npa-envgen-smoke-") as tmp:
        path = Path(tmp) / "envs.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in envs:
                handle.write(json.dumps(row) + "\n")
        if path.stat().st_size <= 0:
            return CheckResult("raw env generation", False, "empty jsonl")
    return CheckResult("raw env generation", True, f"rows={len(envs)}")


def check_genesis_cuda_step() -> CheckResult:
    if not torch.cuda.is_available():
        return CheckResult("genesis cuda step", False, "CUDA unavailable")
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv
    except Exception as exc:
        return CheckResult("genesis cuda step", False, f"import failed: {exc}")
    env = FrankaPickPlaceEnv(
        EnvConfig(
            n_envs=1,
            enable_cameras=False,
            max_episode_steps=8,
            domain_randomize=False,
        )
    )
    obs = env.reset()
    if obs is None:
        return CheckResult("genesis cuda step", False, "reset returned None")
    action = torch.zeros((1, env.num_actions), device=env.device)
    env.step(action)
    return CheckResult("genesis cuda step", True, f"device={env.device}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [check_raw_env_generation, check_genesis_cuda_step]
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
