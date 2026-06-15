"""VLM-signal RL container functional golden eval (real optimizer step on GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from npa.sim2real.rl_signal import SCHEMA_RL_SIGNAL
from npa.workbench.lerobot.policy_container import run_vlm_signal_training_step


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _fixture_signal() -> dict:
    return {
        "schema": SCHEMA_RL_SIGNAL,
        "run_id": "golden-smoke",
        "per_step": [
            {
                "step": 0,
                "reward": 0.8,
                "target": {
                    "nl_correction": "Move gripper closer to the cube center.",
                    "action_delta": [0.01, 0.0, 0.0, 0.0],
                },
            }
        ],
    }


def check_cuda_available() -> CheckResult:
    if not torch.cuda.is_available():
        return CheckResult("cuda available", False, "torch.cuda.is_available() is False")
    return CheckResult("cuda available", True, torch.cuda.get_device_name(0))


def check_vlm_signal_step() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="npa-vlm-rl-smoke-") as tmp:
        signal_path = Path(tmp) / "signal.json"
        output_dir = Path(tmp) / "update"
        signal_path.write_text(json.dumps(_fixture_signal()), encoding="utf-8")
        payload = json.loads(signal_path.read_text(encoding="utf-8"))
        try:
            update = run_vlm_signal_training_step(payload, output_dir=output_dir)
        except Exception as exc:
            return CheckResult("vlm-signal-step", False, str(exc))
        if not Path(update.checkpoint_path).exists():
            return CheckResult("vlm-signal-step", False, "missing checkpoint")
        return CheckResult(
            "vlm-signal-step",
            True,
            f"delta_l2={update.policy_delta_l2:.6f} checkpoint={update.checkpoint_path}",
        )


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [check_cuda_available, check_vlm_signal_step]
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
