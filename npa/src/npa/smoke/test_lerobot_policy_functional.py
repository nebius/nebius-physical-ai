"""LeRobot policy container functional golden eval (short train + eval on GPU)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _run(command: list[str], *, log_path: Path, timeout: int) -> tuple[int, str]:
    env = {
        **os.environ,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true",
    }
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    log_path.write_text(result.stdout)
    return result.returncode, result.stdout


def _find_checkpoint(train_dir: Path) -> Path | None:
    candidate = train_dir / "checkpoints" / "last" / "pretrained_model"
    if candidate.is_dir():
        return candidate
    return None


def check_short_train(state: Path) -> CheckResult:
    train_dir = state / "train"
    log_path = state / "train.log"
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--dataset.repo_id=lerobot/pusht",
        f"--output_dir={train_dir}",
        "--steps=20",
        "--batch_size=8",
        "--num_workers=0",
        "--save_freq=20",
        "--eval_freq=0",
        "--device=cuda",
    ]
    code, output = _run(command, log_path=log_path, timeout=900)
    checkpoint = _find_checkpoint(train_dir)
    if code != 0 or checkpoint is None:
        return CheckResult("short train", False, f"exit={code} checkpoint={checkpoint}")
    return CheckResult("short train", True, str(checkpoint))


def check_short_eval(state: Path) -> CheckResult:
    checkpoint = _find_checkpoint(state / "train")
    if checkpoint is None:
        return CheckResult("short eval", False, "missing checkpoint from train stage")
    eval_dir = state / "eval"
    log_path = state / "eval.log"
    command = [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        f"--output_dir={eval_dir}",
        "--env.type=pusht",
        "--eval.n_episodes=1",
        "--device=cuda",
    ]
    code, _output = _run(command, log_path=log_path, timeout=600)
    if code != 0:
        return CheckResult("short eval", False, f"exit={code}")
    return CheckResult("short eval", True, str(eval_dir))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="npa-policy-smoke-") as tmp:
        state = Path(tmp)
        checks: list[Callable[[], CheckResult]] = [
            lambda: check_short_train(state),
            lambda: check_short_eval(state),
        ]
        failed = 0
        for check in checks:
            result = check()
            state_label = "PASS" if result.ok else "FAIL"
            print(f"[{state_label}] {result.name}: {result.detail}")
            if not result.ok:
                failed += 1
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
