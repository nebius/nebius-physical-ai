"""Standalone LeRobot functional smoke checks.

This script runs a short real training job and evaluates the produced
checkpoint. It is intended for GPU VMs with LeRobot installed.

Run with:
    python -m npa.smoke.test_lerobot_functional
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable

from npa.smoke._versions import expected_lerobot_version, train_env_eval_arg_for_version


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    train_dir: Path
    eval_dir: Path
    checkpoint_dir: Path | None = None
    train_log: Path | None = None
    eval_log: Path | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...<truncated>...\n" + text[-limit:]


def _run_command(
    command: list[str],
    *,
    log_path: Path,
    timeout: int,
) -> tuple[int, str]:
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
    checkpoints_dir = train_dir / "checkpoints"
    candidates = [
        checkpoints_dir / "last" / "pretrained_model",
    ]
    if checkpoints_dir.exists():
        numbered = sorted(
            [
                path
                for path in checkpoints_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
            key=lambda path: int(path.name),
            reverse=True,
        )
        candidates.extend(path / "pretrained_model" for path in numbered)

    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "config.json").exists()
            and (
                (candidate / "model.safetensors").exists()
                or (candidate / "pytorch_model.bin").exists()
            )
        ):
            return candidate
    return None


def check_lerobot_version(state: SmokeState) -> CheckResult:
    try:
        expected = expected_lerobot_version(__file__)
        version = metadata.version("lerobot")
    except Exception as exc:
        return CheckResult("check lerobot version", False, _format_exception(exc))

    if version != expected:
        return CheckResult(
            "check lerobot version",
            False,
            f"expected version: {expected}; found: {version}",
        )
    return CheckResult("check lerobot version", True, f"version: {version}")


def check_lerobot_train(state: SmokeState) -> CheckResult:
    state.train_log = state.root / "lerobot_train.log"
    version = expected_lerobot_version(__file__)
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--dataset.repo_id=lerobot/pusht",
        f"--output_dir={state.train_dir}",
        "--steps=50",
        "--save_freq=50",
        train_env_eval_arg_for_version(version, 1_000_000),
        "--log_freq=10",
        "--batch_size=8",
        "--num_workers=4",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]
    try:
        code, output = _run_command(command, log_path=state.train_log, timeout=1800)
    except Exception as exc:
        return CheckResult("run lerobot-train for 50 steps", False, _format_exception(exc))

    if code != 0:
        return CheckResult(
            "run lerobot-train for 50 steps",
            False,
            f"exit code {code}; log: {state.train_log}\n{_tail(output)}",
        )
    return CheckResult(
        "run lerobot-train for 50 steps",
        True,
        f"output_dir: {state.train_dir}; log: {state.train_log}",
    )


def check_checkpoint_created(state: SmokeState) -> CheckResult:
    checkpoint = _find_checkpoint(state.train_dir)
    state.checkpoint_dir = checkpoint
    if checkpoint is None:
        return CheckResult(
            "checkpoint directory created",
            False,
            f"no pretrained_model checkpoint found under {state.train_dir / 'checkpoints'}",
        )
    return CheckResult("checkpoint directory created", True, str(checkpoint))


def check_lerobot_eval(state: SmokeState) -> CheckResult:
    if state.checkpoint_dir is None:
        return CheckResult(
            "run lerobot-eval against produced checkpoint",
            False,
            "skipped because no checkpoint directory was found",
        )

    state.eval_log = state.root / "lerobot_eval.log"
    command = [
        "lerobot-eval",
        "--policy.type=act",
        f"--policy.pretrained_path={state.checkpoint_dir}",
        "--env.type=pusht",
        f"--output_dir={state.eval_dir}",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--policy.device=cuda",
        "--policy.use_amp=false",
    ]
    try:
        code, output = _run_command(command, log_path=state.eval_log, timeout=900)
    except Exception as exc:
        return CheckResult(
            "run lerobot-eval against produced checkpoint",
            False,
            _format_exception(exc),
        )

    if code != 0:
        return CheckResult(
            "run lerobot-eval against produced checkpoint",
            False,
            f"exit code {code}; log: {state.eval_log}\n{_tail(output)}",
        )
    return CheckResult(
        "run lerobot-eval against produced checkpoint",
        True,
        f"output_dir: {state.eval_dir}; log: {state.eval_log}",
    )


def check_eval_output_created(state: SmokeState) -> CheckResult:
    eval_info = state.eval_dir / "eval_info.json"
    if not eval_info.exists():
        return CheckResult("eval output file exists", False, f"missing {eval_info}")
    return CheckResult("eval output file exists", True, str(eval_info))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="npa_lerobot_functional_"))
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    state = SmokeState(root=root, train_dir=root / "train", eval_dir=root / "eval")

    print(f"Temporary workspace: {root}")
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_lerobot_version,
        check_lerobot_train,
        check_checkpoint_created,
        check_lerobot_eval,
        check_eval_output_created,
    ]
    results: list[CheckResult] = []

    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
