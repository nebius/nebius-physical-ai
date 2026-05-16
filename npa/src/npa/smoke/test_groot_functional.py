"""Standalone NVIDIA GR00T functional smoke checks.

This script runs NVIDIA's current standalone PyTorch inference example on the
included DROID sample dataset. It is intended for GPU GR00T workbench VMs.

Run with:
    python -m npa.smoke.test_groot_functional
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_GROOT_REPO = Path("/opt/groot/Isaac-GR00T")
DEFAULT_MODEL = "nvidia/GR00T-N1.7-3B"
DEFAULT_DATASET = "demo_data/droid_sample"
DEFAULT_EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    repo: Path
    model_path: str
    dataset_path: str
    embodiment_tag: str
    action_horizon: int


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_repo_available(state: SmokeState) -> CheckResult:
    script = state.repo / "scripts" / "deployment" / "standalone_inference_script.py"
    if not script.exists():
        return CheckResult("standalone inference script exists", False, f"missing {script}")
    return CheckResult("standalone inference script exists", True, f"script: {script}")


def check_uv_available(state: SmokeState) -> CheckResult:
    uv = shutil.which("uv")
    if uv is None:
        return CheckResult("uv available", False, "uv not found on PATH")
    return CheckResult("uv available", True, f"executable: {uv}")


def check_standalone_inference(state: SmokeState) -> CheckResult:
    command = [
        "uv",
        "run",
        "python",
        "scripts/deployment/standalone_inference_script.py",
        "--model-path",
        state.model_path,
        "--dataset-path",
        state.dataset_path,
        "--embodiment-tag",
        state.embodiment_tag,
        "--traj-ids",
        "1",
        "--inference-mode",
        "pytorch",
        "--action-horizon",
        str(state.action_horizon),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=state.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(os.environ.get("GROOT_SMOKE_TIMEOUT", "1800")),
            check=False,
        )
    except Exception as exc:
        return CheckResult("run standalone GR00T inference", False, _format_exception(exc))

    output = (result.stdout + "\n" + result.stderr).strip()
    if len(output) > 1000:
        output = output[-1000:]
    if result.returncode != 0:
        return CheckResult(
            "run standalone GR00T inference",
            False,
            f"exit code {result.returncode}; output tail: {output}",
        )
    return CheckResult("run standalone GR00T inference", True, f"output tail: {output}")


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    repo = Path(os.environ.get("GROOT_REPO", str(DEFAULT_GROOT_REPO)))
    state = SmokeState(
        repo=repo,
        model_path=os.environ.get("GROOT_SMOKE_MODEL", DEFAULT_MODEL),
        dataset_path=os.environ.get("GROOT_SMOKE_DATASET", DEFAULT_DATASET),
        embodiment_tag=os.environ.get("GROOT_SMOKE_EMBODIMENT", DEFAULT_EMBODIMENT_TAG),
        action_horizon=int(os.environ.get("GROOT_SMOKE_ACTION_HORIZON", "8")),
    )
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_repo_available,
        check_uv_available,
        check_standalone_inference,
    ]
    results: list[CheckResult] = []

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
