"""Isaac Lab container environment smoke checks."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable


EXPECTED_ISAAC_LAB_VERSION = os.environ.get("ISAAC_LAB_VERSION", "2.3.2.post1")
EXPECTED_ISAAC_SIM_VERSION = os.environ.get("ISAAC_SIM_VERSION", "5.1.0.0")
RT_GPU_MARKERS = ("L40S", "RTX")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _find_isaaclab_root() -> Path | None:
    for candidate in (Path("/workspace/isaaclab"), Path("/workspace/IsaacLab")):
        if (candidate / "isaaclab.sh").is_file():
            return candidate
    return None


def check_import_versions() -> CheckResult:
    try:
        isaaclab = importlib.import_module("isaaclab")
        lab_version = metadata.version("isaaclab")
        if lab_version != EXPECTED_ISAAC_LAB_VERSION:
            return CheckResult(
                "import Isaac Lab core modules",
                False,
                f"expected isaaclab {EXPECTED_ISAAC_LAB_VERSION}; found {lab_version}",
            )
        sim_detail = "isaacsim metadata unavailable"
        for distribution in ("isaacsim", "isaacsim-kernel", "isaacsim-app"):
            try:
                sim_version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
            if sim_version != EXPECTED_ISAAC_SIM_VERSION:
                return CheckResult(
                    "import Isaac Lab core modules",
                    False,
                    f"expected {distribution} {EXPECTED_ISAAC_SIM_VERSION}; found {sim_version}",
                )
            sim_detail = f"{distribution}={sim_version}"
            break
        if not Path("/isaac-sim/python.sh").is_file():
            return CheckResult("import Isaac Lab core modules", False, "/isaac-sim/python.sh not found")
        return CheckResult(
            "import Isaac Lab core modules",
            True,
            f"isaaclab={lab_version}; {sim_detail}; module={isaaclab.__name__}",
        )
    except Exception as exc:
        return CheckResult("import Isaac Lab core modules", False, _format_exception(exc))


def check_rt_gpu() -> CheckResult:
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return CheckResult("verify L40S/RT GPU access", False, "nvidia-smi not found")
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return CheckResult(
                "verify L40S/RT GPU access",
                False,
                f"nvidia-smi exited {result.returncode}: {(result.stderr or result.stdout).strip()}",
            )
        gpu_info = result.stdout.strip()
        if not any(marker in gpu_info for marker in RT_GPU_MARKERS):
            return CheckResult("verify L40S/RT GPU access", False, f"non-RT GPU reported: {gpu_info}")
        return CheckResult("verify L40S/RT GPU access", True, gpu_info)
    except Exception as exc:
        return CheckResult("verify L40S/RT GPU access", False, _format_exception(exc))


def check_launch_runtime_and_cuda() -> CheckResult:
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app
        if simulation_app is None:
            return CheckResult("launch Isaac Sim and check CUDA", False, "AppLauncher.app is None")
        importlib.import_module("isaaclab_tasks")

        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return CheckResult("launch Isaac Sim and check CUDA", False, "torch.cuda.is_available() is false")
        before = torch.cuda.memory_allocated(0)
        tensor = torch.ones((128, 128), device="cuda")
        allocated = torch.cuda.memory_allocated(0) - before
        total = float(tensor.sum().item())
        del tensor
        torch.cuda.empty_cache()
        return CheckResult(
            "launch Isaac Sim and check CUDA",
            True,
            f"headless app launched; gpu0={torch.cuda.get_device_name(0)}; "
            f"allocated_delta_bytes={allocated}; sum={total}",
        )
    except Exception as exc:
        return CheckResult("launch Isaac Sim and check CUDA", False, _format_exception(exc))


def _run_help(command: list[str], cwd: Path) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0 and "usage" in output.lower(), output[-600:]


def check_training_eval_entrypoints() -> CheckResult:
    try:
        root = _find_isaaclab_root()
        if root is None:
            return CheckResult("check training/eval entry points", False, "isaaclab.sh not found")
        isaaclab_sh = root / "isaaclab.sh"
        train_py = root / "scripts/reinforcement_learning/rsl_rl/train.py"
        play_py = root / "scripts/reinforcement_learning/rsl_rl/play.py"
        missing = [str(path) for path in (train_py, play_py) if not path.is_file()]
        if missing:
            return CheckResult("check training/eval entry points", False, "missing: " + ", ".join(missing))

        train_ok, train_output = _run_help([str(isaaclab_sh), "-p", str(train_py), "--help"], root)
        if not train_ok:
            return CheckResult("check training/eval entry points", False, f"train --help failed: {train_output}")
        eval_ok, eval_output = _run_help([str(isaaclab_sh), "-p", str(play_py), "--help"], root)
        if not eval_ok:
            return CheckResult("check training/eval entry points", False, f"eval --help failed: {eval_output}")
        return CheckResult(
            "check training/eval entry points",
            True,
            f"train={train_py}; eval={play_py}",
        )
    except Exception as exc:
        return CheckResult("check training/eval entry points", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_versions,
        check_rt_gpu,
        check_launch_runtime_and_cuda,
        check_training_eval_entrypoints,
    ]
    results = []
    for check in checks:
        result = check()
        results.append(result)
        _print_result(result)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
