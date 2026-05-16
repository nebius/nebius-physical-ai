"""Genesis container functional smoke checks."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


EXPECTED_GENESIS_VERSION = os.environ.get("GENESIS_VERSION", "0.4.6")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    output_path: Path
    gs: Any | None = None
    scene: Any | None = None
    box: Any | None = None
    cuda_memory_delta: int = 0
    output: dict[str, Any] | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _package_version(module: object, distribution_name: str) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _flatten_numbers(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    out: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            out.append(float(item))

    visit(value)
    return out


def _nvidia_smi_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return _format_exception(exc)
    return (result.stdout or result.stderr).strip()


def check_import_genesis(state: SmokeState) -> CheckResult:
    try:
        state.gs = importlib.import_module("genesis")
        version = _package_version(state.gs, "genesis-world")
        if version != EXPECTED_GENESIS_VERSION:
            return CheckResult(
                "check genesis version",
                False,
                f"expected version: {EXPECTED_GENESIS_VERSION}; found: {version}",
            )
        return CheckResult("check genesis version", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("check genesis version", False, _format_exception(exc))


def check_run_gpu_scene(state: SmokeState) -> CheckResult:
    if state.gs is None:
        return CheckResult("run GPU Genesis scene", False, "skipped because genesis import failed")

    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return CheckResult("run GPU Genesis scene", False, "CUDA is not available")

        gs = state.gs
        backend = getattr(gs, "gpu", None)
        if backend is None:
            return CheckResult("run GPU Genesis scene", False, "genesis.gpu backend is unavailable")

        before = torch.cuda.memory_allocated(0)
        marker = torch.ones((512, 512), device="cuda")
        marker_sum = float(marker.sum().item())
        state.cuda_memory_delta = max(0, torch.cuda.memory_allocated(0) - before)

        gs.init(backend=backend, logging_level="warning")
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.1)))
        scene.build()
        for _ in range(int(os.environ.get("GENESIS_SMOKE_STEPS", "8"))):
            scene.step()

        pos = _flatten_numbers(box.get_pos())
        if len(pos) < 3 or not all(math.isfinite(v) for v in pos[:3]):
            return CheckResult("run GPU Genesis scene", False, f"invalid box position: {pos}")

        state.scene = scene
        state.box = box
        state.output = {
            "status": "success",
            "genesis_version": _package_version(gs, "genesis-world"),
            "backend": "gpu",
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_memory_delta_bytes": state.cuda_memory_delta,
            "marker_sum": marker_sum,
            "box_position": pos[:3],
            "nvidia_smi": _nvidia_smi_snapshot(),
            "created_unix": round(time.time(), 3),
        }
        del marker
        torch.cuda.empty_cache()
        return CheckResult(
            "run GPU Genesis scene",
            True,
            f"box position: {pos[:3]}; cuda_memory_delta_bytes: {state.cuda_memory_delta}",
        )
    except Exception as exc:
        return CheckResult("run GPU Genesis scene", False, _format_exception(exc))


def check_output_file(state: SmokeState) -> CheckResult:
    if state.output is None:
        return CheckResult("write simulation output file", False, "skipped because simulation failed")
    try:
        state.output_path.write_text(json.dumps(state.output, indent=2))
        if not state.output_path.exists() or state.output_path.stat().st_size == 0:
            return CheckResult("write simulation output file", False, f"missing {state.output_path}")
        return CheckResult("write simulation output file", True, str(state.output_path))
    except Exception as exc:
        return CheckResult("write simulation output file", False, _format_exception(exc))


def check_gpu_usage_recorded(state: SmokeState) -> CheckResult:
    if state.output is None:
        return CheckResult("verify GPU usage was recorded", False, "skipped because simulation failed")
    if state.cuda_memory_delta <= 0:
        return CheckResult(
            "verify GPU usage was recorded",
            False,
            f"cuda_memory_delta_bytes: {state.cuda_memory_delta}",
        )
    return CheckResult(
        "verify GPU usage was recorded",
        True,
        f"cuda_memory_delta_bytes: {state.cuda_memory_delta}; nvidia-smi: {state.output.get('nvidia_smi')}",
    )


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="npa_genesis_functional_"))
    state = SmokeState(root=root, output_path=root / "genesis_smoke_summary.json")
    print(f"Temporary workspace: {root}")
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_import_genesis,
        check_run_gpu_scene,
        check_output_file,
        check_gpu_usage_recorded,
    ]
    results = []
    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        if os.environ.get("NPA_KEEP_SMOKE_OUTPUT", "").lower() not in {"1", "true", "yes"}:
            shutil.rmtree(root, ignore_errors=True)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
