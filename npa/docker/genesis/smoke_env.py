"""Genesis container environment smoke checks."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Callable


EXPECTED_GENESIS_VERSION = os.environ.get("GENESIS_VERSION", "0.4.6")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


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


def _run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def check_import_genesis() -> CheckResult:
    try:
        genesis = importlib.import_module("genesis")
        version = _package_version(genesis, "genesis-world")
        if version != EXPECTED_GENESIS_VERSION:
            return CheckResult(
                "import genesis",
                False,
                f"expected version: {EXPECTED_GENESIS_VERSION}; found: {version}",
            )
        return CheckResult("import genesis", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("import genesis", False, _format_exception(exc))


def check_cuda_gpu() -> CheckResult:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return CheckResult("check CUDA/GPU access", False, "torch.cuda.is_available() is false")
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        before = torch.cuda.memory_allocated(0)
        tensor = torch.ones((1024, 1024), device="cuda")
        allocated = torch.cuda.memory_allocated(0) - before
        value = float(tensor.sum().item())
        del tensor
        torch.cuda.empty_cache()
        return CheckResult(
            "check CUDA/GPU access",
            True,
            f"devices: {device_count}; gpu0: {device_name}; allocated_delta_bytes: {allocated}; sum: {value}",
        )
    except Exception as exc:
        return CheckResult("check CUDA/GPU access", False, _format_exception(exc))


def check_genesis_python_entrypoint() -> CheckResult:
    try:
        result = _run(
            [
                sys.executable,
                "-c",
                "import genesis as gs; print(getattr(gs, '__version__', 'unknown'))",
            ]
        )
    except Exception as exc:
        return CheckResult("genesis Python entry point responds", False, _format_exception(exc))

    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return CheckResult(
            "genesis Python entry point responds",
            False,
            f"exit code {result.returncode}; output: {output[:500]}",
        )
    return CheckResult("genesis Python entry point responds", True, output)


def check_npa_genesis_help() -> CheckResult:
    npa = shutil.which("npa")
    if npa is None:
        return CheckResult("npa workbench genesis --help", False, "npa executable not found on PATH")
    try:
        result = _run([npa, "workbench", "genesis", "--help"])
    except Exception as exc:
        return CheckResult("npa workbench genesis --help", False, _format_exception(exc))

    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        return CheckResult(
            "npa workbench genesis --help",
            False,
            f"exit code {result.returncode}; output: {output[:500]}",
        )
    return CheckResult("npa workbench genesis --help", True, f"executable: {npa}")


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_genesis,
        check_cuda_gpu,
        check_genesis_python_entrypoint,
        check_npa_genesis_help,
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
