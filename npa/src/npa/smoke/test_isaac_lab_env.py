"""Standalone Isaac Lab environment smoke checks.

Run with:
    python -m npa.smoke.test_isaac_lab_env
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Callable

from npa.smoke._versions import supported_tool_version


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _package_version(module, distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
        return str(version) if version else "unknown"


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_import_isaac_lab() -> CheckResult:
    try:
        expected = supported_tool_version("isaac-lab", __file__)
        isaaclab = importlib.import_module("isaaclab")
        version = _package_version(isaaclab, "isaaclab")
        if version != expected:
            return CheckResult(
                "import isaaclab",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult("import isaaclab", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("import isaaclab", False, _format_exception(exc))


def check_isaac_sim_runtime() -> CheckResult:
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app
        if simulation_app is None:
            return CheckResult("launch Isaac Sim runtime", False, "AppLauncher.app is None")

        update = getattr(simulation_app, "update", None)
        if callable(update):
            update()
        return CheckResult("launch Isaac Sim runtime", True, "headless app launched")
    except Exception as exc:
        return CheckResult("launch Isaac Sim runtime", False, _format_exception(exc))
    finally:
        if simulation_app is not None:
            close = getattr(simulation_app, "close", None)
            if callable(close):
                close()


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_isaac_lab,
        check_isaac_sim_runtime,
    ]
    results: list[CheckResult] = []

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
