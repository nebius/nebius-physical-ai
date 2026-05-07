"""Standalone Genesis environment smoke checks.

Run with:
    python -m npa.smoke.test_genesis_env
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
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_import_genesis() -> CheckResult:
    try:
        expected = supported_tool_version("genesis", __file__)
        genesis = importlib.import_module("genesis")
        version = _package_version(genesis, "genesis-world")
        if version != expected:
            return CheckResult(
                "import genesis",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult("import genesis", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("import genesis", False, _format_exception(exc))


def check_scene_build_and_step() -> CheckResult:
    try:
        gs = importlib.import_module("genesis")

        init_kwargs = {"logging_level": "warning"}
        if hasattr(gs, "cpu"):
            init_kwargs["backend"] = gs.cpu
        gs.init(**init_kwargs)

        scene = gs.Scene(show_viewer=False)
        if hasattr(gs, "morphs") and hasattr(gs.morphs, "Plane"):
            scene.add_entity(gs.morphs.Plane())
        scene.build()
        scene.step()
        return CheckResult("build and step genesis.Scene", True, "CPU backend completed one step")
    except Exception as exc:
        return CheckResult("build and step genesis.Scene", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_genesis,
        check_scene_build_and_step,
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
