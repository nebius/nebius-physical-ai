"""Standalone Voxel51 FiftyOne environment smoke checks.

Run with:
    python -m npa.smoke.test_fiftyone_env
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


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_import_fiftyone() -> CheckResult:
    try:
        expected = supported_tool_version("fiftyone", __file__)
        module = importlib.import_module("fiftyone")
        version = metadata.version("fiftyone")
        if version != expected:
            return CheckResult(
                "import fiftyone",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult(
            "import fiftyone",
            True,
            f"module: {module.__name__}; version: {version}",
        )
    except Exception as exc:
        return CheckResult("import fiftyone", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_fiftyone,
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
