"""FiftyOne container environment smoke checks."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Callable


EXPECTED_FIFTYONE_VERSION = os.environ.get("FIFTYONE_VERSION", "1.15.0")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_import_fiftyone() -> CheckResult:
    try:
        module = importlib.import_module("fiftyone")
        version = metadata.version("fiftyone")
        if version != EXPECTED_FIFTYONE_VERSION:
            return CheckResult(
                "import fiftyone",
                False,
                f"expected version: {EXPECTED_FIFTYONE_VERSION}; found: {version}",
            )
        return CheckResult("import fiftyone", True, f"module: {module.__name__}; version: {version}")
    except Exception as exc:
        return CheckResult("import fiftyone", False, _format_exception(exc))


def check_cli_help() -> CheckResult:
    cli = shutil.which("fiftyone")
    if cli is None:
        return CheckResult("check fiftyone CLI help", False, "fiftyone executable not found")
    try:
        result = subprocess.run(
            [cli, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return CheckResult("check fiftyone CLI help", False, _format_exception(exc))
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "usage" not in output.lower():
        return CheckResult("check fiftyone CLI help", False, f"exit={result.returncode}; output={output[-500:]}")
    return CheckResult("check fiftyone CLI help", True, f"executable: {cli}")


def check_app_config() -> CheckResult:
    try:
        import fiftyone as fo

        return CheckResult(
            "check app server configuration",
            True,
            f"address={fo.config.default_app_address}; port={fo.config.default_app_port}; "
            f"database_dir={fo.config.database_dir}",
        )
    except Exception as exc:
        return CheckResult("check app server configuration", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_fiftyone,
        check_cli_help,
        check_app_config,
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
