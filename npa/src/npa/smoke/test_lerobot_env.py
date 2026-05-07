"""Standalone LeRobot environment smoke checks.

Run with:
    python -m npa.smoke.test_lerobot_env
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
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


def check_import_lerobot() -> CheckResult:
    try:
        expected = supported_tool_version("lerobot", __file__)
        lerobot = importlib.import_module("lerobot")
        version = _package_version(lerobot, "lerobot")
        if version != expected:
            return CheckResult(
                "import lerobot",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult("import lerobot", True, f"version: {version}")
    except Exception as exc:
        return CheckResult("import lerobot", False, _format_exception(exc))


def check_act_config() -> CheckResult:
    try:
        module = importlib.import_module("lerobot.policies.act.configuration_act")
        config = module.ACTConfig()
        return CheckResult(
            "instantiate ACTConfig",
            True,
            f"class: {config.__class__.__module__}.{config.__class__.__name__}",
        )
    except Exception as exc:
        return CheckResult("instantiate ACTConfig", False, _format_exception(exc))


def _check_command_help(command: str) -> CheckResult:
    path = shutil.which(command)
    if path is None:
        return CheckResult(f"{command} --help", False, f"{command} not found on PATH")

    try:
        result = subprocess.run(
            [path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return CheckResult(f"{command} --help", False, _format_exception(exc))

    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        if len(output) > 500:
            output = output[:500] + "..."
        return CheckResult(
            f"{command} --help",
            False,
            f"exit code {result.returncode}; output: {output}",
        )

    return CheckResult(f"{command} --help", True, f"executable: {path}")


def check_lerobot_train_help() -> CheckResult:
    return _check_command_help("lerobot-train")


def check_lerobot_eval_help() -> CheckResult:
    return _check_command_help("lerobot-eval")


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_lerobot,
        check_act_config,
        check_lerobot_train_help,
        check_lerobot_eval_help,
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
