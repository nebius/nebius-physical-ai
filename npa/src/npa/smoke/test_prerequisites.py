"""Standalone prerequisite tool smoke checks.

Run with:
    python -m npa.smoke.test_prerequisites
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

from npa.smoke._versions import supported_tool_version


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _parse_version(output: str) -> str | None:
    match = re.search(r"\b(?:v)?(\d+\.\d+\.\d+)\b", output)
    if match is None:
        return None
    return match.group(1)


def _major(version: str) -> int | None:
    match = re.match(r"^(\d+)\.", version)
    if match is None:
        return None
    return int(match.group(1))


def _compare_versions(tool: str, expected: str, actual: str) -> CheckResult:
    expected_major = _major(expected)
    actual_major = _major(actual)
    if expected_major is None:
        return CheckResult(f"{tool} version", False, f"could not parse expected version: {expected}")
    if actual_major is None:
        return CheckResult(f"{tool} version", False, f"could not parse installed version: {actual}")
    if actual_major != expected_major:
        return CheckResult(
            f"{tool} version",
            False,
            f"expected major version {expected_major} ({expected}); found {actual}",
        )
    if actual != expected:
        return CheckResult(
            f"{tool} version",
            True,
            f"WARN: expected {expected}; found {actual}",
        )
    return CheckResult(f"{tool} version", True, f"version: {actual}")


def _check_tool(
    *,
    executable: str,
    tool_key: str,
    install_hint: str,
) -> CheckResult:
    path = shutil.which(executable)
    if path is None:
        return CheckResult(
            f"{executable} on PATH",
            False,
            f"{executable} not found on PATH. Install it: {install_hint}",
        )

    try:
        expected = supported_tool_version(tool_key, __file__)
        result = subprocess.run(
            [path, "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return CheckResult(f"{executable} version", False, _format_exception(exc))

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        return CheckResult(
            f"{executable} version",
            False,
            f"exit code {result.returncode}; output: {output}",
        )

    actual = _parse_version(output)
    if actual is None:
        return CheckResult(
            f"{executable} version",
            False,
            f"could not parse version from output: {output}",
        )

    comparison = _compare_versions(executable, expected, actual)
    if comparison.detail:
        comparison.detail = f"executable: {path}; {comparison.detail}"
    else:
        comparison.detail = f"executable: {path}"
    return comparison


def check_nebius_cli() -> CheckResult:
    return _check_tool(
        executable="nebius",
        tool_key="nebius-cli",
        install_hint="https://docs.nebius.com/cli/install",
    )


def check_terraform() -> CheckResult:
    return _check_tool(
        executable="terraform",
        tool_key="terraform",
        install_hint="https://developer.hashicorp.com/terraform/install",
    )


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_nebius_cli,
        check_terraform,
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
