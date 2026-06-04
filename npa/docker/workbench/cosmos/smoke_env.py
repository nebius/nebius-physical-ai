"""Cosmos runner image environment smoke checks."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


COSMOS3_HOME = Path("/opt/cosmos/cosmos-framework")
TRANSFER25_HOME = Path("/opt/cosmos/cosmos-transfer2.5")
NPA_ROOT = Path("/opt/npa")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_source_checkouts() -> CheckResult:
    missing = [
        str(path)
        for path in (
            COSMOS3_HOME / "cosmos_framework",
            COSMOS3_HOME / "cosmos_framework" / "scripts" / "inference.py",
            TRANSFER25_HOME / "cosmos_transfer2",
            TRANSFER25_HOME / "examples" / "inference.py",
        )
        if not path.exists()
    ]
    if missing:
        return CheckResult("check Cosmos source checkouts", False, ", ".join(missing))
    return CheckResult("check Cosmos source checkouts", True, f"{COSMOS3_HOME}; {TRANSFER25_HOME}")


def check_cli_tooling() -> CheckResult:
    names = ("uv", "git", "git-lfs", "ffmpeg", "aws", "huggingface-cli")
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        return CheckResult("check CLI tooling", False, ", ".join(missing))
    return CheckResult("check CLI tooling", True, ", ".join(names))


def check_npa_imports() -> CheckResult:
    try:
        module = importlib.import_module("npa.workbench.cosmos.workflows")
        augment_yaml = getattr(module, "COSMOS_AUGMENT_YAML")
        reason_yaml = getattr(module, "COSMOS_REASON_YAML")
        return CheckResult("import NPA Cosmos workflows", True, f"{augment_yaml}; {reason_yaml}")
    except Exception as exc:
        return CheckResult("import NPA Cosmos workflows", False, _format_exception(exc))


def check_guardrails_default_on() -> CheckResult:
    dockerfile = NPA_ROOT / "docker" / "workbench" / "cosmos" / "Dockerfile"
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except Exception as exc:
        return CheckResult("check guardrails default", False, _format_exception(exc))
    if "COSMOS_DISABLE_SAFETY=0" not in text:
        return CheckResult("check guardrails default", False, "COSMOS_DISABLE_SAFETY=0 missing")
    forbidden = ("--no-guardrails", "NPA_COSMOS3_NO_GUARDRAILS", "guardrails=false")
    found = [value for value in forbidden if value in text]
    if found:
        return CheckResult("check guardrails default", False, ", ".join(found))
    return CheckResult("check guardrails default", True, "guardrails on")


def check_python_compile() -> CheckResult:
    files = [
        NPA_ROOT / "src" / "npa" / "workbench" / "cosmos" / "workflows.py",
        NPA_ROOT / "src" / "npa" / "workbench" / "cosmos" / "cosmos3.py",
    ]
    try:
        for path in files:
            subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)
        return CheckResult("compile NPA Cosmos modules", True, str(len(files)))
    except Exception as exc:
        return CheckResult("compile NPA Cosmos modules", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_source_checkouts,
        check_cli_tooling,
        check_npa_imports,
        check_guardrails_default_on,
        check_python_compile,
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
