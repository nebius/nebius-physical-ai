"""Cosmos3-Reason container functional golden eval (module wiring smoke)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_cuda_available() -> CheckResult:
    import torch

    if not torch.cuda.is_available():
        return CheckResult("cuda available", False, "torch.cuda.is_available() is False")
    return CheckResult("cuda available", True, torch.cuda.get_device_name(0))


def check_reason_cache_wiring() -> CheckResult:
    try:
        from npa.cli.workbench import cosmos3 as cosmos3_cli
    except Exception as exc:
        return CheckResult("reason cli wiring", False, str(exc))
    if not hasattr(cosmos3_cli, "app"):
        return CheckResult("reason cli wiring", False, "missing cosmos3 Typer app")
    return CheckResult("reason cli wiring", True, "npa.cli.workbench.cosmos3")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_cuda_available,
        check_reason_cache_wiring,
    ]
    failed = 0
    for check in checks:
        result = check()
        state = "PASS" if result.ok else "FAIL"
        print(f"[{state}] {result.name}: {result.detail}")
        if not result.ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
