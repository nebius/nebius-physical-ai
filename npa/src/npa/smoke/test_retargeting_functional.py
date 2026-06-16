"""Retargeting container functional golden eval (motion-lib validation)."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from npa.workbench.retargeting import validate_motion_lib


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _write_motion_lib(path: Path) -> None:
    payload = {
        "golden_smoke": {
            "root_trans_offset": np.zeros((4, 3), dtype=np.float32),
            "pose_aa": np.zeros((4, 24), dtype=np.float32),
            "dof": np.zeros((4, 23), dtype=np.float32),
            "root_rot": np.zeros((4, 4), dtype=np.float32),
            "fps": 50,
        }
    }
    joblib.dump(payload, path)


def check_validate_motion_lib() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="npa-retarget-smoke-") as tmp:
        motion_path = Path(tmp) / "motion.pkl"
        _write_motion_lib(motion_path)
        try:
            count, files = validate_motion_lib(motion_path)
        except Exception as exc:
            return CheckResult("validate motion lib", False, str(exc))
        if count != 1:
            return CheckResult("validate motion lib", False, f"expected 1 motion, got {count}")
        return CheckResult("validate motion lib", True, f"files={files}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [check_validate_motion_lib]
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
