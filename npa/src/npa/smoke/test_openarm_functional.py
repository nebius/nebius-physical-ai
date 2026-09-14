"""Golden eval: real OpenArm v2 MuJoCo control rollout and artifact validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from npa.workbench.openarm.runtime import _run_mujoco
from npa.workbench.openarm.schemas import OpenArmRunRequest


def main() -> int:
    output = Path("/tmp/npa-openarm-golden")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    request = OpenArmRunRequest(
        simulator="mujoco", output_uri="s3://golden-eval/openarm/", steps=500
    )
    result = _run_mujoco(request, output)
    trace = output / str(result["artifact"]["path"])
    with np.load(trace) as arrays:
        samples = arrays["joint_position"]
        commands = arrays["command"]
        energies = arrays["velocity_energy"]
    checks = {
        "schema": result.get("schema") == "npa.openarm.mujoco_rollout.v1",
        "physics_advanced": float(result.get("simulation_seconds", 0)) > 0,
        "bimanual_joints": samples.ndim == 2 and samples.shape[1] == 14,
        "position_commands": commands.ndim == 2 and commands.shape[1] == 16,
        "finite_state": bool(
            np.isfinite(samples).all() and np.isfinite(energies).all()
        ),
        "artifact_nonempty": trace.stat().st_size > 0,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "result": result,
    }
    (output / "golden_eval.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
