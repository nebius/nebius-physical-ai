"""Run the actual one- and two-node WAM recipe from an explicitly configured Slurm login."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.e2e
RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


@pytest.mark.timeout(0)
def test_native_wam_slurm_scaling():
    required = ("NPA_WAM_LIVE_ROOT", "NPA_WAM_SHARED_ROOT", "NPA_WAM_STEPS")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or any(
        not os.environ.get(key) for key in required
    ):
        pytest.skip(
            "requires explicit Slurm login, prepared inputs and optimizer-step count"
        )
    root = Path(os.environ[required[0]])
    for nodes in (1, 2):
        run = root / f"nodes-{nodes}"
        subprocess.run(
            [
                sys.executable,
                str(RECIPE / "recipe.py"),
                "plan",
                "--shared-root",
                os.environ[required[1]],
                "--run-dir",
                str(run),
                "--name",
                f"nodes-{nodes}",
                "--nodes",
                str(nodes),
                "--steps",
                os.environ[required[2]],
            ],
            check=True,
        )
        subprocess.run(["sbatch", "--wait", str(run / "train.sbatch")], check=True)
        command = [sys.executable, str(RECIPE / "report.py"), "--run-dir", str(run)]
        if nodes > 1:
            command += ["--baseline", str(root / "nodes-1")]
        subprocess.run(command, check=True)
        result = json.loads((run / "measurement.json").read_text())
        assert result["status"] == "measured"
        assert result["gpus"] == nodes * 8
        assert result["measured_steps"] >= 2
