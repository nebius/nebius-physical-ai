"""Real single-pose cuRobo GPU qualification, separate from full benchmark runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

from npa.workbench.curobo.artifacts import build_rrd
from npa.workbench.curobo.runner import execute


def main():
    with tempfile.TemporaryDirectory(prefix="npa-curobo-functional-") as directory:
        root = Path(directory)
        manifest = {
            "problems": [
                {
                    "id": "franka-pose",
                    "start": [0, -1.3, 0, -2.5, 0, 1.0, 0],
                    "goal_pose": {
                        "position_xyz": [0.5, 0.0, 0.3],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    },
                    "cuboids": {
                        "table": {"dims": [2, 2, 0.2], "pose": [0, 0, -0.2, 1, 0, 0, 0]}
                    },
                }
            ]
        }
        report = execute("plan", manifest, root / "output", run_id="curobo-functional")
        assert report["summary"]["kinematic"]["success"] == 1
        artifact = build_rrd(
            root / "output/problems.jsonl",
            root / "planning.rrd",
            run_id="curobo-functional",
        )
        import subprocess

        subprocess.run(
            ["rerun", "rrd", "verify", str(root / "planning.rrd")], check=True
        )
        assert artifact["successful_trajectories"] == 1
        print("cuRobo actual pose, FK trajectory and verified RRD passed")


if __name__ == "__main__":
    main()
