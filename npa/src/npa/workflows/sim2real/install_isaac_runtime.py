"""Materialize Isaac application modules while building the immutable image.

Legacy constants are converted at image build time only. Production Jobs run
these files directly; they never receive or create Python source.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def install(destination: Path) -> None:
    from npa.workflows.sim2real import (
        byo_isaac_eval,
        byo_isaac_policy_rollout,
        isaac_byo_robot_task,
        isaac_physics_task,
        isaac_scenario_task,
    )

    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "isaac_eval.py": byo_isaac_eval.ISAAC_EVAL_SCRIPT,
        "isaac_rollout.py": byo_isaac_policy_rollout.ISAAC_ROLLOUT_SCRIPT,
        "isaac_byo_robot_task.py": isaac_byo_robot_task.module_source(),
        "isaac_scenario_task.py": isaac_scenario_task.module_source(),
        "isaac_robot_train.py": isaac_byo_robot_task.TRAIN_WRAPPER_SCRIPT,
        "isaac_physics_task.py": isaac_physics_task.module_source(),
        "isaac_physics_train.py": isaac_physics_task.TRAIN_WRAPPER_SCRIPT,
    }
    for name, source in sources.items():
        if not source.strip():
            raise RuntimeError(f"empty Isaac runtime module: {name}")
        path = destination / name
        path.write_text(source.rstrip() + "\n", encoding="utf-8")
        path.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    install(args.destination)


if __name__ == "__main__":
    main()
