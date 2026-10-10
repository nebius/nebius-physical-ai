"""npa.workbench.mujoco_manip - MuJoCo contact-rich manipulation workbench.

Real MuJoCo simulation (programmatic MJCF scenes, real contact solving) for
contact-rich manipulation tasks: peg insertion, screw driving, and a reach
smoke task.  The ``mujoco`` import stays lazy inside
:func:`make_env`, so this package imports cleanly on machines without MuJoCo.

The workbench-first SDK surface re-exports the pipeline stage (``run``)
alongside the environment and scripted-policy API; the CLI is a thin client.
"""

from __future__ import annotations

from npa.workflows.byof.mujoco_pipeline import MujocoPipelineError, run
from npa.workbench.mujoco_manip.envs import MujocoManipEnv, make_env
from npa.workbench.mujoco_manip.policies import (
    EXPERT,
    NOISY,
    POLICIES,
    RANDOM,
    get_policy,
)
from npa.workbench.mujoco_manip.scenes import (
    PEG_INSERTION,
    REACH,
    SCREW_DRIVING,
    TASKS,
    TASK_SPECS,
    build_scene,
)

__all__ = [
    "MujocoPipelineError",
    "run",
    "MujocoManipEnv",
    "make_env",
    "EXPERT",
    "NOISY",
    "RANDOM",
    "POLICIES",
    "get_policy",
    "PEG_INSERTION",
    "REACH",
    "SCREW_DRIVING",
    "TASKS",
    "TASK_SPECS",
    "build_scene",
]
