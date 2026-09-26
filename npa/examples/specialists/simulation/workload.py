"""Define matched simulation tasks and validate agent-authored parameter sweeps."""

from __future__ import annotations

import hashlib
import json
from math import isclose
from pathlib import Path

from npa.workbench.token_factory.robot_scene import RobotScene

MASS_SCALES = [0.8, 1.0, 1.2]
FRICTION_SCALES = [0.8, 1.2]
TASKS = {
    "translate": {
        "instruction": "Start the blue cube at (-80, -40) millimeters. Move it +160 mm in x and +80 mm in y onto a red target. Use lighting 1.0.",
        "coordinates": [-0.08, -0.04, 0.08, 0.04],
        "colors": ["blue", "red"],
        "lighting": 1.0,
    },
    "mirror": {
        "instruction": "Start the green cube at (-6, +7) centimeters. Reflect its location across the y axis for the orange target. Use lighting 0.8.",
        "coordinates": [-0.06, 0.07, 0.06, 0.07],
        "colors": ["green", "orange"],
        "lighting": 0.8,
    },
    "diagonal": {
        "instruction": "Start the red cube at (-0.05, -0.05) meters. Place the blue target 12 cm farther in both positive x and positive y. Use lighting 1.1.",
        "coordinates": [-0.05, -0.05, 0.07, 0.07],
        "colors": ["red", "blue"],
        "lighting": 1.1,
    },
    "reverse": {
        "instruction": "Start the orange cube at (+80, +40) millimeters. The green target is at the point obtained by rotating those XY coordinates 180 degrees around the origin. Use lighting 0.9.",
        "coordinates": [0.08, 0.04, -0.08, -0.04],
        "colors": ["orange", "green"],
        "lighting": 0.9,
    },
    "cross": {
        "instruction": "Start the blue cube at x=20 mm and y=-80 mm. Keep x fixed and move 16 centimeters in positive y to the orange target. Use lighting 1.2.",
        "coordinates": [0.02, -0.08, 0.02, 0.08],
        "colors": ["blue", "orange"],
        "lighting": 1.2,
    },
    "offset": {
        "instruction": "The red target is at (-70, -50) millimeters. The green cube starts 140 mm farther in x and 100 mm farther in y than that target. Use lighting 0.7.",
        "coordinates": [0.07, 0.05, -0.07, -0.05],
        "colors": ["green", "red"],
        "lighting": 0.7,
    },
}


def _reference_plan(name):
    task = TASKS[name]
    scene = dict(zip(("object_x", "object_y", "goal_x", "goal_y"), task["coordinates"]))
    scene.update(
        object_color=task["colors"][0],
        target_color=task["colors"][1],
        lighting=task["lighting"],
        reason="Physics feasibility control",
    )
    return {
        "scene": scene,
        "mass_scales": MASS_SCALES,
        "friction_scales": FRICTION_SCALES,
    }


def _task_text(name):
    return (
        TASKS[name]["instruction"]
        + "\nRepair plan.json to satisfy this instruction. Coordinates must be in meters. "
        "The scene keys are object_x, object_y, goal_x, goal_y, object_color, target_color, "
        "lighting, and a nonempty reason. Each object/goal XY coordinate relative to the "
        "table origin must be within [-0.12, 0.12] meters. This bound is on individual "
        "coordinates, not the displacement between object and goal. The cube must move "
        "at least 0.08 m. Colors must differ.\n"
        "Fan out over cube mass multipliers 80%, 100%, 120% and sliding-friction "
        "multipliers 80%, 120%, taking their Cartesian product (six episodes). "
        "Use top-level keys scene, mass_scales, friction_scales. The simulator supplies "
        "a fixed pick/place controller; do not invent measurements or modify its judge.\n"
        "First run validate to capture the broken input. Read this file and plan.json, "
        "make the smallest complete repair, then run validate and simulate. Read the "
        "actual simulation result before reporting success. The simulate operation runs "
        "all six physics cases, retaining videos and state/action traces.\n"
    )


def _prepare_workspace(directory, name):
    directory.mkdir(parents=True)
    (directory / "TASK.md").write_text(_task_text(name))
    broken = _reference_plan(name)
    broken["scene"]["goal_x"] = broken["scene"]["object_x"]
    broken["scene"]["goal_y"] = broken["scene"]["object_y"]
    broken["mass_scales"] = [1.0]
    (directory / "plan.json").write_text(json.dumps(broken, indent=2) + "\n")


def _validate(directory, name):
    path = Path(directory) / "plan.json"
    plan = json.loads(path.read_text())
    if set(plan) != {"scene", "mass_scales", "friction_scales"}:
        raise ValueError("plan must contain scene, mass_scales and friction_scales")
    scene = RobotScene.model_validate(plan["scene"]).model_dump()
    reference = _reference_plan(name)["scene"]
    for key, value in reference.items():
        if key == "reason":
            continue
        equal = (
            isclose(scene[key], value, abs_tol=1e-9)
            if isinstance(value, float)
            else scene[key] == value
        )
        if not equal:
            raise ValueError(f"scene.{key} does not satisfy TASK.md")
    for key, expected in (
        ("mass_scales", MASS_SCALES),
        ("friction_scales", FRICTION_SCALES),
    ):
        if not isinstance(plan[key], list) or any(
            type(value) not in (float, int) for value in plan[key]
        ):
            raise ValueError(f"{key} must be a numeric array")
        if sorted(plan[key]) != expected:
            raise ValueError(
                f"{key} must cover the exact requested parameter axis: {expected}"
            )
    return plan


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
