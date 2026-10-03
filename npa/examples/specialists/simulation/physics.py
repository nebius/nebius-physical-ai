"""Execute and independently replay real Workbench MuJoCo parameter-sweep episodes."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import product
from pathlib import Path
import json
import time

import numpy as np

from npa.adapter.sim_to_lerobot import encode_video
from npa.workbench.token_factory import robot_sim
from workload import _digest, _validate


def _apply_parameters(env, mass, friction):
    import mujoco

    body = env.model.body("object0")
    body.mass[:] *= mass
    body.inertia[:] *= mass
    env.model.geom("object0").friction[0] *= friction
    # Recompute mass constants without resetting the episode's initialized state.
    mujoco.mj_setConst(env.model, mujoco.MjData(env.model))
    mujoco.mj_forward(env.model, env.data)


@contextmanager
def _world(scene, seed, mass, friction):
    with robot_sim._world(scene, seed, 192, 144) as world:
        _apply_parameters(world[0], mass, friction)
        yield world


def _episode(scene, output, seed, mass, friction):
    started = time.perf_counter()
    with _world(scene, seed, mass, friction) as world:
        arrays = robot_sim._rollout(world)
    output.mkdir()
    trace = {key: value for key, value in arrays.items() if not key.startswith("obs_")}
    np.savez_compressed(output / "physics.npz", **trace)
    preview = np.concatenate((arrays["obs_workspace"], arrays["obs_wrist"]), axis=2)
    encode_video(preview, output / "preview.mp4", robot_sim.FPS)
    return {
        **robot_sim.physics_checks(arrays),
        "seed": seed,
        "mass_scale": mass,
        "friction_scale": friction,
        "frames": len(arrays["actions"]),
        "elapsed_seconds": time.perf_counter() - started,
        "trace_sha256": _digest(output / "physics.npz"),
        "video_sha256": _digest(output / "preview.mp4"),
    }


def _simulate(directory, name, seed):
    directory = Path(directory)
    plan = _validate(directory, name)
    plan_digest = _digest(directory / "plan.json")
    root = directory / "runs" / f"{plan_digest}-{seed}"
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (mass, friction) in enumerate(
        product(plan["mass_scales"], plan["friction_scales"])
    ):
        result = _episode(
            plan["scene"], root / f"episode-{index}", seed + index, mass, friction
        )
        rows.append(result)
    report = {
        "task": name,
        "plan_sha256": plan_digest,
        "episodes": rows,
        "accepted": sum(row["accepted"] for row in rows),
        "total": len(rows),
        "simulator_versions": robot_sim.runtime_versions(),
        "controller": "scripted_privileged_state_pick_place_v1",
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (directory / "latest.json").write_text(
        json.dumps({"report": str(report_path.relative_to(directory))})
    )
    return report


def _step_error(env, arrays, index, observation, info):
    errors = [
        np.max(np.abs(robot_sim._robot_state(env) - arrays["next_state"][index])),
        np.max(
            np.abs(observation["achieved_goal"] - arrays["next_object_position"][index])
        ),
        np.max(
            np.abs(
                observation["observation"][:3] - arrays["next_gripper_position"][index]
            )
        ),
        float(robot_sim._finger_contacts(env) != arrays["finger_contacts"][index]),
        float(bool(info["is_success"]) != arrays["environment_success"][index]),
    ]
    return float(max(errors))


def _replay(scene, path, row):
    arrays = dict(np.load(path, allow_pickle=False))
    if _digest(path) != row["trace_sha256"]:
        raise ValueError("physics trace digest changed")
    maximum = 0.0
    with _world(scene, row["seed"], row["mass_scale"], row["friction_scale"]) as world:
        env = world[0]
        for index, action in enumerate(arrays["actions"]):
            observation, _, _, _, info = env.step(action)
            maximum = max(maximum, _step_error(env, arrays, index, observation, info))
    if maximum > 1e-6:
        raise ValueError(f"recorded trajectory did not replay: {maximum}")
    return {**robot_sim.physics_checks(arrays), "maximum_replay_error": maximum}


def _score(directory, name):
    directory = Path(directory)
    plan = _validate(directory, name)
    relative = json.loads((directory / "latest.json").read_text())["report"]
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve() / "runs"):
        raise ValueError("report escaped the task run directory")
    report = json.loads(path.read_text())
    if report["plan_sha256"] != _digest(directory / "plan.json"):
        raise ValueError("simulation does not match the current plan")
    expected = list(product(plan["mass_scales"], plan["friction_scales"]))
    actual = [(row["mass_scale"], row["friction_scale"]) for row in report["episodes"]]
    if actual != expected:
        raise ValueError("simulation omitted or duplicated parameter cases")
    for index, row in enumerate(report["episodes"]):
        episode = path.parent / f"episode-{index}"
        if _digest(episode / "preview.mp4") != row["video_sha256"]:
            raise ValueError("video digest changed")
        verdict = _replay(plan["scene"], episode / "physics.npz", row)
        if verdict["accepted"] != row["accepted"]:
            raise ValueError("stored outcome disagrees with physics")
    return {
        "task": name,
        "accepted": sum(row["accepted"] for row in report["episodes"]),
        "total": len(expected),
        "replay_verified": True,
    }
