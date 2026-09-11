"""Verify physical timing in the actual generated Isaac capture and action code."""

from __future__ import annotations

import ast
import os
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from npa.workflows.sim2real.byo_isaac_eval import ISAAC_EVAL_SCRIPT
from npa.workflows.sim2real.byo_isaac_policy_rollout import ISAAC_ROLLOUT_SCRIPT
from npa.workflows.sim2real.isaac_simulation_clock import SimulationClock


@pytest.mark.parametrize(
    "duration", [None, True, False, "invalid", 0, -0.02, float("inf"), float("nan")]
)
def test_clock_rejects_unknown_or_invalid_physics_duration(duration):
    with pytest.raises(ValueError, match="step_dt must be finite and positive"):
        SimulationClock(duration)


def test_sampling_does_not_advance_physics():
    clock = SimulationClock(0.02)
    assert clock.sample()["timestamp_seconds"] == 0
    clock.advance()
    assert clock.sample() == clock.sample()
    assert clock.sample()["timestamp_seconds"] == 0.02
    assert clock.sample()["completed_simulation_steps"] == 1


def _run_nodes(nodes, namespace, tmp_path):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    script = tmp_path / "generated_capture.py"
    script.write_text(ast.unparse(module))
    return runpy.run_path(str(script), init_globals=namespace)


def _capture_program(script):
    tree = ast.parse(script)
    capture = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "capture"
    )
    main = next(
        node for node in tree.body if isinstance(node, ast.Try) and capture in node.body
    )
    clock = [
        node
        for node in main.body
        if "simulation_clock =" in ast.unparse(node) and isinstance(node, ast.Assign)
    ]
    imports = [
        node
        for node in main.body
        if isinstance(node, ast.ImportFrom)
        and node.module.endswith("isaac_simulation_clock")
    ]
    loop = _capture_loop(main)
    final = next(
        node
        for node in main.body
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith("capture(")
    )
    return [*imports, *clock, capture, loop, final]


def _capture_loop(main):
    loop = next(
        node
        for node in main.body
        if isinstance(node, ast.For) and ast.unparse(node.target) == "_step"
    )
    # Preserve the ordering and capture predicates of the generated loop while
    # excluding policy inference and metric calculations unrelated to this test.
    body = [
        node
        for node in loop.body
        if ast.unparse(node).startswith(
            ("obs, _, dones, extras = env.step", "simulation_clock.advance()")
        )
        or (isinstance(node, ast.If) and "capture(_step" in ast.unparse(node))
    ]
    loop = ast.For(target=loop.target, iter=loop.iter, body=body, orelse=[])
    return loop


def _camera_environment(duration):
    torch = pytest.importorskip("torch")
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={"rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)}
        )
    )
    environment = SimpleNamespace(step_dt=duration, scene={"primary": sensor})
    return environment


def _capture_settings():
    return {
        "CAPTURE_WIDTH": 2,
        "CAPTURE_HEIGHT": 2,
        "PNG_COMPRESS_LEVEL": 1,
        "CAMERA_VIEWS": [{"name": "primary"}],
        "SIM_DEVICE": "cuda:0",
    }


def _capture_namespace(tmp_path, duration, fps):
    return {
        "np": np,
        "os": os,
        "env": SimpleNamespace(
            unwrapped=_camera_environment(duration),
            step=lambda actions: (None, None, None, None),
        ),
        "actions": None,
        "N": 1,
        "STEPS": 3,
        "HORIZON_STEPS": 3,
        "CAPTURE_STRIDE": 2,
        "CAPTURE_FPS": fps,
        **_capture_settings(),
        "FRAMES_DIR": str(tmp_path),
        "rend_root": str(tmp_path),
        "rollout_ids": ["episode"],
        "CKPT_URI": "s3://fixture/checkpoint.pt",
        "_camera_key": lambda view: view,
        "_env_id": lambda index: "episode",
        "_have_pil": True,
        "_PILImage": Image,
        "_write_rgb_png": lambda path, rgb: Image.fromarray(rgb).save(path),
        "capture_pointcloud": lambda: None,
        "frame_names": {0: {"primary": []}},
        "frame_metadata": {0: {"primary": []}},
        "active": np.array([True]),
        "newly_terminal": np.array([False]),
        "completed": np.array([False]),
        "SAMPLE_INDEX": {0: 0, 2: 1},
    }


@pytest.mark.parametrize(
    "script", [ISAAC_EVAL_SCRIPT, ISAAC_ROLLOUT_SCRIPT], ids=["evaluation", "rollout"]
)
@pytest.mark.parametrize("duration", [0.02, 0.005])
@pytest.mark.parametrize("fps", [10, 30])
def test_generated_capture_uses_completed_physical_steps(
    script, duration, fps, tmp_path
):
    result = _run_nodes(
        _capture_program(script), _capture_namespace(tmp_path, duration, fps), tmp_path
    )
    frames = result["frame_metadata"][0]["primary"]
    assert [frame["sim_step"] for frame in frames] == [0, 2, 3]
    assert [frame["completed_simulation_steps"] for frame in frames] == [1, 3, 3]
    assert [frame["timestamp_seconds"] for frame in frames] == pytest.approx(
        [duration, 3 * duration, 3 * duration]
    )
    assert all(frame["simulation_step_seconds"] == duration for frame in frames)
    assert all(
        frame["timestamp_timebase"] == "simulation_elapsed_since_rollout_start"
        for frame in frames
    )
    assert result["simulation_clock"].completed_steps == 3
    assert len(list((tmp_path / "episode").glob("*.png"))) == 3


@pytest.mark.parametrize(
    "script", [ISAAC_EVAL_SCRIPT, ISAAC_ROLLOUT_SCRIPT], ids=["evaluation", "rollout"]
)
def test_generated_capture_rejects_invalid_step_duration(script, tmp_path):
    namespace = _capture_namespace(tmp_path, float("nan"), 10)
    with pytest.raises(ValueError, match="step_dt"):
        _run_nodes(_capture_program(script), namespace, tmp_path)
    assert not (tmp_path / "episode").exists()


def _action_namespace():
    torch = pytest.importorskip("torch")
    clock = SimulationClock(0.02)
    for _ in range(3):
        clock.advance()
    namespace = {
        "simulation_clock": clock,
        "actions_log": {0: []},
        "i": 0,
        "decision_step": 1,
        "_step": 2,
        "a_np": np.zeros((1, 2)),
        "scenario": {},
        "goal_change": 0.0,
        "ee_change": 0.0,
        "obj": torch.zeros((1, 3)),
    }
    for name in (
        "goal_distance",
        "ee_distance",
        "contact_now",
        "gripper_closed",
        "stable_grasp_now",
        "lift_m",
        "stable_place_now",
        "done_np",
    ):
        namespace[name] = np.array([0.0])
    return namespace


def test_generated_action_row_uses_the_same_post_step_clock(tmp_path):
    namespace = _action_namespace()
    append = next(
        node
        for node in ast.walk(ast.parse(ISAAC_ROLLOUT_SCRIPT))
        if isinstance(node, ast.Expr)
        and ast.unparse(node).startswith("actions_log[i].append(")
    )
    result = _run_nodes([append], namespace, tmp_path)
    row = result["actions_log"][0][0]
    assert row["sim_step"] == 2 and row["step"] == 1
    assert row["timestamp_seconds"] == 0.06
    assert row["simulation_step_seconds"] == 0.02
    assert row["completed_simulation_steps"] == 3
