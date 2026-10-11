"""Tests for the mujoco_manip workbench surface.

Real MuJoCo simulation throughout (no stubs): scene XML parses, the expert
policy succeeds on every task, and the random baseline fails.  The whole
module skips when the ``mujoco`` package is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from npa.cli.workbench import mujoco as mujoco_cli
from npa.workbench import mujoco_manip as wb
from npa.workbench.mujoco_manip.envs import (
    MujocoManipEnv,
    MujocoManipError,
    make_env,
)
from npa.workbench.mujoco_manip.policies import get_policy
from npa.workbench.mujoco_manip.scenes import TASKS, build_scene
from npa.workflows.byof import mujoco_pipeline as pipe

mujoco = pytest.importorskip("mujoco")


def _rollout(task: str, policy: str, seed: int, max_steps: int = 500):
    env = make_env(task, max_steps=max_steps)
    policy_fn = get_policy(task, policy, seed=seed)
    obs = env.reset(seed=seed)
    info: dict = {}
    for _ in range(max_steps):
        obs, _, done, info = env.step(policy_fn(obs))
        if done:
            break
    return info


def test_sdk_surface_is_workbench_first():
    assert wb.run is pipe.run
    assert wb.MujocoPipelineError is pipe.MujocoPipelineError
    for name in ("run", "make_env", "get_policy", "TASKS", "build_scene"):
        assert name in wb.__all__
    assert not hasattr(wb, "__npa_cli_module__")


def test_cli_registers_run():
    names = {c.name for c in mujoco_cli.app.registered_commands}
    assert "run" in names


@pytest.mark.parametrize("task", TASKS)
def test_scenes_parse(task):
    model = mujoco.MjModel.from_xml_string(build_scene(task))
    assert model.nq >= 4
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip") >= 0


def test_unknown_scene_raises():
    with pytest.raises(KeyError):
        build_scene("nope")


def test_env_interface():
    env = make_env("peg_insertion")
    assert isinstance(env, MujocoManipEnv)
    assert env.action_space["shape"] == (4,)
    obs = env.reset(seed=0)
    assert obs.shape == env.observation_space["shape"]
    assert obs.shape[0] == 2 * env.model.nq + 6
    obs2, reward, done, info = env.step(np.zeros(4))
    assert obs2.shape == obs.shape
    assert isinstance(info["success"], bool)
    assert "tip_pos" in info


def test_env_rejects_bad_task_and_action():
    with pytest.raises(MujocoManipError, match="unknown mujoco_manip task"):
        make_env("nope")
    env = make_env("reach")
    env.reset(seed=0)
    with pytest.raises(MujocoManipError, match="4-D action"):
        env.step(np.zeros(3))


def test_get_policy_rejects_unknown():
    with pytest.raises(MujocoManipError, match="unknown mujoco_manip task"):
        get_policy("nope", "expert")
    with pytest.raises(MujocoManipError, match="unknown mujoco_manip policy"):
        get_policy("reach", "oracle")


@pytest.mark.parametrize("task", TASKS)
def test_expert_succeeds(task):
    for seed in (0, 1):
        info = _rollout(task, "expert", seed)
        assert info["success"] is True, f"expert failed {task} seed {seed}"


def test_random_baseline_fails_peg_insertion():
    for seed in (0, 1, 2):
        info = _rollout("peg_insertion", "random", seed)
        assert info["success"] is False


def test_noisy_is_no_worse_than_random_on_screw():
    noisy_wins = sum(
        _rollout("screw_driving", "noisy", seed)["success"] for seed in (0, 1, 2)
    )
    assert noisy_wins >= 1


def test_pipeline_run_writes_report(tmp_path):
    out = tmp_path / "report.json"
    report = pipe.run(
        task="reach",
        policy="scripted:expert",
        output_path=str(out),
        episodes=2,
        seed=0,
    )
    assert report["status"] == "completed"
    assert report["schema"] == pipe.SCHEMA_RUN
    payload = json.loads(out.read_text())
    assert payload["summary"]["success_rate"] == pytest.approx(1.0)
    assert payload["summary"]["episodes"] == 2
    assert len(payload["episodes"]) == 2
    first = payload["episodes"][0]
    assert first["success"] is True
    assert len(first["trajectory_obs"]) > 0


def test_pipeline_run_rejects_bad_inputs(tmp_path):
    with pytest.raises(pipe.MujocoPipelineError, match="task is required"):
        pipe.run(
            task="", policy="scripted:expert", output_path=str(tmp_path / "x.json")
        )
    with pytest.raises(pipe.MujocoPipelineError, match="episodes"):
        pipe.run(
            task="reach",
            policy="scripted:expert",
            output_path=str(tmp_path / "x.json"),
            episodes=0,
        )


def test_cli_run_rejects_local_output_path(tmp_path):
    # Invoke through a parent group, mirroring how `npa workbench` registers
    # the mujoco app via add_typer(..., name="mujoco"): a single-command Typer
    # app collapses the command name on direct invocation, but the real CLI
    # path is `npa workbench mujoco run ...`.
    import typer

    parent = typer.Typer()
    parent.add_typer(mujoco_cli.app, name="mujoco")
    runner = CliRunner()
    out = tmp_path / "cli-report.json"
    result = runner.invoke(
        parent,
        [
            "mujoco",
            "run",
            "--task",
            "reach",
            "--policy",
            "scripted:expert",
            "--episodes",
            "1",
            "--output-path",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "S3 handoff contract" in result.output
    assert not out.exists()
