"""Reject incomplete benchmark coverage and preserve real simulator state in sweeps."""

from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    directory = root / "npa/examples/specialists/simulation"
    modules = {}
    for name in ("workload", "physics", "experiment", "mcp_bridge", "score"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(__import__("sys").modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _write_plan(workload, directory, plan=None):
    directory.mkdir(exist_ok=True)
    plan = plan or workload._reference_plan("translate")
    (directory / "plan.json").write_text(json.dumps(plan))
    return plan


@pytest.mark.parametrize(
    "task", ["translate", "mirror", "diagonal", "reverse", "cross", "offset"]
)
def test_broken_tasks_fail_and_complete_reference_passes(benchmark, tmp_path, task):
    workload = benchmark["workload"]
    workspace = tmp_path / task
    workload._prepare_workspace(workspace, task)
    with pytest.raises(ValueError):
        workload._validate(workspace, task)
    (workspace / "plan.json").write_text(json.dumps(workload._reference_plan(task)))
    assert len(workload._validate(workspace, task)["mass_scales"]) == 3


@pytest.mark.parametrize(
    "axis,values",
    [
        ("mass_scales", [1.0]),
        ("mass_scales", [0.8, 1.0, 1.0]),
        ("friction_scales", [0.8]),
        ("friction_scales", "0.8,1.2"),
    ],
)
def test_incomplete_or_duplicate_parameter_coverage_is_rejected(
    benchmark, tmp_path, axis, values
):
    workload = benchmark["workload"]
    plan = workload._reference_plan("translate")
    plan[axis] = values
    _write_plan(workload, tmp_path, plan)
    with pytest.raises(ValueError, match=axis):
        workload._validate(tmp_path, "translate")


def test_valid_physical_scene_must_also_satisfy_task(benchmark, tmp_path):
    workload = benchmark["workload"]
    plan = workload._reference_plan("translate")
    plan["scene"]["goal_y"] = 0.08
    _write_plan(workload, tmp_path, plan)
    with pytest.raises(ValueError, match="TASK.md"):
        workload._validate(tmp_path, "translate")


def test_parameter_order_and_numeric_roundoff_are_not_failures(benchmark, tmp_path):
    workload = benchmark["workload"]
    plan = copy.deepcopy(workload._reference_plan("translate"))
    plan["mass_scales"].reverse()
    plan["scene"]["goal_x"] += 1e-12
    _write_plan(workload, tmp_path, plan)
    assert workload._validate(tmp_path, "translate")["scene"]["goal_x"] > 0.08


def test_grader_rejects_missing_episodes_before_reading_claimed_artifacts(
    benchmark, tmp_path
):
    workload, physics = benchmark["workload"], benchmark["physics"]
    _write_plan(workload, tmp_path)
    run = tmp_path / "runs" / "incomplete"
    run.mkdir(parents=True)
    report = {"plan_sha256": workload._digest(tmp_path / "plan.json"), "episodes": []}
    (run / "report.json").write_text(json.dumps(report))
    (tmp_path / "latest.json").write_text(
        json.dumps({"report": "runs/incomplete/report.json"})
    )
    with pytest.raises(ValueError, match="omitted or duplicated"):
        physics._score(tmp_path, "translate")


def test_astra_baseline_has_parallel_batch_but_no_extra_source_access(
    benchmark, tmp_path
):
    pytest.importorskip("mcp.server.mcpserver")
    import asyncio

    path = benchmark["experiment"]._prepare(tmp_path / "run", 11, 0)
    server = benchmark["mcp_bridge"]._server(path)
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} >= {"read_file", "edit_file", "run_operations"}
    response = asyncio.run(
        server.call_tool(
            "read_file", {"specialist": "translate", "path": "../team.json"}
        )
    )
    assert json.loads(response.content[0].text)["ok"] is False


def test_mass_change_does_not_reset_initialized_simulator_state(benchmark):
    pytest.importorskip("mujoco")
    gym = pytest.importorskip("gymnasium")
    robotics = pytest.importorskip("gymnasium_robotics")
    import numpy as np

    gym.register_envs(robotics)
    env = gym.make("FetchPickAndPlace-v4").unwrapped
    try:
        env.reset(seed=11)
        before = env.data.qpos.copy()
        original_mass = float(env.model.body("object0").mass[0])
        benchmark["physics"]._apply_parameters(env, 0.8, 1.2)
        np.testing.assert_array_equal(env.data.qpos, before)
        assert float(env.model.body("object0").mass[0]) == pytest.approx(
            original_mass * 0.8
        )
    finally:
        env.close()


def test_price_accounting_counts_reasoning_once_and_handles_cache_writes(benchmark):
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 600,
        "cache_write_input_tokens": 100,
        "output_tokens": 200,
        "reasoning_output_tokens": 150,
    }
    prices = {
        "input_price_per_million_tokens": 10,
        "cached_input_price_per_million_tokens": 1,
        "cache_write_price_per_million_tokens": 12.5,
        "output_price_per_million_tokens": 50,
    }
    assert benchmark["score"]._astra_cost(usage, prices) == pytest.approx(0.01485)


def test_failed_specialist_does_not_get_a_falsely_complete_cost(benchmark):
    receipts = {
        "task": {
            "status": "needs_attention",
            "events": [
                {
                    "type": "model",
                    "model": "synthetic",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
                {"type": "needs_attention", "error": "ValueError"},
            ],
        }
    }
    price = {
        "synthetic": {
            "input_price_per_million_tokens": 1,
            "output_price_per_million_tokens": 2,
        }
    }
    result = benchmark["score"]._specialist_usage(receipts, price)
    assert result["recorded_cost_usd"] > 0
    assert result["usage_complete"] is False


def test_rejected_response_with_usage_still_has_measurable_cost(benchmark):
    receipts = {
        "task": {
            "status": "needs_attention",
            "events": [
                {
                    "type": "model",
                    "model": "synthetic",
                    "accepted": False,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
                {"type": "needs_attention", "error": "ValueError"},
            ],
        }
    }
    prices = {
        "synthetic": {
            "input_price_per_million_tokens": 1,
            "output_price_per_million_tokens": 2,
        }
    }
    result = benchmark["score"]._specialist_usage(receipts, prices)
    assert result["usage_complete"] is True
    assert result["recorded_cost_usd"] == pytest.approx(0.00014)


def test_later_recovery_cannot_inflate_original_trial_success(
    benchmark, monkeypatch, tmp_path
):
    def unexpected(*args):
        pytest.fail("must not grade artifacts created after the trial receipt snapshot")

    monkeypatch.setattr(benchmark["score"], "_score", unexpected)
    result = benchmark["score"]._task_outcome(tmp_path, "translate", {"events": []})
    assert result["accepted"] == 0


@pytest.mark.skipif(
    os.environ.get("NPA_SIMULATION_BENCHMARK_LIVE") != "1",
    reason="Opt-in real MuJoCo rendering",
)
def test_open_gripper_negative_control_fails_real_physics(
    benchmark, tmp_path, monkeypatch
):
    physics = benchmark["physics"]
    phases = physics.robot_sim._phases
    monkeypatch.setattr(
        physics.robot_sim,
        "_phases",
        lambda obj, goal: [
            (name, target, 1.0, steps) for name, target, _, steps in phases(obj, goal)
        ],
    )
    scene = benchmark["workload"]._reference_plan("translate")["scene"]
    row = physics._episode(scene, tmp_path / "failed-grasp", 11, 1.0, 1.2)
    assert row["accepted"] is False
    assert row["checks"]["bilateral_grasp_contact"] is False
    assert row["checks"]["lifted"] is False
    assert (
        physics._replay(scene, tmp_path / "failed-grasp/physics.npz", row)["accepted"]
        is False
    )


@pytest.mark.skipif(
    os.environ.get("NPA_SIMULATION_BENCHMARK_LIVE") != "1",
    reason="Opt-in real MuJoCo rendering",
)
def test_changed_state_fails_replay_even_with_matching_file_hash(benchmark, tmp_path):
    import numpy as np

    physics, workload = benchmark["physics"], benchmark["workload"]
    scene = workload._reference_plan("translate")["scene"]
    row = physics._episode(scene, tmp_path / "episode", 11, 1.0, 1.2)
    path = tmp_path / "episode/physics.npz"
    assert physics._replay(scene, path, row)["accepted"] is True
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["next_state"][0, 0] += 0.1
    np.savez_compressed(path, **arrays)
    row["trace_sha256"] = workload._digest(path)
    with pytest.raises(ValueError, match="did not replay"):
        physics._replay(scene, path, row)


def test_infrastructure_interruption_keeps_partial_usage_and_unknown_total(
    benchmark, tmp_path
):
    execution = {
        "arm": "specialists",
        "seed": 53,
        "agent_tool_seconds": None,
        "infrastructure_failure": "Host storage was exhausted",
    }
    event = {
        "type": "model",
        "model": "synthetic",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    receipts = {name: {"events": [event]} for name in benchmark["workload"].TASKS}
    (tmp_path / "execution.json").write_text(json.dumps(execution))
    (tmp_path / "task-receipts.json").write_text(json.dumps(receipts))
    prices = {
        "synthetic": {
            "input_price_per_million_tokens": 1,
            "output_price_per_million_tokens": 2,
        }
    }
    result = benchmark["score"]._score_run(tmp_path, prices)
    assert result["recorded_cost_usd"] > 0
    assert result["estimated_cost_usd"] is None
    assert result["agent_tool_seconds"] is None
    assert result["infrastructure_failure"] == execution["infrastructure_failure"]
