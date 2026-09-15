"""Exercise sparse reset bookkeeping in the complete generated Isaac rollout loop."""

from __future__ import annotations

import ast
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.sim2real.byo_isaac_policy_rollout import ISAAC_ROLLOUT_SCRIPT


HORIZON = 300
SAMPLE_STEPS = [int(index * (HORIZON - 1) / 31) for index in range(32)]


def _native_program():
    """Keep native initialization, capture, every loop statement and final capture."""
    tree = ast.parse(ISAAC_ROLLOUT_SCRIPT)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.FunctionDef) and child.name == "capture"
            for child in node.body
        )
    )
    start = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "previous_goal_distance"
            for target in node.targets
        )
    )
    end = next(
        index
        for index, node in enumerate(main.body[start:], start)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "capture"
    )
    return ast.fix_missing_locations(
        ast.Module(body=main.body[start : end + 1], type_ignores=[])
    )


class _Scene(dict):
    def __init__(self, count, torch):
        super().__init__()
        self.env_origins = torch.zeros(count, 3)


def _scene(count, torch):
    scene = _Scene(count, torch)
    scene["object"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(count, 3),
            root_lin_vel_w=torch.zeros(count, 3),
        )
    )
    scene["ee_frame"] = SimpleNamespace(
        data=SimpleNamespace(target_pos_w=torch.zeros(count, 1, 3))
    )
    scene["object_contact"] = SimpleNamespace(
        data=SimpleNamespace(net_forces_w_history=torch.ones(count, 1, 1, 3))
    )
    scene["primary"] = SimpleNamespace(
        data=SimpleNamespace(
            output={"rgb": torch.zeros(count, 2, 2, 3, dtype=torch.uint8)}
        )
    )
    return scene


class _Environment:
    """Return deterministic post-reset state, like an autoresetting vector env."""

    def __init__(self, resets, torch, *, stable=False):
        self.torch = torch
        self.resets = resets
        self.stable = stable
        self.num_envs = len(resets)
        self.unwrapped = self
        self.step_dt = 0.02
        self.step_calls = 0
        self.policy_calls = 0
        self.scene = _scene(self.num_envs, torch)
        self.goal = torch.zeros(self.num_envs, 3)
        self.command_manager = SimpleNamespace(get_command=lambda _: self.goal)
        self.npa_scenario_rows = [
            {"scenario_config_digest": f"fixture-scenario-{index}"}
            for index in range(self.num_envs)
        ]
        self.npa_scenario_indices = torch.arange(self.num_envs)
        self._update_state(-1)

    def _episode(self, index, step):
        prior_resets = [reset for reset in self.resets[index] if reset <= step]
        generation = len(prior_resets)
        age = step - prior_resets[-1] if prior_resets else step + 1
        return generation, age

    def _update_state(self, step):
        for index in range(self.num_envs):
            generation, age = self._episode(index, step)
            base_z = 1.0 + 0.2 * generation + 0.1 * index
            height = base_z + (0.02 if self.stable else 0.002) * age
            distance = 0.01 if self.stable else 0.8 - 0.001 * age
            ee_distance = 0.02 if self.stable else 0.3 - 0.0005 * age
            self.scene["object"].data.root_pos_w[index] = self.torch.tensor(
                [0.0, 0.0, height]
            )
            self.scene["ee_frame"].data.target_pos_w[index, 0] = self.torch.tensor(
                [ee_distance, 0.0, height]
            )
            self.goal[index] = self.torch.tensor([distance, 0.0, height])
            # With no rerender on reset, RGB can still belong to the prior episode.
            camera_generation = generation - int(step in self.resets[index])
            self.scene["primary"].data.output["rgb"][index, :, :, 0] = camera_generation
            self.scene["primary"].data.output["rgb"][index, :, :, 1] = index

    def policy(self, _observations):
        self.policy_calls += 1
        return -self.torch.ones(self.num_envs, 8)

    def step(self, actions):
        assert actions.shape == (self.num_envs, 8)
        step = self.step_calls
        self.step_calls += 1
        self._update_state(step)
        done = self.torch.tensor([step in resets for resets in self.resets])
        return self.torch.zeros(self.num_envs, 36), None, done, {}


def _simulation_namespace(environment):
    count, torch = environment.num_envs, environment.torch
    return {
        "np": np,
        "torch": torch,
        "os": os,
        "N": count,
        "HORIZON_STEPS": HORIZON,
        "SAMPLE_INDEX": {step: index for index, step in enumerate(SAMPLE_STEPS)},
        "env": environment,
        "uenv": environment,
        "policy": environment.policy,
        "_batched_obs": lambda value: value,
        "obs": torch.zeros(count, 36),
        "gripper_start": 7,
        "actions_log": {index: [] for index in range(count)},
    }


def _camera_namespace(tmp_path, count, encoded_pixels):
    def record_pixels(path, pixels):
        # Exercise the real camera loop/metadata without testing PNG encoding.
        encoded_pixels[str(Path(path))] = np.asarray(pixels).copy()

    return {
        "rollout_ids": [f"fixture-rollout-{index}" for index in range(count)],
        "frame_names": {index: {"primary": []} for index in range(count)},
        "frame_metadata": {index: {"primary": []} for index in range(count)},
        "CAMERA_VIEWS": [{"name": "primary"}],
        "SIM_DEVICE": "cuda:0",
        "CAPTURE_STRIDE": 1,
        "CAPTURE_WIDTH": 2,
        "CAPTURE_HEIGHT": 2,
        "FRAMES_DIR": str(tmp_path),
        "CKPT_URI": "s3://fixture/checkpoint.pt",
        "_camera_key": lambda name: name,
        "_write_rgb_png": record_pixels,
    }


def _run_native(tmp_path, resets, *, stable=False):
    torch = pytest.importorskip("torch")
    environment = _Environment(resets, torch, stable=stable)
    encoded_pixels = {}
    namespace = {
        **_simulation_namespace(environment),
        **_camera_namespace(tmp_path, len(resets), encoded_pixels),
    }
    script = tmp_path / "generated_rollout_loop.py"
    script.write_text(ast.unparse(_native_program()))
    result = runpy.run_path(str(script), init_globals=namespace)
    assert environment.step_calls == environment.policy_calls == HORIZON
    assert result["simulation_clock"].completed_steps == HORIZON
    return result, encoded_pixels


def _assert_action_boundary(row, scheduled, previous_step):
    step = row["sim_step"]
    boundary = row["episode_boundary"]
    generation = sum(reset <= step for reset in scheduled)
    pending = [reset for reset in scheduled if previous_step < reset <= step]
    assert boundary["schema"] == "npa.sim2real.episode_boundary.v1"
    assert boundary["simulator_episode_id"] == generation
    assert boundary["action_episode_id"] == sum(reset < step for reset in scheduled)
    assert boundary["reset_events"] == [
        {
            "sim_step": reset,
            "terminated_episode_id": scheduled.index(reset),
            "next_episode_id": scheduled.index(reset) + 1,
        }
        for reset in pending
    ]
    assert boundary["reset_on_current_step"] is (step in scheduled)
    assert boundary["action_outcome_valid"] is (step not in scheduled)
    assert boundary["temporal_credit_valid"] is (not pending)
    truth = row["simulator_ground_truth"]
    assert truth["terminated"] is bool(pending)
    if pending:
        assert truth["termination_reason"] == "task_or_timeout"


def _assert_episode_frames(result, index, scheduled, tmp_path, pixels):
    frames = result["frame_metadata"][index]["primary"]
    assert [frame["sim_step"] for frame in frames] == [*SAMPLE_STEPS, HORIZON]
    assert len(frames) == 33
    for frame in frames:
        completed = min(frame["sim_step"] + 1, HORIZON)
        generation = sum(reset < completed for reset in scheduled)
        assert frame["completed_simulation_steps"] == completed
        assert frame["timestamp_seconds"] == pytest.approx(completed * 0.02)
        assert frame["episode_id"] == f"fixture-rollout-{index}"
        path = tmp_path / frame["episode_id"] / frame["path"]
        camera_generation, camera_index = pixels[str(path)][0, 0, :2].tolist()
        assert camera_index == index
        if completed - 1 in scheduled:
            assert frame["simulator_episode_id"] is None
            assert camera_generation == generation - 1
        else:
            assert frame["simulator_episode_id"] == camera_generation == generation
    assert frames[-1]["timestamp_seconds"] == frames[-2]["timestamp_seconds"] == 6.0


@pytest.mark.parametrize(
    "resets",
    [
        ((), ()),
        ((249,), ()),
        ((250,), ()),
        ((243, 247), (49, 149)),
        ((249,), (250,)),
        ((0,), (299,)),
    ],
    ids=["no-reset", "unsampled-reset", "sampled-reset", "multiple", "per-env", "ends"],
)
def test_native_loop_preserves_every_reset_and_camera_generation(tmp_path, resets):
    result, pixels = _run_native(tmp_path, resets)
    for index, scheduled in enumerate(resets):
        rows = result["actions_log"][index]
        assert len(rows) == 32
        previous_step = -1
        for row in rows:
            _assert_action_boundary(row, scheduled, previous_step)
            previous_step = row["sim_step"]
        _assert_episode_frames(result, index, scheduled, tmp_path, pixels)


@pytest.mark.parametrize("reset_step", [249, 250])
def test_native_distances_and_lift_rebase_then_resume_within_episode(
    tmp_path, reset_step
):
    result, _pixels = _run_native(tmp_path, ((reset_step,), ()))
    reset_rows = result["actions_log"][0]
    boundary_truth = reset_rows[26]["simulator_ground_truth"]
    assert boundary_truth["object_goal_distance_change_m"] == 0.0
    assert boundary_truth["end_effector_distance_change_m"] == 0.0
    assert boundary_truth["object_lift_m"] == pytest.approx((250 - reset_step) * 0.002)
    resumed = reset_rows[27]
    assert resumed["episode_boundary"]["temporal_credit_valid"] is True
    truth = resumed["simulator_ground_truth"]
    delta_steps = resumed["sim_step"] - reset_rows[26]["sim_step"]
    assert truth["object_goal_distance_change_m"] == pytest.approx(
        delta_steps * 0.001, abs=1e-6
    )
    assert truth["end_effector_distance_change_m"] == pytest.approx(
        delta_steps * 0.0005, abs=1e-6
    )
    assert truth["object_lift_m"] == pytest.approx(
        (resumed["sim_step"] - reset_step) * 0.002
    )
    # Initial lift uses the actual pre-loop reset state, not the first post-step state.
    assert reset_rows[0]["simulator_ground_truth"]["object_lift_m"] == pytest.approx(
        0.002
    )
    # Resetting one environment must not clear the other environment's baselines.
    unaffected = result["actions_log"][1][26]["simulator_ground_truth"]
    assert unaffected["object_goal_distance_change_m"] == pytest.approx(0.009, abs=1e-6)
    assert unaffected["end_effector_distance_change_m"] == pytest.approx(
        0.0045, abs=1e-6
    )
    assert unaffected["object_lift_m"] == pytest.approx(0.502)


@pytest.mark.parametrize("reset_step", [249, 250])
def test_native_stability_requires_new_episode_dwell(tmp_path, reset_step):
    result, _pixels = _run_native(tmp_path, ((reset_step,), ()), stable=True)
    reset_rows = result["actions_log"][0]
    for field in ("stable_grasp", "placement_stable"):
        assert reset_rows[25]["simulator_ground_truth"][field] is True
        assert reset_rows[26]["simulator_ground_truth"][field] is False
        assert reset_rows[27]["simulator_ground_truth"][field] is True
        assert result["actions_log"][1][26]["simulator_ground_truth"][field] is True
