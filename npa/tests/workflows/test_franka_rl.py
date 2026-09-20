"""Regression tests for Franka geometric success, held-out selection, and stage failure."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows import franka_rl
from npa.workflows.franka_rl_metrics import PlacementMetrics, rank_checkpoint
from npa.workflows.franka_rl_report import summarize


@pytest.fixture
def recipe():
    return franka_rl._recipe(
        SimpleNamespace(
            run_id="test-franka",
            seed=42,
            iterations=1500,
            num_envs=4096,
            eval_episodes=2,
            minimum_success=0.7,
        )
    )


def test_success_requires_consecutive_lifted_slow_steps(recipe):
    metrics = PlacementMetrics(1, recipe)
    for _ in range(19):
        metrics.update([0.01], [0.01], [0.3], [False])
    assert not metrics.success[0]
    metrics.update([0.1], [0.01], [0.3], [False])
    assert metrics.streak[0] == 0
    for _ in range(20):
        metrics.update([0.01], [0.01], [0.3], [False])
    assert metrics.success[0]


@pytest.mark.parametrize(
    "distance,speed,height", [(0.05, 0.01, 0.3), (0.01, 0.03, 0.3), (0.01, 0.01, 0.1)]
)
def test_fast_passes_or_unlifted_object_never_count(recipe, distance, speed, height):
    metrics = PlacementMetrics(1, recipe)
    for _ in range(25):
        metrics.update([distance], [speed], [height], [False])
    assert not metrics.success[0]


def test_automatic_reset_cannot_complete_or_restart_stability(recipe):
    metrics = PlacementMetrics(1, recipe)
    for _ in range(19):
        metrics.update([0.01], [0.01], [0.3], [False])
    metrics.update([0.01], [0.01], [0.3], [True])
    for _ in range(25):
        metrics.update([0.01], [0.01], [0.3], [False])
    assert not metrics.success[0]
    assert metrics.longest[0] == 19


def test_domain_departure_revokes_prior_success_without_erasing_hold_trace(recipe):
    metrics = PlacementMetrics(2, recipe)
    for _ in range(recipe["stable_steps"]):
        metrics.update([0.01, 0.01], [0, 0], [0.2, 0.2], [False, False])
    assert metrics.success.tolist() == [True, True]
    metrics.update([1000, 0.01], [1000, 0], [1000, 0.2], [True, False], [True, False])
    assert metrics.success.tolist() == [False, True]
    assert metrics.lifted.tolist() == [False, True]
    assert metrics.domain_exit.tolist() == [True, False]
    assert metrics.longest[0] == recipe["stable_steps"]
    assert metrics.steps[0] == recipe["stable_steps"]
    for _ in range(25):
        metrics.update([0.01, 0.01], [0, 0], [0.2, 0.2], [False, False], [False, False])
    assert not metrics.success[0]


@pytest.mark.parametrize(
    "distance,speed,height,done",
    [
        ([np.nan], [0], [1], [False]),
        ([-1], [0], [1], [False]),
        ([0], [-1], [1], [False]),
        ([0, 0], [0], [1], [False]),
        ([0], [0], [1], [False, True]),
    ],
)
def test_invalid_simulator_geometry_fails(recipe, distance, speed, height, done):
    with pytest.raises(ValueError):
        PlacementMetrics(1, recipe).update(distance, speed, height, done)


def _validation_row(**changes):
    return {
        "split": "validation",
        "checkpoint_sha256": "a" * 64,
        "iteration": 500,
        "env_index": 0,
        "closest_distance_m": 0.04,
        "success": True,
    } | changes


def test_validation_ranking_prefers_success_before_distance_and_earlier_ties():
    success = [_validation_row()]
    miss = [_validation_row(success=False, closest_distance_m=0.001)]
    assert rank_checkpoint(success) > rank_checkpoint(miss)
    assert rank_checkpoint(success) > rank_checkpoint([_validation_row(iteration=1000)])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [_validation_row(split="test")],
        [_validation_row(), _validation_row()],
        [_validation_row(), _validation_row(env_index=1, checkpoint_sha256="b" * 64)],
        [_validation_row(closest_distance_m=float("inf"))],
    ],
)
def test_selection_rejects_test_data_mixed_weights_duplicates_and_nonfinite(rows):
    with pytest.raises(ValueError):
        rank_checkpoint(rows)


@pytest.fixture
def evaluation(recipe):
    validity = {
        "verified": True,
        "contract": recipe["simulation_validity"],
        "checked_batches": 251,
    }
    rows = [
        {
            "split": "test",
            "condition": condition,
            "env_index": index,
            "reset_seed": recipe["test_seed"],
            "arm": arm,
            "success": arm == "trained",
            "checkpoint_sha256": "a" * 64 if arm == "trained" else "b" * 64,
            "closest_distance_m": 0.01,
            "longest_stable_steps": 20 if arm == "trained" else 0,
        }
        for arm in ("initial", "trained")
        for condition in recipe["conditions"]
        for index in range(2)
    ]
    for row in rows:
        row["initial_state_sha256"] = f"{row['env_index']:064x}"
        row["simulation_validity"] = validity
    return {
        "recipe": recipe,
        "trials": rows,
        "validation": [dict(_validation_row(), simulation_validity=validity)],
        "training": {"physics": {"simulation_validity": validity}},
        "selection": {"selected_checkpoint_sha256": "a" * 64},
    }


def test_complete_simulation_success_never_claims_physical_transfer(evaluation):
    report = summarize(evaluation)
    assert report["simulation_qualified"]
    assert report["paired_delta_95ci"] == [1.0, 1.0]
    assert not report["ready_for_robot_deployment"]
    assert not report["physical_robot_tested"]


@pytest.mark.parametrize("source", ["training", "validation", "test"])
def test_missing_measured_validity_rejects_successful_report(evaluation, source):
    if source == "training":
        evaluation["training"]["physics"].pop("simulation_validity")
    else:
        rows = evaluation["trials" if source == "test" else "validation"]
        rows[0].pop("simulation_validity")
    with pytest.raises(ValueError, match="simulation validity evidence"):
        summarize(evaluation)


def test_historical_geometry_alone_cannot_qualify_a_new_report(evaluation):
    evaluation["recipe"].pop("simulation_validity")
    report = summarize(evaluation)
    assert report["success_rates"]["trained"]["nominal"] == 1
    assert not report["simulation_validity_verified"]
    assert not report["simulation_qualified"]


@pytest.mark.parametrize("robot_type", ["ur10e_robotiq85", "kinova_jaco7"])
def test_report_hashes_exported_robot_recording(
    evaluation, robot_type, tmp_path, monkeypatch
):
    from npa.workflows import franka_rl_report as reports
    from npa.workflows.lerobot_transfer_data import file_sha256

    evaluation["recipe"].pop("visual_eval", None)
    evaluation["recipe"].pop("embodiment", None)
    source, output = tmp_path / "evaluated", tmp_path / "reported"
    (source / "trajectories").mkdir(parents=True)
    (source / "evaluation.json").write_text(json.dumps(evaluation))
    (source / "trajectories/meta.json").write_text(
        json.dumps({"robot_type": robot_type})
    )

    def export(captured, destination, visual):
        metadata = json.loads((captured / "meta.json").read_text())
        path = destination / (metadata["robot_type"] + ".rrd")
        path.write_bytes(b"recording fixture")
        return path

    monkeypatch.setattr(reports, "_convert_trajectories", export)
    monkeypatch.setattr(reports, "_plot", lambda *args: None)
    reports.report_results(source, output)
    report = json.loads((output / "report.json").read_text())
    assert report["recording_file"] == robot_type + ".rrd"
    assert report["recording_sha256"] == file_sha256(output / report["recording_file"])


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "wrong_checkpoint", "wrong_seed", "wrong_split"],
)
def test_report_rejects_incomplete_or_mixed_test_evidence(evaluation, mutation):
    changed = deepcopy(evaluation)
    if mutation == "missing":
        changed["trials"].pop()
    elif mutation == "duplicate":
        changed["trials"].append(changed["trials"][0])
    else:
        key, value = {
            "wrong_checkpoint": ("checkpoint_sha256", "c" * 64),
            "wrong_seed": ("reset_seed", 42),
            "wrong_split": ("split", "validation"),
        }[mutation]
        changed["trials"][-1][key] = value
    with pytest.raises(ValueError):
        summarize(changed)


def test_report_rejects_unpaired_start_states_and_false_success(evaluation):
    changed = deepcopy(evaluation)
    changed["trials"][-1]["initial_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different physical resets"):
        summarize(changed)
    changed = deepcopy(evaluation)
    changed["trials"][-1]["longest_stable_steps"] = 0
    with pytest.raises(ValueError, match="sustained physical event"):
        summarize(changed)


def test_native_failure_preserves_log_and_never_reports_completion(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        franka_rl.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=7),
    )
    with pytest.raises(RuntimeError, match="exit 7"):
        franka_rl._run_native("train", tmp_path / "input", tmp_path / "output")
    assert (tmp_path / "output/runtime.log").exists()
    assert not (tmp_path / "output/training.json").exists()


@pytest.mark.parametrize(
    "stage,receipt",
    [
        ("train", "training.json"),
        ("validate", "validation.json"),
        ("test", "evaluation.json"),
        ("capture", "meta.json"),
    ],
)
def test_isaac_zero_exit_without_completion_record_is_a_failure(
    tmp_path, monkeypatch, stage, receipt
):
    monkeypatch.setattr(
        franka_rl.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(RuntimeError, match=f"required {receipt}"):
        franka_rl._native_step(stage, tmp_path / "input", tmp_path / "output")


@pytest.mark.parametrize(
    "message",
    ["PhysX error: GPU buffer overflow", "the simulation will miss interactions"],
)
def test_physics_error_rejects_even_zero_exit_and_valid_completion(
    tmp_path, monkeypatch, message
):
    def simulate(argv, **kwargs):
        output = Path(argv[argv.index("--output-path") + 1])
        (output / "training.json").write_text('{"schema":"npa.franka-rl.training.v1"}')
        kwargs["stdout"].write(message + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(franka_rl.subprocess, "run", simulate)
    with pytest.raises(RuntimeError, match="reported invalid physics"):
        franka_rl._native_step("train", tmp_path / "input", tmp_path / "output")


def test_scene_restores_sealed_physx_pair_capacity(tmp_path, monkeypatch, recipe):
    import sys
    from npa.workflows import franka_rl_environment
    from npa.workflows.sim2real import isaac_assets_compat

    config = SimpleNamespace(
        scene=SimpleNamespace(),
        sim=SimpleNamespace(
            physics=SimpleNamespace(gpu_total_aggregate_pairs_capacity=16384)
        ),
        commands=SimpleNamespace(object_pose=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        SimpleNamespace(load_cfg_from_registry=lambda *args: config),
    )
    monkeypatch.setattr(
        isaac_assets_compat, "remap_moved_franka_usd", lambda *args: None
    )
    monkeypatch.setattr(franka_rl_environment, "_physics_events", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.managers",
        SimpleNamespace(TerminationTermCfg=SimpleNamespace),
    )
    config.terminations = SimpleNamespace()
    actual = franka_rl_environment.environment_config(recipe, training=True)
    assert actual.sim.physics.gpu_total_aggregate_pairs_capacity == 2**21
    assert (
        actual.sim.physics.gpu_total_aggregate_pairs_capacity
        == recipe["physics_capacity"]["gpu_total_aggregate_pairs_capacity"]
    )


@pytest.mark.parametrize("capture_exit", [0, 1])
def test_evaluation_uses_fresh_processes_and_requires_capture_before_publication(
    tmp_path, monkeypatch, capture_exit
):
    calls, published = [], []
    (tmp_path / "recipe.json").write_text("{}")

    def simulate(argv, **kwargs):
        stage = argv[3]
        calls.append(stage)
        output = Path(argv[argv.index("--output-path") + 1])
        receipts = {
            "validate": ("validation.json", {"schema": "npa.franka-rl.validation.v1"}),
            "test": ("evaluation.json", {"schema": "npa.franka-rl.evaluation.v1"}),
            "capture": ("meta.json", {"format": "npa_isaac_lab_rollout_v2"}),
        }
        filename, record = receipts[stage]
        (output / filename).write_text(json.dumps(record))
        return SimpleNamespace(returncode=capture_exit if stage == "capture" else 0)

    monkeypatch.setattr(franka_rl.subprocess, "run", simulate)
    monkeypatch.setattr(franka_rl, "materialize", lambda *args: tmp_path)
    monkeypatch.setattr(
        franka_rl, "publish", lambda root, destination: published.append(destination)
    )
    argv = [
        "evaluate",
        "--input-path",
        str(tmp_path),
        "--output-path",
        str(tmp_path / "published"),
    ]
    if capture_exit:
        with pytest.raises(RuntimeError, match="capture failed"):
            franka_rl.main(argv)
    else:
        assert franka_rl.main(argv) == 0
    assert calls == ["validate", "test", "capture"]
    assert len(published) == 1
    if capture_exit:
        assert published[0].startswith(str(tmp_path / "published-failures") + "/")
    else:
        assert published == [str(tmp_path / "published")]


def test_initial_native_checkpoint_is_decodable_without_a_training_logger(tmp_path):
    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_runtime import _save_initial

    runner = SimpleNamespace(
        current_learning_iteration=0,
        alg=SimpleNamespace(
            save=lambda: {"actor_state_dict": {"weight": torch.ones(2)}}
        ),
    )
    path = tmp_path / "initial.pt"
    _save_initial(runner, path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["iter"] == 0 and payload["infos"] is None
    assert torch.equal(payload["actor_state_dict"]["weight"], torch.ones(2))


def test_test_process_rejects_changed_selected_weights_before_opening_test_stream(
    tmp_path, monkeypatch, recipe
):
    from npa.workflows import franka_rl_eval
    from npa.workflows.lerobot_transfer_data import file_sha256, write_json

    training, output = tmp_path / "training", tmp_path / "output"
    checkpoint = training / "checkpoints/model_500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen validation weights")
    selection = {
        "iteration": 500,
        "selected_checkpoint_sha256": file_sha256(checkpoint),
        "test_data_used": False,
    }
    write_json(output / "selection.json", selection)
    write_json(output / "validation.json", {"recipe": recipe, "selection": selection})
    (output / "selected.pt").write_bytes(b"changed weights")
    calls = []
    monkeypatch.setattr(franka_rl_eval, "_test", lambda *args: calls.append(True))
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        franka_rl_eval.evaluate_checkpoints(training, output, recipe, {})
    assert not calls


def test_isaac_physics_evidence_decodes_actual_warp_arrays():
    pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from npa.workflows.franka_rl_environment import physics_evidence

    material = wp.array(
        [[[0.5, 0.25, 0.0]], [[1.5, 1.25, 0.0]]], dtype=wp.float32, device="cpu"
    )
    masses = wp.array([[0.7], [1.3]], dtype=wp.float32, device="cpu")
    view = SimpleNamespace(
        get_material_properties=lambda: material, get_masses=lambda: masses
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene={"object": SimpleNamespace(root_view=view)},
            npa_embodiment_evidence={"robot_type": "franka"},
            cfg=SimpleNamespace(
                sim=SimpleNamespace(
                    physics=SimpleNamespace(gpu_total_aggregate_pairs_capacity=2**21)
                )
            ),
            event_manager=SimpleNamespace(
                active_terms={"startup": ["npa_object_mass", "npa_object_material"]}
            ),
        )
    )
    evidence = physics_evidence(env)
    assert evidence["mass_kg"] == pytest.approx({"min": 0.7, "max": 1.3})
    assert evidence["static_friction"] == {"min": 0.5, "max": 1.5}
    assert evidence["dynamic_friction"] == {"min": 0.25, "max": 1.25}
    assert evidence["physics_capacity"] == {"gpu_total_aggregate_pairs_capacity": 2**21}
    assert evidence["embodiment"] == {"robot_type": "franka"}


def test_isaac_proxy_geometry_uses_explicit_tensors_for_goal_and_reset_hash(
    monkeypatch,
):
    import sys

    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_eval import _initial_state_hashes, _observe

    def transform(position, quaternion, command):
        assert all(
            isinstance(value, torch.Tensor) for value in (position, quaternion, command)
        )
        return position + command, quaternion

    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.math",
        SimpleNamespace(combine_frame_transforms=transform),
    )

    def proxy(values):
        return SimpleNamespace(torch=torch.tensor(values, dtype=torch.float32))

    robot = SimpleNamespace(
        root_pos_w=proxy([[1, 0, 0], [10, 0, 0]]),
        root_quat_w=proxy([[0, 0, 0, 1]] * 2),
        joint_pos=proxy([[0] * 9] * 2),
    )
    obj = SimpleNamespace(
        root_pos_w=proxy([[1.5, 0, 0.3], [10.8, 0, 0.3]]),
        root_quat_w=proxy([[0, 0, 0, 1]] * 2),
        root_lin_vel_w=proxy([[0, 0.02, 0]] * 2),
    )
    scene = type("Scene", (dict,), {})(
        robot=SimpleNamespace(data=robot), object=SimpleNamespace(data=obj)
    )
    scene.env_origins = torch.zeros((2, 3))
    command = torch.tensor([[0.5, 0, 0.3, 0, 0, 0, 1]] * 2)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            scene=scene,
            command_manager=SimpleNamespace(get_command=lambda name: command),
        )
    )
    distance, speed, height = _observe(env)
    assert distance == pytest.approx([0, 0.3], abs=1e-6)
    assert speed == pytest.approx([0.02, 0.02]) and height == pytest.approx([0.3, 0.3])
    hashes = _initial_state_hashes(env)
    assert hashes == _initial_state_hashes(env) and len(set(hashes)) == 2


def test_isaac_proxy_camera_returns_owned_rgb_pixels():
    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_capture import _frame

    pixels = torch.full((1, 480, 640, 3), 128, dtype=torch.uint8)
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=pixels)})
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(scene={"npa_rollout_camera": camera})
    )
    frame = _frame(env)
    pixels.zero_()
    assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
    assert np.all(frame == 128)


@pytest.mark.parametrize("condition", ["nominal", "delay"])
def test_capture_can_reset_tensors_retained_from_preceding_inference_episode(
    tmp_path, monkeypatch, recipe, condition
):
    torch = pytest.importorskip("torch")
    from npa.workflows import franka_rl_capture, franka_rl_eval

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros((1, 9))))
    )
    wrapped = SimpleNamespace(
        unwrapped=SimpleNamespace(scene={"robot": robot}, seed=lambda seed: None),
        clip_actions=None,
        num_actions=8,
        device="cpu",
        metric=torch.zeros(1),
        steps=0,
    )

    def reset():
        wrapped.metric[0] = 0.0
        return torch.zeros((1, 36)), {}

    def step(action):
        wrapped.metric = torch.ones(1)
        wrapped.steps += 1
        return torch.zeros((1, 36)), None, torch.tensor([False]), {}

    def frame(env):
        pixels = np.zeros((480, 640, 3), dtype=np.uint8)
        pixels[0, 0] = 255
        pixels[1, 1] = wrapped.steps
        return pixels

    wrapped.reset, wrapped.step = reset, step
    monkeypatch.setattr(franka_rl_capture, "_orient_camera", lambda env: None)
    monkeypatch.setattr(franka_rl_capture, "_frame", frame)
    monkeypatch.setattr(
        franka_rl_capture, "task_domain_exits", lambda env: torch.tensor([False])
    )
    monkeypatch.setattr(
        franka_rl_eval,
        "_observe",
        lambda env: (np.array([0.2]), np.array([0.0]), np.array([0.0])),
    )
    monkeypatch.setattr(franka_rl_eval, "_initial_state_hashes", lambda env: ["a" * 64])
    recipe = dict(recipe, episode_steps=2)
    for index in range(2):
        result = franka_rl_capture._capture_episode(
            wrapped,
            lambda obs: torch.ones((1, 8)),
            tmp_path / str(index),
            recipe,
            condition=condition,
        )
        assert result["length"] == 2
        applied = np.load(tmp_path / str(index) / "actions.npy")
        np.testing.assert_array_equal(applied[0], 0 if condition == "delay" else 1)
        np.testing.assert_array_equal(applied[1], 1)
    assert wrapped.metric.is_inference() and wrapped.steps == 4
    with pytest.raises(RuntimeError, match="outside InferenceMode"):
        wrapped.metric.zero_()


def test_prepare_seals_disjoint_streams_without_isaac(tmp_path, recipe):
    output = tmp_path / "prepared"
    assert (
        franka_rl.main(
            ["prepare", "--run-id", "test-franka", "--output-path", str(output)]
        )
        == 0
    )
    sealed = json.loads((output / "recipe.json").read_text())
    assert (
        len(
            {
                sealed[key]
                for key in ("seed", "validation_seed", "test_seed", "capture_seed")
            }
        )
        == 4
    )
    assert sealed["task"] == "Isaac-Lift-Cube-Franka-v0"
    assert set(json.loads((output / "checksums.json").read_text())) == {
        "recipe.json",
        *sealed["assets"]["files"],
    }
    assert sealed["assets"]["target"] == "spool"
    assert sealed["visual_eval"]["model"] == "MiniMaxAI/MiniMax-M3"


def test_workflow_uses_real_stages_and_render_capable_gpu():
    import yaml

    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/testing/franka-rl-transfer.yaml"
    )
    workflow = yaml.safe_load(path.read_text())
    assert workflow["resources"]["isaac"]["accelerators"] == "RTXPRO6000:1"
    assert "@sha256:" in workflow["resources"]["isaac"]["image"]
    for stage in ("prepare", "train", "evaluate", "visual-evaluate", "report"):
        assert workflow["states"][stage]["run"]["argv"][:4] == [
            "python3",
            "-m",
            "npa.workflows.franka_rl",
            stage,
        ]


def test_every_franka_worker_uses_staged_source_instead_of_stale_image_modules(
    monkeypatch,
):
    import yaml

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )
    from npa.orchestration.npa_workflow.spec import load_spec

    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/staged-source/npa/")
    path = (
        Path(__file__).resolve().parents[3]
        / "workflows/testing/franka-rl-transfer.yaml"
    )
    spec = load_spec(path)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="test-franka"),
        run_id="test-franka",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    tasks = [doc for doc in yaml.safe_load_all(rendered) if doc and "envs" in doc]
    assert len(tasks) == 5
    assert all(task["envs"]["NPA_SRC_OVERLAY"] == "1" for task in tasks)
    assert all(
        task["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/staged-source/npa/"
        for task in tasks
    )


def test_franka_recording_decodes_camera_and_all_named_joints_without_synthetic_pose(
    tmp_path,
):
    pytest.importorskip("rerun")
    from rerun.recording import load_recording

    from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
    from npa.workflows.franka_rl_recording import write_recording

    raw = tmp_path / "raw" / "episode_000000"
    raw.mkdir(parents=True)
    np.save(raw / "state.npy", np.zeros((3, 9), dtype=np.float32))
    np.save(raw / "actions.npy", np.ones((3, 8), dtype=np.float32))
    np.save(raw / "rgb.npy", np.zeros((3, 32, 32, 3), dtype=np.uint8))
    states, actions = [f"joint{i}" for i in range(9)], [f"action{i}" for i in range(8)]
    dataset = tmp_path / "lerobot"
    convert(
        raw.parent,
        dataset,
        fps=50,
        robot_type="franka",
        spec=LeRobotFeatureSpec(states, actions, "franka"),
    )
    metadata = {
        "run_id": "fixture-franka",
        "genuine_simulator_pixels": True,
        "num_episodes": 1,
        "episode_results": [{"success": False}],
        "state_names": states,
        "action_names": actions,
    }
    output = tmp_path / "franka.rrd"
    counts = write_recording(dataset, output, metadata)
    assert len(counts) == 18 and set(counts.values()) == {3}
    chunks = list(load_recording(output).chunks())
    entities = {str(chunk.entity_path) for chunk in chunks}
    assert "/episodes/000000/video" in entities
    assert not any("transform" in entity or "skeleton" in entity for entity in entities)


@pytest.mark.parametrize("fail_last", [False, True])
def test_visual_capture_grid_uses_separate_native_processes_and_requires_every_case(
    tmp_path, monkeypatch, recipe, fail_last
):
    from npa.workflows import franka_rl_capture_merge

    recipe["visual_eval"] = {
        "arms": ["initial", "trained"],
        "conditions": list(recipe["conditions"]),
    }
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    (tmp_path / "initial.pt").write_bytes(b"initial checkpoint fixture")
    calls, merged = [], []

    def native(stage, prepared, output, *extra):
        output.mkdir(parents=True, exist_ok=True)
        calls.append((stage, extra))
        if fail_last and extra == ("--capture-arm", "trained", "--condition", "delay"):
            raise RuntimeError("final capture failed")

    monkeypatch.setattr(franka_rl, "_native_step", native)
    monkeypatch.setattr(
        franka_rl_capture_merge, "merge_captures", lambda *args: merged.append(True)
    )
    if fail_last:
        with pytest.raises(RuntimeError, match="final capture failed"):
            franka_rl._run_native("evaluate", tmp_path, tmp_path / "output")
    else:
        franka_rl._run_native("evaluate", tmp_path, tmp_path / "output")
    assert [stage for stage, _ in calls] == ["validate", "test"] + ["capture"] * 8
    assert {extra for stage, extra in calls if stage == "capture"} == {
        ("--capture-arm", arm, "--condition", condition)
        for arm in ("initial", "trained")
        for condition in recipe["conditions"]
    }
    assert merged == ([] if fail_last else [True])
