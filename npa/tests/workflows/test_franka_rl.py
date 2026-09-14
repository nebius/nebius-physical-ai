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
    return franka_rl._recipe(SimpleNamespace(run_id="test-franka", seed=42, iterations=1500, num_envs=4096,
                                            eval_episodes=2, minimum_success=0.7))


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


@pytest.mark.parametrize("distance,speed,height", [(0.05, 0.01, 0.3), (0.01, 0.03, 0.3), (0.01, 0.01, 0.1)])
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


@pytest.mark.parametrize("distance,speed,height,done", [([np.nan], [0], [1], [False]),
    ([-1], [0], [1], [False]), ([0], [-1], [1], [False]), ([0, 0], [0], [1], [False]),
    ([0], [0], [1], [False, True])])
def test_invalid_simulator_geometry_fails(recipe, distance, speed, height, done):
    with pytest.raises(ValueError):
        PlacementMetrics(1, recipe).update(distance, speed, height, done)


def _validation_row(**changes):
    return {"split": "validation", "checkpoint_sha256": "a" * 64, "iteration": 500,
            "env_index": 0, "closest_distance_m": 0.04, "success": True} | changes


def test_validation_ranking_prefers_success_before_distance_and_earlier_ties():
    success = [_validation_row()]
    miss = [_validation_row(success=False, closest_distance_m=0.001)]
    assert rank_checkpoint(success) > rank_checkpoint(miss)
    assert rank_checkpoint(success) > rank_checkpoint([_validation_row(iteration=1000)])


@pytest.mark.parametrize("rows", [[], [_validation_row(split="test")],
    [_validation_row(), _validation_row()],
    [_validation_row(), _validation_row(env_index=1, checkpoint_sha256="b" * 64)],
    [_validation_row(closest_distance_m=float("inf"))]])
def test_selection_rejects_test_data_mixed_weights_duplicates_and_nonfinite(rows):
    with pytest.raises(ValueError):
        rank_checkpoint(rows)


@pytest.fixture
def evaluation(recipe):
    rows = [{"split": "test", "condition": condition, "env_index": index,
             "reset_seed": recipe["test_seed"], "arm": arm, "success": arm == "trained",
             "checkpoint_sha256": "a" * 64 if arm == "trained" else "b" * 64,
             "closest_distance_m": 0.01, "longest_stable_steps": 20 if arm == "trained" else 0}
            for arm in ("initial", "trained") for condition in recipe["conditions"] for index in range(2)]
    for row in rows:
        row["initial_state_sha256"] = f"{row['env_index']:064x}"
    return {"recipe": recipe, "trials": rows, "validation": [_validation_row()],
            "selection": {"selected_checkpoint_sha256": "a" * 64}}


def test_complete_simulation_success_never_claims_physical_transfer(evaluation):
    report = summarize(evaluation)
    assert report["simulation_qualified"]
    assert report["paired_delta_95ci"] == [1.0, 1.0]
    assert not report["ready_for_robot_deployment"]
    assert not report["physical_robot_tested"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_checkpoint", "wrong_seed", "wrong_split"])
def test_report_rejects_incomplete_or_mixed_test_evidence(evaluation, mutation):
    changed = deepcopy(evaluation)
    if mutation == "missing":
        changed["trials"].pop()
    elif mutation == "duplicate":
        changed["trials"].append(changed["trials"][0])
    else:
        key, value = {"wrong_checkpoint": ("checkpoint_sha256", "c" * 64),
                      "wrong_seed": ("reset_seed", 42), "wrong_split": ("split", "validation")}[mutation]
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


def test_native_failure_preserves_log_and_never_reports_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(franka_rl.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(returncode=7))
    with pytest.raises(RuntimeError, match="exit 7"):
        franka_rl._run_native("train", tmp_path / "input", tmp_path / "output")
    assert (tmp_path / "output/runtime.log").exists()
    assert not (tmp_path / "output/training.json").exists()


@pytest.mark.parametrize("stage,receipt", [("train", "training.json"), ("evaluate", "evaluation.json")])
def test_isaac_zero_exit_without_completion_record_is_a_failure(tmp_path, monkeypatch, stage, receipt):
    monkeypatch.setattr(franka_rl.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match=f"required {receipt}"):
        franka_rl._run_native(stage, tmp_path / "input", tmp_path / "output")


def test_initial_native_checkpoint_is_decodable_without_a_training_logger(tmp_path):
    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_runtime import _save_initial

    runner = SimpleNamespace(current_learning_iteration=0,
                             alg=SimpleNamespace(save=lambda: {"actor_state_dict": {"weight": torch.ones(2)}}))
    path = tmp_path / "initial.pt"
    _save_initial(runner, path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["iter"] == 0 and payload["infos"] is None
    assert torch.equal(payload["actor_state_dict"]["weight"], torch.ones(2))


def test_isaac_physics_evidence_decodes_actual_warp_arrays():
    pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    from npa.workflows.franka_rl_environment import physics_evidence

    material = wp.array([[[0.5, 0.25, 0.0]], [[1.5, 1.25, 0.0]]], dtype=wp.float32, device="cpu")
    masses = wp.array([[0.7], [1.3]], dtype=wp.float32, device="cpu")
    view = SimpleNamespace(get_material_properties=lambda: material, get_masses=lambda: masses)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"object": SimpleNamespace(root_view=view)},
        event_manager=SimpleNamespace(active_terms={"startup": ["npa_object_mass", "npa_object_material"]})))
    evidence = physics_evidence(env)
    assert evidence["mass_kg"] == pytest.approx({"min": 0.7, "max": 1.3})
    assert evidence["static_friction"] == {"min": 0.5, "max": 1.5}
    assert evidence["dynamic_friction"] == {"min": 0.25, "max": 1.25}


def test_prepare_seals_disjoint_streams_without_isaac(tmp_path, recipe):
    output = tmp_path / "prepared"
    assert franka_rl.main(["prepare", "--run-id", "test-franka", "--output-path", str(output)]) == 0
    sealed = json.loads((output / "recipe.json").read_text())
    assert len({sealed[key] for key in ("seed", "validation_seed", "test_seed", "capture_seed")}) == 4
    assert sealed["task"] == "Isaac-Lift-Cube-Franka-v0"
    assert set(json.loads((output / "checksums.json").read_text())) == {"recipe.json"}


def test_workflow_uses_real_stages_and_render_capable_gpu():
    import yaml

    path = Path(__file__).resolve().parents[3] / "workflows/testing/franka-rl-transfer.yaml"
    workflow = yaml.safe_load(path.read_text())
    assert workflow["resources"]["isaac"]["accelerators"] == "RTXPRO6000:1"
    assert "@sha256:" in workflow["resources"]["isaac"]["image"]
    for stage in ("prepare", "train", "evaluate", "report"):
        assert workflow["states"][stage]["run"]["argv"][:4] == ["python3", "-m", "npa.workflows.franka_rl", stage]


def test_every_franka_worker_uses_staged_source_instead_of_stale_image_modules(monkeypatch):
    import yaml

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions, render_skypilot_yaml
    from npa.orchestration.npa_workflow.spec import load_spec

    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/staged-source/npa/")
    path = Path(__file__).resolve().parents[3] / "workflows/testing/franka-rl-transfer.yaml"
    spec = load_spec(path)
    rendered = render_skypilot_yaml(spec, build_plan(spec, run_id="test-franka"), run_id="test-franka",
                                   options=SkypilotRenderOptions(materialize_registry_secrets=False))
    tasks = [doc for doc in yaml.safe_load_all(rendered) if doc and "envs" in doc]
    assert len(tasks) == 4
    assert all(task["envs"]["NPA_SRC_OVERLAY"] == "1" for task in tasks)
    assert all(task["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/staged-source/npa/" for task in tasks)


def test_franka_recording_decodes_camera_and_all_named_joints_without_synthetic_pose(tmp_path):
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
    convert(raw.parent, dataset, fps=50, robot_type="franka", spec=LeRobotFeatureSpec(states, actions, "franka"))
    metadata = {"run_id": "fixture-franka", "genuine_simulator_pixels": True, "num_episodes": 1,
                "episode_results": [{"success": False}], "state_names": states, "action_names": actions}
    output = tmp_path / "franka.rrd"
    counts = write_recording(dataset, output, metadata)
    assert len(counts) == 18 and set(counts.values()) == {3}
    chunks = list(load_recording(output).chunks())
    entities = {str(chunk.entity_path) for chunk in chunks}
    assert "/episodes/000000/video" in entities
    assert not any("transform" in entity or "skeleton" in entity for entity in entities)
