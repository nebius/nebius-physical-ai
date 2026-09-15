"""Native policy handoffs must reject incomplete and unmeasured success claims."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from npa.workbench.cosmos.policy_artifacts import materialize_bundle, policy_workspace, publish_bundle
from npa.workbench.cosmos.policy_contract import ACTION_CONTRACT, EvalSettings, TrainSettings, validate_summary
from npa.workbench.cosmos.policy_eval import EVAL_SCHEMA, evaluation_argv, server_argv
from npa.workbench.cosmos.policy_feedback import FEEDBACK_SCHEMA, generate_failure_candidates, policy_feedback
from npa.workbench.cosmos.policy_train import _collect_checkpoint, train_policy, training_argv


def _summary(settings: EvalSettings, failures: int = 0) -> dict:
    tasks = []
    for task_id in settings.task_ids:
        count = failures if task_id == settings.task_ids[0] else 0
        episodes = [{"episode": i, "success": i >= count, "steps": 80}
                    for i in range(settings.trials_per_task)]
        successes = sum(episode["success"] for episode in episodes)
        tasks.append({"task_id": task_id, "task_description": f"place the object in container {task_id}",
                      "episodes": len(episodes), "successes": successes,
                      "success_rate": successes / len(episodes), "episode_results": episodes})
    total = len(tasks) * settings.trials_per_task
    successes = sum(task["successes"] for task in tasks)
    return {**ACTION_CONTRACT, "task_results": tasks, "total_episodes": total,
            "total_successes": successes, "overall_success_rate": successes / total,
            "num_trials_per_task": settings.trials_per_task, "selected_task_ids": settings.task_ids}


def _evaluation(path: Path, settings: EvalSettings, failures: int = 0) -> Path:
    report = {"schema": EVAL_SCHEMA, "status": "succeeded", "settings": settings.model_dump(),
              "summary": _summary(settings, failures), "training_manifest": str(path / "training.json")}
    target = path / "evaluation.json"
    target.write_text(json.dumps(report))
    return target


@pytest.mark.parametrize("corruption", ["missing_task", "duplicate_episode", "success_count", "nan_rate", "selected_tasks", "missing_episode", "wrong_action", "boolean_success"])
def test_incomplete_or_inconsistent_native_evidence_is_rejected(corruption):
    settings = EvalSettings(trials_per_task=2, task_ids=[0, 1])
    summary = _summary(settings)
    if corruption == "missing_task":
        summary["task_results"].pop()
    elif corruption == "duplicate_episode":
        summary["task_results"][0]["episode_results"][1]["episode"] = 0
    elif corruption == "success_count":
        summary["total_successes"] = 0
    elif corruption == "nan_rate":
        summary["overall_success_rate"] = float("nan")
    elif corruption == "selected_tasks":
        summary["selected_task_ids"] = [0]
    elif corruption == "missing_episode":
        summary["task_results"][0]["episode_results"].pop()
    elif corruption == "wrong_action":
        summary["action_dim"] = 7
    else:
        summary["task_results"][0]["episode_results"][0]["success"] = "true"
    with pytest.raises(ValueError):
        validate_summary(summary, settings)


@pytest.mark.parametrize("settings", [EvalSettings(task_ids=[0]), EvalSettings(trials_per_task=1)])
def test_perfect_functional_smoke_cannot_qualify_checkpoint(tmp_path, settings):
    manifest = _evaluation(tmp_path, settings)
    result = policy_feedback(input_path=str(manifest), output_path=str(tmp_path / "feedback"))
    assert result["qualified"] is False
    assert result["selected_checkpoint"] is None
    assert result["improvement_proven"] is False


def test_full_benchmark_qualifies_without_claiming_relative_improvement(tmp_path):
    manifest = _evaluation(tmp_path, EvalSettings(), failures=2)
    result = policy_feedback(input_path=str(manifest), output_path=str(tmp_path / "feedback"))
    assert result["qualified"] is True
    assert result["success_rate"] == 498 / 500
    assert result["improvement_proven"] is False
    assert len(result["targets"]) == 1
    assert result["targets"][0]["failed_episodes"] == [0, 1]
    assert result["generated_video_training_eligible"] is False


def test_no_failed_task_does_not_run_video_model(tmp_path, monkeypatch):
    feedback = tmp_path / "feedback.json"
    feedback.write_text(json.dumps({"schema": FEEDBACK_SCHEMA, "status": "succeeded", "targets": []}))
    def unexpected(**kwargs):
        pytest.fail("no failed task should invoke generation")
    monkeypatch.setattr("npa.workbench.cosmos.generate.generate_and_publish", unexpected)
    result = generate_failure_candidates(input_path=str(feedback), output_path=str(tmp_path / "candidates"))
    assert result["generation_needed"] is False
    assert result["candidates"] == []


def test_bundle_content_corruption_stops_checkpoint_handoff(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"real checkpoint fixture")
    output = tmp_path / "published"
    publish_bundle(root, str(output), {"schema": "test", "status": "succeeded"}, "training.json")
    materialize_bundle(str(output / "training.json"), tmp_path / "valid", "test")
    (output / "weights.bin").write_bytes(b"changed checkpoint data")
    with pytest.raises(ValueError, match="verification"):
        materialize_bundle(str(output / "training.json"), tmp_path / "invalid", "test")


def test_bundle_cannot_reference_an_unrelated_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"weights")
    output = tmp_path / "published"
    report = publish_bundle(root, str(output), {"schema": "test", "status": "succeeded"}, "training.json")
    report["artifacts"][0]["uri"] = str(tmp_path / "unrelated-private-file")
    (output / "training.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="own prefix"):
        materialize_bundle(str(output / "training.json"), tmp_path / "invalid", "test")


def test_missing_optimizer_state_cannot_be_published_as_training(tmp_path):
    checkpoint = tmp_path / "output/job/checkpoints/iter_000000005/model"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_bytes(b"metadata")
    (checkpoint / "__0_0.distcp").write_bytes(b"weights")
    with pytest.raises(ValueError, match="optim"):
        _collect_checkpoint(tmp_path / "output", tmp_path / "artifacts", TrainSettings(iterations=5))


def test_native_failure_retains_diagnostics_without_success_marker(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="native failed"):
        with policy_workspace(str(output), "train") as root:
            (root / "bootstrap.log").write_text("upstream dependency failure")
            raise RuntimeError("native failed")
    failure = json.loads((output / "failure/failure.json").read_text())
    assert failure["status"] == "failed"
    assert not (output / "training.json").exists()
    assert (output / "failure/bootstrap.log").read_text() == "upstream dependency failure"


def test_settings_error_occurs_before_runtime_fetch(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text('{"unknown_override": true}')
    def unexpected(*args):
        pytest.fail("invalid settings must not start runtime setup")
    monkeypatch.setattr("npa.workbench.cosmos.policy_train.prepare_training_runtime", unexpected)
    with pytest.raises(ValidationError):
        train_policy(input_path=str(settings), output_path=str(tmp_path / "output"))


def test_native_training_preserves_global_batch_and_action_recipe():
    settings = TrainSettings()
    argv = training_argv(Path("/runtime"), settings)
    assert "model.config.parallelism.data_parallel_replicate_degree=1" in argv
    assert "trainer.grad_accum_iter=2" in argv
    assert settings.processes * settings.samples_per_rank * settings.gradient_accumulation == 2048
    assert not any("encode_exact_durations" in value for value in argv)
    assert argv[argv.index("--sft-toml") + 1].endswith("action_policy_libero_10_nano.toml")


def test_train_eval_action_contract_and_loopback_binding_match():
    repo, bundle = Path("/runtime"), Path("/artifacts")
    server = server_argv(repo, bundle, bundle / "checkpoint", 8123, 7)
    evaluation = evaluation_argv(Path("/sim/python"), repo, Path("/out"), 8123, EvalSettings(seed=7))
    assert server[server.index("--host") + 1] == "127.0.0.1"
    assert server[server.index("--raw-action-dim") + 1] == evaluation[evaluation.index("--action_dim") + 1]
    assert server[server.index("--action-normalization") + 1] == "quantile_rot"
    assert evaluation[evaluation.index("--camera") + 1] == "agentview,wrist"
    assert "--max_steps" not in evaluation


@pytest.mark.parametrize("tasks", [[], [1, 1], [-1], [10]])
def test_invalid_task_selection_is_rejected_before_gpu_work(tasks):
    with pytest.raises(ValidationError):
        EvalSettings(task_ids=tasks)


def test_native_summary_validation_does_not_mutate_evidence():
    settings = EvalSettings(trials_per_task=1)
    summary = _summary(settings)
    before = copy.deepcopy(summary)
    validate_summary(summary, settings)
    assert summary == before
