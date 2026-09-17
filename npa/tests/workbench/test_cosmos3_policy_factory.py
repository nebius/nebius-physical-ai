"""Native policy handoffs must reject incomplete and unmeasured success claims."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from npa.workbench.cosmos.policy_artifacts import materialize_bundle, policy_workspace, publish_bundle
from npa.workbench.cosmos.policy_contract import ACTION_CONTRACT, MODEL_REVISION, EvalSettings, TrainSettings, validate_summary
from npa.workbench.cosmos.policy_eval import EVAL_SCHEMA, _evaluate_native, evaluation_argv, server_argv
from npa.workbench.cosmos.policy_feedback import FEEDBACK_SCHEMA, generate_failure_candidates, policy_feedback
from npa.workbench.cosmos.policy_train import _collect_checkpoint, _fetch_inputs, _verify_processor_revision, train_policy, training_argv


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


def test_input_fetch_does_not_require_hub_in_lightweight_npa_environment(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr("npa.workbench.cosmos.generate.resolve_hf_token", lambda: "")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"fps": 20}')
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "data/chunk-000/file-000.parquet").touch()
    (tmp_path / "artifacts").mkdir()
    repo = tmp_path / "native"

    def native_fetch(argv, **kwargs):
        assert argv[0] == str(repo / ".venv/bin/python")
        assert Path(argv[1]).name == "policy_inputs.py"
        assert kwargs["cwd"] == repo
        Path(argv[argv.index("--output-path") + 1]).write_text(json.dumps(
            {"dataset": str(dataset), "model": str(tmp_path / "model"), "vae": str(tmp_path / "vae")}))

    monkeypatch.setattr("npa.workbench.cosmos.policy_train.run_native", native_fetch)
    actual = _fetch_inputs(tmp_path, repo, {})
    assert actual[0] == dataset
    assert json.loads((tmp_path / "artifacts/dataset.json").read_text())["info_sha256"]


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


@pytest.mark.parametrize("invalid_episode", [
    {"error": "Empty action chunk from server"},
    {"error": "Simulator initialization failed", "steps": 0},
    {"steps": 0}, {"steps": None}, {"steps": True}, {"steps": 1.5},
])
def test_runtime_errors_cannot_become_policy_failure_feedback(tmp_path, invalid_episode):
    manifest = _evaluation(tmp_path, EvalSettings(trials_per_task=1, task_ids=[0]), failures=1)
    report = json.loads(manifest.read_text())
    report["summary"]["task_results"][0]["episode_results"][0].update(invalid_episode)
    manifest.write_text(json.dumps(report))
    output = tmp_path / "feedback"
    with pytest.raises(ValueError, match="evaluation"):
        policy_feedback(input_path=str(manifest), output_path=str(output))
    assert not (output / "feedback.json").exists()


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


@pytest.mark.parametrize("revision", [None, "different-revision", MODEL_REVISION])
def test_native_processor_registry_cannot_drift_from_pinned_model(tmp_path, revision):
    snapshot = tmp_path / "model/snapshots" / MODEL_REVISION
    if revision is not None:
        ref = tmp_path / "model/refs/main"
        ref.parent.mkdir(parents=True)
        ref.write_text(revision)
    if revision == MODEL_REVISION:
        _verify_processor_revision(snapshot)
    else:
        with pytest.raises(ValueError, match="processor registry"):
            _verify_processor_revision(snapshot)


@pytest.mark.parametrize("has_config", [True, False])
def test_completed_checkpoint_handoff_preserves_bytes_without_disk_duplication(tmp_path, has_config):
    job = tmp_path / "output/job"
    checkpoint = job / "checkpoints/iter_000000005"
    for component in ("model", "optim", "scheduler", "trainer"):
        folder = checkpoint / component
        folder.mkdir(parents=True)
        (folder / ".metadata").write_bytes(b"metadata")
        (folder / "__0_0.distcp").write_bytes(component.encode())
    inode = (checkpoint / "model/__0_0.distcp").stat().st_ino
    if not has_config:
        with pytest.raises(ValueError, match="resolved configuration"):
            _collect_checkpoint(tmp_path / "output", tmp_path / "artifacts", TrainSettings(iterations=5))
        assert checkpoint.is_dir()
        return
    (job / "config.yaml").write_text("resolved: true\n")
    relative = _collect_checkpoint(tmp_path / "output", tmp_path / "artifacts", TrainSettings(iterations=5))
    staged = tmp_path / "artifacts" / relative
    assert (staged / "model/__0_0.distcp").stat().st_ino == inode
    assert (staged / "optim/__0_0.distcp").read_bytes() == b"optim"
    assert (tmp_path / "artifacts/job/config.yaml").read_text() == "resolved: true\n"
    assert not checkpoint.exists()


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
    assert server[server.index("--checkpoint-path") + 1] == str(bundle / "checkpoint/model")
    assert server[server.index("--raw-action-dim") + 1] == evaluation[evaluation.index("--action_dim") + 1]
    assert server[server.index("--action-normalization") + 1] == "quantile_rot"
    assert "dataloader_train.dataloader.datasets.libero.dataset.root=null" in server
    assert "dataloader_train=null" not in server
    assert evaluation[evaluation.index("--camera") + 1] == "agentview,wrist"
    assert "--max_steps" not in evaluation


@pytest.mark.parametrize("matching_checkpoint", [True, False])
def test_native_resolved_model_directory_controls_rollout_identity(tmp_path, monkeypatch, matching_checkpoint):
    import npa.workbench.cosmos.policy_eval as evaluation

    checkpoint = tmp_path / "job/checkpoints/iter_000000005"
    native_path = checkpoint / "model" if matching_checkpoint else tmp_path / "different/model"
    process = SimpleNamespace(terminate=lambda: None, wait=lambda **kwargs: 0)
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(evaluation, "_wait_ready", lambda *args: {"checkpoint": str(native_path)})
    calls = []
    monkeypatch.setattr(evaluation, "run_native", lambda *args, **kwargs: calls.append(args))
    def invoke():
        _evaluate_native(tmp_path, tmp_path, checkpoint, tmp_path / "python", {}, tmp_path,
                         EvalSettings(trials_per_task=1, task_ids=[0]))
    if matching_checkpoint:
        invoke()
        assert len(calls) == 1
        assert json.loads((tmp_path / "server-info.json").read_text())["checkpoint"] == str(native_path)
    else:
        with pytest.raises(ValueError, match="different checkpoint"):
            invoke()
        assert not calls


def test_evaluation_setup_failure_does_not_download_checkpoint(tmp_path, monkeypatch):
    import npa.workbench.cosmos.policy_eval as evaluation

    def fail_setup(root, *, guardrails):
        assert guardrails is True
        raise RuntimeError("native runtime dependency failed")

    def unexpected_download(*args, **kwargs):
        pytest.fail("checkpoint download preceded runtime preflight")

    monkeypatch.setattr(evaluation, "prepare_training_runtime", fail_setup)
    monkeypatch.setattr(evaluation, "materialize_bundle", unexpected_download)
    with pytest.raises(RuntimeError, match="dependency failed"):
        evaluation.evaluate_policy(input_path=str(tmp_path / "training.json"),
                                   output_path=str(tmp_path / "result"), trials_per_task=1, task_ids="0")
    assert (tmp_path / "result/failure/failure.json").is_file()


def test_native_config_failure_precedes_simulator_setup_and_checkpoint_download(tmp_path, monkeypatch):
    import npa.workbench.cosmos.policy_eval as evaluation

    monkeypatch.setattr(evaluation, "prepare_training_runtime", lambda root, **kwargs: (root, {}))
    def fail_config(argv, **kwargs):
        assert "dataloader_train.dataloader.datasets.libero.dataset.root=null" in argv
        assert evaluation.EXPERIMENT in argv
        raise RuntimeError("native config interpolation failed")
    def unexpected(*args, **kwargs):
        pytest.fail("native configuration must resolve before simulator setup or checkpoint download")
    monkeypatch.setattr(evaluation, "run_native", fail_config)
    monkeypatch.setattr(evaluation, "prepare_simulation_runtime", unexpected)
    monkeypatch.setattr(evaluation, "materialize_bundle", unexpected)
    with pytest.raises(RuntimeError, match="config interpolation"):
        evaluation.evaluate_policy(input_path=str(tmp_path / "training.json"),
                                   output_path=str(tmp_path / "result"), trials_per_task=1, task_ids="0")


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
