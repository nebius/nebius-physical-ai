"""Protect public VLA split isolation, native execution contracts and portable launch packaging."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workflows.policy_training import public_vla as runner
from npa.workflows.policy_training import public_vla_data as data
from npa.workflows.policy_training import public_vla_eval as evaluation
from npa.workflows.policy_training import public_vla_launch as launch
from npa.workflows.policy_training import public_vla_train as training
from npa.workflows.policy_training import public_vla_report as report
from npa.workflows.policy_training import public_vla_export as export
from npa.workflows.policy_training import public_vla_collect as collect


def _corpus(root):
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    rows, episodes = [], []
    for index in range(20):
        episodes.append({"episode_index": index, "length": 3})
        for frame in range(3):
            rows.append(
                {
                    "episode_index": index,
                    "task_index": 0,
                    "action": [float(index), float(frame)],
                    "observation.state": [float(index)],
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "frames.parquet")
    pq.write_table(
        pa.Table.from_pylist(episodes), root / "meta" / "episodes" / "episodes.parquet"
    )
    tasks = pd.DataFrame({"task_index": [0]}, index=["put the cube in the bowl"])
    pq.write_table(pa.Table.from_pandas(tasks), root / "meta" / "tasks.parquet")


def test_public_v3_without_episode_tasks_uses_frame_task_indices(tmp_path):
    _corpus(tmp_path)
    result = data.prepare_corpus(tmp_path, tmp_path / "corpus.json", 42)
    assert {tuple(r["tasks"]) for r in result["episodes"]} == {
        ("put the cube in the bowl",)
    }
    assert sorted(r["partition"] for r in result["episodes"]).count("train") == 18
    assert len({r["trajectory_sha256"] for r in result["episodes"]}) == 20


def test_normalization_uses_only_selected_training_episodes(tmp_path):
    _corpus(tmp_path)
    manifest = data.prepare_corpus(tmp_path, tmp_path / "corpus.json", 42)
    selection = data.training_view(tmp_path, manifest, tmp_path / "view", None)
    stats = json.loads((tmp_path / "view/meta/stats.json").read_text())
    expected = np.mean(selection["episodes"])
    assert stats["action"]["mean"][0] == pytest.approx(expected)
    assert stats["action"]["count"] == [54]
    held_out = {
        r["episode_index"] for r in manifest["episodes"] if r["partition"] != "train"
    }
    assert not held_out.intersection(selection["episodes"])
    assert (tmp_path / "view/videos").resolve() == tmp_path / "videos"


def test_duplicate_trajectories_stay_together_and_flagged_rows_remain():
    rows = [
        {
            "tasks": ["task"],
            "trajectory_sha256": str(i),
            "quality_flags": [],
            "partition": "excluded",
        }
        for i in range(12)
    ]
    rows += [dict(rows[0]), dict(rows[1], quality_flags=["nonfinite"])]
    data._assign_partitions(rows, 9)
    assert rows[0]["partition"] == rows[-2]["partition"]
    assert rows[-1]["partition"] == "excluded"


@pytest.mark.parametrize(
    "override",
    [
        {"batch_size": True},
        {"seed": -1},
        {"task_id": 10},
        {"evaluation_episodes": 26},
        {"minimum_success": float("nan")},
        {"minimum_success": True},
        {"unknown": "value"},
    ],
)
def test_invalid_recipes_fail_before_submission(tmp_path, override):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        runner._recipe(path)


def test_output_directory_cannot_change_recipe(tmp_path):
    recipe = runner._recipe(None)
    runner._bind_recipe(tmp_path, recipe)
    with pytest.raises(ValueError, match="different recipe"):
        runner._bind_recipe(tmp_path, recipe | {"seed": 99})


def _gate_evidence(root, validation, test):
    recipe = runner._recipe(None)
    data.write_json(root / "recipe.json", recipe)
    policy = root / "candidate"
    policy.mkdir()
    (policy / "model.safetensors").write_bytes(b"unit-test-checkpoint")
    data.write_json(policy / "config.json", {"type": "unit-test"})
    data.write_json(
        root / "selection.json",
        {"selected": "generalist", "validation_pass": True, "minimum_success": 0.1},
    )
    for phase, score in [("generalist", validation), ("test", test)]:
        output = root / "evaluation" / phase
        data.write_json(
            output / "eval_info.json",
            {"overall": {"n_episodes": 10, "pc_success": score}},
        )
        data.write_json(
            output / "request.json",
            runner._evaluation_request(recipe, policy, 10 if phase == "test" else 0),
        )
        (output / "videos").mkdir()
        for episode in range(10):
            (output / "videos" / f"{episode}.mp4").touch()


@pytest.mark.parametrize("validation,test", [(60, 80), (80, 60), (60, 60)])
def test_failed_quality_gate_blocks_export_and_completion(
    tmp_path, monkeypatch, validation, test
):
    _gate_evidence(tmp_path, validation, test)
    monkeypatch.setattr(
        export, "export_policy", lambda *args: pytest.fail("rejected policy exported")
    )
    with pytest.raises(ValueError, match="export is blocked"):
        runner._finish(tmp_path, tmp_path / "candidate")
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["quality_gate_passed"] is False
    assert selection["minimum_success"] == 0.7
    assert not (tmp_path / "completed.json").exists()


@pytest.mark.parametrize("score", [-1, 101, float("nan"), float("inf"), True])
def test_invalid_scores_cannot_promote_a_checkpoint(tmp_path, score):
    _gate_evidence(tmp_path, 80, 80)
    path = tmp_path / "evaluation/test/eval_info.json"
    path.write_text(json.dumps({"overall": {"n_episodes": 10, "pc_success": score}}))
    with pytest.raises(ValueError, match="evaluation"):
        runner._require_quality_gate(tmp_path, tmp_path / "candidate")


def test_export_verification_failure_does_not_mark_run_complete(tmp_path, monkeypatch):
    _gate_evidence(tmp_path, 80, 80)
    monkeypatch.setattr(export, "export_policy", lambda *args: {})

    def failed_verification(*args):
        raise ValueError("offline verification failed")

    monkeypatch.setattr(runner, "_execute", failed_verification)
    with pytest.raises(ValueError, match="offline verification failed"):
        runner._finish(tmp_path, tmp_path / "candidate")
    assert not (tmp_path / "completed.json").exists()


def test_passing_gate_recomputes_approval_from_bound_recipe(tmp_path):
    _gate_evidence(tmp_path, 70, 70)
    selection = runner._require_quality_gate(tmp_path, tmp_path / "candidate")
    assert selection["minimum_success"] == 0.7
    assert selection["quality_gate_passed"] is True


@pytest.mark.parametrize("change", ["weights", "config", "holdout", "missing"])
def test_passing_scores_cannot_approve_different_policy_or_holdout(tmp_path, change):
    _gate_evidence(tmp_path, 80, 80)
    request_path = tmp_path / "evaluation/test/request.json"
    if change == "weights":
        (tmp_path / "candidate/model.safetensors").write_bytes(b"different-checkpoint")
    elif change == "config":
        data.write_json(tmp_path / "candidate/config.json", {"type": "different"})
    elif change == "holdout":
        request = json.loads(request_path.read_text())
        data.write_json(request_path, request | {"initial_state_offset": 0})
    else:
        request_path.unlink()
    with pytest.raises(ValueError, match="identity"):
        runner._require_quality_gate(tmp_path, tmp_path / "candidate")


def test_fixed_validation_and_test_states_are_disjoint():
    def original(instance, episode):
        instance._init_states = list(range(50))
        instance._reset_stride = 10
        instance.init_state_id = episode

    observed = []
    for offset in [0, 10]:
        states = []
        for episode in range(10):
            env = SimpleNamespace()
            evaluation._partitioned_reset(original, offset)(env, episode)
            states.append(env.init_state_id)
        observed.append(set(states))
    assert not observed[0].intersection(observed[1])
    with pytest.raises(ValueError, match="wrap"):
        evaluation._partitioned_reset(original, 45)(SimpleNamespace(), 0)


def test_observer_records_native_optimizer_metrics_without_changing_results(tmp_path):
    tracker = SimpleNamespace(steps=7, metrics={"loss": SimpleNamespace(val=0.123)})
    outputs = {"native": "result"}
    wrapped = training._observe_updates(lambda: (tracker, outputs), tmp_path)
    assert wrapped() == (tracker, outputs)
    row = json.loads((tmp_path / "metrics-rank-0.jsonl").read_text())
    assert row["step"] == 8 and row["loss"] == 0.123


def test_source_archive_contains_runner_dependencies_and_no_workspace_files():
    with tarfile.open(
        fileobj=io.BytesIO(launch.source_archive()), mode="r:gz"
    ) as archive:
        names = archive.getnames()
    assert "npa/__init__.py" in names
    assert "npa/workflows/policy_training/checkpoints.py" in names
    assert "npa/workflows/policy_training/public_vla_train.py" in names
    assert all(
        name.startswith("npa/") and name.endswith((".py", ".html", ".txt"))
        for name in names
    )


def test_kubernetes_job_runs_pinned_image_without_credentials_or_deadline():
    objects = launch.kubernetes_objects("npa-vla-0123456789ab", b"source", {})
    job = objects[-1]["spec"]
    pod = job["template"]["spec"]
    assert "activeDeadlineSeconds" not in job
    assert pod["automountServiceAccountToken"] is False
    assert "@sha256:" in pod["containers"][0]["image"]
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "secret" not in json.dumps(objects).lower()
    mounts = pod["containers"][0]["volumeMounts"]
    shared = next(mount for mount in mounts if mount["name"] == "shm")
    assert Path(shared["mountPath"]).parts == ("/", "dev", "shm")
    volume = next(volume for volume in pod["volumes"] if volume["name"] == "shm")
    assert volume["emptyDir"] == {"medium": "Memory", "sizeLimit": "16Gi"}
    assert all("hostPath" not in volume for volume in pod["volumes"])


def test_slurm_quotes_paths_and_allocates_a_real_gpu():
    shared = Path("/shared/a space/run")
    script = launch.slurm_script(shared, shared / "recipe.json")
    assert "#SBATCH --gpus=1" in script
    assert "--container-image=" in script and launch.MODULE in script
    assert "ghcr.io#nebius/nebius-physical-ai/npa-lerobot:sha256:" in script
    assert "--time=" not in script
    with pytest.raises(ValueError):
        launch.slurm_script(Path("/"), Path("/recipe.json"))


def test_report_runtime_drops_private_extra_fields(tmp_path):
    path = tmp_path / "evidence/generalist/runtime-rank-0.json"
    path.parent.mkdir(parents=True)
    actual = {
        "gpu": "test GPU",
        "torch": "test",
        "parameters": 3,
        "trainable_parameters": 2,
        "world_size": 1,
        "rank": 0,
    }
    path.write_text(json.dumps(actual | {"private_endpoint": "must-not-export"}))
    assert report._runtime(tmp_path) == actual


def test_portable_policy_removes_training_paths_and_localizes_tokenizer(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"unit-test-checkpoint")
    (source / "train_config.json").write_text('{"private_path":"not-exported"}')
    (source / "config.json").write_text(
        json.dumps(
            {
                "vlm_model_name": "/private/backbone",
                "pretrained_path": "/private/checkpoint",
            }
        )
    )
    (source / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"config": {"tokenizer_name": "/private/backbone"}}]})
    )
    destination = tmp_path / "export"
    export._copy_policy(source, destination)
    assert not (destination / "train_config.json").exists()
    config = json.loads((destination / "config.json").read_text())
    assert config["vlm_model_name"] == "./backbone"
    assert config["load_vlm_weights"] is False
    assert "pretrained_path" not in config
    assert "/private" not in (destination / "policy_preprocessor.json").read_text()


def test_report_refuses_incomplete_training_history(tmp_path):
    data.write_json(tmp_path / "plans/generalist.json", {"steps": 20})
    path = tmp_path / "evidence/generalist/metrics-rank-0.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"step":10,"loss":0.2}\n')
    with pytest.raises(ValueError, match="final step"):
        report._training(tmp_path, "generalist")


@pytest.mark.parametrize("status", [{}, {"active": 1}, {"succeeded": 1}])
def test_resume_never_replaces_pending_active_or_completed_jobs(
    tmp_path, monkeypatch, status
):
    receipt = {
        "backend": "kubernetes",
        "image": launch.IMAGE,
        "namespace": "npa-vla-0123456789ab",
        "kubectl": ["kubectl"],
    }
    data.write_json(tmp_path / "receipt.json", receipt)
    data.write_json(
        tmp_path / "resources.json",
        {
            "items": launch.kubernetes_objects(receipt["namespace"], b"source", {}),
        },
    )
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps({"status": status}),
    )
    writes = []
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: writes.append(a))
    with pytest.raises(ValueError, match="absent or failed"):
        launch._resume(SimpleNamespace(output_path=tmp_path, backend="kubernetes"))
    assert writes == []


def test_collection_rejects_receipts_that_inject_kubectl_commands():
    receipt = {
        "backend": "kubernetes",
        "image": launch.IMAGE,
        "namespace": "npa-vla-0123456789ab",
        "kubectl": ["kubectl", "delete", "nodes"],
    }
    with pytest.raises(ValueError, match="unsupported"):
        collect._kubectl(receipt)
