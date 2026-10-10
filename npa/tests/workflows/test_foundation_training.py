"""Check foundation-training event isolation, corpus selection and checkpoint safety."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.workflows.policy_training.checkpoints import (
    enqueue_evaluation,
    publish_checkpoint,
)
from npa.workflows.policy_training.contracts import digest
from npa.workflows.policy_training.data import _write_curated
from npa.workflows.policy_training.distributed import slurm_script
from npa.workflows.policy_training.events import event_workflow
from npa.workflows.policy_training.selection import select_training_data

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "npa/tests/fixtures/policy-training-slurm.json"


@pytest.mark.parametrize(
    "kind,benchmark,first",
    [
        ("corpus-ready", False, "curate"),
        ("dataset-arrived", False, "curate"),
        ("dataset-arrived", True, "curate"),
        ("code-changed", False, "finetune"),
    ],
)
def test_event_paths_validate_and_do_not_retrain_unnecessarily(
    tmp_path, kind, benchmark, first
):
    source = yaml.safe_load(TEMPLATE.read_text())
    event = {
        "kind": kind,
        "benchmark": benchmark,
        "episodes_uri": "s3://example-bucket/episodes.json",
        "split_uri": "s3://example-bucket/splits.json",
        "approved_checkpoint_uri": "s3://example-bucket/approved.json",
    }
    result = event_workflow(source, event)
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(result))
    spec = load_spec(path)
    validate_spec(spec)
    steps = build_plan(spec, run_id="test", assume_decision="promote_checkpoint").steps
    assert steps[0].state == first
    if kind != "corpus-ready":
        assert all("pretrain" not in step.state for step in steps)
    if kind == "dataset-arrived" and not benchmark:
        assert len(steps) == 1
    if kind == "code-changed" or benchmark:
        fine = next(step for step in steps if step.state == "finetune")
        assert fine.inputs[0]["uri"] == event["approved_checkpoint_uri"]
    assert source == yaml.safe_load(TEMPLATE.read_text())


def test_code_event_requires_an_approved_checkpoint():
    with pytest.raises(ValueError, match="approved_checkpoint_uri"):
        event_workflow(
            yaml.safe_load(TEMPLATE.read_text()),
            {"kind": "code-changed", "split_uri": "split.json"},
        )


def _selection(tmp_path, phase="finetune"):
    source = {
        "schema": "npa.policy.episodes.v1",
        "episodes": [
            {
                "dataset_uri": "s3://example-bucket/corpus-a",
                "episode_index": 0,
                "task_id": "pick",
            },
            {
                "dataset_uri": "s3://example-bucket/corpus-b",
                "episode_index": 1,
                "task_id": "pick",
            },
            {
                "dataset_uri": "s3://example-bucket/corpus-a",
                "episode_index": 2,
                "task_id": "place",
            },
        ],
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(source))
    request = {
        "stage": phase,
        "partition": "train",
        "result_uri": str(tmp_path / "attempt/result.json"),
        "dataset": {"uri": str(path), "sha256": digest(source)},
    }
    policy = {
        "task_ids": ["pick"],
        "dataset_weights": {
            "s3://example-bucket/corpus-a": 3,
            "s3://example-bucket/corpus-b": 1,
        },
    }
    return request, {"training_selection": {phase: policy}}


def test_finetuning_selects_only_requested_training_tasks(tmp_path):
    request, settings = _selection(tmp_path)
    select_training_data(request, settings)
    selected = json.loads(Path(request["dataset"]["uri"]).read_text())
    assert [e["episode_index"] for e in selected["episodes"]] == [0, 1]
    assert [m["probability"] for m in selected["mixture"]] == [0.75, 0.25]
    assert request["dataset"]["sha256"] == digest(selected)


@pytest.mark.parametrize(
    "bad", ["holdout", "empty", "missing-weight", "nan", "boolean"]
)
def test_invalid_training_mixtures_fail_before_launch(tmp_path, bad):
    request, settings = _selection(tmp_path)
    policy = settings["training_selection"]["finetune"]
    if bad == "holdout":
        request["partition"] = "holdout_2"
    if bad == "empty":
        policy["task_ids"] = []
    if bad == "missing-weight":
        policy["dataset_weights"].pop("s3://example-bucket/corpus-a")
    if bad == "nan":
        policy["dataset_weights"]["s3://example-bucket/corpus-a"] = float("nan")
    if bad == "boolean":
        policy["dataset_weights"]["s3://example-bucket/corpus-a"] = True
    with pytest.raises(ValueError):
        select_training_data(request, settings)
    assert not (tmp_path / "attempt/result.json").exists()


def test_flagging_retains_failed_episodes_when_all_data_is_selected(tmp_path):
    episodes = [{"episode_index": i, "inhouse_keep": bool(i)} for i in range(2)]
    path = tmp_path / "curated.json"
    _write_curated(episodes, {digest(episodes[1])}, {"selection": "all"}, str(path))
    output = json.loads(path.read_text())
    assert output["episodes"] == episodes
    assert output["quality_pass_count"] == 1
    assert output["selected_count"] == 2
    assert [r["quality_pass"] for r in output["review"]] == [False, True]


def test_public_curation_does_not_invent_inhouse_review_labels(tmp_path):
    episodes = [{"episode_index": index} for index in range(2)]
    path = tmp_path / "curated.json"
    _write_curated(episodes, {digest(episodes[0])}, {"selection": "all"}, str(path))
    output = json.loads(path.read_text())
    assert output["inhouse_reviewed_count"] == 0
    assert output["inhouse_disagreement_count"] is None
    assert all(row["inhouse_keep"] is None for row in output["review"])


def test_curation_comparison_counts_only_supplied_reviews(tmp_path):
    episodes = [{"episode_index": 0}, {"episode_index": 1, "inhouse_keep": False}]
    path = tmp_path / "curated.json"
    selected = {digest(episode) for episode in episodes}
    _write_curated(episodes, selected, {"selection": "all"}, str(path))
    output = json.loads(path.read_text())
    assert output["inhouse_reviewed_count"] == 1
    assert output["inhouse_disagreement_count"] == 1


def _checkpoint(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    roles = {
        role: [role + ".bin"]
        for role in ("model", "optimizer", "scheduler", "rng", "sampler")
    }
    for role in roles:
        (directory / (role + ".bin")).write_bytes(role.encode())
    return directory, roles


def test_checkpoint_event_requires_all_recovery_state(tmp_path):
    directory, roles = _checkpoint(tmp_path)
    roles.pop("rng")
    with pytest.raises(ValueError, match="RNG"):
        publish_checkpoint(directory, 10, roles, tmp_path / "events")
    assert not (tmp_path / "events").exists()


def test_checkpoint_event_submits_once_and_rejects_changed_weights(
    tmp_path, monkeypatch
):
    directory, roles = _checkpoint(tmp_path)
    event = publish_checkpoint(directory, 10, roles, tmp_path / "events")
    script = tmp_path / "evaluate.sbatch"
    script.write_text("#!/bin/bash\ntrue\n")
    calls = []

    def submit(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="12345;cluster\n")

    monkeypatch.setattr(subprocess, "run", submit)
    first = enqueue_evaluation(event, script, tmp_path / "ledger")
    assert enqueue_evaluation(event, script, tmp_path / "ledger") == first
    assert len(calls) == 1 and calls[0][-2] == "--checkpoint-event"
    (directory / "model.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        enqueue_evaluation(event, script, tmp_path / "ledger")


def test_ambiguous_slurm_submit_cannot_duplicate_a_checkpoint_job(
    tmp_path, monkeypatch
):
    directory, roles = _checkpoint(tmp_path)
    event = publish_checkpoint(directory, 10, roles, tmp_path / "events")
    script = tmp_path / "evaluate.sbatch"
    script.write_text("#!/bin/bash\ntrue\n")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
    )
    with pytest.raises(RuntimeError, match="reconcile"):
        enqueue_evaluation(event, script, tmp_path / "ledger")
    with pytest.raises(RuntimeError, match="ambiguous"):
        enqueue_evaluation(event, script, tmp_path / "ledger")


def _recipe(tmp_path):
    return {
        "mode": "pretrain",
        "nodes": 2,
        "gpus_per_node": 8,
        "cpus_per_node": 32,
        "memory_gib": 256,
        "master_port": 29500,
        "image": "ghcr.io/example/trainer@sha256:" + "a" * 64,
        "shared_path": str(tmp_path),
        "working_directory": str(tmp_path / "code"),
        "output_path": str(tmp_path / "outputs"),
        "trainer_argv": ["train.py", "--config", "train.json"],
    }


def test_slurm_recipe_quotes_literal_arguments_and_uses_rank_per_node(tmp_path):
    recipe = _recipe(tmp_path)
    recipe["trainer_argv"] += ["--label", "$(touch leaked); `touch leaked`"]
    rendered = slurm_script(recipe)
    script = tmp_path / "train.sbatch"
    script.write_text(rendered)
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "'$(touch leaked); `touch leaked`'" in rendered
    assert '--node-rank="$SLURM_PROCID"' in rendered
    assert "#SBATCH --ntasks-per-node=1" in rendered
    assert "--kill-on-bad-exit=1" in rendered
    assert "--max-restarts" not in rendered and "#SBATCH --time" not in rendered


@pytest.mark.parametrize(
    "field,value",
    [
        ("nodes", True),
        ("master_port", 70000),
        ("image", "trainer:latest"),
        ("output_path", "/outside/shared-root"),
        ("mode", "train"),
    ],
)
def test_invalid_distributed_recipes_fail(tmp_path, field, value):
    recipe = _recipe(tmp_path)
    recipe[field] = value
    with pytest.raises(ValueError):
        slurm_script(recipe)


def test_rendered_job_preserves_literal_trainer_argv(tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    marker = tmp_path / "must-not-exist"
    payload = "$(touch " + str(marker) + ")"
    recipe["trainer_argv"] += ["--label", payload]
    Path(recipe["working_directory"]).mkdir()
    binary = tmp_path / "bin"
    binary.mkdir()
    commands = {
        "scontrol": '#!/bin/bash\nprintf "worker-a\\nworker-b\\n"\n',
        "srun": '#!/bin/bash\nwhile [[ "$1" == --* ]]; do shift; done\nexec "$@"\n',
        "torchrun": '#!/bin/bash\nprintf "%s\\0" "$@" > "$NPA_TEST_TRACE"\n',
    }
    for name, source in commands.items():
        executable = binary / name
        executable.write_text(source)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + ":/usr/bin:/bin")
    for key, value in {
        "SLURM_NNODES": "2",
        "SLURM_PROCID": "1",
        "SLURM_JOB_NODELIST": "worker-[a,b]",
        "NPA_TEST_TRACE": str(tmp_path / "argv"),
    }.items():
        monkeypatch.setenv(key, value)
    script = tmp_path / "train.sbatch"
    script.write_text(slurm_script(recipe))
    subprocess.run(["bash", str(script)], check=True, capture_output=True)
    args = (tmp_path / "argv").read_bytes().decode().split("\0")[:-1]
    assert args[:5] == [
        "--nnodes=2",
        "--nproc-per-node=8",
        "--node-rank=1",
        "--master-addr=worker-a",
        "--master-port=29500",
    ]
    assert args[5:] == recipe["trainer_argv"]
    assert not marker.exists()
