"""Exercise policy split isolation, measured gates, and real batch-command boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from npa.workflows.policy_training.contracts import digest, probability
from npa.workflows.policy_training.data import split
from npa.workflows.policy_training.gate import gate

batch_module = importlib.import_module("npa.workflows.policy_training.batch")


def _write(path, value):
    write_json_uri(str(path), value)
    return str(path)


def _episodes(count=100):
    return {
        "schema": "npa.policy.episodes.v1",
        "episodes": [
            {
                "dataset_uri": "s3://example-bucket/dataset",
                "episode_index": index,
                "group_id": f"capture-{index // 2}",
                "source": "real" if index % 2 else "synthetic",
            }
            for index in range(count)
        ],
    }


def _split(tmp_path):
    source = _write(tmp_path / "episodes.json", _episodes())
    index = str(tmp_path / "splits/index.json")
    split(source, index, "test-seed")
    return index


def test_split_keeps_real_and_synthetic_siblings_together(tmp_path):
    index = _split(tmp_path)
    partitions = read_json_uri(index)["partitions"]
    groups = {}
    for name, record in partitions.items():
        episodes = read_json_uri(record["uri"])["episodes"]
        groups[name] = {e["group_id"] for e in episodes}
        assert digest(read_json_uri(record["uri"])) == record["sha256"]
    assert [len(groups[k]) for k in ("train", "holdout_1", "holdout_2")] == [46, 2, 2]
    assert not groups["train"] & groups["holdout_1"]
    assert not groups["train"] & groups["holdout_2"]
    assert not groups["holdout_1"] & groups["holdout_2"]
    before = Path(index).read_bytes()
    _split(tmp_path)
    assert Path(index).read_bytes() == before


@pytest.mark.parametrize("case", ["few", "duplicate", "missing-group", "bad-index"])
def test_invalid_episode_inputs_fail(tmp_path, case):
    payload = _episodes(10 if case == "few" else 100)
    if case == "duplicate":
        payload["episodes"].append(payload["episodes"][0])
    if case == "missing-group":
        payload["episodes"][0]["group_id"] = ""
    if case == "bad-index":
        payload["episodes"][0]["episode_index"] = True
    uri = _write(tmp_path / "episodes.json", payload)
    with pytest.raises(ValueError):
        split(uri, str(tmp_path / "splits/index.json"), "seed")


def _gate_inputs(tmp_path, phase="pretrain"):
    split_uri = _split(tmp_path)
    candidate = {
        "checkpoint": {"uri": "s3://example-bucket/checkpoint", "sha256": "a" * 64}
    }
    candidate_uri = _write(tmp_path / "candidate.json", candidate)
    policy_uri = _write(
        tmp_path / "policy.json", {phase: {"system_1": 0.8, "system_2": 0.9}}
    )
    partition = "holdout_1" if phase == "pretrain" else "holdout_2"
    request = batch_module._request(
        f"evaluate-{phase}", partition, split_uri, candidate_uri, "test", "result.json"
    )
    evaluation = {
        "schema": "npa.policy.batch-result.v1",
        "status": "completed",
        "request": request,
        "request_sha256": digest(request),
        **candidate,
        "systems": {
            "system_1": {"successes": 8, "trials": 10},
            "system_2": {"successes": 9, "trials": 10},
        },
    }
    evaluation_uri = _write(tmp_path / "evaluation.json", evaluation)
    output_uri = str(tmp_path / "decision.json")
    return (evaluation_uri, candidate_uri, split_uri, policy_uri, phase, output_uri)


@pytest.mark.parametrize("phase", ["pretrain", "finetune"])
def test_gate_requires_each_system_to_pass(tmp_path, phase):
    arguments = _gate_inputs(tmp_path, phase)
    gate(*arguments)
    assert read_json_uri(arguments[-1])["decision"] == "promote_checkpoint"
    payload = read_json_uri(arguments[0])
    payload["systems"]["system_2"]["successes"] = 8
    write_json_uri(arguments[0], payload)
    gate(*arguments)
    assert read_json_uri(arguments[-1])["decision"] == "loop_back"


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "nan",
        "boolean",
        "missing-system",
        "zero-trials",
        "wrong-checkpoint",
        "wrong-holdout",
        "stale-request",
    ],
)
def test_gate_rejects_invalid_evidence(tmp_path, case):
    arguments = _gate_inputs(tmp_path)
    payload = read_json_uri(arguments[0])
    if case in {"empty", "nan", "boolean"}:
        policy = (
            {}
            if case == "empty"
            else {"system_1": float("nan") if case == "nan" else True}
        )
        write_json_uri(arguments[3], {"pretrain": policy})
    if case == "missing-system":
        del payload["systems"]["system_2"]
    if case == "zero-trials":
        payload["systems"]["system_1"]["trials"] = 0
    if case == "wrong-checkpoint":
        payload["checkpoint"] = {"uri": "wrong", "sha256": "b" * 64}
    if case == "wrong-holdout":
        payload["request"]["partition"] = "holdout_2"
    if case == "stale-request":
        payload["request_sha256"] = "b" * 64
    write_json_uri(arguments[0], payload)
    with pytest.raises((ValueError, KeyError)):
        gate(*arguments)
    assert not Path(arguments[-1]).exists()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, 2, "0.9"])
def test_probability_rejects_invalid_thresholds(value):
    with pytest.raises(ValueError):
        probability(value)


def _completing_batch_client():
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        request = read_json_uri(command[-1])
        write_json_uri(
            request["result_uri"],
            {
                "schema": "npa.policy.batch-result.v1",
                "status": "completed",
                "request_sha256": digest(request),
                "checkpoint": {
                    "uri": "s3://example-bucket/checkpoint",
                    "sha256": "a" * 64,
                },
            },
        )
        return SimpleNamespace(returncode=0)

    return run, commands


def test_batch_waits_for_completion_and_binds_current_request(tmp_path, monkeypatch):
    settings = _write(
        tmp_path / "batch.json",
        {
            "transport": "local",
            "scripts": {"pretrain": "/shared/train.sh"},
        },
    )
    run, commands = _completing_batch_client()
    monkeypatch.setattr(batch_module.subprocess, "run", run)
    index = _split(tmp_path)
    first = str(tmp_path / "train/1/candidate.json")
    second = str(tmp_path / "train/2/candidate.json")
    for iteration, output in enumerate((first, second), start=1):
        batch_module.batch(
            settings, "pretrain", "train", index, "", output, "test", iteration
        )
    assert "--wait" in commands[0]
    assert "--parsable" in commands[0]
    assert "checkpoint" not in read_json_uri(first)["request"]
    assert (
        read_json_uri(second)["request"]["checkpoint"]
        == read_json_uri(first)["checkpoint"]
    )
    assert (
        read_json_uri(first)["request_sha256"]
        != read_json_uri(second)["request_sha256"]
    )


def test_failed_batch_cancels_without_exposing_worker_output(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=9 if command[0] == "sbatch" else 0,
            stderr="private-worker-details",
        )

    monkeypatch.setattr(batch_module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="exit code 9") as error:
        batch_module._submit(
            {"transport": "local", "scripts": {"pretrain": "/job.sh"}},
            "pretrain",
            "request.json",
        )
    assert "private-worker-details" not in str(error.value)
    assert commands[1][0] == "scancel"
    assert commands[0][3].split("=", 1)[1] == commands[1][1].split("=", 1)[1]


def test_soperator_uses_argv_transport_not_a_shell(monkeypatch):
    settings = {
        "transport": "soperator",
        "context": "example-context",
        "namespace": "slurm",
        "login_pod": "login-0",
        "login_container": "sshd",
    }
    calls = []
    monkeypatch.setattr(
        batch_module, "_pod_execute", lambda config, argv: calls.append(argv) or 0
    )
    assert batch_module._execute(settings, ["sbatch", "/job.sh"]) == 0
    assert calls == [["chroot", "/mnt/jail", "sbatch", "/job.sh"]]


@pytest.mark.parametrize("stage", ["pretrain", "finetune"])
def test_training_cannot_use_holdout(tmp_path, stage):
    with pytest.raises(ValueError, match="partition"):
        batch_module.batch(
            "unused", stage, "holdout_2", "unused", "", "unused", "test", 1
        )


def test_stale_batch_result_cannot_be_published(tmp_path, monkeypatch):
    settings = _write(tmp_path / "settings.json", {})
    output = str(tmp_path / "train/1/candidate.json")

    def submit(settings, stage, request_uri):
        request = read_json_uri(request_uri)
        write_json_uri(
            request["result_uri"],
            {
                "schema": "npa.policy.batch-result.v1",
                "request_sha256": "old",
            },
        )

    monkeypatch.setattr(batch_module, "_submit", submit)
    with pytest.raises(ValueError, match="current request"):
        batch_module.batch(
            settings, "pretrain", "train", _split(tmp_path), "", output, "test", 1
        )
    assert not Path(output).exists()
