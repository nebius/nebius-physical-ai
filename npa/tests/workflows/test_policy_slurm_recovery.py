"""Prove submission reconciliation, non-progress rejection and private failure retention."""

import importlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from npa.workflows.policy_training import slurm, slurm_transport, diagnostics
from npa.workflows.policy_training.contracts import digest

batch = importlib.import_module("npa.workflows.policy_training.batch")


def _request(tmp_path):
    uri = str(tmp_path / "attempt/request.json")
    write_json_uri(uri, {"stage": "pretrain", "run_id": "test"})
    return uri


def _settings():
    return {"transport": "local", "scripts": {"pretrain": "/shared/train.sh"}}


def test_disconnect_after_submission_adopts_job_without_duplicate_or_cancel(
    tmp_path, monkeypatch
):
    uri = _request(tmp_path)
    commands = []
    name = None

    def execute(settings, command, evidence):
        nonlocal name
        commands.append(command)
        if command[0] == "sbatch":
            name = command[2].split("=", 1)[1]
            raise ConnectionError("lost submission response")
        return f"123|{name}"

    monkeypatch.setattr(slurm, "_execute", execute)
    with pytest.raises(ConnectionError):
        slurm._submission(_settings(), "pretrain", uri)
    receipt, _ = slurm._submission(_settings(), "pretrain", uri)
    assert receipt["job_id"] == "123"
    assert sum(command[0] == "sbatch" for command in commands) == 1
    assert all(
        command[0] != "scancel" and "--wait" not in command for command in commands
    )


@pytest.mark.parametrize("query", ["", "123|{name}\n124|{name}"])
def test_ambiguous_submission_never_resubmits(tmp_path, monkeypatch, query):
    uri = _request(tmp_path)
    commands = []

    def execute(settings, command, evidence):
        commands.append(command)
        if command[0] == "sbatch":
            raise ConnectionError("unconfirmed submit")
        name = read_json_uri(evidence)["job_name"]
        return query.format(name=name)

    monkeypatch.setattr(slurm, "_execute", execute)
    with pytest.raises(ConnectionError):
        slurm._submission(_settings(), "pretrain", uri)
    with pytest.raises(RuntimeError, match="unresolved"):
        slurm._submission(_settings(), "pretrain", uri)
    assert sum(command[0] == "sbatch" for command in commands) == 1


def test_poll_disconnect_keeps_job_and_resume_adopts_same_id(tmp_path, monkeypatch):
    uri = _request(tmp_path)
    commands = []
    monkeypatch.setattr(
        slurm,
        "_execute",
        lambda settings, command, evidence: commands.append(command) or "123",
    )
    monkeypatch.setattr(
        slurm,
        "_state",
        lambda *args: (_ for _ in ()).throw(ConnectionError("poll lost")),
    )
    with pytest.raises(ConnectionError):
        slurm._submit_slurm(_settings(), "pretrain", uri)
    monkeypatch.setattr(
        slurm, "_state", lambda settings, receipt, evidence: receipt["job_id"] == "123"
    )
    slurm._submit_slurm(_settings(), "pretrain", uri)
    assert [command[0] for command in commands] == ["sbatch"]


def test_cancel_failure_preserves_interrupt_and_private_evidence(
    tmp_path, monkeypatch, capsys
):
    uri = _request(tmp_path)
    calls = []

    def execute(settings, command, evidence):
        calls.append(command)
        if command[0] == "scancel":
            raise OSError("PRIVATE_SENTINEL")
        return "123"

    monkeypatch.setattr(slurm, "_execute", execute)
    monkeypatch.setattr(
        slurm,
        "_state",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt("cancelled")),
    )
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        slurm._submit_slurm(_settings(), "pretrain", uri)
    assert calls[-1] == ["scancel", "123"]
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().err
    assert any(
        "PRIVATE_SENTINEL" in path.read_text()
        for path in tmp_path.rglob("diagnostics/*.json")
    )


@pytest.mark.parametrize(
    "state,code,expected",
    [
        ("RUNNING", "0:0", False),
        ("COMPLETED", "0:0", True),
        ("FAILED", "1:0", None),
        ("COMPLETED", "0:9", None),
        ("NEW_STATE", "0:0", None),
    ],
)
def test_completion_requires_exact_terminal_accounting(
    tmp_path, monkeypatch, state, code, expected
):
    receipt = {
        "job_id": "123",
        "job_name": "npa-policy-test",
        "accounting_start": "2026-01-01",
    }
    monkeypatch.setattr(
        slurm,
        "_execute",
        lambda settings, command, uri: (
            "" if command[0] == "squeue" else f"123|npa-policy-test|{state}|{code}"
        ),
    )
    if expected is None:
        with pytest.raises(RuntimeError):
            slurm._state(_settings(), receipt, str(tmp_path / "submission.json"))
    else:
        assert (
            slurm._state(_settings(), receipt, str(tmp_path / "submission.json"))
            is expected
        )


def test_changed_submission_identity_is_rejected(tmp_path, monkeypatch):
    uri = _request(tmp_path)
    monkeypatch.setattr(slurm, "_execute", lambda *args: "123")
    slurm._submission(_settings(), "pretrain", uri)
    write_json_uri(uri, {"stage": "pretrain", "run_id": "different"})
    with pytest.raises(ValueError, match="identity changed"):
        slurm._submission(_settings(), "pretrain", uri)


@pytest.mark.parametrize("stage", ["pretrain", "finetune"])
def test_renamed_checkpoint_with_unchanged_weights_is_rejected(stage):
    request = {
        "stage": stage,
        "checkpoint": {"uri": "/prior/model", "sha256": "a" * 64},
    }
    result = {
        "schema": "npa.policy.batch-result.v1",
        "status": "completed",
        "engine": "operator-trainer",
        "request_sha256": digest(request),
        "checkpoint": {"uri": "/new/model", "sha256": "a" * 64},
    }
    with pytest.raises(ValueError, match="unchanged checkpoint"):
        batch._validate_result(request, result)
    result["checkpoint"]["sha256"] = "b" * 64
    batch._validate_result(request, result)


@pytest.mark.parametrize("engine", [None, "", "numpy-planar-reference"])
def test_production_result_requires_non_reference_engine(engine):
    request = {"stage": "pretrain"}
    result = {
        "schema": "npa.policy.batch-result.v1",
        "status": "completed",
        "engine": engine,
        "request_sha256": digest(request),
        "checkpoint": {"uri": "/model", "sha256": "a" * 64},
    }
    with pytest.raises(ValueError, match="engine"):
        batch._validate_result(request, result)


def test_soperator_transport_preserves_argv_and_output(monkeypatch):
    commands = []
    monkeypatch.setattr(
        slurm_transport,
        "_pod_execute",
        lambda settings, command, evidence: (
            commands.append(command)
            or subprocess.CompletedProcess(command, 0, "123\n", "")
        ),
    )
    assert (
        slurm_transport._execute(
            {"transport": "soperator"}, ["sbatch", "/job.sh"], "/unused/result.json"
        )
        == "123"
    )
    assert commands == [["chroot", "/mnt/jail", "sbatch", "/job.sh"]]


def test_worker_stderr_is_saved_privately(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        slurm_transport.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="", stderr="PRIVATE_SENTINEL"
        ),
    )
    with pytest.raises(RuntimeError, match="exit code 9") as error:
        slurm_transport._execute(_settings(), ["sbatch"], str(tmp_path / "result.json"))
    assert "PRIVATE_SENTINEL" not in str(error.value) + capsys.readouterr().err
    payload = json.loads(next(tmp_path.rglob("diagnostics/*.json")).read_text())
    assert payload["stderr"] == "PRIVATE_SENTINEL"


def test_diagnostic_write_failure_cannot_mask_primary_error(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnostics,
        "write_json_uri",
        lambda *args: (_ for _ in ()).throw(OSError("PRIVATE_STORAGE")),
    )
    diagnostics._failure("/private/result.json", ValueError("primary"))
    assert "PRIVATE_STORAGE" not in capsys.readouterr().err
