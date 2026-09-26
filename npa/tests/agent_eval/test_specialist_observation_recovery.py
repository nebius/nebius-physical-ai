"""Exercise opt-in recovery with real journals, process loss and concurrent ownership."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.call_policy import _classification
from npa.agent_backend.specialists.config import (
    Operation,
    Profile,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.recovery import _require_operations
from npa.agent_backend.specialists.storage_errors import StorageFailure
from npa.agent_backend.specialists.store import TaskStore, UncertainOperation
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


@pytest.fixture
def team(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("unchanged")
    audit = tmp_path / "observations.log"
    command = (
        "from pathlib import Path; "
        f"p=Path({str(audit)!r}); p.write_text((p.read_text() if p.exists() else '')+'observed\\n'); "
        "print(Path('value.txt').read_text())"
    )
    profile = Profile(
        name="scene",
        description="Observe a workflow",
        model="synthetic",
        workspace=workspace,
        read_paths=["value.txt"],
        write_paths=["value.txt"],
        required_operations=["status"],
        operations={
            "status": Operation(
                argv=["{python}", "-c", command],
                description="Read status",
                observation_only=True,
            ),
            "submit": Operation(
                argv=["{python}", "-c", "raise SystemExit(91)"],
                description="Effectful command",
            ),
        },
    )
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="scene"
    )
    return SpecialistTeam(config)


def _call(name="status", identity="observation", tool="run_operation"):
    return {
        "id": identity,
        "function": {"name": tool, "arguments": json.dumps({"name": name})},
    }


def _interrupt(team, *, call=None, task_id="task", classified=True):
    call = call or _call()
    profile = team.config.profiles[0]
    team.submit(
        "Observe the existing workflow", specialist=profile.name, task_id=task_id
    )
    invocation = call["function"]
    metadata = _classification(profile, invocation) if classified else None
    team.store._begin_call(task_id, call["id"], invocation, classification=metadata)
    team.store._update(
        task_id, "needs_attention", error="Original interrupted observation"
    )
    return call


def _trigger(team, sql):
    with team.store._connection() as connection:
        connection.execute(sql)


def test_default_policy_hash_remains_compatible_and_flag_is_strict(team):
    profile = team.config.profiles[0]
    profile.operations["status"].observation_only = False
    legacy = profile.model_dump(
        mode="json",
        exclude={
            "fallback_models",
            "model_router",
            "require_model_route",
            "routing_model",
            "model_criteria",
            "compact_context",
        },
    )
    for operation in legacy["operations"].values():
        operation.pop("handoff_on_failure")
        operation.pop("observation_only")
        operation.pop("wait_for")
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    assert fingerprint(profile) == expected
    profile.operations["status"].observation_only = True
    assert fingerprint(profile) != expected
    for invalid in ("true", 1, None):
        with pytest.raises(ValueError):
            Operation(argv=["true"], description="Observe", observation_only=invalid)


def test_additive_migration_keeps_old_receipts_unclassified(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir(mode=0o700)
    path = directory / "tasks.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE calls(task_id TEXT,call_id TEXT,digest TEXT,status TEXT,result TEXT,PRIMARY KEY(task_id,call_id))"
        )
        connection.execute(
            "INSERT INTO calls VALUES('old','call','digest','started',NULL)"
        )
    path.chmod(0o600)
    with ThreadPoolExecutor(max_workers=4) as pool:
        stores = list(pool.map(lambda _: TaskStore(directory), range(4)))
    for store in stores:
        assert store._calls("old") == [
            {
                "call_id": "call",
                "digest": "digest",
                "status": "started",
                "classification": None,
            }
        ]


def test_dismissal_is_failed_idempotent_and_does_not_release_task(team, monkeypatch):
    call = _interrupt(team)
    before = team.status("task")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("dismissal executed a command")
    )
    result = team.dismiss_interrupted_observation("task", call["id"])
    assert result["ok"] is False and result["outcome"] == "failed"
    assert team.dismiss_interrupted_observation("task", call["id"]) == result
    after = team.status("task")
    assert after["status"] == "needs_attention" and after["error"] == before["error"]
    assert after["events"][:-1] == before["events"]
    assert after["events"][-1]["previous_status"] == "started"
    assert after["calls"][0]["classification"] == before["calls"][0]["classification"]
    assert (
        WorkbenchTools(team.config.profiles[0], team.store, "task").execute(call)
        == result
    )
    messages = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)},
    ]
    with pytest.raises(ValueError, match="status"):
        _require_operations(messages, ["status"])
    assert team.work_once("scene") is None


@pytest.mark.parametrize(
    "kind", ["legacy", "submit", "edit", "unknown", "policy", "digest", "declaration"]
)
def test_unsafe_or_changed_calls_cannot_be_dismissed(team, kind):
    call = _call(
        name="submit"
        if kind == "submit"
        else "unknown"
        if kind == "unknown"
        else "status",
        tool="edit_file" if kind == "edit" else "run_operation",
    )
    _interrupt(team, call=call, classified=kind != "legacy")
    if kind == "policy":
        team.config.profiles[0].instructions = "Changed grant"
    if kind == "declaration":
        team.config.profiles[0].operations["status"].observation_only = False
    if kind == "digest":
        _trigger(team, "UPDATE calls SET digest='tampered'")
    before = team.status("task")
    with pytest.raises(ValueError):
        team.dismiss_interrupted_observation("task", call["id"])
    assert team.status("task") == before


@pytest.mark.parametrize("status", ["queued", "running", "completed", "cancelled"])
def test_only_attention_boundary_can_be_dismissed(team, status):
    _interrupt(team)
    team.store._update("task", status)
    with pytest.raises(ValueError, match="awaiting attention"):
        team.dismiss_interrupted_observation("task", "observation")
    assert team.store._calls("task")[0]["status"] == "started"


def test_mixed_unsafe_uncertainty_and_other_queued_work_stay_blocked(team):
    _interrupt(team)
    _interrupt(team, task_id="other", call=_call("submit", "write"))
    with pytest.raises(ValueError, match="not an observation"):
        team.dismiss_interrupted_observation("task", "observation")
    team.store._finish_call("other", "write", {"ok": False})
    team.store._update("other", "queued")
    with pytest.raises(ValueError, match="active or queued"):
        team.dismiss_interrupted_observation("task", "observation")
    assert team.store._calls("task")[0]["status"] == "started"


def test_classification_cannot_be_changed_or_retrofitted(team):
    _interrupt(team)
    profile = team.config.profiles[0]
    changed = _classification(profile, _call()["function"])
    changed["observation_only"] = False
    with pytest.raises(ValueError, match="classification changed"):
        team.store._begin_call(
            "task", "observation", _call()["function"], classification=changed
        )
    with pytest.raises(StorageFailure):
        _trigger(team, "UPDATE call_policies SET classification='{}'")
    _interrupt(team, task_id="legacy", classified=False)
    with pytest.raises(UncertainOperation):
        WorkbenchTools(profile, team.store, "legacy").execute(_call())
    assert team.store._calls("legacy")[0]["classification"] is None


def test_dismissal_event_failure_rolls_back_receipt(team):
    _interrupt(team)
    before = team.status("task")
    _trigger(
        team,
        "CREATE TRIGGER fail_event BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'secret storage diagnostic'); END",
    )
    with pytest.raises(StorageFailure) as failure:
        team.dismiss_interrupted_observation("task", "observation")
    assert failure.value.diagnostic["phase"] == "observation_dismissal"
    assert "secret" not in str(failure.value)
    assert team.status("task") == before


def test_commit_failure_rolls_back_both_resolution_and_history(team, monkeypatch):
    _interrupt(team)
    before = team.status("task")
    connect = sqlite3.connect

    class FailedCommit(sqlite3.Connection):
        def __exit__(self, kind, error, traceback):
            if kind is None and self.in_transaction:
                self.rollback()
                failure = sqlite3.OperationalError("private failed fsync")
                failure.sqlite_errorcode = sqlite3.SQLITE_IOERR_FSYNC
                failure.sqlite_errorname = "SQLITE_IOERR_FSYNC"
                raise failure
            return super().__exit__(kind, error, traceback)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            sqlite3, "connect", lambda *a, **kw: connect(*a, factory=FailedCommit, **kw)
        )
        with pytest.raises(StorageFailure) as failure:
            team.dismiss_interrupted_observation("task", "observation")
    assert failure.value.diagnostic["sqlite_errorname"] == "SQLITE_IOERR_FSYNC"
    assert team.status("task") == before


def test_failed_start_record_never_runs_command_or_persists_classification(team):
    team.submit("Observe", task_id="task", specialist="scene")
    _trigger(
        team,
        "CREATE TRIGGER fail_start BEFORE INSERT ON events BEGIN SELECT RAISE(ABORT, 'private'); END",
    )
    with pytest.raises(StorageFailure) as failure:
        WorkbenchTools(team.config.profiles[0], team.store, "task").execute(_call())
    assert failure.value.diagnostic["phase"] == "tool_start"
    assert not team.store._calls("task")
    with team.store._connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM call_policies").fetchone()[0] == 0
        )
    assert not team.config.state_directory.parent.joinpath("observations.log").exists()


def test_successful_observation_cannot_be_replaced_with_dismissal(team):
    team.submit("Observe", task_id="task", specialist="scene")
    WorkbenchTools(team.config.profiles[0], team.store, "task").execute(_call())
    team.store._update("task", "needs_attention", error="Later model failure")
    before = team.status("task")
    with pytest.raises(ValueError, match="different result"):
        team.dismiss_interrupted_observation("task", "observation")
    assert team.status("task") == before


def test_model_arguments_cannot_declare_an_operation_safe(team):
    profile = team.config.profiles[0]
    profile.operations["status"].observation_only = False
    team.submit("Observe", task_id="task", specialist="scene")
    call = _call()
    call["function"]["arguments"] = json.dumps(
        {"name": "status", "observation_only": True}
    )
    result = WorkbenchTools(profile, team.store, "task").execute(call)
    assert result["ok"] is False
    assert team.store._calls("task")[0]["classification"]["observation_only"] is False
    assert not team.config.state_directory.parent.joinpath("observations.log").exists()


def test_concurrent_dismissal_has_one_event_and_active_owner_blocks(team):
    _interrupt(team)
    with team._ownership("scene"), pytest.raises(BlockingIOError):
        team.dismiss_interrupted_observation("task", "observation")
    barrier = threading.Barrier(2)

    def dismiss():
        barrier.wait()
        try:
            return team.dismiss_interrupted_observation("task", "observation")
        except BlockingIOError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: dismiss(), range(2)))
    assert any(result and result["outcome"] == "failed" for result in outcomes)
    assert (
        len(
            [
                event
                for event in team.status("task")["events"]
                if event["type"] == "interrupted_observation_dismissed"
            ]
        )
        == 1
    )
    with pytest.raises(UncertainOperation):
        team.store._finish_call("task", "observation", {"ok": True})


def test_process_loss_after_real_observation_never_replays(team, tmp_path):
    team.submit("Read status", task_id="task", specialist="scene")
    config = tmp_path / "team.json"
    config.write_text(team.config.model_dump_json())
    program = """
import os, sys
from npa.agent_backend.specialists.config import load_config
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools
team = SpecialistTeam(load_config(sys.argv[1]))
team.store._finish_call = lambda *args: os._exit(79)
WorkbenchTools(team.config.profiles[0], team.store, 'task').execute({
    'id':'observation', 'function':{'name':'run_operation','arguments':'{"name": "status"}'}})
"""
    process = subprocess.run([sys.executable, "-c", program, str(config)], check=False)
    assert process.returncode == 79
    audit = tmp_path / "observations.log"
    assert audit.read_text() == "observed\n"
    executor = WorkbenchTools(team.config.profiles[0], team.store, "task")
    with pytest.raises(UncertainOperation):
        executor.execute(_call())
    team.store._update("task", "needs_attention", error="Process lost")
    result = team.dismiss_interrupted_observation("task", "observation")
    assert executor.execute(_call()) == result
    assert audit.read_text() == "observed\n"
    assert (tmp_path / "workspace/value.txt").read_text() == "unchanged"


def test_real_receipt_failure_keeps_classification_and_safe_diagnostic(team):
    team.submit("Read status", task_id="task", specialist="scene")
    _trigger(
        team,
        "CREATE TRIGGER fail_receipt BEFORE UPDATE ON calls BEGIN SELECT RAISE(ABORT, 'secret storage diagnostic'); END",
    )
    executor = WorkbenchTools(team.config.profiles[0], team.store, "task")
    with pytest.raises(StorageFailure) as failure:
        executor.execute(_call())
    assert failure.value.diagnostic["phase"] == "tool_receipt"
    assert failure.value.diagnostic["sqlite_errorname"] == "SQLITE_CONSTRAINT_TRIGGER"
    assert "secret" not in str(failure.value)
    assert team.store._calls("task")[0]["status"] == "started"
    assert team.store._calls("task")[0]["classification"]["observation_only"] is True
    assert (
        team.config.state_directory.parent.joinpath("observations.log").read_text()
        == "observed\n"
    )


@pytest.mark.parametrize(
    "phase", ["tool_start", "worker_heartbeat", "graph_checkpoint"]
)
def test_sqlite_failure_is_classified_without_raw_text(team, monkeypatch, phase):
    def fail(*args, **kwargs):
        error = sqlite3.OperationalError("credential=private-secret path=/private/file")
        error.sqlite_errorcode, error.sqlite_errorname = (
            sqlite3.SQLITE_IOERR,
            "SQLITE_IOERR",
        )
        raise error

    monkeypatch.setattr(sqlite3, "connect", fail)
    with pytest.raises(StorageFailure) as failure:
        if phase == "tool_start":
            team.store._begin_call("task", "call", _call()["function"])
        elif phase == "worker_heartbeat":
            team.store._heartbeat("scene")
        else:
            with team._graph(team.config.profiles[0], "task"):
                pytest.fail("graph opened unavailable storage")
    assert failure.value.diagnostic == {
        "phase": phase,
        "sqlite_errorcode": sqlite3.SQLITE_IOERR,
        "sqlite_errorname": "SQLITE_IOERR",
    }
    assert "private" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_worker_retains_storage_error_phase_and_does_not_continue(team, monkeypatch):
    team.submit("Observe", task_id="task", specialist="scene")

    @contextmanager
    def graph(*args):
        def invoke(*args, **kwargs):
            raise StorageFailure("tool_receipt", sqlite3.OperationalError("secret"))

        yield SimpleNamespace(
            get_state=lambda _: SimpleNamespace(values={}, next=["tools"]),
            invoke=invoke,
        )

    monkeypatch.setattr(team, "_graph", graph)
    assert team.work_once("scene")["status"] == "needs_attention"
    event = team.status("task")["events"][-1]
    assert event["storage_failure"]["phase"] == "tool_receipt"
    assert "secret" not in json.dumps(team.status("task"))
    assert team.work_once("scene") is None
