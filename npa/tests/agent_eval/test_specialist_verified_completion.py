"""Prove bounded runtime completion requires authoritative current verification."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.config import Operation, Profile, TeamConfig, fingerprint
from npa.agent_backend.specialists.team import SpecialistTeam


class _Client:
    def __init__(self, calls):
        self.responses = iter(calls)
        self.requests = []

    def chat_completion(self, **request):
        self.requests.append(request)
        calls = next(self.responses)
        return {
            "model": "synthetic",
            "choices": [{
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"content": "Verified" if not calls else "", "tool_calls": calls},
            }],
        }


def _call(identity, tool="run_operation", **arguments):
    return {"id": identity, "type": "function", "function": {"name": tool, "arguments": json.dumps(arguments)}}


def _setup(tmp_path, *, finish=True, command="print('verified')", required=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("old")
    profile = Profile(
        name="repair", description="Repair", model="synthetic", workspace=workspace,
        read_paths=["value.txt"], write_paths=["value.txt"],
        finish_on_verified_operations=finish,
        required_operations=required or ["verify"],
        operations={name: Operation(argv=["{python}", "-c", command], description=name) for name in (required or ["verify"])},
    )
    config = TeamConfig(state_directory=tmp_path / "state", profiles=[profile], default_profile="repair")
    return SimpleNamespace(workspace=workspace, profile=profile, config=config)


def _team(setup, responses):
    client = _Client(responses)
    team = SpecialistTeam(setup.config, clients={"repair": client})
    team.submit("Verify the bounded task", specialist="repair", task_id="task")
    return team, client


def _finish(team):
    # A finite scripted conversation must terminate at its declared boundary.
    for _ in range(12):
        result = team.work_once("repair")
        if result["status"] != "running":
            return result
    pytest.fail("scripted task did not terminate")


def test_opt_in_finishes_after_verification_without_another_model_call(tmp_path):
    setup = _setup(tmp_path)
    team, client = _team(setup, [[_call("verify", name="verify")]])
    assert _finish(team)["status"] == "completed"
    assert len(client.requests) == 1
    report = team.task_report("task")
    assert report["required_operations_passed"] is True
    assert report["completion_kind"] == "runtime_verified"
    assert report["model_completed"] is False


def test_default_retains_model_final_answer_behavior(tmp_path):
    setup = _setup(tmp_path, finish=False)
    team, client = _team(setup, [[_call("verify", name="verify")], []])
    assert _finish(team)["status"] == "completed"
    assert len(client.requests) == 2
    assert team.task_report("task")["completion_kind"] == "model"
    assert team.task_report("task")["model_completed"] is True


def test_missing_or_failed_check_cannot_trigger_runtime_completion(tmp_path):
    setup = _setup(tmp_path, required=["verify", "export"])
    team, _client = _team(setup, [[_call("verify", name="verify")], []])
    assert _finish(team)["status"] == "needs_attention"
    assert team.task_report("task")["completion_kind"] is None


def test_queued_edit_invalidates_an_earlier_successful_check(tmp_path):
    setup = _setup(tmp_path)
    edit = _call("edit", "edit_file", path="value.txt", expected_sha256=hashlib.sha256(b"old").hexdigest(), old="old", new="new")
    team, client = _team(setup, [[_call("early", name="verify"), edit], [_call("latest", name="verify")]])
    assert _finish(team)["status"] == "completed"
    assert len(client.requests) == 2
    assert team.task_report("task")["required_operations"]["verify"]["call_id"] == "latest"


def test_unjournaled_source_drift_prevents_runtime_completion(tmp_path):
    setup = _setup(tmp_path, command="from pathlib import Path; Path('value.txt').write_text('outside recorded edit')")
    edit = _call("edit", "edit_file", path="value.txt", expected_sha256=hashlib.sha256(b"old").hexdigest(), old="old", new="new")
    team, _client = _team(setup, [[edit, _call("verify", name="verify")]])
    team.work_once("repair")
    team.work_once("repair")
    team.work_once("repair")
    report = team.task_report("task")
    assert report["source_matches_last_edit"] is False
    assert report["completion_kind"] is None
    assert not any(e["type"] == "verified_completion" for e in team.status("task")["events"])


def test_resume_after_verified_tool_boundary_needs_no_model_or_tool_replay(tmp_path):
    setup = _setup(tmp_path, command="from pathlib import Path; p=Path('runs'); p.write_text(p.read_text()+'x' if p.exists() else 'x')")
    team, client = _team(setup, [[_call("verify", name="verify")]])
    team.work_once("repair")
    team.work_once("repair")
    restarted = SpecialistTeam(setup.config, clients={"repair": client})
    assert restarted.status("task")["status"] == "completed"
    assert restarted.work_once("repair") is None
    assert len(client.requests) == 1
    assert (setup.workspace / "runs").read_text() == "x"


def test_false_default_preserves_existing_task_fingerprints():
    profile = Profile(
        name="repair", description="Repair", model="synthetic",
        workspace=Path("/bounded-workspace"), required_operations=["verify"],
        operations={"verify": Operation(argv=["{python}", "-c", "print('verified')"], description="verify")},
    )
    # Recorded using the unchanged upstream fingerprint function and legacy schema.
    original = "da80c6c569b8678cd2f51993c5cc5c03b8f25b413edc2dd6c3cd3465c12c9af7"
    assert fingerprint(profile) == original
    assert fingerprint(profile.model_copy(update={"finish_on_verified_operations": True})) != original


def test_opt_in_requires_a_nonempty_acceptance_contract(tmp_path):
    with pytest.raises(ValueError, match="requires required_operations"):
        Profile(name="repair", description="Repair", model="synthetic", workspace=tmp_path, finish_on_verified_operations=True)


def test_source_is_rechecked_at_final_completion_boundary(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    edit = _call("edit", "edit_file", path="value.txt", expected_sha256=hashlib.sha256(b"old").hexdigest(), old="old", new="new")
    team, _client = _team(setup, [[edit, _call("verify", name="verify")]])
    original = team._complete

    def complete(identity, state):
        (setup.workspace / "value.txt").write_text("changed after verifier")
        return original(identity, state)

    monkeypatch.setattr(team, "_complete", complete)
    assert _finish(team)["status"] == "needs_attention"
    assert team.task_report("task")["completion_kind"] is None


def test_cancellation_at_completion_boundary_is_not_overwritten(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    team, _client = _team(setup, [[_call("verify", name="verify")]])
    original = team._complete

    def complete(identity, state):
        team.cancel(identity)
        return original(identity, state)

    monkeypatch.setattr(team, "_complete", complete)
    assert _finish(team)["status"] == "cancelled"
    assert team.task_report("task")["completion_kind"] is None


@pytest.mark.parametrize("scope", ["task", "profile"])
def test_pause_at_atomic_completion_write_is_not_overwritten(tmp_path, monkeypatch, scope):
    setup = _setup(tmp_path)
    team, _client = _team(setup, [[_call("verify", name="verify")]])
    original = team.store._update

    def update(identity, status, **kwargs):
        if status == "completed":
            if scope == "task":
                team.pause(task_id=identity)
            else:
                team.pause(specialist="repair")
        return original(identity, status, **kwargs)

    monkeypatch.setattr(team.store, "_update", update)
    team.work_once("repair")
    team.work_once("repair")
    report = team.task_report("task")
    assert report["paused"] is True
    assert report["status_completed"] is False


def test_failed_verifier_never_produces_runtime_completion(tmp_path):
    setup = _setup(tmp_path, command="raise SystemExit(3)")
    team, _client = _team(setup, [[_call("verify", name="verify")], []])
    assert _finish(team)["status"] == "needs_attention"
    assert not any(e["type"] == "verified_completion" for e in team.status("task")["events"])


@pytest.mark.parametrize("blocked_by", ["pause", "uncertain-effect"])
def test_operator_pause_and_unresolved_effects_prevent_completion(tmp_path, monkeypatch, blocked_by):
    from npa.agent_backend.specialists.tools import WorkbenchTools

    setup = _setup(tmp_path)
    team, _client = _team(setup, [[_call("verify", name="verify")]])
    original = WorkbenchTools._run_operation

    def operation(tools, arguments):
        result = original(tools, arguments)
        if blocked_by == "pause":
            team.pause(task_id="task")
        else:
            team.store._begin_call("task", "unresolved", {"name": "unresolved"})
        return result

    monkeypatch.setattr(WorkbenchTools, "_run_operation", operation)
    team.work_once("repair")
    team.work_once("repair")
    assert team.status("task")["status"] != "completed"
    assert not any(e["type"] == "verified_completion" for e in team.status("task")["events"])
