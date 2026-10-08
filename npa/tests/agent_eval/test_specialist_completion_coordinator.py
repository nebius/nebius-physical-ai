"""Verify coordinator inference occurs only at delegation and actionable boundaries."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def completion_modules(monkeypatch):
    import sys

    root = Path(__file__).resolve().parents[3] / "npa/examples/specialists/workflows"
    modules = {}
    for name in ("evidence", "workflow_bridge", "completion_coordinator", "experiment"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


class _Team:
    def __init__(self):
        self.tasks = []
        self.waits = []
        self.events = {}
        self.config = SimpleNamespace(profiles=[])
        self.store = SimpleNamespace(
            _list=lambda: self.tasks,
            _paused_profiles=lambda: [],
            _events=lambda identity: self.events.get(identity, []),
        )

    def add(self, name):
        self.tasks.append(
            {
                "id": name,
                "task_id": name,
                "profile": name,
                "paused": False,
                "status": "running",
            }
        )

    def task_report(self, task_id):
        return dict(next(task for task in self.tasks if task["id"] == task_id))

    def wait_for_attention(self, task_ids):
        self.waits.append(list(task_ids))
        task = next(task for task in self.tasks if task["id"] == task_ids[0])
        task["status"] = "completed"
        return {"attention_task_ids": [task["id"]]}


def _passing_task(team, name):
    team.add(name)
    team.config.profiles.append(
        SimpleNamespace(name=name, required_operations=["verify"])
    )
    team.tasks[-1].update(
        task_id=name,
        specialist=name,
        required_operations={
            "verify": {
                "call_id": "verify",
                "ok": True,
                "returncode": 0,
                "receipt_sha256": "receipt",
            }
        },
        required_operations_passed=True,
        policy_matches=True,
        source_matches_last_edit=True,
        source_binding="recorded_edits_only",
        changes=[],
        uncertain_calls=[],
        latest_edit_epoch=0,
        required_operation_epochs={"verify": 2},
    )
    team.events[name] = [
        {"type": "tool", "name": "run_operation", "call_id": "verify", "at": 2}
    ]


def _coordinate(module, team, directory, invoke):
    return module.coordinate(
        team, directory / "team.json", directory, "medium", "Repair", [], invoke
    )


def test_all_assignments_verified_finish_without_success_review(
    completion_modules, tmp_path
):
    team, phases = _Team(), []

    def invoke(*args, phase):
        phases.append(phase)
        _passing_task(team, "simulation")
        _passing_task(team, "dataset")
        return 0

    assert (
        _coordinate(completion_modules.completion_coordinator, team, tmp_path, invoke)
        == 0
    )
    assert phases == ["delegate"]
    assert team.waits == [["simulation", "dataset"], ["dataset"]]
    result = json.loads((tmp_path / "completion-result.json").read_text())
    assert result["acceptance_scope"] == "configured_required_operations"
    assert result["assignments"] == {
        "simulation": {"simulation": "worker"},
        "dataset": {"dataset": "worker"},
    }
    assert result["worker_reports"]["dataset"]["required_operations_passed"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"required_operations": {}, "required_operations_passed": True},
        {"required_operations": {"verify": None}},
        {"required_operations_passed": False},
        {"source_matches_last_edit": False},
        {"policy_matches": False},
        {"uncertain_calls": [{"status": "started"}]},
        {"source_binding": "model_claim"},
        {"paused": True},
    ],
)
def test_unproven_contract_still_requires_review(completion_modules, tmp_path, change):
    team, phases = _Team(), []

    def invoke(*args, phase):
        phases.append(phase)
        if phase == "delegate":
            _passing_task(team, "simulation")
            team.tasks[-1].update(status="completed", **change)
        return 0

    assert (
        _coordinate(completion_modules.completion_coordinator, team, tmp_path, invoke)
        == 0
    )
    assert phases == ["delegate", "review"]
    assert not (tmp_path / "completion-result.json").exists()


def test_missing_configured_assignment_is_not_vacuous_success(
    completion_modules, tmp_path
):
    team, phases = _Team(), []

    def invoke(*args, phase):
        phases.append(phase)
        if phase == "delegate":
            _passing_task(team, "simulation")
            team.config.profiles.append(
                SimpleNamespace(name="dataset", required_operations=["verify"])
            )
        return 0

    _coordinate(completion_modules.completion_coordinator, team, tmp_path, invoke)
    assert phases == ["delegate", "review"]
    assert not (tmp_path / "completion-result.json").exists()


def test_verified_takeover_finishes_after_recovery_review(
    completion_modules, tmp_path, monkeypatch
):
    module, team, phases, recovery = (
        completion_modules.completion_coordinator,
        _Team(),
        [],
        {},
    )
    monkeypatch.setattr(module, "_coordinator_reports", lambda *_: recovery)

    def invoke(*args, phase):
        phases.append(phase)
        if phase == "delegate":
            _passing_task(team, "simulation")
            team.tasks[-1].update(status="needs_attention", error="Original failure")
        else:
            team.tasks[0]["status"] = "cancelled"
            report = copy.deepcopy(team.tasks[0])
            report.update(task_id="coordinator-simulation", status="running")
            recovery["simulation"] = report
        return 0

    assert _coordinate(module, team, tmp_path, invoke) == 0
    assert phases == ["delegate", "review"]
    result = json.loads((tmp_path / "completion-result.json").read_text())
    assert result["assignments"]["simulation"]["simulation"] == "coordinator_recovery"
    assert result["worker_reports"]["simulation"]["error"] == "Original failure"
    assert result["worker_reports"]["simulation"]["status"] == "cancelled"


@pytest.mark.parametrize(
    "fault", ["wrong_profile", "uncertain", "unbound_edit", "no_takeover"]
)
def test_coordinator_recovery_requires_exact_resolved_assignment(
    completion_modules, tmp_path, monkeypatch, fault
):
    module, team = completion_modules.completion_coordinator, _Team()
    _passing_task(team, "simulation")
    worker = team.tasks[0]
    worker.update(status="cancelled")
    recovery = copy.deepcopy(worker)
    recovery.update(task_id="coordinator-simulation", status="completed")
    if fault == "wrong_profile":
        recovery["specialist"] = "another"
    elif fault == "uncertain":
        worker["uncertain_calls"] = [{"status": "started"}]
    elif fault == "no_takeover":
        worker["status"] = "needs_attention"
    else:
        worker["changes"] = [
            {"path": "source.py", "sha256": "new", "matches_last_edit": False}
        ]
    assert not module._complete(
        team, tmp_path, {"simulation": worker}, {"simulation": recovery}
    )
    assert not (tmp_path / "completion-result.json").exists()


def test_coordinator_uncertainty_blocks_earlier_worker_success(
    completion_modules, tmp_path
):
    module, team = completion_modules.completion_coordinator, _Team()
    _passing_task(team, "simulation")
    team.tasks[0]["status"] = "completed"
    recovery = copy.deepcopy(team.tasks[0])
    recovery["uncertain_calls"] = [{"status": "started"}]
    assert not module._complete(
        team, tmp_path, module._reports(team), {"simulation": recovery}
    )


@pytest.fixture
def journal_team(tmp_path, completion_modules):
    from npa.agent_backend.specialists.config import Profile, Operation, TeamConfig
    from npa.agent_backend.specialists.team import SpecialistTeam

    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "example.txt").write_text("broken")
    profile = Profile(
        name="repair",
        description="Repair source",
        model="test",
        workspace=workspace,
        write_paths=["example.txt"],
        required_operations=["verify"],
        operations={
            "verify": Operation(
                description="Verify current bytes",
                argv=[
                    "{python}",
                    "-c",
                    "from pathlib import Path; assert Path('example.txt').read_text() == 'fixed'",
                ],
            )
        },
    )
    team = SpecialistTeam(
        TeamConfig(
            state_directory=tmp_path / "workers",
            profiles=[profile],
            default_profile="repair",
        )
    )
    coordinator = completion_modules.workflow_bridge._coordinator_store(team, tmp_path)
    team.submit("Repair", specialist="repair", task_id="worker")
    return team, coordinator


def _journal_call(team, store, task_id, name, arguments, call_id):
    from npa.agent_backend.specialists.tools import WorkbenchTools

    return WorkbenchTools(team.config.profiles[0], store, task_id).execute(
        {
            "id": call_id,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    )


def _journal_edit(team, store, task_id, before, after, identity):
    return _journal_call(
        team,
        store,
        task_id,
        "edit_file",
        {
            "path": "example.txt",
            "expected_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "old": before,
            "new": after,
        },
        identity,
    )


def _journal_verify(team, store, identity):
    return _journal_call(
        team, store, "coordinator-repair", "run_operation", {"name": "verify"}, identity
    )


def _journal_takeover(team):
    team.store._update("worker", "needs_attention", error="Original failure")
    team.store._event(
        "worker", {"type": "needs_attention", "error": "Original failure"}
    )
    team.store._event(
        "worker",
        {
            "type": "coordinator_takeover_requested",
            "previous_error": "Original failure",
            "previous_status": "needs_attention",
            "unrelated_trace": "do not copy",
        },
    )
    team.cancel("worker")


def test_current_source_drift_invalidates_real_worker_receipts(
    completion_modules, journal_team, tmp_path
):
    team, _ = journal_team
    module = completion_modules.completion_coordinator
    assert _journal_edit(team, team.store, "worker", "broken", "fixed", "edit")["ok"]
    assert _journal_call(
        team, team.store, "worker", "run_operation", {"name": "verify"}, "check"
    )["ok"]
    team.store._update("worker", "completed")
    assert module._complete(team, tmp_path, module._reports(team), {})
    accepted = (tmp_path / "completion-result.json").read_bytes()
    (team.config.profiles[0].workspace / "example.txt").write_text("changed externally")
    assert not module._complete(team, tmp_path, module._reports(team), {})
    assert (tmp_path / "completion-result.json").read_bytes() == accepted


def test_recovery_cannot_reuse_check_before_newer_worker_edit(
    completion_modules, journal_team, tmp_path
):
    team, coordinator = journal_team
    module = completion_modules.completion_coordinator
    _journal_edit(team, coordinator, "coordinator-repair", "broken", "fixed", "repair")
    _journal_verify(team, coordinator, "old-check")
    _journal_edit(team, team.store, "worker", "fixed", "broken", "regress")
    _journal_edit(team, team.store, "worker", "broken", "fixed", "restore")
    _journal_takeover(team)
    original = team.task_report("worker")
    assert not module._complete(
        team,
        tmp_path,
        module._reports(team),
        module._coordinator_reports(team, tmp_path),
    )
    assert not (tmp_path / "completion-result.json").exists()
    _journal_verify(team, coordinator, "fresh-check")
    assert module._complete(
        team,
        tmp_path,
        module._reports(team),
        module._coordinator_reports(team, tmp_path),
    )
    assert team.task_report("worker") == original
    report = json.loads((tmp_path / "completion-result.json").read_text())[
        "worker_reports"
    ]["worker"]
    assert report["error"] == ""
    assert [item["error"] for item in report["failure_history"]] == [
        "Original failure"
    ] * 2
    assert "do not copy" not in json.dumps(report)


def test_no_coordinator_process_runs_during_wait(completion_modules, tmp_path):
    team, calls = _Team(), []

    def invoke(*args, phase):
        calls.append((phase, args[-1]))
        if phase == "delegate":
            team.add("simulation")
            team.add("dataset")
        else:
            assert len(team.waits) == 2
            assert "Task reports:" in args[-1]
            assert "Previous conversation" not in args[-1]
        return 0

    code = completion_modules.completion_coordinator.coordinate(
        team,
        tmp_path / "team.json",
        tmp_path,
        "medium",
        "Repair and verify",
        [],
        invoke,
    )
    assert code == 0
    assert [phase for phase, _ in calls] == ["delegate", "review"]
    assert team.waits == [["simulation", "dataset"], ["dataset"]]
    events = json.loads((tmp_path / "coordination.json").read_text())["events"]
    assert len(events) == 2 and all(event["model_calls"] == 0 for event in events)


@pytest.mark.parametrize("exit_code", [0, 9])
def test_empty_or_failed_delegation_does_not_fake_completion(
    completion_modules, tmp_path, exit_code
):
    team = _Team()
    code = completion_modules.completion_coordinator.coordinate(
        team,
        tmp_path / "team.json",
        tmp_path,
        "medium",
        "Task",
        [],
        lambda *args, **kwargs: exit_code,
    )
    assert code == (exit_code or 1)
    assert team.waits == []


def test_blocked_worker_gets_one_review_without_automatic_replay(
    completion_modules, tmp_path
):
    team, phases = _Team(), []

    def invoke(*args, phase):
        phases.append(phase)
        if phase == "delegate":
            team.add("blocked")
            team.tasks[0]["status"] = "needs_attention"
        else:
            assert "needs_attention" in args[-1]
        return 0

    completion_modules.completion_coordinator.coordinate(
        team, tmp_path / "team.json", tmp_path, "medium", "Task", [], invoke
    )
    assert phases == ["delegate", "review"]
    assert team.tasks[0]["status"] == "needs_attention"
    assert team.waits == []


def test_baseline_keeps_identical_task_and_direct_tools(
    completion_modules, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        completion_modules.experiment, "_codex", lambda *args: calls.append(args) or 0
    )
    code = completion_modules.experiment._coordinate(
        None,
        "config",
        "directory",
        "astra-only",
        "medium",
        None,
        "same prompt",
        "completion",
    )
    assert code == 0
    assert calls == [("config", "directory", "astra-only", "medium", "same prompt")]


@pytest.mark.parametrize("finishes_during_review", [False, True])
def test_failure_during_delegation_is_reviewed_before_waiting_on_peer(
    completion_modules, tmp_path, finishes_during_review
):
    team, phases = _Team(), []

    def invoke(*args, phase):
        phases.append(phase)
        if phase == "delegate":
            team.add("blocked")
            team.tasks[0]["status"] = "needs_attention"
            team.add("peer")
        elif len(phases) == 2:
            assert team.waits == []
            assert "needs_attention" in args[-1]
            if finishes_during_review:
                team.tasks[1]["status"] = "completed"
        else:
            assert team.tasks[1]["status"] == "completed"
            assert '"status": "completed"' in args[-1]
        return 0

    completion_modules.completion_coordinator.coordinate(
        team, tmp_path / "team.json", tmp_path, "medium", "Task", [], invoke
    )
    assert phases == ["delegate", "review", "review"]
    assert team.waits == ([] if finishes_during_review else [["peer"]])


def test_fresh_turns_use_frozen_task_despite_external_prompt_edit(
    completion_modules, tmp_path, monkeypatch
):
    original = tmp_path / "external.txt"
    original.write_text("changed externally")
    (tmp_path / "prompt.txt").write_text("frozen common task")
    observed = []
    monkeypatch.setattr(
        completion_modules.experiment, "_workspace_policies", lambda _: []
    )
    monkeypatch.setattr(
        completion_modules.completion_coordinator,
        "coordinate",
        lambda *args: observed.append(args[4]) or 0,
    )
    completion_modules.experiment._coordinate(
        None,
        "config",
        tmp_path,
        "astra-tofa",
        "medium",
        original,
        "unused continuous prompt",
        "completion",
    )
    assert observed == ["frozen common task"]
