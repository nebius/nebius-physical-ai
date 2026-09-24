"""Verify coordinator inference occurs only at delegation and actionable boundaries."""

from __future__ import annotations

import importlib.util
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
        self.config = SimpleNamespace(profiles=[])
        self.store = SimpleNamespace(
            _list=lambda: self.tasks, _paused_profiles=lambda: []
        )

    def add(self, name):
        self.tasks.append(
            {"id": name, "profile": name, "paused": False, "status": "running"}
        )

    def task_report(self, task_id):
        return dict(next(task for task in self.tasks if task["id"] == task_id))

    def wait_for_attention(self, task_ids):
        self.waits.append(list(task_ids))
        task = next(task for task in self.tasks if task["id"] == task_ids[0])
        task["status"] = "completed"
        return {"attention_task_ids": [task["id"]]}


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
