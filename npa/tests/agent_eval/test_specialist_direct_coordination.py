"""Verify explicit dispatch avoids Astra only when durable task and usage evidence permits it."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.config import Operation, Profile, TeamConfig
from npa.agent_backend.specialists.team import SpecialistTeam


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "npa/examples/specialists"
    loaded = {}
    for name in (
        "evidence",
        "workflow_bridge",
        "completion_coordinator",
        "experiment",
        "workflow_usage",
    ):
        path = root / (
            "workflow_usage.py" if name == "workflow_usage" else f"workflows/{name}.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


class _Client:
    def __init__(self):
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        message = {"role": "assistant", "content": "Verified"}
        if len(self.calls) == 1:
            message["tool_calls"] = [
                {
                    "id": "verify-call",
                    "type": "function",
                    "function": {
                        "name": "run_operation",
                        "arguments": '{"name":"verify"}',
                    },
                }
            ]
        return {
            "model": arguments["model"],
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if len(self.calls) == 1 else "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    profiles = []
    for index in range(2):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        profiles.append(
            Profile(
                name=f"worker-{index}",
                model="test/model",
                description="One explicit assignment",
                instructions=f"Repair task {index}",
                workspace=workspace,
                required_operations=["verify"],
                operations={
                    "verify": Operation(
                        argv=["{python}", "-c", "print('verified')"],
                        description="Check completed artifacts",
                    )
                },
            )
        )
    config = TeamConfig(
        state_directory=tmp_path / "state",
        profiles=profiles,
        default_profile=profiles[0].name,
    )
    clients = {profile.name: _Client() for profile in profiles}
    team = SpecialistTeam(config, clients=clients)
    monkeypatch.setattr(
        team, "wait_for_attention", lambda identities: _run_workers(team, identities)
    )
    directory = tmp_path / "run"
    directory.mkdir()
    return SimpleNamespace(team=team, directory=directory, clients=clients)


def _run_workers(team, identities):
    for identity in identities:
        task = team.store._get(identity)
        for _ in range(6):
            task = team.work_once(task["profile"])
            if task["status"] != "running":
                break
        assert task["status"] == "completed"
    return {"attention_task_ids": identities}


def _coordinate(modules, setup, invoke):
    return modules.completion_coordinator.coordinate_specialists_first(
        setup.team,
        setup.directory / "team.json",
        setup.directory,
        "medium",
        "Complete both supplied tasks",
        [],
        invoke,
    )


def test_real_journals_prove_every_assignment_without_any_astra_invocation(
    modules, setup
):
    def invoke(*args, **kwargs):
        pytest.fail("successful explicit assignments must not invoke Astra")

    assert _coordinate(modules, setup, invoke) == 0
    result = json.loads((setup.directory / "completion-result.json").read_text())
    assert set(result["assignments"]) == {"worker-0", "worker-1"}
    assert all(
        report["required_operations_passed"]
        for report in result["worker_reports"].values()
    )
    dispatch = json.loads((setup.directory / "dispatch.json").read_text())
    for profile in setup.team.config.profiles:
        task = setup.team.store._get(dispatch["assignments"][profile.name]["task_id"])
        assert task["profile"] == profile.name and task["status"] == "completed"
        assert profile.instructions in task["goal"]
        assert "Complete both supplied tasks" in task["goal"]
        assert len(setup.clients[profile.name].calls) == 2
    assert (
        json.loads((setup.directory / "coordinator-config.json").read_text())["turns"]
        == []
    )
    usage = modules.evidence._specialist_usage(
        modules.evidence._receipts(setup.team.store)
    )
    assert usage["usage_complete"] is True
    assert len(usage["responses"]) == 4


def test_failure_escalates_only_to_review_with_original_failed_receipts(
    modules, setup, monkeypatch
):
    def fail(identities):
        for identity in identities:
            setup.team.store._update(
                identity, "needs_attention", error="Native verification failed"
            )
        return {"attention_task_ids": identities}

    monkeypatch.setattr(setup.team, "wait_for_attention", fail)
    invocations = []

    def invoke(*args, phase):
        invocations.append((args, phase))
        assert "needs_attention" in args[-1]
        assert "Native verification failed" in args[-1]
        return 0

    assert _coordinate(modules, setup, invoke) == 0
    assert [phase for _, phase in invocations] == ["review"]
    assert invocations[0][0][2:4] == ("astra-tofa", "medium")
    assert not (setup.directory / "completion-result.json").exists()


@pytest.mark.parametrize("defect", ["instructions", "checks", "existing", "empty"])
def test_invalid_dispatch_refuses_before_creating_any_task(modules, setup, defect):
    if defect == "instructions":
        setup.team.config.profiles[-1].instructions = "  "
    elif defect == "checks":
        setup.team.config.profiles[-1].required_operations = []
    elif defect == "empty":
        setup.team.config.profiles = []
    else:
        setup.team.submit("Original task", specialist="worker-0", task_id="original")
    before = copy.deepcopy(setup.team.store._list())
    with pytest.raises(ValueError):
        _coordinate(modules, setup, lambda *_args, **_kwargs: pytest.fail("no Astra"))
    assert setup.team.store._list() == before
    assert not (setup.directory / "dispatch.json").exists()


def test_long_profiles_have_distinct_legal_task_ids(modules, setup):
    for index, profile in enumerate(setup.team.config.profiles):
        profile.name = "worker-" + "a" * 120 + str(index)
    modules.completion_coordinator._dispatch_configured(
        setup.team, setup.directory, "Request"
    )
    tasks = setup.team.store._list()
    assert len(tasks) == 2 and len({task["id"] for task in tasks}) == 2
    assert all(len(task["id"]) <= 128 for task in tasks)
    assert {task["profile"] for task in tasks} == {
        profile.name for profile in setup.team.config.profiles
    }


@pytest.mark.parametrize("arm", ["astra-only", "astra-tofa"])
def test_experiment_keeps_baseline_direct_and_selects_opt_in_dispatch(
    modules, setup, monkeypatch, arm
):
    calls = []
    monkeypatch.setattr(
        modules.experiment, "_codex", lambda *args: calls.append("astra") or 0
    )
    monkeypatch.setattr(
        modules.completion_coordinator,
        "coordinate_specialists_first",
        lambda *args: calls.append("dispatch") or 0,
    )
    prompt = setup.directory / "prompt.txt"
    prompt.write_text("Operator request")
    assert (
        modules.experiment._coordinate(
            setup.team,
            setup.directory / "team.json",
            setup.directory,
            arm,
            "medium",
            prompt,
            "Final prompt",
            "specialists-first",
        )
        == 0
    )
    assert calls == (["astra"] if arm == "astra-only" else ["dispatch"])


def _zero_documents(directory):
    (directory / "protocol.json").write_text(
        json.dumps({"arm": "astra-tofa", "coordination": "specialists-first"})
    )
    (directory / "coordinator-config.json").write_text(
        json.dumps({"strategy": "specialists-first", "turns": []})
    )


def test_explicit_zero_has_both_records_and_no_events(modules, tmp_path):
    _zero_documents(tmp_path)
    result = modules.evidence._astra_usage(tmp_path, "astra-tofa")
    assert result["usage_complete"] is True
    assert result["matched_tool_scope"] is True
    assert result["turns"] == []
    assert result["invocations"]["verification"] == "explicit_zero_invocations"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_protocol",
        "missing_invocations",
        "completion_mode",
        "baseline_arm",
        "started_event",
        "failed_event",
        "malformed_event",
        "declared_invocation",
    ],
)
def test_missing_or_contradictory_evidence_never_means_zero_astra(
    modules, tmp_path, defect
):
    _zero_documents(tmp_path)
    protocol = tmp_path / "protocol.json"
    config = tmp_path / "coordinator-config.json"
    if defect == "missing_protocol":
        protocol.unlink()
    elif defect == "missing_invocations":
        config.unlink()
    elif defect in {"completion_mode", "baseline_arm"}:
        data = json.loads(protocol.read_text())
        data.update(
            {"coordination": "completion"}
            if defect == "completion_mode"
            else {"arm": "astra-only"}
        )
        protocol.write_text(json.dumps(data))
    elif defect == "declared_invocation":
        config.write_text(
            json.dumps(
                {
                    "strategy": "specialists-first",
                    "turns": [{"exit_code": 0, "ended_epoch": 1}],
                }
            )
        )
    else:
        event = {
            "started_event": '{"type":"turn.started"}',
            "failed_event": '{"type":"turn.failed"}',
            "malformed_event": "not json",
        }[defect]
        (tmp_path / "codex.jsonl").write_text(event + "\n")
    assert (
        modules.evidence._astra_usage(tmp_path, "astra-tofa")["usage_complete"] is False
    )


def _accounting_inputs(modules, directory):
    _zero_documents(directory)
    usage = {
        "usage_complete": True,
        "astra": modules.evidence._astra_usage(directory, "astra-tofa"),
        "specialists": {
            "usage_complete": True,
            "responses": [
                {
                    "model": model,
                    "accepted": accepted,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }
                for model, accepted in [("primary", False), ("backup", True)]
            ],
        },
    }
    execution = {
        "arm": "astra-tofa",
        "coordination": "specialists-first",
        "exit_code": 0,
        "snapshot_errors": {},
        "unfinished_tasks": [],
    }
    rate = {"input_price_per_million_tokens": 2, "output_price_per_million_tokens": 4}
    prices = {
        "models": {
            name: {"cache_policy": "full-input", "rate_options": [rate]}
            for name in ("primary", "backup", "astra")
        }
    }
    return usage, execution, prices


def test_zero_astra_preserves_every_specialist_and_rejected_fallback_cost(
    modules, tmp_path
):
    usage, execution, prices = _accounting_inputs(modules, tmp_path)
    report = modules.workflow_usage.summarize_usage(
        usage, execution, prices, coordinator_model="astra"
    )
    assert report["usage_complete"] is True
    assert report["estimated_cost_usd"] == pytest.approx(0.00056)
    assert set(report["models"]) == {"primary", "backup"}
    assert report["models"]["primary"]["rejected_specialist_responses"] == 1
    assert report["models"]["backup"]["accepted_specialist_responses"] == 1
    assert report["totals"]["tokens"]["input_tokens"] == 200


def test_zero_astra_does_not_hide_unknown_provider_failure_usage(modules, tmp_path):
    usage, execution, prices = _accounting_inputs(modules, tmp_path)
    usage["specialists"]["failures"] = [
        {
            "type": "provider_failure",
            "model": "primary",
            "usage": {},
            "usage_missing": True,
        }
    ]
    usage["specialists"]["usage_complete"] = False
    usage["usage_complete"] = False
    report = modules.workflow_usage.summarize_usage(
        usage, execution, prices, coordinator_model="astra"
    )
    assert report["usage_complete"] is False
    assert "provider_failure_usage_unknown" in report["incomplete_reasons"]
    assert report["estimated_cost_range_usd"] is None
    assert report["models"]["primary"]["unpriced_records"] == 1
    assert report["models"]["backup"]["accepted_specialist_responses"] == 1


@pytest.mark.parametrize(
    "defect",
    ["mode", "arm", "missing_invocations", "usage_missing", "nonzero_invocations"],
)
def test_summary_rejects_undeclared_zero_even_with_specialist_tokens(
    modules, tmp_path, defect
):
    usage, execution, prices = _accounting_inputs(modules, tmp_path)
    if defect == "mode":
        execution["coordination"] = "completion"
    elif defect == "arm":
        execution["arm"] = "astra-only"
    elif defect == "missing_invocations":
        del usage["astra"]["invocations"]
    elif defect == "usage_missing":
        usage["specialists"]["responses"][0]["usage"] = {}
    else:
        usage["astra"]["invocations"]["expected_turns"] = 1
    report = modules.workflow_usage.summarize_usage(
        usage, execution, prices, coordinator_model="astra"
    )
    assert report["usage_complete"] is False
    assert report["estimated_cost_usd"] is None
    assert set(report["models"]) == {"primary", "backup"}
