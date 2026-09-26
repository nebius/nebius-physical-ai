"""Prove opt-in failure handoffs retain receipts, ownership and uncertain-effect safety."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.config import (
    ModelEndpoint,
    ObservationWait,
    Operation,
    Profile,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


def _call(identity, tool="run_operation", **arguments):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": tool, "arguments": json.dumps(arguments)},
    }


def _response(model, *calls):
    message = {"role": "assistant", "content": "Verified" if not calls else ""}
    if calls:
        message["tool_calls"] = list(calls)
    return {
        "model": model,
        "choices": [
            {"message": message, "finish_reason": "tool_calls" if calls else "stop"}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        return next(self.responses)


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "value.txt").write_text("invalid")
    operation = Operation(
        argv=[
            "{python}",
            "-c",
            "from pathlib import Path; "
            "p=Path('invocations'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
            "print('independent verifier result'); "
            "raise SystemExit(0 if Path('value.txt').read_text()=='valid' else 3)",
        ],
        description="Verify existing artifacts; finished failure is authoritative",
        handoff_on_failure=True,
    )
    profile = Profile(
        name="repair",
        description="Repair",
        model="test/primary",
        workspace=workspace,
        read_paths=["value.txt"],
        write_paths=["value.txt"],
        compact_context=True,
        operations={"verify": operation},
        required_operations=["verify"],
        fallback_models=[ModelEndpoint(model="test/backup")],
    )
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="repair"
    )
    return SimpleNamespace(profile=profile, config=config, workspace=workspace)


def _team(setup, responses):
    client = _Client(responses)
    team = SpecialistTeam(setup.config, clients={"repair": client})
    team.submit("Repair and verify", specialist="repair", task_id="task")
    return team, client


def _snapshot(team):
    with team._graph(team.config.profiles[0], "task") as graph:
        return graph.get_state({"configurable": {"thread_id": "task"}}).values


def _finish(team):
    for _ in range(20):
        result = team.work_once("repair")
        if result["status"] != "running":
            return result
    pytest.fail("finite scripted graph did not terminate")


def _repair_responses(repairing_model):
    return [
        _response("test/primary", _call("failed", name="verify")),
        _response(repairing_model, _call("read", "read_file", path="value.txt")),
        _response(
            repairing_model,
            _call(
                "edit",
                "edit_file",
                path="value.txt",
                expected_sha256=hashlib.sha256(b"invalid").hexdigest(),
                old="invalid",
                new="valid",
            ),
        ),
        _response(repairing_model, _call("passed", name="verify")),
        _response(repairing_model),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_real_failure_backup_repairs_with_full_receipt_after_restart(setup, enabled):
    setup.profile.operations["verify"].handoff_on_failure = enabled
    repairing_model = "test/backup" if enabled else "test/primary"
    team, client = _team(setup, _repair_responses(repairing_model))
    team.work_once("repair")
    team.work_once("repair")
    recorded = _snapshot(team)["messages"][-1]
    assert json.loads(recorded["content"])["returncode"] == 3
    assert (setup.workspace / "invocations").read_text() == "x"
    restarted = SpecialistTeam(setup.config, clients=team.clients)
    assert _finish(restarted)["status"] == "completed"
    assert [call["model"] for call in client.calls] == [
        "test/primary",
        *[repairing_model] * 4,
    ]
    assert recorded in client.calls[1]["messages"]
    assert client.calls[0]["extra"] == client.calls[1]["extra"]
    events = restarted.status("task")["events"]
    assert sum(event["type"] == "model_fallback" for event in events) == int(enabled)
    assert (
        sum(
            event["usage"]["total_tokens"]
            for event in events
            if event["type"] == "model"
        )
        == 125
    )
    assert len(restarted.store._calls("task")) == 4
    assert (setup.workspace / "invocations").read_text() == "xx"
    assert restarted.task_report("task")["required_operations_passed"] is True
    if enabled:
        assert "operation executed" in client.calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("backups", [0, 1])
def test_exhaustion_retains_each_failed_receipt_without_extra_inference(setup, backups):
    if not backups:
        setup.profile.fallback_models = []
    team, client = _team(
        setup,
        [
            _response("test/primary", _call("first", name="verify")),
            _response("test/backup", _call("second", name="verify")),
        ],
    )
    assert _finish(team)["status"] == "needs_attention"
    assert len(client.calls) == backups + 1
    calls = team.store._calls("task")
    assert len(calls) == backups + 1
    assert all(call["status"] == "completed" for call in calls)
    receipts = [
        message for message in _snapshot(team)["messages"] if message["role"] == "tool"
    ]
    assert len(receipts) == backups + 1
    assert all(
        json.loads(message["content"])["returncode"] == 3 for message in receipts
    )
    assert team.work_once("repair") is None


def test_lost_receipt_blocks_handoff_and_replay(setup, monkeypatch):
    team, client = _team(
        setup, [_response("test/primary", _call("lost", name="verify"))]
    )
    team.work_once("repair")
    original = team.store._finish_call
    monkeypatch.setattr(
        team.store, "_finish_call", lambda *_: (_ for _ in ()).throw(OSError())
    )
    assert team.work_once("repair")["status"] == "needs_attention"
    monkeypatch.setattr(team.store, "_finish_call", original)
    assert (setup.workspace / "invocations").read_text() == "x"
    assert len(team.task_report("task")["uncertain_calls"]) == 1
    assert team.work_once("repair") is None
    assert len(client.calls) == 1
    assert not any(
        event["type"] == "model_fallback" for event in team.status("task")["events"]
    )


def test_completed_receipt_reused_before_checkpoint_without_rerunning(setup):
    team, client = _team(
        setup, [_response("test/primary", _call("once", name="verify"))]
    )
    team.work_once("repair")
    tools = WorkbenchTools(setup.profile, team.store, "task")
    tools.execute(_snapshot(team)["pending"][0])
    restarted = SpecialistTeam(setup.config, clients=team.clients)
    restarted.work_once("repair")
    restarted.work_once("repair")
    assert _snapshot(restarted)["model_index"] == 1
    assert len(client.calls) == 1
    assert (setup.workspace / "invocations").read_text() == "x"


def test_unknown_effect_in_remaining_batch_blocks_handoff(setup, monkeypatch):
    team, client = _team(
        setup,
        [
            _response(
                "test/primary",
                _call("known", name="verify"),
                _call("uncertain", name="verify"),
            )
        ],
    )
    team.work_once("repair")
    team.work_once("repair")
    monkeypatch.setattr(
        team.store, "_finish_call", lambda *_: (_ for _ in ()).throw(OSError())
    )
    assert team.work_once("repair")["status"] == "needs_attention"
    assert len(client.calls) == 1
    assert len(team.task_report("task")["uncertain_calls"]) == 1
    assert not any(
        event["type"] == "model_fallback" for event in team.status("task")["events"]
    )


@pytest.mark.parametrize("terminal", ["failed", "succeeded", "unconfigured"])
def test_wait_polls_without_inference_and_only_terminal_failure_hands_off(
    setup, terminal
):
    status = setup.workspace / "status.txt"
    status.write_text("pending")
    setup.profile.operations["verify"] = Operation(
        argv=[
            "{python}",
            "-c",
            "from pathlib import Path; import json; print(json.dumps(dict(status=Path('status.txt').read_text())))",
        ],
        description="Observe",
        observation_only=True,
        handoff_on_failure=True,
        wait_for=ObservationWait(
            field="/status",
            pending_values=["pending"],
            success_values=["succeeded"],
            failure_values=["failed"],
            poll_interval=0.001,
        ),
    )
    team, client = _team(
        setup, [_response("test/primary", _call("wait", name="verify"))]
    )
    team.work_once("repair")
    team.work_once("repair")
    assert not _snapshot(team).get("operation_failure")
    status.write_text(terminal)
    import time

    time.sleep(0.01)
    team.work_once("repair")
    assert bool(_snapshot(team).get("operation_failure")) == (terminal == "failed")
    assert len(client.calls) == 1


def test_signal_termination_is_not_authority_for_handoff(setup):
    setup.profile.operations["verify"].argv = [
        "{python}",
        "-c",
        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
    ]
    team, _ = _team(
        setup, [_response("test/primary", _call("terminated", name="verify"))]
    )
    team.work_once("repair")
    team.work_once("repair")
    state = _snapshot(team)
    assert json.loads(state["messages"][-1]["content"])["returncode"] < 0
    assert not state["operation_failure"]


def test_policy_default_strictness_and_changes_require_new_task(setup):
    original = Operation(argv=["true"], description="Check")
    assert original.handoff_on_failure is False
    for invalid in ("true", 1, None):
        with pytest.raises(ValueError):
            Operation(argv=["true"], description="Check", handoff_on_failure=invalid)
    team, client = _team(setup, [])
    before = fingerprint(setup.profile)
    setup.profile.operations["verify"].handoff_on_failure = False
    assert fingerprint(setup.profile) != before
    assert team.work_once("repair")["status"] == "needs_attention"
    assert client.calls == []
