"""Prove durable model-free observation waits, interruption safety and compact handoffs."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.config import (
    ObservationWait,
    Operation,
    Profile,
    TeamConfig,
)
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools
from npa.agent_backend.specialists.waiting import (
    _poll_call,
    _pointer,
    _observation_result,
)


def _response(*, final=False):
    message = {"role": "assistant", "content": "Verified the workflow" if final else ""}
    if not final:
        message["tool_calls"] = [
            {
                "id": "wait-for-run",
                "type": "function",
                "function": {
                    "name": "run_operation",
                    "arguments": '{"name":"observe"}',
                },
            }
        ]
    return {
        "model": "synthetic",
        "choices": [
            {
                "message": message,
                "finish_reason": "stop" if final else "tool_calls",
            }
        ],
    }


class _Client:
    def __init__(self):
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        return _response(final=len(self.calls) > 1)


@pytest.fixture
def waiting(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    status = workspace / "status.json"
    status.write_text('{"status":"running"}')
    profile = _profile(workspace)
    config = TeamConfig(
        state_directory=tmp_path / "state", profiles=[profile], default_profile="scene"
    )
    client = _Client()
    team = SpecialistTeam(config, clients={"scene": client})
    team.submit("Observe completion", specialist="scene", task_id="task")
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        "npa.agent_backend.specialists.waiting.time.time", lambda: clock.now
    )
    return SimpleNamespace(team=team, client=client, status=status, clock=clock)


def _profile(workspace):
    return Profile(
        name="scene",
        description="Observe one authorized run",
        model="synthetic",
        workspace=workspace,
        read_paths=["status.json"],
        write_paths=["source.txt"],
        required_operations=["observe"],
        operations={
            "observe": Operation(
                argv=[
                    "{python}",
                    "-c",
                    "from pathlib import Path; print(Path('status.json').read_text())",
                ],
                description="Wait for the existing run",
                observation_only=True,
                wait_for=ObservationWait(
                    field="/status",
                    pending_values=["running"],
                    success_values=["succeeded"],
                    failure_values=["failed"],
                    poll_interval=10,
                ),
            )
        },
    )


def _observe(waiting, value):
    waiting.clock.now += 10
    waiting.status.write_text(json.dumps({"status": value}))
    return waiting.team.work_once("scene")


def test_pending_observations_never_call_model_or_claim_completion(waiting):
    waiting.team.work_once("scene")
    for _ in range(4):
        assert _observe(waiting, "running")["status"] == "running"
        assert len(waiting.client.calls) == 1
        assert waiting.team.task_report("task")["required_operations_passed"] is False
    _observe(waiting, "succeeded")
    assert len(waiting.client.calls) == 1
    assert waiting.team.work_once("scene")["status"] == "completed"
    assert len(waiting.client.calls) == 2
    messages = waiting.client.calls[-1]["messages"]
    receipts = [
        json.loads(message["content"])
        for message in messages
        if message["role"] == "tool"
    ]
    assert len(receipts) == 1
    assert receipts[0]["observation"]["status"] == "succeeded"
    assert waiting.team.task_report("task")["required_operations_passed"] is True
    assert len(waiting.team.store._calls("task")) == 5


def test_poll_deadline_and_completed_checkpoint_survive_restart(waiting):
    waiting.team.work_once("scene")
    _observe(waiting, "running")
    restarted = SpecialistTeam(waiting.team.config, clients={"scene": waiting.client})
    assert restarted.work_once("scene")["next_observation_at"] == waiting.clock.now + 10
    assert len(restarted.store._calls("task")) == 1
    waiting.team = restarted
    _observe(waiting, "succeeded")
    assert restarted.work_once("scene")["status"] == "completed"
    assert len(waiting.client.calls) == 2


def test_receipt_before_checkpoint_is_reused_without_repoll(waiting, monkeypatch):
    waiting.team.work_once("scene")
    profile = waiting.team.config.profiles[0]
    tools = WorkbenchTools(profile, waiting.team.store, "task")
    call = _response()["choices"][0]["message"]["tool_calls"][0]
    tools.execute(_poll_call(call, 0))
    monkeypatch.setattr(
        WorkbenchTools, "_run_operation", lambda *_: pytest.fail("replayed poll")
    )
    assert waiting.team.work_once("scene")["status"] == "running"
    assert len(waiting.team.store._calls("task")) == 1
    assert len(waiting.client.calls) == 1


def test_interrupted_poll_requires_reconciliation_without_replay(waiting, monkeypatch):
    waiting.team.work_once("scene")
    original = waiting.team.store._finish_call
    monkeypatch.setattr(
        waiting.team.store, "_finish_call", lambda *_: (_ for _ in ()).throw(OSError())
    )
    assert waiting.team.work_once("scene")["status"] == "needs_attention"
    monkeypatch.setattr(waiting.team.store, "_finish_call", original)
    assert waiting.team.work_once("scene") is None
    assert len(waiting.client.calls) == 1
    assert len(waiting.team.task_report("task")["uncertain_calls"]) == 1
    waiting.team.dismiss_interrupted_observation(
        "task", waiting.team.store._calls("task")[0]["call_id"]
    )
    assert waiting.team.status("task")["status"] == "needs_attention"


@pytest.mark.parametrize("control", ["task_pause", "profile_pause", "cancel"])
def test_control_during_wait_prevents_any_more_polls_or_model_calls(waiting, control):
    waiting.team.work_once("scene")
    _observe(waiting, "running")
    if control == "task_pause":
        waiting.team.pause(task_id="task")
    elif control == "profile_pause":
        waiting.team.pause(specialist="scene")
    else:
        waiting.team.cancel("task")
    assert _observe(waiting, "succeeded") is None
    assert len(waiting.team.store._calls("task")) == 1
    assert len(waiting.client.calls) == 1
    report = waiting.team.wait_for_attention(["task"])
    assert report["attention_task_ids"] == ["task"]
    assert report["tasks"]["task"]["required_operations_passed"] is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"status":"failed"}', "failed"),
        ('{"status":"unknown"}', "invalid"),
        ('{"status":null}', "invalid"),
        ('{"other":"running"}', "invalid"),
        ("not JSON", "invalid"),
    ],
)
def test_bad_observation_wakes_specialist_instead_of_waiting_forever(
    waiting, text, expected
):
    waiting.team.config.profiles[0].required_operations = []
    waiting.team = SpecialistTeam(
        waiting.team.config, clients={"scene": waiting.client}
    )
    waiting.team.submit("Observe", specialist="scene", task_id="bad-observation")
    waiting.team.cancel("task")
    waiting.team.work_once("scene")
    waiting.status.write_text(text)
    waiting.team.work_once("scene")
    assert waiting.team.work_once("scene")["status"] == "completed"
    result = json.loads(waiting.client.calls[-1]["messages"][-1]["content"])
    assert result["ok"] is False
    assert result["observation"]["status"] == expected
    assert len(waiting.client.calls) == 2


def test_wait_for_attention_ignores_routine_events_and_caller_stop_is_local(waiting):
    stopped = threading.Event()
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            waiting.team.wait_for_attention,
            ["task"],
            poll_interval=0.01,
            stop_event=stopped,
        )
        waiting.team.store._event(
            "task", {"type": "tool", "name": "read_file", "result": {}}
        )
        assert not future.done()
        stopped.set()
        result = future.result(timeout=5)
    assert result["interrupted"] is True
    assert result["attention_task_ids"] == []
    assert waiting.team.status("task")["status"] == "queued"
    assert waiting.client.calls == []
    assert "events" not in result["tasks"]["task"]


def test_compact_report_rejects_source_drift_and_omits_raw_content(waiting):
    profile = waiting.team.config.profiles[0]
    tools = WorkbenchTools(profile, waiting.team.store, "task")
    edit = {
        "id": "edit",
        "function": {
            "name": "edit_file",
            "arguments": json.dumps(
                {
                    "path": "source.txt",
                    "expected_sha256": hashlib.sha256(b"").hexdigest(),
                    "old": "",
                    "new": "private source content",
                }
            ),
        },
    }
    tools.execute(edit)
    waiting.team.work_once("scene")
    _observe(waiting, "succeeded")
    waiting.team.work_once("scene")
    report = waiting.team.task_report("task")
    assert report["required_operations_passed"] is True
    assert "private source content" not in json.dumps(report)
    (profile.workspace / "source.txt").write_text("external modification")
    changed = waiting.team.task_report("task")
    assert changed["source_matches_last_edit"] is False
    assert changed["required_operations_passed"] is False


def test_model_final_without_required_checks_is_not_artifact_acceptance(waiting):
    waiting.team.store._update("task", "completed", result="Everything passed")
    report = waiting.team.task_report("task")
    assert report["model_completed"] is True
    assert report["required_operations_passed"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"pending_values": ["succeeded"]},
        {"success_values": [""]},
        {"field": "status"},
        {"field": "/bad~2field"},
        {"poll_interval": float("nan")},
        {"poll_interval": 0},
    ],
)
def test_wait_policy_rejects_ambiguous_configuration(waiting, change):
    policy = waiting.team.config.profiles[0].operations["observe"].wait_for.model_dump()
    with pytest.raises(ValueError):
        ObservationWait.model_validate({**policy, **change})


def test_effectful_command_cannot_opt_into_repeated_execution(waiting):
    operation = waiting.team.config.profiles[0].operations["observe"].model_dump()
    with pytest.raises(ValueError, match="observation_only"):
        Operation.model_validate({**operation, "observation_only": False})
    assert _pointer({"a/b": {"~state": "running"}}, "/a~1b/~0state") == "running"
    assert _pointer({"waves": [{"status": "running"}]}, "/waves/0/status") == "running"


def test_nonzero_command_cannot_hide_behind_pending_or_success_json(waiting):
    policy = waiting.team.config.profiles[0].operations["observe"].wait_for
    for state in ("running", "succeeded"):
        result = _observation_result(
            {
                "ok": False,
                "returncode": 1,
                "stdout": json.dumps({"status": state}),
            },
            policy,
        )
        assert result["ok"] is False
        assert result["observation"]["status"] == "command_failed"


def test_changed_policy_cannot_certify_old_required_receipts(waiting):
    waiting.team.work_once("scene")
    _observe(waiting, "succeeded")
    waiting.team.work_once("scene")
    assert waiting.team.task_report("task")["required_operations_passed"] is True
    waiting.team.config.profiles[0].operations["observe"].argv = ["other-command"]
    report = waiting.team.task_report("task")
    assert report["policy_matches"] is False
    assert report["required_operations_passed"] is False


@pytest.mark.parametrize(
    "pointer", ["/waves/-1/status", "/waves/00/status", "/waves/1/status"]
)
def test_json_pointer_rejects_invalid_array_indices(pointer):
    with pytest.raises(ValueError):
        _pointer({"waves": [{"status": "running"}]}, pointer)
