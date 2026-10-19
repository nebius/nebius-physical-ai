"""Require actual deterministic checks for host-completed model responses and recovery."""

import hashlib
import json

import pytest

from npa.agent_backend.specialists.config import (
    AgentConfiguration,
    ConfigurationRouting,
    ContextEndpoint,
    Operation,
    Profile,
    TeamConfig,
    fingerprint,
)
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completion(self, **arguments):
        self.calls.append(arguments)
        message = next(self.responses)
        return {
            "model": "synthetic/model",
            "choices": [
                {
                    "finish_reason": "tool_calls"
                    if message.get("tool_calls")
                    else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }


@pytest.fixture
def setup(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    target = workspace / "src/config.json"
    target.write_text('{"value":"broken"}')
    endpoint = {
        "model": "synthetic/model",
        "context_window": 100000,
        "model_options": {"max_tokens": 1000},
    }
    candidate = AgentConfiguration(
        **endpoint,
        name="single",
        description="Repair",
        tools_verified=True,
        input_price_per_million=1.0,
        output_price_per_million=1.0,
    )
    profile = Profile(
        name="worker",
        description="Repair",
        model="synthetic/model",
        workspace=workspace,
        read_paths=["src"],
        write_paths=["src/config.json"],
        configuration_routing=ConfigurationRouting(
            classifier=ContextEndpoint(**endpoint), candidates=[candidate]
        ),
        required_operations=["verify"],
        finish_on_verified_operations=True,
        verify_on_completion=True,
        operations={
            "verify": Operation(
                argv=[
                    "{python}",
                    "-c",
                    'import json; assert json.load(open("src/config.json"))["value"] == "fixed"',
                ],
                description="Verify config",
                deterministic_inputs=["src/config.json"],
                handoff_on_failure=True,
            )
        },
    )
    return (
        target,
        profile,
        TeamConfig(
            state_directory=tmp_path / "state",
            profiles=[profile],
            default_profile="worker",
        ),
    )


def _response(text="Done", call=None):
    result = {"role": "assistant", "content": text}
    if call:
        result["tool_calls"] = [call]
    return result


def test_host_checks_accept_current_correct_artifact_without_model_verify_call(setup):
    target, profile, config = setup
    target.write_text('{"value":"fixed"}')
    client = _Client([_response()])
    team = SpecialistTeam(config, clients={"worker": client})
    team.submit("Verify", specialist="worker", task_id="task")
    assert team.work_once("worker")["status"] == "running"
    assert team.work_once("worker")["status"] == "completed"
    report = team.task_report("task")
    assert (
        report["completion_kind"] == "runtime_verified"
        and not report["model_completed"]
    )
    assert report["required_operations_passed"] and len(client.calls) == 1
    assert any(
        event["type"] == "host_verification" for event in team.status("task")["events"]
    )


@pytest.mark.parametrize("tool", ["edit_file", "replace_json"])
def test_failed_host_check_does_not_promote_and_last_model_can_repair(setup, tool):
    target, profile, config = setup
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    edit = {
        "id": "repair",
        "type": "function",
        "function": {
            "name": tool,
            "arguments": json.dumps(
                {
                    "path": "src/config.json",
                    "expected_sha256": digest,
                    **({"old": "broken", "new": "fixed"} if tool == "edit_file" else {"json_pointer": "/value", "value": "fixed"}),
                }
            ),
        },
    }
    client = _Client([_response(), _response(call=edit), _response()])
    team = SpecialistTeam(config, clients={"worker": client})
    team.submit("Repair", specialist="worker", task_id="task")
    task = team.work_once("worker")
    while task["status"] == "running":
        task = team.work_once("worker")
    assert (
        task["status"] == "completed"
        and team.task_report("task")["required_operations_passed"]
    )
    assert team.task_report("task")["failure_count"] == 1
    assert all(item["matches_last_edit"] for item in team.task_report("task")["changes"])
    assert any(
        event["type"] == "configuration_self_recovery"
        for event in team.status("task")["events"]
    )


def test_unchanged_failed_host_checks_end_in_attention(setup):
    _, profile, config = setup
    client = _Client([_response(), _response()])
    team = SpecialistTeam(config, clients={"worker": client})
    team.submit("Repair", specialist="worker", task_id="task")
    task = team.work_once("worker")
    while task["status"] == "running":
        task = team.work_once("worker")
    assert (
        task["status"] == "needs_attention"
        and "same failed input hashes" in task["error"]
    )
    assert len(client.calls) == 2


def test_changed_inputs_during_verifier_invalidate_success(setup):
    target, profile, config = setup
    profile.operations["verify"].argv = [
        "{python}",
        "-c",
        'from pathlib import Path; Path("src/config.json").write_text("changed")',
    ]
    team = SpecialistTeam(config)
    team.submit("Verify", specialist="worker", task_id="task")
    tools = WorkbenchTools(profile, team.store, "task")
    result = tools.execute(
        {
            "id": "check",
            "function": {"name": "run_operation", "arguments": '{"name":"verify"}'},
        }
    )
    assert (
        result["returncode"] == 0
        and not result["ok"]
        and not result["inputs_unchanged"]
    )


def test_external_edit_after_check_invalidates_receipt_even_without_agent_edit(setup):
    target, profile, config = setup
    target.write_text('{"value":"fixed"}')
    team = SpecialistTeam(config)
    team.submit("Verify", specialist="worker", task_id="task")
    tools = WorkbenchTools(profile, team.store, "task")
    tools.execute(
        {
            "id": "check",
            "function": {"name": "run_operation", "arguments": '{"name":"verify"}'},
        }
    )
    assert team.task_report("task")["required_operations_passed"]
    target.write_text('{"value":"changed"}')
    assert not team.task_report("task")["required_operations_passed"]


def test_host_checks_require_explicit_deterministic_scope_and_bind_policy(setup):
    _, profile, _ = setup
    original = fingerprint(profile)
    data = profile.model_dump()
    data["operations"]["verify"]["deterministic_inputs"] = []
    with pytest.raises(ValueError, match="deterministic required checks"):
        Profile.model_validate(data)
    data = profile.model_dump()
    data["operations"]["verify"]["deterministic_inputs"] = ["../outside"]
    with pytest.raises(ValueError):
        Profile.model_validate(data)
    profile.verify_on_completion = False
    assert fingerprint(profile) != original


def test_json_assignment_invalidates_passed_checks_and_binds_current_source(setup):
    from npa.agent_backend.specialists.recovery import _RejectedGeneration, _require_operations

    target, profile, config = setup
    target.write_text('{"value":"fixed"}')
    team = SpecialistTeam(config)
    team.submit("Verify", specialist="worker", task_id="task")
    tools = WorkbenchTools(profile, team.store, "task")
    calls = [
        {"id": "check", "function": {"name": "run_operation", "arguments": '{"name":"verify"}'}},
        {"id": "change", "function": {"name": "replace_json", "arguments": json.dumps({
            "path": "src/config.json", "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "json_pointer": "/value", "value": "wrong"})}},
    ]
    messages = []
    for call in calls:
        receipt = tools.execute(call)
        assert receipt["ok"]
        messages.extend([{"role": "assistant", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(receipt)}])
    report = team.task_report("task")
    assert not report["required_operations_passed"]
    assert report["required_operations"]["verify"] is None
    assert report["changes"][0]["matches_last_edit"]
    with pytest.raises(_RejectedGeneration, match="after the latest edit"):
        _require_operations(messages, ["verify"])
