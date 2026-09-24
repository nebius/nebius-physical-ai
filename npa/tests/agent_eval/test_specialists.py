"""Prove durable multi-specialist execution, replay safety and public tool boundaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import signal
import subprocess
import threading
import time

from fastapi.testclient import TestClient
import pytest

from npa.agent_backend.specialists.config import Operation, Profile, TeamConfig
from npa.agent_backend.specialists.service import create_app
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


def _response(
    model, *, text="", tool="", arguments=None, call_id="call-one", finish=None
):
    message = {"role": "assistant", "content": text}
    if tool:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments or {})},
            }
        ]
    return {
        "id": "synthetic-response",
        "model": model,
        "choices": [
            {
                "message": message,
                "finish_reason": finish or ("tool_calls" if tool else "stop"),
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


class _Client:
    def __init__(self, responses, barrier=None):
        self.responses = iter(responses)
        self.calls = []
        self.barrier = barrier

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.barrier:
            self.barrier.wait(timeout=10)
        return next(self.responses)


@pytest.fixture
def configuration(tmp_path):
    profiles = []
    for name in ("simulation", "data"):
        workspace = tmp_path / name
        (workspace / "src").mkdir(parents=True)
        (workspace / "src/value.txt").write_text("broken\n")
        profiles.append(
            Profile(
                name=name,
                description="Inspect and improve " + name,
                model="synthetic/" + name,
                workspace=workspace,
                read_paths=["src"],
                write_paths=["src"],
                operations={
                    "check": Operation(
                        argv=[
                            "{python}",
                            "-c",
                            "from pathlib import Path; assert Path('src/value.txt').read_text() == 'fixed\\n'",
                        ],
                        description="Check the repaired value",
                    )
                },
            )
        )
    return TeamConfig(
        state_directory=tmp_path / "state",
        profiles=profiles,
        default_profile="simulation",
    )


def _team(configuration, responses):
    return SpecialistTeam(configuration, clients={"simulation": _Client(responses)})


def test_tool_checkpoint_survives_new_coordinator(configuration):
    model = configuration.profiles[0].model
    client = _Client(
        [
            _response(model, tool="read_file", arguments={"path": "src/value.txt"}),
            _response(model, text="Read the source successfully."),
        ]
    )
    first = SpecialistTeam(configuration, clients={"simulation": client})
    first.submit("Inspect the source", task_id="restart", specialist="simulation")
    assert first.work_once("simulation")["status"] == "running"
    assert first.store._calls("restart") == []
    second = SpecialistTeam(configuration, clients={"simulation": client})
    second.work_once("simulation")
    assert len(client.calls) == 1
    assert second.work_once("simulation")["status"] == "completed"
    assert len(client.calls) == 2
    assert (
        json.loads(client.calls[-1]["messages"][-1]["content"])["content"] == "broken\n"
    )


def test_real_edit_command_and_patch_complete(configuration):
    model = configuration.profiles[0].model
    responses = [
        _response(
            model,
            tool="edit_file",
            arguments={
                "path": "src/value.txt",
                "expected_sha256": hashlib.sha256(b"broken\n").hexdigest(),
                "old": "broken",
                "new": "fixed",
            },
        ),
        _response(
            model,
            tool="run_operation",
            arguments={"name": "check"},
            call_id="check-one",
        ),
        _response(model, text="Fixed and checked the source."),
    ]
    team = _team(configuration, responses)
    team.submit("Repair and test", task_id="repair", specialist="simulation")
    for _ in range(5):
        result = team.work_once("simulation")
    assert result["status"] == "completed"
    assert (
        configuration.profiles[0].workspace.joinpath("src/value.txt").read_text()
        == "fixed\n"
    )
    assert "-broken\n+fixed\n" in team.patch("repair")
    operations = [
        event for event in team.status("repair")["events"] if event["type"] == "tool"
    ]
    assert operations[-1]["result"]["returncode"] == 0


def test_specialists_reach_model_concurrently(configuration):
    barrier = threading.Barrier(2)
    clients = {
        profile.name: _Client([_response(profile.model, text=profile.name)], barrier)
        for profile in configuration.profiles
    }
    team = SpecialistTeam(configuration, clients=clients)
    for profile in configuration.profiles:
        team.submit("Inspect", specialist=profile.name, task_id=profile.name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(team.work_once, ["simulation", "data"]))
    assert all(result["status"] == "completed" for result in results)


def test_completed_tool_receipt_prevents_reexecution_after_checkpoint_gap(
    configuration, monkeypatch
):
    model = configuration.profiles[0].model
    team = _team(
        configuration,
        [_response(model, tool="run_operation", arguments={"name": "check"})],
    )
    team.submit("Check", specialist="simulation", task_id="gap")
    team.work_once("simulation")
    call = _response(model, tool="run_operation", arguments={"name": "check"})[
        "choices"
    ][0]["message"]["tool_calls"][0]
    executor = WorkbenchTools(configuration.profiles[0], team.store, "gap")
    expected = executor.execute(call)
    monkeypatch.setattr(
        executor.__class__,
        "_run_operation",
        lambda *_: pytest.fail("operation replayed"),
    )
    team.work_once("simulation")
    assert team.store._calls("gap")[0]["status"] == "completed"
    assert expected["returncode"] == 1


def test_uncertain_call_requires_operator_receipt(configuration):
    _backup(configuration.profiles[0])
    model = configuration.profiles[0].model
    response = _response(model, tool="run_operation", arguments={"name": "check"})
    team = _team(configuration, [response, _response(model, text="Reconciled")])
    team.submit("Check", specialist="simulation", task_id="uncertain")
    team.work_once("simulation")
    call = response["choices"][0]["message"]["tool_calls"][0]
    invocation = {"name": "run_operation", "arguments": call["function"]["arguments"]}
    team.store._begin_call("uncertain", "call-one", invocation)
    assert team.work_once("simulation")["status"] == "needs_attention"
    with pytest.raises(ValueError, match="reconcile"):
        team.pause(task_id="uncertain", paused=False)
    team.reconcile(
        "uncertain", call_id="call-one", result={"ok": True, "returncode": 0}
    )
    team.work_once("simulation")
    assert team.work_once("simulation")["status"] == "completed"


def test_duplicate_id_does_not_reroute_or_infer(configuration, monkeypatch):
    team = _team(configuration, [])
    first = team.submit("Inspect", task_id="same")
    monkeypatch.setattr(
        team, "_route", lambda *_: pytest.fail("duplicate routed again")
    )
    assert team.submit("Inspect", task_id="same") == first
    with pytest.raises(ValueError, match="different request"):
        team.submit("Changed goal", task_id="same")


def test_paused_profile_and_task_do_not_block_other_specialist(configuration):
    clients = {
        item.name: _Client([_response(item.model, text="Done")])
        for item in configuration.profiles
    }
    team = SpecialistTeam(configuration, clients=clients)
    team.submit("Inspect", specialist="simulation", task_id="paused")
    team.submit("Inspect next", specialist="simulation", task_id="next")
    team.submit("Inspect", specialist="data", task_id="independent")
    team.pause(task_id="paused")
    assert team.work_once("simulation") is None
    assert team.work_once("data")["status"] == "completed"
    team.pause(task_id="paused", paused=False)
    team.pause(specialist="simulation")
    assert team.work_once("simulation") is None
    team.pause(specialist="simulation", paused=False)
    assert team.work_once("simulation")["status"] == "completed"


@pytest.mark.parametrize(
    "mutation", ["model", "length", "empty", "invalid_arguments", "missing_choices"]
)
def test_unusable_model_response_never_executes_tools(configuration, mutation):
    model = configuration.profiles[0].model
    response = _response(model, tool="run_operation", arguments={"name": "check"})
    if mutation == "model":
        response["model"] = "different/model"
    elif mutation == "length":
        response["choices"][0]["finish_reason"] = "length"
    elif mutation == "empty":
        response = _response(model, text="")
    elif mutation == "invalid_arguments":
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            "{"
        )
    else:
        del response["choices"]
    response["usage"]["prompt_tokens_details"] = {"cached_tokens": 10}
    team = _team(configuration, [response])
    team.submit("Check", specialist="simulation", task_id="invalid")
    assert team.work_once("simulation")["status"] == "needs_attention"
    assert team.store._calls("invalid") == []
    events = [
        event for event in team.status("invalid")["events"] if event["type"] == "model"
    ]
    assert len(events) == 1
    event = events[0]
    assert event["accepted"] is False
    assert event["tools"] == []
    assert event["model"] == response["model"]
    assert event["usage"]["total_tokens"] == 25
    assert event["usage"]["cached_tokens"] == 10
    assert event["finish_reason"] == response.get("choices", [{}])[0].get(
        "finish_reason"
    )


@pytest.mark.parametrize(
    "path",
    ["../outside", "/etc/passwd", ".git/config", "outside.txt", "src/../../outside"],
)
def test_tool_paths_cannot_escape_grants(configuration, path):
    team = _team(configuration, [])
    tools = WorkbenchTools(configuration.profiles[0], team.store, "scope")
    call = _response("test", tool="read_file", arguments={"path": path})["choices"][0][
        "message"
    ]["tool_calls"][0]
    assert tools.execute(call)["ok"] is False


def test_symlinks_hardlinks_stale_edits_and_unknown_operations_are_rejected(
    configuration, tmp_path
):
    profile = configuration.profiles[0]
    external = tmp_path / "outside.txt"
    external.write_text("private")
    (profile.workspace / "src/link.txt").symlink_to(external)
    os.link(external, profile.workspace / "src/hard.txt")
    team = _team(configuration, [])
    executor = WorkbenchTools(profile, team.store, "scope")
    invocations = [
        ("read_file", {"path": "src/link.txt"}),
        ("read_file", {"path": "src/hard.txt"}),
        (
            "edit_file",
            {
                "path": "src/value.txt",
                "expected_sha256": "stale",
                "old": "broken",
                "new": "fixed",
            },
        ),
        ("run_operation", {"name": "shell"}),
    ]
    for index, (tool, arguments) in enumerate(invocations):
        call = _response(
            "test", tool=tool, arguments=arguments, call_id=f"call-{index}"
        )["choices"][0]["message"]["tool_calls"][0]
        assert executor.execute(call)["ok"] is False
    assert external.read_text() == "private"


def test_changed_profile_does_not_reuse_old_grants(configuration):
    team = _team(configuration, [])
    team.submit("Inspect", task_id="policy", specialist="simulation")
    configuration.profiles[0].write_paths.append("docs")
    assert team.work_once("simulation")["status"] == "needs_attention"


def test_followup_and_authentication_share_real_coordinator(configuration, monkeypatch):
    team = _team(configuration, [])
    monkeypatch.setenv("NPA_SPECIALISTS_TOKEN", "synthetic-service-credential-long")
    app = create_app("unused", with_workers=False, team=team)
    headers = {"Authorization": "Bearer synthetic-service-credential-long"}
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert client.post("/api/tasks", json={"goal": "inspect"}).status_code == 401
        result = client.post(
            "/api/tasks",
            headers=headers,
            json={"goal": "inspect", "task_id": "http-task"},
        )
        assert result.status_code == 200
        assert client.post(
            "/api/tasks/http-task/pause", headers=headers, json={"paused": True}
        ).json()["paused"]
        assert client.get("/api/tasks/missing", headers=headers).status_code == 404
        team.store._update("http-task", "completed", result="Verified original result")
        result = client.post(
            "/api/tasks",
            headers=headers,
            json={"goal": "continue", "parent_id": "http-task"},
        )
        assert "Verified original result" in result.json()["goal"]
        assert (
            "synthetic-service-credential-long"
            not in client.get("/api/status", headers=headers).text
        )


def test_jev_fallback_and_explicit_assignment(configuration, monkeypatch):
    configuration.router = "jev"
    team = _team(configuration, [])
    calls = []

    def classify(text, candidates, **kwargs):
        calls.append((text, candidates))
        return {"status": "accepted", "selected_model": "data"}

    monkeypatch.setattr(
        "npa.agent_backend.specialists.team.classify_generation_model", classify
    )
    assert team.submit("data task")["profile"] == "data"
    assert team.submit("explicit", specialist="simulation")["profile"] == "simulation"
    assert len(calls) == 1
    monkeypatch.setattr(
        "npa.agent_backend.specialists.team.classify_generation_model",
        lambda *a, **k: {"status": "unavailable"},
    )
    assert team.submit("fallback")["profile"] == "simulation"


def test_multi_call_response_runs_each_effect_at_its_own_checkpoint(configuration):
    model = configuration.profiles[0].model
    response = _response(model, tool="read_file", arguments={"path": "src/value.txt"})
    second = _response(
        model, tool="run_operation", arguments={"name": "check"}, call_id="second"
    )["choices"][0]["message"]["tool_calls"][0]
    response["choices"][0]["message"]["tool_calls"].append(second)
    client = _Client([response, _response(model, text="Both results inspected")])
    team = SpecialistTeam(configuration, clients={"simulation": client})
    team.submit("Read and check", specialist="simulation", task_id="batch")
    team.work_once("simulation")
    team.work_once("simulation")
    assert len(team.store._calls("batch")) == 1
    restarted = SpecialistTeam(configuration, clients={"simulation": client})
    restarted.work_once("simulation")
    assert len(team.store._calls("batch")) == 2
    assert len(client.calls) == 1
    assert restarted.work_once("simulation")["status"] == "completed"


def test_profile_file_lock_rejects_concurrent_worker(configuration):
    team = _team(configuration, [])
    with team._ownership("simulation"):
        with pytest.raises(BlockingIOError):
            team.work_once("simulation")
    assert team.work_once("simulation") is None


def test_duplicate_call_ids_rejected_before_any_effect(configuration):
    model = configuration.profiles[0].model
    response = _response(model, tool="read_file", arguments={"path": "src/value.txt"})
    message = response["choices"][0]["message"]
    message["tool_calls"].append(message["tool_calls"][0].copy())
    team = _team(configuration, [response])
    team.submit("Inspect", specialist="simulation", task_id="duplicate-call")
    assert team.work_once("simulation")["status"] == "needs_attention"
    assert team.store._calls("duplicate-call") == []


def test_operation_environment_does_not_implicitly_forward_model_credentials(
    configuration, monkeypatch
):
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "private-fixture-value")
    profile = configuration.profiles[0]
    profile.operations["environment"] = Operation(
        argv=[
            "{python}",
            "-c",
            "import os; assert 'NEBIUS_TOKEN_FACTORY_KEY' not in os.environ",
        ],
        description="Assert credential isolation",
    )
    team = _team(configuration, [])
    call = _response(
        profile.model, tool="run_operation", arguments={"name": "environment"}
    )["choices"][0]["message"]["tool_calls"][0]
    assert (
        WorkbenchTools(profile, team.store, "environment").execute(call)["returncode"]
        == 0
    )


def test_cancel_during_model_call_cannot_be_overwritten(configuration):
    entered, release = threading.Event(), threading.Event()
    model = configuration.profiles[0].model

    class SlowClient:
        def chat_completion(self, **kwargs):
            entered.set()
            assert release.wait(10)
            return _response(model, tool="run_operation", arguments={"name": "check"})

    team = SpecialistTeam(configuration, clients={"simulation": SlowClient()})
    team.submit("Check", specialist="simulation", task_id="cancel-inflight")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(team.work_once, "simulation")
        assert entered.wait(10)
        team.cancel("cancel-inflight")
        release.set()
        assert pending.result()["status"] == "cancelled"
    assert team.work_once("simulation") is None
    assert team.store._calls("cancel-inflight") == []


def test_listing_stays_inside_scope_and_ignores_symlinks(configuration, tmp_path):
    profile = configuration.profiles[0]
    (profile.workspace / "src/link").symlink_to(tmp_path, target_is_directory=True)
    team = _team(configuration, [])
    call = _response(profile.model, tool="list_files", arguments={"path": "src"})[
        "choices"
    ][0]["message"]["tool_calls"][0]
    result = WorkbenchTools(profile, team.store, "list").execute(call)
    assert result["paths"] == ["src/value.txt"]


def test_patch_without_trailing_newline_can_be_applied_by_git(configuration):
    profile = configuration.profiles[0]
    path = profile.workspace / "src/value.txt"
    path.write_text("broken")
    team = _team(configuration, [])
    team.submit("Repair", specialist="simulation", task_id="patch-check")
    call = _response(
        profile.model,
        tool="edit_file",
        arguments={
            "path": "src/value.txt",
            "expected_sha256": hashlib.sha256(b"broken").hexdigest(),
            "old": "broken",
            "new": "fixed",
        },
    )["choices"][0]["message"]["tool_calls"][0]
    assert WorkbenchTools(profile, team.store, "patch-check").execute(call)["ok"]
    path.write_text("broken")
    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=team.patch("patch-check"),
        cwd=profile.workspace,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_uses_shared_task_store_and_returns_json(configuration, tmp_path):
    from typer.testing import CliRunner
    from npa.cli.main import app

    path = tmp_path / "team.json"
    path.write_text(configuration.model_dump_json())
    prefix = ["workbench", "specialists", "--config", str(path)]
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            *prefix,
            "submit",
            "Inspect",
            "--specialist",
            "simulation",
            "--task-id",
            "cli-task",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["id"] == "cli-task"
    result = runner.invoke(app, [*prefix, "cancel", "cli-task"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["status"] == "cancelled"
    assert SpecialistTeam(configuration).status("cli-task")["status"] == "cancelled"


@pytest.mark.parametrize(
    "calls", [[None], [{"id": "call", "function": None}], {"id": "call"}]
)
def test_malformed_tool_calls_surface_attention_without_worker_crash(
    configuration, calls
):
    model = configuration.profiles[0].model
    response = _response(model, tool="read_file", arguments={"path": "src/value.txt"})
    response["choices"][0]["message"]["tool_calls"] = calls
    team = _team(configuration, [response])
    team.submit("Inspect", specialist="simulation", task_id="malformed")
    assert team.work_once("simulation")["status"] == "needs_attention"
    assert team.store._calls("malformed") == []


def _wait_for_workers(team, previous=None):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        workers = team.store._workers()
        if len(workers) == 2 and all(
            previous is None or worker["pid"] != previous[name]["pid"]
            for name, worker in workers.items()
        ):
            return workers
        time.sleep(0.05)
    pytest.fail("supervised workers did not report new heartbeats")


def test_supervisor_restarts_real_workers_and_cleans_up(configuration, tmp_path):
    from npa.agent_backend.specialists.worker import supervise

    path = tmp_path / "team.json"
    path.write_text(configuration.model_dump_json())
    team = SpecialistTeam(configuration)
    with supervise(str(path)):
        first = _wait_for_workers(team)
        for worker in first.values():
            os.kill(worker["pid"], signal.SIGTERM)
        second = _wait_for_workers(team, first)
    for worker in second.values():
        with pytest.raises(ProcessLookupError):
            os.kill(worker["pid"], 0)


def test_supervisor_startup_failure_propagates(configuration, tmp_path, monkeypatch):
    from npa.agent_backend.specialists.worker import supervise

    path = tmp_path / "team.json"
    path.write_text(configuration.model_dump_json())

    def reject_start(*args, **kwargs):
        raise OSError("process creation failed")

    monkeypatch.setattr(subprocess, "Popen", reject_start)
    with pytest.raises(OSError, match="process creation failed"):
        with supervise(str(path)):
            pytest.fail("service reported ready without starting workers")


def _backup(profile):
    from npa.agent_backend.specialists.config import ModelEndpoint

    endpoint = ModelEndpoint(
        model="synthetic/backup",
        base_url="http://localhost:8089/v1",
        key_env="BACKUP_INFERENCE_KEY",
        model_options={"temperature": 1, "top_p": 0.95},
    )
    profile.fallback_models = [endpoint]
    return endpoint


def _rejected_response(model, failure):
    response = _response(
        model, tool="run_operation", arguments={"name": "check"}, finish="length"
    )
    if failure == "arguments":
        response["choices"][0]["finish_reason"] = "tool_calls"
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
            "{"
        )
    elif failure == "empty":
        response = _response(model)
    return response


@pytest.mark.parametrize("failure", ["length", "arguments", "empty"])
def test_rejected_generation_checkpoints_backup_and_preserves_tools(
    configuration, failure
):
    profile = configuration.profiles[0]
    backup = _backup(profile)
    rejected = _rejected_response(profile.model, failure)
    client = _Client(
        [
            rejected,
            _response(
                backup.model, tool="read_file", arguments={"path": "src/value.txt"}
            ),
            _response(backup.model, text="Source inspected"),
        ]
    )
    team = SpecialistTeam(configuration, clients={"simulation": client})
    team.submit("Inspect", specialist="simulation", task_id="recover")
    assert team.work_once("simulation")["status"] == "running"
    assert team.store._calls("recover") == []
    restarted = SpecialistTeam(configuration, clients={"simulation": client})
    for _ in range(3):
        result = restarted.work_once("simulation")
    assert result["status"] == "completed"
    expected_models = [profile.model, backup.model, backup.model]
    assert [call["model"] for call in client.calls] == expected_models
    assert client.calls[1]["extra"]["temperature"] == 1
    assert client.calls[0]["extra"]["tools"] == client.calls[1]["extra"]["tools"]
    events = restarted.status("recover")["events"]
    assert sum(e["usage"]["total_tokens"] for e in events if e["type"] == "model") == 75
    handoff = next(e for e in events if e["type"] == "model_fallback")
    assert (handoff["from_model"], handoff["to_model"]) == (profile.model, backup.model)
    assert len(restarted.store._calls("recover")) == 1


def test_exhausted_backups_preserve_all_rejected_usage(configuration):
    profile = configuration.profiles[0]
    backup = _backup(profile)
    team = _team(
        configuration,
        [_response(model, finish="length") for model in (profile.model, backup.model)],
    )
    team.submit("Inspect", specialist="simulation", task_id="exhausted")
    assert team.work_once("simulation")["status"] == "running"
    assert team.work_once("simulation")["status"] == "needs_attention"
    assert team.store._calls("exhausted") == []
    events = [e for e in team.status("exhausted")["events"] if e["type"] == "model"]
    assert len(events) == 2 and all(not e["accepted"] for e in events)
    assert sum(e["usage"]["total_tokens"] for e in events) == 50


@pytest.mark.parametrize(
    "failure", ["identity", "content_filter", "refusal", "truncated_refusal"]
)
def test_integrity_and_refusal_failures_never_use_backup(configuration, failure):
    profile = configuration.profiles[0]
    _backup(profile)
    response = _response(profile.model, text="No")
    if failure == "identity":
        response["model"] = "unexpected/model"
    elif failure in {"refusal", "truncated_refusal"}:
        response["choices"][0]["message"]["refusal"] = "No"
        if failure == "truncated_refusal":
            response["choices"][0]["finish_reason"] = "length"
    else:
        response["choices"][0]["finish_reason"] = failure
    team = _team(configuration, [response])
    team.submit("Inspect", specialist="simulation", task_id="terminal")
    assert team.work_once("simulation")["status"] == "needs_attention"
    assert not any(
        e["type"] == "model_fallback" for e in team.status("terminal")["events"]
    )


def test_backup_uses_its_own_endpoint_and_credential(configuration, monkeypatch):
    from npa.agent_backend.specialists import graph

    profile = configuration.profiles[0]
    backup = _backup(profile)
    monkeypatch.setenv(profile.key_env, "synthetic-primary-secret")
    monkeypatch.setenv(backup.key_env, "synthetic-backup-secret")
    configs = []
    responses = iter(
        [
            _response(profile.model, finish="length"),
            _response(backup.model, text="Done"),
        ]
    )

    def client(*, config, retry_attempts):
        assert retry_attempts == 1
        configs.append(config)
        return _Client([next(responses)])

    monkeypatch.setattr(graph, "TokenFactoryClient", client)
    team = SpecialistTeam(configuration)
    team.submit("Inspect", specialist="simulation", task_id="endpoints")
    team.work_once("simulation")
    assert team.work_once("simulation")["status"] == "completed"
    assert [c.base_url for c in configs] == [profile.base_url, backup.base_url]
    assert [c.api_key for c in configs] == [
        "synthetic-primary-secret",
        "synthetic-backup-secret",
    ]


def test_completion_gate_hands_false_success_to_backup(configuration):
    profile = configuration.profiles[0]
    profile.required_operations = ["check"]
    profile.workspace.joinpath("src/value.txt").write_text("fixed\n")
    backup = _backup(profile)
    team = _team(
        configuration,
        [
            _response(profile.model, text="Checks passed"),
            _response(backup.model, tool="run_operation", arguments={"name": "check"}),
            _response(backup.model, text="Check actually passed"),
        ],
    )
    team.submit("Verify", specialist="simulation", task_id="gate")
    for _ in range(4):
        result = team.work_once("simulation")
    assert result["status"] == "completed"
    events = team.status("gate")["events"]
    assert next(e for e in events if e["type"] == "model")["accepted"] is False
    assert "check" in next(e for e in events if e["type"] == "model_fallback")["reason"]


def test_completion_gate_invalidates_old_and_failed_checks():
    from npa.agent_backend.specialists.recovery import _require_operations

    messages = []

    def receipt(tool, arguments, result):
        response = _response(
            "synthetic", tool=tool, arguments=arguments, call_id=str(len(messages))
        )
        message = response["choices"][0]["message"]
        messages.extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0]["id"],
                    "content": json.dumps(result),
                },
            ]
        )

    receipt("run_operation", {"name": "check"}, {"ok": True, "returncode": 0})
    _require_operations(messages, ["check"])
    receipt("edit_file", {}, {"ok": True})
    with pytest.raises(ValueError, match="check"):
        _require_operations(messages, ["check"])
    receipt("run_operation", {"name": "check"}, {"ok": True, "returncode": 0})
    _require_operations(messages, ["check"])
    receipt("run_operation", {"name": "check"}, {"ok": False, "returncode": 1})
    with pytest.raises(ValueError, match="check"):
        _require_operations(messages, ["check"])


def test_legacy_fingerprint_and_new_recovery_policy_binding(configuration):
    from npa.agent_backend.specialists.config import fingerprint

    profile = configuration.profiles[0]
    legacy = profile.model_dump(
        mode="json",
        exclude={
            "fallback_models",
            "required_operations",
            "model_router",
            "require_model_route",
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
    team = _team(configuration, [])
    team.submit("Inspect", specialist="simulation", task_id="changed-recovery")
    _backup(profile)
    assert fingerprint(profile) != expected
    assert team.work_once("simulation")["status"] == "needs_attention"


@pytest.mark.parametrize(
    "options",
    [
        {"temperature": True},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"top_p": 1.1},
        {"tools": []},
    ],
)
def test_sampling_and_endpoint_policy_validation(configuration, options):
    from npa.agent_backend.specialists.config import ModelEndpoint

    with pytest.raises(ValueError):
        ModelEndpoint(model="synthetic/model", model_options=options)
    with pytest.raises(ValueError):
        ModelEndpoint(model="synthetic/model", write_paths=["src"])
    with pytest.raises(ValueError, match="required_operations"):
        Profile.model_validate(
            {
                **configuration.profiles[0].model_dump(),
                "required_operations": ["missing"],
            }
        )


def test_completion_uses_recovered_receipt_when_event_was_not_written(
    configuration, monkeypatch
):
    profile = configuration.profiles[0]
    profile.required_operations = ["check"]
    profile.workspace.joinpath("src/value.txt").write_text("fixed\n")
    response = _response(
        profile.model, tool="run_operation", arguments={"name": "check"}
    )
    team = _team(configuration, [response, _response(profile.model, text="Verified")])
    team.submit("Check", specialist="simulation", task_id="receipt-gap")
    team.work_once("simulation")
    executor = WorkbenchTools(profile, team.store, "receipt-gap")
    call = response["choices"][0]["message"]["tool_calls"][0]
    executor.execute(call)
    with team.store._connection() as connection:
        connection.execute("DELETE FROM events WHERE task_id=?", ("receipt-gap",))
    monkeypatch.setattr(
        WorkbenchTools, "_run_operation", lambda *_: pytest.fail("effect repeated")
    )
    team.work_once("simulation")
    assert team.work_once("simulation")["status"] == "completed"


def test_cross_model_handoff_preserves_receipts_without_provider_reasoning(
    configuration,
):
    profile = configuration.profiles[0]
    backup = _backup(profile)
    read = _response(
        profile.model, tool="read_file", arguments={"path": "src/value.txt"}
    )
    read["choices"][0]["message"]["reasoning_content"] = (
        "Provider-specific hidden state"
    )
    client = _Client(
        [
            read,
            _response(profile.model, finish="length"),
            _response(backup.model, text="Inspected"),
        ]
    )
    team = SpecialistTeam(configuration, clients={"simulation": client})
    team.submit("Inspect", specialist="simulation", task_id="handoff-history")
    for _ in range(4):
        result = team.work_once("simulation")
    assert result["status"] == "completed"
    messages = client.calls[-1]["messages"]
    assert all("reasoning_content" not in message for message in messages)
    receipt = next(message for message in messages if message["role"] == "tool")
    assert json.loads(receipt["content"])["content"] == "broken\n"
    assert receipt["tool_call_id"] == "call-one"
    assert len(team.store._calls("handoff-history")) == 1


def test_cached_old_operation_receipt_cannot_certify_a_later_edit():
    from npa.agent_backend.specialists.recovery import _require_operations

    check = _response(
        "synthetic",
        tool="run_operation",
        arguments={"name": "check"},
        call_id="old-check",
    )["choices"][0]["message"]
    edit = _response("synthetic", tool="edit_file", call_id="new-edit")["choices"][0][
        "message"
    ]
    check_receipt = {
        "role": "tool",
        "tool_call_id": "old-check",
        "content": json.dumps({"ok": True, "returncode": 0}),
    }
    edit_receipt = {
        "role": "tool",
        "tool_call_id": "new-edit",
        "content": json.dumps({"ok": True}),
    }
    messages = [check, check_receipt, edit, edit_receipt, check, check_receipt]
    with pytest.raises(ValueError, match="check"):
        _require_operations(messages, ["check"])
