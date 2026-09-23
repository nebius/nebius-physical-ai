"""Prove matched coordinator grants, durable delegation and honest failure evidence."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest


@pytest.fixture
def workflow_experiment(monkeypatch):
    directory = (
        Path(__file__).resolve().parents[3] / "npa/examples/specialists/workflows"
    )
    modules = {}
    for name in ("evidence", "workflow_bridge", "experiment"):
        spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _profile(tmp_path, name):
    workspace = tmp_path / name
    workspace.mkdir()
    (workspace / "workflow.yaml").write_text("broken: true\n")
    return {
        "name": name,
        "description": "Repair and verify one scene",
        "instructions": "Read real receipts before finishing",
        "model": "synthetic",
        "workspace": str(workspace),
        "read_paths": ["workflow.yaml"],
        "write_paths": ["workflow.yaml"],
        "required_operations": ["verify"],
        "operations": {
            "verify": {
                "argv": [
                    "{python}",
                    "-c",
                    "import sys; print('real-output'); print('diagnostic', file=sys.stderr); sys.exit(3)",
                ],
                "description": "Independent artifact verification",
            }
        },
    }


@pytest.fixture
def prepared(tmp_path):
    profiles = [_profile(tmp_path, name) for name in ("scene-a", "scene-b")]
    path = tmp_path / "team.json"
    path.write_text(
        json.dumps(
            {
                "state_directory": str(tmp_path / "state"),
                "profiles": profiles,
                "default_profile": "scene-a",
                "router": "explicit",
            }
        )
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Complete both scene workflows and verify their artifacts.")
    return path, prompt


def _bridge(modules, prepared, directory, hybrid=True):
    directory.mkdir(mode=0o700)
    return modules["workflow_bridge"]._Bridge(prepared[0], directory, hybrid)


def _write_usage(directory, usage):
    event = {"type": "turn.completed", "usage": usage}
    (directory / "codex.jsonl").write_text(json.dumps(event) + "\n")


def _enable_observations(prepared):
    config = json.loads(prepared[0].read_text())
    for profile in config["profiles"]:
        profile["operations"]["status"] = {
            "argv": [
                "{python}",
                "-c",
                "from pathlib import Path; print(Path('workflow.yaml').read_text())",
            ],
            "description": "Observe existing workflow settings",
            "observation_only": True,
        }
    prepared[0].write_text(json.dumps(config))


def _interrupted_observation(bridge):
    from npa.agent_backend.specialists.call_policy import _classification

    bridge._delegate("scene-a", "task", "Inspect workflow")
    profile = bridge.team.config.profile("scene-a")
    invocation = {"name": "run_operation", "arguments": json.dumps({"name": "status"})}
    bridge.team.store._begin_call(
        "task", "read", invocation, classification=_classification(profile, invocation)
    )
    bridge.team.store._update("task", "needs_attention", error="Original interruption")


def test_opt_in_observation_dismissal_then_takeover_preserves_remote_work(
    workflow_experiment, prepared, tmp_path, monkeypatch
):
    _enable_observations(prepared)
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    _interrupted_observation(bridge)
    with pytest.raises(RuntimeError, match="uncertain"):
        bridge._take_over("task")
    before = bridge.team.status("task")
    monkeypatch.setattr(
        workflow_experiment["workflow_bridge"].WorkbenchTools,
        "execute",
        lambda *_: pytest.fail("recovery executed a command"),
    )
    result = bridge._dismiss_observation("task", "read")
    assert result["ok"] is False and result["dismissed"] is True
    assert bridge.team.status("task")["status"] == "needs_attention"
    assert bridge._take_over("task")["remote_workloads_cancelled"] is False
    after = bridge.team.status("task")
    assert after["events"][: len(before["events"])] == before["events"]
    assert after["calls"][0]["classification"] == before["calls"][0]["classification"]


def test_fresh_declared_observation_does_not_resolve_uncertain_submission(
    workflow_experiment, prepared, tmp_path
):
    _enable_observations(prepared)
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Inspect workflow")
    bridge.team.store._begin_call("task", "uncertain-submit", {"name": "submit"})
    bridge.team.store._update("task", "needs_attention")
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "status"}))
    assert result["ok"] is True and result["stdout"] == "broken: true\n\n"
    assert bridge.team.status("task")["calls"][0]["status"] == "started"
    assert bridge.team.status("task")["status"] == "needs_attention"
    denied = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "verify"}))
    assert denied["ok"] is False
    with pytest.raises(RuntimeError, match="uncertain"):
        bridge._take_over("task")
    with bridge.team._ownership("scene-a"):
        locked = asyncio.run(
            bridge._direct("scene-a", "run_operation", {"name": "status"})
        )
    assert locked["error_type"] == "BlockingIOError"


@pytest.mark.parametrize("boundary", ["queued", "running", "changed-policy"])
def test_observation_never_bypasses_active_owner_or_original_policy(
    workflow_experiment, prepared, tmp_path, boundary
):
    _enable_observations(prepared)
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    _interrupted_observation(bridge)
    if boundary == "changed-policy":
        bridge.team.config.profiles[0].instructions += " changed"
    else:
        bridge.team.store._update("task", boundary)
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "status"}))
    assert result["ok"] is False
    assert not bridge.store._calls("coordinator-scene-a")


def test_dismissal_refuses_coordinator_uncertainty_and_active_ownership(
    workflow_experiment, prepared, tmp_path
):
    _enable_observations(prepared)
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    _interrupted_observation(bridge)
    with bridge.team._ownership("scene-a"), pytest.raises(BlockingIOError):
        bridge._dismiss_observation("task", "read")
    bridge.store._begin_call("coordinator-scene-a", "submit", {"name": "submit"})
    with pytest.raises(RuntimeError, match="uncertain"):
        bridge._dismiss_observation("task", "read")
    assert bridge.team.status("task")["calls"][0]["status"] == "started"


@pytest.mark.parametrize("observations", [False, True])
def test_mcp_grants_recovery_only_when_explicitly_configured(
    workflow_experiment, prepared, tmp_path, observations
):
    pytest.importorskip("mcp.server.mcpserver")
    if observations:
        _enable_observations(prepared)
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    module = workflow_experiment["workflow_bridge"]
    for hybrid in (False, True):
        names = {
            tool.name
            for tool in asyncio.run(
                module._server(prepared[0], directory, hybrid).list_tools()
            )
        }
        assert ("dismiss_interrupted_observation" in names) is (hybrid and observations)


def test_scope_grants_recovery_only_when_explicitly_configured(
    workflow_experiment, prepared, tmp_path
):
    _enable_observations(prepared)
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    directory.joinpath("team.json").write_bytes(prepared[0].read_bytes())
    for hybrid in (False, True):
        arm = "astra-tofa" if hybrid else "astra-only"
        settings = workflow_experiment["experiment"]._settings(
            prepared[0], directory, arm, "medium"
        )
        assert (
            "mcp_servers.workbench.tools.dismiss_interrupted_observation.approval_mode"
            in settings
        ) is hybrid
    event = {
        "type": "item.completed",
        "item": {
            "type": "mcp_tool_call",
            "server": "workbench",
            "tool": "dismiss_interrupted_observation",
        },
    }
    directory.joinpath("codex.jsonl").write_text(json.dumps(event))
    evidence = workflow_experiment["evidence"]
    assert evidence._astra_usage(directory, "astra-tofa")["matched_tool_scope"] is True
    assert evidence._astra_usage(directory, "astra-only")["matched_tool_scope"] is False
    directory.joinpath("team.json").unlink()
    assert evidence._astra_usage(directory, "astra-tofa")["matched_tool_scope"] is False


def test_mcp_storage_failure_does_not_claim_persisted_receipt(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    with bridge.store._connection() as connection:
        connection.execute(
            "CREATE TRIGGER receipt_failure BEFORE UPDATE ON calls BEGIN SELECT RAISE(ABORT, 'private diagnostic'); END"
        )
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "verify"}))
    assert result["error_type"] == "StorageFailure"
    assert result["receipt_persisted"] is False
    assert result["storage_failure"]["phase"] == "tool_receipt"
    assert "private diagnostic" not in json.dumps(result)
    assert bridge.store._calls("coordinator-scene-a")[0]["status"] == "started"
    assert bridge.store._calls("supervisor")[0]["status"] == "started"


def test_both_arms_use_same_astra_effort_and_direct_tools(
    workflow_experiment, tmp_path, prepared
):
    experiment = workflow_experiment["experiment"]
    argv = [
        experiment._astra_argv(prepared[0], tmp_path, arm, "medium")
        for arm in ("astra-only", "astra-tofa")
    ]
    for command in argv:
        assert command[command.index("-m") + 1] == "gpt-6-astra"
        assert all(
            feature in command
            for feature in ("shell_tool", "unified_exec", "multi_agent")
        )
        assert 'model_reasoning_effort="medium"' in command
        for tool in workflow_experiment["evidence"].DIRECT_TOOLS:
            assert (
                f'mcp_servers.workbench.tools.{tool}.approval_mode="approve"' in command
            )
    assert not any("tools.delegate" in value for value in argv[0])
    assert any("tools.delegate" in value for value in argv[1])


def test_real_direct_command_keeps_receipts_out_of_worker_queue(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "verify"}))
    assert result["returncode"] == 3
    assert result["stdout"] == "real-output\n"
    assert result["stderr"] == "diagnostic\n"
    assert bridge.team.store._list() == []
    receipts = bridge.store._events("coordinator-scene-a")
    assert receipts[-1]["result"] == result
    assert bridge.store._events("supervisor")[-1]["result"] == result
    assert all(
        call["status"] == "completed" for call in bridge.store._calls("supervisor")
    )


def test_delegation_is_idempotent_and_rejects_changed_goals(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    first = bridge._delegate("scene-a", "request-1", "Repair and verify")
    assert bridge._delegate("scene-a", "request-1", "Repair and verify") == first
    assert len(bridge.team.store._list()) == 1
    with pytest.raises(ValueError, match="different request"):
        bridge._delegate("scene-a", "request-1", "Different goal")
    with pytest.raises(ValueError, match="owns"):
        bridge._delegate("scene-a", "request-2", "A concurrent task")


@pytest.mark.parametrize("status", ["queued", "running", "needs_attention"])
def test_active_or_blocked_worker_prevents_coordinator_effects(
    workflow_experiment, prepared, tmp_path, status
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    bridge.team.store._update("task", status)
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "verify"}))
    assert result["ok"] is False and "owns" in result["error"]
    assert bridge.store._calls("coordinator-scene-a") == []
    assert bridge.store._events("supervisor")[-1]["result"] == result


def test_uncertain_operation_is_never_replayed_via_new_task(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge.store._begin_call("coordinator-scene-a", "interrupted", {"name": "submit"})
    result = asyncio.run(bridge._direct("scene-a", "run_operation", {"name": "verify"}))
    assert result["error_type"] == "UncertainOperation"
    with pytest.raises(RuntimeError, match="uncertain"):
        bridge._delegate("scene-a", "new-task", "Try again")
    assert len(bridge.store._calls("coordinator-scene-a")) == 1


def test_profile_lock_excludes_delegate_during_direct_operation(
    workflow_experiment, prepared, tmp_path, monkeypatch
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    entered, release = threading.Event(), threading.Event()

    def execute(*args):
        entered.set()
        assert release.wait(5)
        return {"ok": True}

    monkeypatch.setattr(
        workflow_experiment["workflow_bridge"].WorkbenchTools, "execute", execute
    )
    worker = threading.Thread(
        target=bridge._invoke, args=("scene-a", "run_operation", {"name": "verify"})
    )
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(BlockingIOError):
            bridge._delegate("scene-a", "task", "Repair")
    finally:
        release.set()
        worker.join()
    assert bridge._delegate("scene-a", "task", "Repair")["status"] == "queued"


def test_async_wait_yields_and_returns_new_receipts(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")

    async def exercise():
        waiting = asyncio.create_task(bridge._wait("task", 0, 2))
        await asyncio.sleep(0.01)
        assert not waiting.done()
        bridge.team.store._event("task", {"type": "tool", "result": {"ok": True}})
        return await waiting

    result = asyncio.run(exercise())
    assert result["changed"] is True
    assert result["last_sequence"] > 0
    assert result["events"][0]["result"]["ok"] is True


def test_observation_interval_does_not_cancel_or_stop_task(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    result = asyncio.run(bridge._wait("task", 0, 0.001))
    assert result["changed"] is False
    assert bridge.team.status("task")["status"] == "queued"
    assert bridge.team.status("task")["paused"] is False
    with pytest.raises(ValueError, match="at most 60"):
        asyncio.run(bridge._wait("task", 0, 61))


def test_takeover_keeps_receipts_and_never_operates_on_remote_workflows(
    workflow_experiment, prepared, tmp_path, monkeypatch
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    bridge.team.store._begin_call("task", "submit", {"name": "run_operation"})
    bridge.team.store._finish_call(
        "task", "submit", {"ok": True, "operation": "submit"}
    )
    bridge.team.store._event(
        "task", {"type": "tool", "call_id": "submit", "result": {"ok": True}}
    )
    bridge.team.store._update("task", "needs_attention", error="Rejected model output")
    before = bridge.team.status("task")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            workflow_experiment["workflow_bridge"].WorkbenchTools,
            "execute",
            lambda *_: pytest.fail("takeover must never run an operation"),
        )
        result = bridge._take_over("task")
    after = bridge.team.status("task")
    assert result["status"] == "cancelled"
    assert result["remote_workloads_cancelled"] is False
    assert after["calls"] == before["calls"]
    assert after["events"][:-1] == before["events"]
    assert after["events"][-1]["previous_error"] == "Rejected model output"
    read = asyncio.run(
        bridge._direct("scene-a", "read_file", {"path": "workflow.yaml"})
    )
    assert read["ok"] is True


@pytest.mark.parametrize("owner", ["specialist", "coordinator"])
def test_takeover_refuses_uncertain_effects_without_cancelling_task(
    workflow_experiment, prepared, tmp_path, owner
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    bridge.team.store._update("task", "needs_attention")
    store = bridge.team.store if owner == "specialist" else bridge.store
    identity = "task" if owner == "specialist" else "coordinator-scene-a"
    store._begin_call(identity, "interrupted", {"name": "submit"})
    with pytest.raises(RuntimeError, match="uncertain"):
        bridge._take_over("task")
    assert bridge.team.status("task")["status"] == "needs_attention"
    assert store._calls(identity)[0]["status"] == "started"


@pytest.mark.parametrize("status", ["queued", "running", "completed", "cancelled"])
def test_takeover_requires_an_explicit_failed_boundary(
    workflow_experiment, prepared, tmp_path, status
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    bridge.team.store._update("task", status)
    with pytest.raises(ValueError, match="awaiting attention"):
        bridge._take_over("task")
    assert bridge.team.status("task")["status"] == status


def test_takeover_requires_exclusive_profile_ownership(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "task", "Repair")
    bridge.team.store._update("task", "needs_attention")
    with bridge.team._ownership("scene-a"), pytest.raises(BlockingIOError):
        bridge._take_over("task")
    assert bridge.team.status("task")["status"] == "needs_attention"


def test_mcp_surface_is_matched_and_delegation_calls_are_recorded(
    workflow_experiment, prepared, tmp_path
):
    pytest.importorskip("mcp.server.mcpserver")
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    module = workflow_experiment["workflow_bridge"]
    baseline = module._server(prepared[0], directory, False)
    hybrid = module._server(prepared[0], directory, True)
    basic = {tool.name for tool in asyncio.run(baseline.list_tools())}
    extended = {tool.name for tool in asyncio.run(hybrid.list_tools())}
    assert extended - basic == {
        "delegate",
        "specialist_status",
        "wait_specialist",
        "wait_specialists",
        "take_over",
    }
    response = asyncio.run(
        hybrid.call_tool(
            "delegate",
            {
                "specialist": "scene-a",
                "task_id": "mcp-task",
                "goal": "Repair and verify",
            },
        )
    )
    assert json.loads(response.content[0].text)["status"] == "queued"
    bridge = module._Bridge(prepared[0], directory, True)
    assert bridge.store._events("supervisor")[-1]["name"] == "delegate"
    response = asyncio.run(
        hybrid.call_tool(
            "wait_specialists",
            {"task_ids": ["mcp-task"], "observation_seconds": 0.001},
        )
    )
    observed = json.loads(response.content[0].text)
    assert observed["tasks"]["mcp-task"]["status"] == "queued"
    assert observed["attention_task_ids"] == []
    assert bridge.store._events("supervisor")[-1]["name"] == "wait_specialists"


@pytest.mark.parametrize("hybrid", [False, True])
def test_mcp_read_ranges_preserve_full_file_hash(
    workflow_experiment, prepared, tmp_path, hybrid
):
    import hashlib

    pytest.importorskip("mcp.server.mcpserver")
    directory = tmp_path / "range-evidence"
    directory.mkdir(mode=0o700)
    path = tmp_path / "scene-a/workflow.yaml"
    path.write_text("first: 1\nsecond: 2\nthird: 3\n")
    server = workflow_experiment["workflow_bridge"]._server(
        prepared[0], directory, hybrid
    )
    response = asyncio.run(
        server.call_tool(
            "read_file",
            {
                "specialist": "scene-a",
                "path": "workflow.yaml",
                "start_line": 2,
                "end_line": 2,
            },
        )
    )
    observed = json.loads(response.content[0].text)
    assert observed["ok"] is True and observed["content"] == "second: 2\n"
    assert observed["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed["total_lines"] == 3


def test_supervisor_failure_still_freezes_receipts(
    workflow_experiment, prepared, tmp_path, monkeypatch
):
    experiment = workflow_experiment["experiment"]

    def fail(*args):
        raise OSError("synthetic coordinator startup failure")

    monkeypatch.setattr(experiment, "_codex", fail)
    output = tmp_path / "evidence"
    with pytest.raises(OSError):
        experiment._run(*prepared, output, "astra-only", "medium")
    execution = json.loads((output / "execution.json").read_text())
    assert execution["error_type"] == "OSError"
    assert execution["remote_workloads_cancelled"] is False
    assert execution["snapshot_errors"] == {}
    assert (output / "coordinator-receipts.json").exists()
    assert json.loads((output / "usage.json").read_text())["usage_complete"] is False


def test_hybrid_snapshots_usage_after_workers_reach_boundary(
    workflow_experiment, prepared, tmp_path, monkeypatch
):
    experiment = workflow_experiment["experiment"]
    team = experiment.SpecialistTeam(experiment.load_config(prepared[0]))

    @contextmanager
    def supervise(config):
        yield
        assert "scene-a" in team.store._paused_profiles()
        team.store._event(
            "task",
            {
                "type": "model",
                "model": "synthetic",
                "usage": {"prompt_tokens": 13, "completion_tokens": 7},
            },
        )

    def coordinator(config, directory, arm, effort, prompt):
        team.submit("Repair", specialist="scene-a", task_id="task")
        _write_usage(directory, {"input_tokens": 20, "output_tokens": 5})
        return 1

    monkeypatch.setattr(experiment, "supervise", supervise)
    monkeypatch.setattr(experiment, "_codex", coordinator)
    output = tmp_path / "evidence"
    result = experiment._run(*prepared, output, "astra-tofa", "medium")
    usage = json.loads((output / "usage.json").read_text())
    assert usage["specialists"]["responses"][0]["usage"]["prompt_tokens"] == 13
    assert usage["usage_complete"] is False
    assert result["unfinished_tasks"] == ["task"]
    frozen = json.loads((output / "coordinator-end-task-receipts.json").read_text())
    assert frozen["task"]["events"] == []


def test_usage_retains_bad_output_and_detects_extra_codex_tools(
    workflow_experiment, tmp_path
):
    (tmp_path / "codex.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"type": "item.completed", "item": {"type": "command_execution"}}
                ),
                "unfinished",
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 8, "output_tokens": 3},
                    }
                ),
            ]
        )
    )
    result = workflow_experiment["evidence"]._astra_usage(tmp_path)
    assert result["turns"][0]["input_tokens"] == 8
    assert result["malformed_lines"] == [2]
    assert result["usage_complete"] is False
    assert result["matched_tool_scope"] is False


@pytest.mark.parametrize(
    "arm,item,accepted",
    [
        ("astra-only", {"type": "agent_message"}, True),
        ("astra-only", {"type": "reasoning"}, True),
        ("astra-only", {"type": "web_search"}, False),
        ("astra-tofa", {"type": "unknown_tool"}, False),
        ("astra-tofa", None, False),
        ("astra-only", "invalid item", False),
        *[
            (
                arm,
                {"type": "mcp_tool_call", "server": server, "tool": tool},
                accepted,
            )
            for arm, server, tool, accepted in (
                ("astra-only", "workbench", "run_operations", True),
                ("astra-tofa", "workbench", "delegate", True),
                ("astra-only", "workbench", "delegate", False),
                ("astra-tofa", "external", "run_operations", False),
                ("astra-tofa", "workbench", "arbitrary_shell", False),
            )
        ],
    ],
)
def test_usage_audits_observed_tools_against_arm_grants(
    workflow_experiment, tmp_path, arm, item, accepted
):
    event = {"type": "item.completed", "item": item}
    (tmp_path / "codex.jsonl").write_text(json.dumps(event) + "\n")
    result = workflow_experiment["evidence"]._astra_usage(tmp_path, arm)
    assert result["matched_tool_scope"] is accepted
    assert result["out_of_scope_events"] == ([] if accepted else [event])


def test_unparseable_events_cannot_prove_tool_scope(workflow_experiment, tmp_path):
    (tmp_path / "codex.jsonl").write_text("truncated tool event\n")
    result = workflow_experiment["evidence"]._astra_usage(tmp_path)
    assert result["matched_tool_scope"] is False
    assert result["malformed_lines"] == [1]


def test_team_wait_ignores_routine_events_and_reports_attention_from_any_worker(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    for profile, task in (("scene-a", "first"), ("scene-b", "second")):
        bridge._delegate(profile, task, "Operate workflow")
        bridge.team.store._update(task, "running")
        bridge.team.store._event(task, {"type": "model", "usage": {}})

    async def observe():
        waiting = asyncio.create_task(bridge._wait_all(["first", "second"], {}, 0.05))
        await asyncio.sleep(0)
        assert not waiting.done(), "routine receipts must not wake the coordinator"
        bridge.team.store._update("second", "needs_attention", error="fixture blocker")
        bridge.team.store._event("second", {"type": "needs_attention"})
        return await waiting

    result = asyncio.run(observe())
    assert result["attention_task_ids"] == ["second"]
    assert result["tasks"]["first"]["status"] == "running"
    assert result["tasks"]["second"]["error"] == "fixture blocker"
    assert all("events" not in state for state in result["tasks"].values())
    assert bridge.team.store._calls("first") == []
    assert bridge.team.status("second")["status"] == "needs_attention"


def test_team_wait_consumes_terminal_cursor_without_stopping_running_work(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    bridge._delegate("scene-a", "finished", "Operate workflow")
    bridge.team.store._update("finished", "completed")
    bridge.team.store._event("finished", {"type": "completed"})
    bridge._delegate("scene-b", "active", "Operate workflow")
    bridge.team.store._update("active", "running")
    first = asyncio.run(bridge._wait_all(["finished", "active"], {}, 0.01))
    assert first["attention_task_ids"] == ["finished"]
    bridge.team.store._event("active", {"type": "tool", "result": {"ok": True}})
    second = asyncio.run(
        bridge._wait_all(["finished", "active"], first["after_sequences"], 0.01)
    )
    assert second["attention_task_ids"] == []
    assert second["after_sequences"]["active"] > first["after_sequences"]["active"]
    assert bridge.team.status("active")["status"] == "running"


@pytest.mark.parametrize(
    "tasks,cursors,seconds",
    [
        ([], {}, 1),
        (["a", "a"], {}, 1),
        (["a"], {"b": 0}, 1),
        (["a"], {"a": -1}, 1),
        (["a"], {"a": True}, 1),
        (["a"], {}, 0),
        (["a"], {}, 61),
    ],
)
def test_team_wait_rejects_invalid_observation_requests(
    workflow_experiment, prepared, tmp_path, tasks, cursors, seconds
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    with pytest.raises(ValueError):
        asyncio.run(bridge._wait_all(tasks, cursors, seconds))


def _source_receipt(bridge):
    from npa.agent_backend.specialists.tools import WorkbenchTools

    bridge._delegate("scene-a", "task", "Review source")
    profile = bridge.team.config.profile("scene-a")
    (profile.workspace / "workflow.yaml").write_text("source: " + "α" * 10000)
    executor = WorkbenchTools(profile, bridge.team.store, "task")
    receipt = executor.execute(
        {
            "id": "source-read",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "workflow.yaml"}),
            },
        }
    )
    assert receipt["ok"] is True
    return receipt, executor


def test_status_omits_source_text_without_losing_receipts_or_uncertain_effects(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    receipt, executor = _source_receipt(bridge)
    failed = executor.execute(
        {
            "id": "check",
            "function": {"name": "run_operation", "arguments": '{"name":"verify"}'},
        }
    )
    bridge.team.store._begin_call("task", "unknown", {"name": "submit"})
    original = bridge.team.status("task")["events"]
    compact = bridge._status("task")
    full = bridge._status("task", include_read_content=True)
    reads = [event for event in compact["events"] if event.get("name") == "read_file"]
    assert reads[0]["result"] == {
        **{key: value for key, value in receipt.items() if key != "content"},
        "content_omitted": True,
        "content_bytes": len(receipt["content"].encode("utf-8")),
    }
    operations = [
        event for event in compact["events"] if event.get("name") == "run_operation"
    ]
    assert operations[0]["result"] == failed
    assert failed["returncode"] == 3 and failed["stderr"] == "diagnostic\n"
    assert compact["uncertain_calls"] == full["uncertain_calls"]
    assert compact["uncertain_calls"][0]["call_id"] == "unknown"
    assert compact["last_sequence"] == full["last_sequence"]
    assert len(json.dumps(compact)) < len(json.dumps(full)) / 3
    assert full["events"] == original == bridge.team.status("task")["events"]


def test_status_cursor_and_single_wait_preserve_compact_source_metadata(
    workflow_experiment, prepared, tmp_path
):
    bridge = _bridge(workflow_experiment, prepared, tmp_path / "evidence")
    _source_receipt(bridge)
    first = bridge._status("task")
    waited = asyncio.run(bridge._wait("task", 0, 0.01))
    assert waited["events"] == first["events"]
    assert bridge._status("task", first["last_sequence"])["events"] == []
    bridge.team.store._event("task", {"type": "needs_attention", "error": "failure"})
    later = bridge._status("task", first["last_sequence"])
    assert len(later["events"]) == 1
    assert later["events"][0]["error"] == "failure"


def test_mcp_status_requires_explicit_opt_in_for_historical_source_content(
    workflow_experiment, prepared, tmp_path
):
    pytest.importorskip("mcp.server.mcpserver")
    directory = tmp_path / "evidence"
    bridge = _bridge(workflow_experiment, prepared, directory)
    receipt, _ = _source_receipt(bridge)
    server = workflow_experiment["workflow_bridge"]._server(
        prepared[0], directory, True
    )
    responses = []
    for include in (False, True):
        arguments = {"task_id": "task", "include_read_content": include}
        result = asyncio.run(server.call_tool("specialist_status", arguments))
        events = json.loads(result.content[0].text)["events"]
        responses.append(
            next(
                event["result"] for event in events if event.get("name") == "read_file"
            )
        )
    assert "content" not in responses[0]
    assert responses[1] == receipt
