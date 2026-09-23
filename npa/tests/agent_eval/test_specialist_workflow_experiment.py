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


def test_both_arms_use_same_astra_effort_and_direct_tools(
    workflow_experiment, tmp_path
):
    experiment = workflow_experiment["experiment"]
    argv = [
        experiment._astra_argv(tmp_path / "team.json", tmp_path, arm, "medium")
        for arm in ("astra-only", "astra-tofa")
    ]
    for command in argv:
        assert command[command.index("-m") + 1] == "gpt-6-astra"
        assert all(
            feature in command
            for feature in ("shell_tool", "unified_exec", "multi_agent")
        )
        assert 'model_reasoning_effort="medium"' in command
        for tool in experiment.DIRECT_TOOLS:
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
