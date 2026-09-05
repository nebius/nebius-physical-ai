"""Exercise the rendered session's actual cleanup functions without a GPU."""
from __future__ import annotations

import ast
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import pytest
import yaml


class ProcessError(Exception):
    pass


class ProcessGone(ProcessError):
    pass


class Child:
    def __init__(self, pid, *, exit_on="kill", zombie=False):
        self.pid = pid
        self.created = float(pid)
        self.running = True
        self.exit_on = exit_on
        self.zombie = zombie
        self.actions = []
        self.terminate_error = False
        self.gone_during_discovery = False

    def create_time(self):
        if self.gone_during_discovery:
            raise ProcessGone()
        return self.created

    def is_running(self):
        return self.running

    def status(self):
        return "zombie" if self.zombie else "running"

    def terminate(self):
        self.actions.append("terminate")
        if self.terminate_error:
            raise ProcessError()
        if self.exit_on == "terminate":
            self.running = False

    def kill(self):
        self.actions.append("kill")
        if self.exit_on == "kill":
            self.running = False


@pytest.fixture
def session():
    workflow = Path(__file__).parents[2] / "workflows/workbench/npa-workflows/ray-clip-development-session.yaml"
    shell = yaml.safe_load(workflow.read_text())["states"]["application-session"]["run"]["shell"]
    bootstrap = shell.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    parsed = ast.parse(bootstrap)
    names = {"remember_children", "alive_owned_children", "shutdown_owned_processes"}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    finalbody = next(node.finalbody for node in parsed.body if isinstance(node, ast.Try) and node.finalbody)
    children = []
    waits = []
    published = []
    cli_events = []
    cli = SimpleNamespace(pid=101, returncode=None)
    cli.poll = lambda: cli.returncode
    cli.send_signal = lambda value: cli_events.append(("signal", value))

    def wait():
        cli_events.append(("exit", 1))
        cli.returncode = 1
        return 1

    cli.wait = wait

    def wait_procs(items, timeout):
        waits.append([item.pid for item in items])
        return [], items

    backend = SimpleNamespace(
        Process=lambda pid: SimpleNamespace(children=lambda recursive: children),
        NoSuchProcess=ProcessGone, Error=ProcessError, STATUS_ZOMBIE="zombie",
        wait_procs=wait_procs,
    )
    namespace = {
        "psutil": backend, "process": cli, "time": time, "signal": signal,
        "owned_children": {}, "receipt": {}, "requested": True, "rank": 0,
        "publish": lambda name, receipt: published.append((name, dict(receipt))),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(workflow), "exec"), namespace)
    return SimpleNamespace(namespace=namespace, children=children, waits=waits,
                           backend=backend, cli_events=cli_events, published=published,
                           finalbody=finalbody, workflow=workflow)


def test_cli_exit_waits_for_asynchronous_owned_child_exit(session):
    child = Child(201)
    unrelated = Child(301)
    session.children.append(child)

    def asynchronous_exit(items, timeout):
        assert session.cli_events[-1] == ("exit", 1)
        session.waits.append([item.pid for item in items])
        for item in items:
            item.running = False
        return items, []

    session.backend.wait_procs = asynchronous_exit
    result = session.namespace["shutdown_owned_processes"]()
    assert session.waits[0] == [child.pid]
    assert result["surviving_owned_children"] == []
    assert result["shutdown_signals"] == {"terminate": [], "kill": []}
    assert result["shutdown_phases"][0]["before"] == 1
    assert result["shutdown_phases"][0]["after"] == 0
    assert result["shutdown_signal_counts"] == {"terminate": 0, "kill": 0}
    assert child.actions == [] and unrelated.actions == [] and unrelated.running


def test_fallback_targets_only_live_owned_identity_and_excludes_pid_reuse(session):
    child = Child(201)
    zombie = Child(202, zombie=True)
    reused = Child(203)
    unrelated = Child(301)
    session.children.extend([child, zombie])
    session.namespace["owned_children"][(reused.pid, reused.created - 1)] = reused

    result = session.namespace["shutdown_owned_processes"]()
    assert child.actions == ["terminate", "kill"]
    assert result["shutdown_signals"] == {"terminate": [201], "kill": [201]}
    assert [phase["after"] for phase in result["shutdown_phases"]] == [1, 1, 0]
    assert result["surviving_owned_children"] == []
    assert zombie.actions == reused.actions == unrelated.actions == []
    assert unrelated.running and reused.running


def test_shutdown_error_attempts_other_owned_children_and_fails_absence_gate(session):
    stuck = Child(201, exit_on="never")
    stuck.terminate_error = True
    healthy = Child(202)
    session.children.extend([stuck, healthy])
    code = compile(ast.Module(body=session.finalbody, type_ignores=[]), str(session.workflow), "exec")

    with pytest.raises(RuntimeError, match="shutdown could not be verified"):
        exec(code, session.namespace)

    assert stuck.actions == healthy.actions == ["terminate", "kill"]
    name, receipt = session.published[0]
    assert name == "rank-0-cleanup.json"
    assert receipt["surviving_owned_children"] == [stuck.pid]
    assert receipt["shutdown_errors"] == [
        {"pid": stuck.pid, "operation": "terminate", "error_type": "ProcessError"}
    ]
    assert receipt["ray_cli_exit_code"] == 1


def test_child_discovery_race_does_not_skip_other_owned_processes(session):
    gone = Child(201)
    gone.gone_during_discovery = True
    survivor = Child(202)
    session.children.extend([gone, survivor])

    session.namespace["remember_children"]()
    assert list(session.namespace["owned_children"]) == [(survivor.pid, survivor.created)]
