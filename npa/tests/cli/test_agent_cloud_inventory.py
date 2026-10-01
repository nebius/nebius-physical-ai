"""Exercise unavailable inventory, metadata-only identity and owned cleanup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.cli import agent_resources as resources
from npa.cli.agent_chat import format_infra_backends
from npa.cli.agent_workflow import resolve_workflow_infrastructure


METADATA = {"NPA_NEBIUS_CREDENTIAL_SOURCE": "instance_metadata"}


@pytest.mark.parametrize("source", ["", "configured_profile", "operator-profile"])
def test_unstaged_or_unsupported_identity_never_launches(monkeypatch, source):
    spawn = Mock(side_effect=AssertionError("credentials must reject before launch"))
    monkeypatch.setattr(resources, "run_bounded_agent_command", spawn)
    environment = {"NPA_NEBIUS_CREDENTIAL_SOURCE": source}
    clusters = resources.discover_mk8s_clusters("project-fixture", environment)
    accelerators = resources.discover_mk8s_accelerators(
        "cluster-fixture", [], environment
    )
    for result in (clusters, accelerators["accelerator_discovery"]):
        assert result["status"] == "unavailable"
        assert result["error"]["kind"] == "credential_unavailable"
    assert "items" not in clusters and "available_accelerators" not in accelerators
    spawn.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        (TimeoutError("agent command timed out"), "timeout"),
        (TimeoutError("a prior agent cloud command has not exited"), "cleanup_pending"),
        (OSError("private details must not escape"), "command_unavailable"),
    ],
)
def test_inventory_execution_failures_are_not_empty(monkeypatch, failure, kind):
    monkeypatch.setattr(
        resources, "run_bounded_agent_command", Mock(side_effect=failure)
    )
    result = resources.discover_mk8s_clusters("project-fixture", METADATA)
    assert result["status"] == "unavailable" and result["error"]["kind"] == kind
    assert "items" not in result and "private details" not in json.dumps(result)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "[]",
        "null",
        "broken",
        '{"items": {}}',
        '{"error": "private detail"}',
        '{"items": [false]}',
        '{"items": [], "next_page_token": "private"}',
    ],
)
def test_invalid_or_incomplete_inventory_is_not_empty(monkeypatch, stdout):
    monkeypatch.setattr(
        resources,
        "run_bounded_agent_command",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout, ""),
    )
    result = resources.discover_mk8s_clusters("project-fixture", METADATA)
    assert result["status"] == "unavailable" and "items" not in result
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("stdout", ['{"items": []}', "{}"])
def test_successful_empty_inventory_remains_available(monkeypatch, stdout):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(resources, "run_bounded_agent_command", run)
    assert resources.discover_mk8s_clusters("project-fixture", METADATA) == {
        "status": "available",
        "items": [],
    }
    result = resources.discover_mk8s_accelerators(
        "cluster-fixture", ["nebius", "--profile", "operator"], METADATA
    )
    assert result == {
        "accelerator_discovery": {"status": "available"},
        "available_accelerators": [],
        "gpu_platforms": [],
    }
    for command, kwargs in calls:
        assert command[1:5] == [
            "--config",
            "/root/.nebius/config.yaml",
            "--profile",
            "cursor-sa",
        ]
        assert "--all" in command and "operator" not in command
        assert kwargs["env"]["NPA_NEBIUS_CREDENTIAL_SOURCE"] == "instance_metadata"


def test_cluster_keeps_failed_accelerator_discovery_unknown(monkeypatch):
    responses = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                '{"items": [{"metadata": {"id": "cluster-fixture", "name": "cluster"}}]}',
                "",
            ),
            subprocess.CompletedProcess([], 1, "", "PermissionDenied private-details"),
        ]
    )
    monkeypatch.setattr(
        resources, "run_bounded_agent_command", lambda *a, **k: next(responses)
    )
    inventory = resources.discover_mk8s_clusters("project-fixture", METADATA)
    raw = inventory["items"][0]["raw"]
    assert inventory["status"] == "available"
    assert raw["accelerator_discovery"]["status"] == "unavailable"
    assert raw["accelerator_discovery"]["error"]["kind"] == "permission_denied"
    assert "available_accelerators" not in raw and "private-details" not in json.dumps(
        raw
    )
    selected = resolve_workflow_infrastructure({"cloud_clusters": inventory["items"]})
    assert selected["accelerator_discovery_status"] == "unavailable"


@pytest.mark.parametrize("configured", [False, True])
def test_unavailable_inventory_reaches_agent_consumers(tmp_path, configured):
    config = {"projects": {"demo": {"k8s_context": "chosen"} if configured else {}}}
    inventory = resources.assemble_k8s_backend_inventory(
        config=config,
        alias="demo",
        clusters_root=tmp_path / "clusters",
        cloud_clusters=[],
        cloud_discovery={"status": "unavailable", "error": {"kind": "timeout"}},
        npa_ready=True,
        npa_error="",
        terraform_dir=tmp_path,
    )
    assert inventory["has_infra"] is (True if configured else None)
    selected = resolve_workflow_infrastructure(inventory)
    assert selected["source"] == ("configured" if configured else "unavailable")
    assert selected["cloud_discovery_status"] == "unavailable"
    reply = format_infra_backends({"infra": inventory})
    assert "absence" in reply and "unavailable" in reply
    assert "No Kubernetes infra" not in reply
    assert "Ask the Agent to deploy" not in reply


def test_uncertain_cleanup_retains_breaker_and_backs_off(monkeypatch):
    process = SimpleNamespace(pid=981100, returncode=None, wait=Mock())
    monkeypatch.setattr(
        resources, "_AGENT_ABANDONED_PROCESS_GROUPS", {process.pid: process}
    )
    monkeypatch.setattr(resources, "_AGENT_ACTIVE_PROCESSES", {process.pid: process})
    monkeypatch.setattr(resources, "_AGENT_COMMAND_BREAKER_OPEN", True)
    monkeypatch.setattr(
        resources, "_agent_process_exited_without_reaping", lambda p: True
    )
    monkeypatch.setattr(
        resources, "_agent_process_group_has_other_members", lambda *a: None
    )
    kill = Mock()
    monkeypatch.setattr(resources, "_kill_agent_process_group", kill)
    delays = []

    def observe(delay):
        delays.append(delay)
        if len(delays) == 8:
            raise InterruptedError("end observation")

    monkeypatch.setattr(resources.time, "sleep", observe)
    with pytest.raises(InterruptedError):
        resources._reap_abandoned_agent_processes()
    assert delays == [0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 5.0, 5.0]
    assert resources._AGENT_COMMAND_BREAKER_OPEN
    assert process.pid in resources._AGENT_ABANDONED_PROCESS_GROUPS
    process.wait.assert_not_called()
    kill.assert_not_called()


def test_unowned_child_never_releases_breaker_or_signals_group(monkeypatch):
    process = SimpleNamespace(pid=981101, returncode=0, wait=Mock())
    monkeypatch.setattr(
        resources, "_agent_process_exited_without_reaping", lambda p: None
    )
    inspect = Mock(side_effect=AssertionError("PID ownership is unknown"))
    monkeypatch.setattr(resources, "_agent_process_group_has_other_members", inspect)
    assert resources._reap_owned_agent_group(process.pid, process) is False
    inspect.assert_not_called()
    process.wait.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux waitid and procfs")
def test_real_linux_cleanup_recovers_only_after_proc_visibility(monkeypatch):
    monkeypatch.setattr(resources, "_AGENT_ACTIVE_PROCESSES", {})
    monkeypatch.setattr(resources, "_AGENT_ABANDONED_PROCESS_GROUPS", {})
    monkeypatch.setattr(resources, "_AGENT_COMMAND_BREAKER_OPEN", False)
    monkeypatch.setattr(resources, "_AGENT_COMMAND_REAPER", None)
    original = Path.iterdir

    def inaccessible(path):
        if str(path) == "/proc":
            raise PermissionError("synthetic procfs denial")
        return original(path)

    with monkeypatch.context() as restricted:
        restricted.setattr(Path, "iterdir", inaccessible)
        with pytest.raises(TimeoutError, match="timed out"):
            resources.run_bounded_agent_command(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.05
            )
        time.sleep(0.3)
        assert resources._AGENT_COMMAND_BREAKER_OPEN
        assert resources._AGENT_ABANDONED_PROCESS_GROUPS
        with pytest.raises(TimeoutError, match="prior agent cloud"):
            resources.run_bounded_agent_command(
                [sys.executable, "-c", "pass"], timeout_s=1
            )
    deadline = time.monotonic() + 6
    while resources._AGENT_COMMAND_BREAKER_OPEN and time.monotonic() < deadline:
        time.sleep(0.02)
    assert resources._AGENT_COMMAND_BREAKER_OPEN is False
    assert resources._AGENT_ABANDONED_PROCESS_GROUPS == {}
