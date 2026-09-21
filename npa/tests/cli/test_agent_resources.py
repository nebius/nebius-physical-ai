from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from npa.cli import agent_resources
from npa.cli.agent_resources import (
    build_resource_inventory,
    category_payload,
    configured_k8s_backends,
    discover_mk8s_accelerators,
    discover_nebius_categories,
    format_resource_inventory,
    inventory_summary,
    merge_configured_references,
    prepare_agent_cloud_environment,
    run_bounded_agent_command,
    run_resource_discovery_command,
)


class _Pipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _NeverExits:
    def __init__(self, pid: int, barrier: threading.Barrier | None = None) -> None:
        self.pid = pid
        self.returncode = None
        self.stdout = _Pipe()
        self.stderr = _Pipe()
        self.barrier = barrier
        self.communicate_calls = 0
        self.poll_calls = 0

    def communicate(self, *, timeout: float):
        self.communicate_calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        raise subprocess.TimeoutExpired(["nebius"], timeout)

    def poll(self):
        self.poll_calls += 1
        return None


def _isolated_process_registry(monkeypatch) -> None:
    monkeypatch.setattr(agent_resources, "_AGENT_ACTIVE_PROCESSES", {})
    monkeypatch.setattr(agent_resources, "_AGENT_ABANDONED_PROCESS_GROUPS", {})
    monkeypatch.setattr(agent_resources, "_AGENT_COMMAND_BREAKER_OPEN", False)
    monkeypatch.setattr(agent_resources, "_AGENT_COMMAND_REAPER", None)


def test_cloud_environment_requires_provenance_and_scrubs_ambient_tokens() -> None:
    with pytest.raises(ValueError, match="credential source"):
        prepare_agent_cloud_environment({"NEBIUS_IAM_TOKEN": "secret"})

    environment, source = prepare_agent_cloud_environment(
        {
            "NPA_NEBIUS_CREDENTIAL_SOURCE": "instance_metadata",
            "NEBIUS_IAM_TOKEN": "secret",
            "NEBIUS_IAM_TOKEN_FILE": "/private/token",
            "NPA_NEBIUS_IAM_TOKEN": "secret",
            "NPA_NEBIUS_IAM_TOKEN_FILE": "/private/other-token",
            "NPA_REUSE_IAM_TOKEN": "1",
            "TF_VAR_iam_token": "secret",
            "IAM_TOKEN": "secret",
        }
    )

    assert source == "instance_metadata"
    assert environment["NEBIUS_PROFILE"] == "cursor-sa"
    assert not (agent_resources._AMBIENT_NEBIUS_TOKEN_KEYS & set(environment))


def test_resource_discovery_rejects_unknown_source_and_scrubs_tokens(
    monkeypatch,
) -> None:
    command = ["nebius", "--profile", "cursor-sa", "iam", "project", "list"]
    assert run_resource_discovery_command(command, command_env={}) == (
        2,
        "",
        "agent credential source is unavailable",
    )
    seen: dict[str, object] = {}

    def bounded(argv, *, env, timeout_s):
        seen.update(argv=list(argv), env=dict(env), timeout=timeout_s)
        return subprocess.CompletedProcess(argv, 0, '{"items": []}', "")

    monkeypatch.setattr(agent_resources, "run_bounded_agent_command", bounded)
    environment = {
        "NPA_NEBIUS_CREDENTIAL_SOURCE": "configured_profile",
        "NEBIUS_IAM_TOKEN": "must-not-propagate",
    }

    assert run_resource_discovery_command(command, command_env=environment) == (
        0,
        '{"items": []}',
        "",
    )
    assert seen["timeout"] == 30
    assert "NEBIUS_IAM_TOKEN" not in seen["env"]


def test_bounded_command_times_out_without_waiting_and_opens_breaker(
    monkeypatch,
) -> None:
    _isolated_process_registry(monkeypatch)
    process = _NeverExits(981001)
    starts = 0
    killed: list[int] = []

    def popen(*_args, **_kwargs):
        nonlocal starts
        starts += 1
        return process

    monkeypatch.setattr(agent_resources.subprocess, "Popen", popen)
    monkeypatch.setattr(agent_resources, "_kill_agent_process_group", killed.append)
    monkeypatch.setattr(agent_resources, "_start_agent_process_reaper", lambda: None)

    with pytest.raises(TimeoutError, match="timed out"):
        run_bounded_agent_command(["nebius"], timeout_s=0.01)
    with pytest.raises(TimeoutError, match="prior agent cloud command"):
        run_bounded_agent_command(["nebius"], timeout_s=0.01)

    assert starts == 1
    assert process.communicate_calls == 1
    assert killed == [process.pid]
    assert process.stdout.closed and process.stderr.closed
    assert agent_resources._AGENT_COMMAND_BREAKER_OPEN is True


def test_concurrent_timeouts_retain_every_process_group(monkeypatch) -> None:
    _isolated_process_registry(monkeypatch)
    barrier = threading.Barrier(2)
    processes = [_NeverExits(981011, barrier), _NeverExits(981012, barrier)]
    starts = iter(processes)
    failures: list[type[BaseException]] = []

    monkeypatch.setattr(
        agent_resources.subprocess, "Popen", lambda *_args, **_kwargs: next(starts)
    )
    monkeypatch.setattr(agent_resources, "_kill_agent_process_group", lambda _pid: None)
    monkeypatch.setattr(agent_resources, "_start_agent_process_reaper", lambda: None)

    def invoke() -> None:
        try:
            run_bounded_agent_command(["nebius"], timeout_s=0.01)
        except BaseException as exc:
            failures.append(type(exc))

    workers = [threading.Thread(target=invoke) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert failures == [TimeoutError, TimeoutError]
    assert set(agent_resources._AGENT_ABANDONED_PROCESS_GROUPS) == {
        process.pid for process in processes
    }


def test_reaper_clears_breaker_only_after_process_groups_exit(monkeypatch) -> None:
    _isolated_process_registry(monkeypatch)
    process = _NeverExits(981021)
    agent_resources._AGENT_ACTIVE_PROCESSES[process.pid] = process
    agent_resources._AGENT_ABANDONED_PROCESS_GROUPS[process.pid] = process
    agent_resources._AGENT_COMMAND_BREAKER_OPEN = True
    checks = iter((True, False))

    monkeypatch.setattr(
        agent_resources, "_agent_process_group_exists", lambda _pid: next(checks)
    )
    monkeypatch.setattr(agent_resources.time, "sleep", lambda _seconds: None)

    agent_resources._reap_abandoned_agent_processes()

    assert process.poll_calls == 3
    assert agent_resources._AGENT_ABANDONED_PROCESS_GROUPS == {}
    assert agent_resources._AGENT_ACTIVE_PROCESSES == {}
    assert agent_resources._AGENT_COMMAND_BREAKER_OPEN is False


def test_bounded_command_kills_and_reaps_real_descendant_group(monkeypatch) -> None:
    _isolated_process_registry(monkeypatch)
    child = (
        "import subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "time.sleep(60)"
    )
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="timed out"):
        run_bounded_agent_command(
            [sys.executable, "-c", child],
            timeout_s=0.1,
        )

    assert time.monotonic() - started < 2
    deadline = time.monotonic() + 3
    while agent_resources._AGENT_COMMAND_BREAKER_OPEN and time.monotonic() < deadline:
        time.sleep(0.02)
    assert agent_resources._AGENT_COMMAND_BREAKER_OPEN is False
    assert agent_resources._AGENT_ABANDONED_PROCESS_GROUPS == {}


def test_k8s_grounding_normalizes_legacy_config_and_live_node_groups(
    monkeypatch,
) -> None:
    configured = configured_k8s_backends(
        {
            "k8s_context": "customer-context",
            "container_registry": "registry.example/customer",
        },
        "customer",
    )
    assert configured[0]["context"] == "customer-context"
    assert configured[0]["raw"] == {"container_registry": "registry.example/customer"}

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "items": [
                    {"spec": {"template": {"resources": {"platform": "gpu-rtx6000"}}}},
                    {"spec": {"template": {"resources": {"platform": "cpu-d3"}}}},
                ]
            }
        )

    monkeypatch.setattr(
        "npa.cli.agent_resources.run_bounded_agent_command",
        lambda *_a, **_kw: Result(),
    )
    discovered = discover_mk8s_accelerators(
        "cluster-id",
        ["nebius"],
        {"NPA_NEBIUS_CREDENTIAL_SOURCE": "instance_metadata"},
    )
    assert discovered == {
        "available_accelerators": ["RTXPRO6000"],
        "gpu_platforms": ["cpu-d3", "gpu-rtx6000"],
        "gpu_accelerator": "RTXPRO6000",
    }


def test_nested_k8s_grounding_keeps_secret_names_but_redacts_secret_values() -> None:
    configured = configured_k8s_backends(
        {
            "kubernetes": {
                "context": "customer-context",
                "image_pull_secrets": "registry-pull",
                "env_secret_names": "runtime-env",
                "api_token": "must-not-leak",
                "registry_password": "must-not-leak",
            }
        },
        "customer",
    )

    assert configured[0]["raw"]["image_pull_secrets"] == "registry-pull"
    assert configured[0]["raw"]["env_secret_names"] == "runtime-env"
    assert "api_token" not in configured[0]["raw"]
    assert "registry_password" not in configured[0]["raw"]


def test_build_inventory_prefers_metadata_profile_and_includes_local_resources() -> (
    None
):
    inventory = build_resource_inventory(
        config={
            "default_project": "demo",
            "projects": {
                "demo": {
                    "project_id": "project-test",
                    "tenant_id": "tenant-test",
                    "region": "us-central1",
                }
            },
        },
        env={"NEBIUS_PROFILE": "stale-profile", "NPA_AGENT_NAME": "paidf"},
        state={"latest_submit": {"run_id": "run-test"}},
        tool_refs=["workbench.cosmos_evaluator.evaluate", "workbench.fiftyone.curate"],
        runner=_runner_for(
            {"compute instance": {"items": [{"metadata": {"name": "paidf"}}]}}
        ),
        generated_at="2026-08-09T00:00:00Z",
        metadata_token_available=True,
        force_refresh=True,
    )

    assert inventory["context"] == {
        "project_alias": "demo",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "region": "us-central1",
        "profile": "cursor-sa",
        "profile_status": "authenticated",
    }
    by_id = {item["id"]: item for item in inventory["categories"]}
    assert by_id["compute"]["configured"][0]["name"] == "paidf"
    assert by_id["workbench"]["discovered_count"] == 2
    assert by_id["workflows"]["discovered"][0]["name"] == "run-test"


def _runner_for(payloads):
    def run(command):
        service = command[3]
        operation = " ".join(command[3:5])
        value = payloads.get(operation, payloads.get(service, {"items": []}))
        if isinstance(value, tuple):
            return value
        return 0, json.dumps(value), ""

    return run


def test_discovers_non_empty_and_empty_categories_without_secrets() -> None:
    categories = discover_nebius_categories(
        project_id="project-test",
        tenant_id="tenant-test",
        profile="cursor-sa",
        runner=_runner_for(
            {
                "iam project": {
                    "metadata": {"id": "project-test", "name": "demo-project"}
                },
                "iam tenant": {
                    "metadata": {"id": "tenant-test", "name": "demo-tenant"}
                },
                "compute instance": {
                    "items": [
                        {
                            "metadata": {"id": "instance-test", "name": "agent-demo"},
                            "status": {"state": "RUNNING", "token": "must-not-leak"},
                            "credentials": {"password": "must-not-leak"},
                        }
                    ]
                },
            }
        ),
    )

    by_id = {item["id"]: item for item in categories}
    assert by_id["compute"]["status"] == "discovered"
    assert by_id["compute"]["discovered_count"] == 1
    assert by_id["compute"]["discovered"][0] == {
        "kind": "instance",
        "source": "nebius_cli",
        "id": "instance-test",
        "name": "agent-demo",
        "status": "RUNNING",
    }
    assert by_id["kubernetes"]["status"] == "empty"
    rendered = json.dumps(categories)
    assert "must-not-leak" not in rendered
    assert "password" not in rendered


def test_permission_error_is_honest_and_keeps_configured_reference() -> None:
    categories = discover_nebius_categories(
        project_id="project-test",
        tenant_id="tenant-test",
        profile="cursor-sa",
        runner=lambda _command: (
            1,
            "",
            "PermissionDenied opaque-debug-token-should-not-leak",
        ),
    )
    categories = merge_configured_references(
        categories,
        {
            "storage": [
                {
                    "kind": "bucket",
                    "name": "configured-bucket",
                    "source": "staged_credentials",
                }
            ]
        },
    )
    storage = next(item for item in categories if item["id"] == "storage")
    assert storage["status"] == "error"
    assert storage["configured_count"] == 1
    assert storage["discovered_count"] == 0
    assert storage["error"] == {
        "kind": "permission_denied",
        "message": "Credentials are authenticated but cannot enumerate this resource category.",
    }
    assert "opaque-debug-token" not in json.dumps(storage)


def test_category_states_and_summary_are_explicit() -> None:
    categories = [
        category_payload("a", "A", discovered=[{"name": "one"}]),
        category_payload(
            "b", "B", configured=[{"name": "two"}], discovery_attempted=False
        ),
        category_payload("c", "C"),
        category_payload(
            "d", "D", error={"kind": "authentication_error", "message": "no"}
        ),
    ]
    assert [item["status"] for item in categories] == [
        "discovered",
        "configured",
        "empty",
        "error",
    ]
    assert inventory_summary(categories) == {
        "categories": 4,
        "discovered_categories": 1,
        "configured_only_categories": 1,
        "empty_categories": 1,
        "error_categories": 1,
        "configured_resources": 1,
        "discovered_resources": 1,
    }


def test_grounded_inventory_reply_reports_counts_and_errors() -> None:
    inventory = {
        "context": {
            "project_alias": "demo",
            "project_id": "project-test",
            "tenant_id": "tenant-test",
            "region": "us-central1",
            "profile": "cursor-sa",
        },
        "categories": [
            category_payload("compute", "Compute", discovered=[{"name": "agent-demo"}]),
            category_payload(
                "storage",
                "Object storage",
                configured=[{"name": "configured-bucket"}],
                error={"kind": "permission_denied", "message": "Not enumerable."},
            ),
        ],
    }
    reply = format_resource_inventory(inventory)
    assert "**Tenant resources**" in reply
    assert "**discovered_resources**: `1`" in reply
    assert "status=`error`" in reply
    assert "discovery_error=`permission_denied`" in reply
