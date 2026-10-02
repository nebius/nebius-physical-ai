"""Verify the cloud controller is isolated from jobs and handoff copies only owned state."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest


@pytest.fixture
def controller(monkeypatch):
    """Load checked-out controller code.

    Args:
        monkeypatch: Isolated script imports.
    Returns:
        Controller bundle module.
    Raises:
        ImportError: Checked-out scripts cannot be imported.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("ci_cpu_runner_controller")


def _config():
    return dict(
        owner="test-pool",
        project_id="project-test",
        image_id="image-test",
        subnet_id="subnet-test",
        worker_security_group_id="workers-test",
        preset="4vcpu-16gb",
        profile="operator-profile",
    )


def test_controller_has_its_own_identity_but_workers_do_not(controller):
    """Keep cloud-management identity off every job VM.

    Args:
        controller: Checked-out provisioning module.
    Returns:
        None.
    Raises:
        AssertionError: Worker credentials or execution paths gain controller access.
    """
    config = _config()
    request = controller._controller_request(
        config, "account-test", "controller-test", "ssh-ed25519 synthetic"
    )
    assert request["spec"]["service_account_id"] == "account-test"
    assert request["spec"]["recovery_policy"] == "RECOVER"
    assert request["spec"]["network_interfaces"][0]["security_groups"] == [
        {"id": "controller-test"}
    ]
    init = json.loads(request["spec"]["cloud_init_user_data"].split("\n", 1)[1])
    assert ["rm", "-f", "/etc/sudoers.d/runner"] in init["runcmd"]
    assert "--jitconfig" not in request["spec"]["cloud_init_user_data"]
    runtime = controller._runtime_config(config, "controller-test")
    worker = controller._instance_request(runtime, "worker-test", "single-job-only")
    assert "service_account_id" not in worker["spec"]
    assert runtime["github_auth"] == "app"
    assert runtime["quota_scope"] == "project"
    assert runtime["profile"] != config["profile"]


def test_handoff_never_copies_operator_auth_or_supervisor_state(controller, tmp_path):
    """Copy explicit controller inputs while excluding unrelated private files.

    Args:
        controller: Checked-out bundle builder.
        tmp_path: Isolated operator state and bundle destination.
    Returns:
        None.
    Raises:
        AssertionError: Operator credentials, lock files or host state enter the bundle.
    """
    root = tmp_path / "state"
    root.mkdir()
    for folder in ("workers", "retired"):
        (root / folder).mkdir()
    (root / "operator-token").write_text("never-copy")
    (root / "controller.plist").write_text("never-copy")
    (root / "remote-controller.json").write_text("never-copy")
    (root / "routing.json").write_text('{"previous": null}')
    config = controller._runtime_config(_config(), "controller-test")
    destination = tmp_path / "bundle.tar.gz"
    controller._write_bundle(Path(__file__).parents[2], root, config, destination)
    assert destination.stat().st_mode & 0o077 == 0
    with tarfile.open(destination) as archive:
        assert "state/controller-stopped" in archive.getnames()
        assert not any(
            "operator-token" in name
            or "remote-controller.json" in name
            or "controller.plist" in name
            for name in archive.getnames()
        )
        assert json.load(archive.extractfile("state/config.json")) == config


def test_remote_commands_require_pinned_host_identity(controller, tmp_path):
    """Require explicit key and host verification for controller administration.

    Args:
        controller: Loads the checked-out script import path.
        tmp_path: Private synthetic SSH state.
    Returns:
        None.
    Raises:
        AssertionError: SSH may accept another host or arbitrary remote commands.
    """
    remote = importlib.import_module("ci_cpu_runner_remote")
    for name in ("controller-key", "controller-known-hosts"):
        (tmp_path / name).touch(mode=0o600)
    args = remote._ssh_command(tmp_path, {"host": "127.0.0.1"}, "down")
    assert "StrictHostKeyChecking=yes" in args
    assert "IdentitiesOnly=yes" in args
    assert args[-2:] == ["/usr/local/sbin/npa-ci-controller", "down"]
    with pytest.raises(ValueError, match="Unsupported"):
        remote._ssh_command(tmp_path, {"host": "127.0.0.1"}, "serve")


def test_cloud_quota_checks_do_not_need_tenant_wide_access(
    controller, tmp_path, monkeypatch
):
    """Retain live project quota checks under a project-scoped service account.

    Args:
        controller: Loads the checked-out script import path.
        tmp_path: Private diagnostics directory.
        monkeypatch: Substitutes synthetic inherited project quota responses.
    Returns:
        None.
    Raises:
        AssertionError: Controller requests broader tenant inventory permissions.
    """
    cloud = importlib.import_module("ci_cpu_runner_cloud")
    config = {**_config(), "quota_scope": "project", "region": "us-central1"}
    requests = []
    quotas = [
        {
            "metadata": {"name": name},
            "spec": {"region": config["region"]},
            "status": {"unit": "byte" if name.endswith("network-ssd") else "count"},
        }
        for name in cloud._capacity_requirements(config, 1)
    ]
    monkeypatch.setattr(
        cloud,
        "_nebius_items",
        lambda root, config, service, parent: requests.append(parent) or quotas,
    )
    cloud._check_capacity(tmp_path, config, 1)
    assert requests == [config["project_id"]]


def test_shutdown_marker_is_removed_when_provider_stop_fails(
    controller, tmp_path, monkeypatch
):
    """Allow supervisor retries when the provider rejects controller shutdown.

    Args:
        controller: Loads the checked-out script import path.
        tmp_path: Private controller state.
        monkeypatch: Injects a failed provider stop after ownership verification.
    Returns:
        None.
    Raises:
        AssertionError: A failed stop disables recovery or is reported as success.
    """
    remote = importlib.import_module("ci_cpu_runner_remote")
    monkeypatch.setattr(remote, "_controller_instance", lambda *a: {})

    def fail(*args):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(remote, "_nebius", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        remote._stop_controller(tmp_path, {"controller_instance_id": "controller-test"})
    assert not (tmp_path / "controller-stopped").exists()


def test_controller_boot_never_retries_a_host_identity_failure(
    controller, tmp_path, monkeypatch
):
    """Fail immediately when SSH reports a changed host identity.

    Args:
        controller, tmp_path, monkeypatch: Module, private keys, and fake SSH result.
    Returns:
        None.
    Raises:
        AssertionError: Host-key failures are retried as ordinary boot delay.
    """
    remote = importlib.import_module("ci_cpu_runner_remote")
    for name in ("controller-key", "controller-known-hosts"):
        (tmp_path / name).touch(mode=0o600)
    result = subprocess.CompletedProcess(
        ["ssh"], 255, b"", b"REMOTE HOST IDENTIFICATION HAS CHANGED"
    )
    monkeypatch.setattr(remote.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="SSH identity"):
        remote._wake_controller(
            tmp_path, {}, {"host": "127.0.0.1"}, {"status": {"state": "RUNNING"}}
        )


def test_disconnect_is_not_success_while_workers_remain(
    controller, tmp_path, monkeypatch
):
    """Require provider proof before treating shutdown SSH disconnect as success.

    Args:
        controller, tmp_path, monkeypatch: Module, private state, and cloud responses.
    Returns:
        None.
    Raises:
        AssertionError: A stopped controller hides an abandoned worker VM.
    """
    remote = importlib.import_module("ci_cpu_runner_remote")
    monkeypatch.setattr(
        remote, "_controller_instance", lambda *a: {"status": {"state": "STOPPED"}}
    )
    worker = {"metadata": {"id": "worker-test", "labels": {"npa-owner": "test-pool"}}}
    monkeypatch.setattr(remote, "_nebius_items", lambda *a: [worker])
    with pytest.raises(RuntimeError, match="worker removal"):
        remote._verify_remote_shutdown(
            tmp_path, _config(), {"instance_id": "controller-test"}
        )
