"""Exercise CPU-runner isolation, ownership, routing rollback, and draining."""

from __future__ import annotations

import importlib
from concurrent.futures import Future
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (
        importlib.import_module("ci_cpu_runner_cloud"),
        importlib.import_module("ci_cpu_runners"),
    )


@pytest.fixture
def config():
    return {
        "repository": "example/workbench",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "region": "us-central1",
        "owner": "test-pool",
        "label": "npa-ci-test-pool",
        "variable": "NPA_CI_SECURITY_RUNNER",
        "workers": 2,
        "preset": "4vcpu-16gb",
        "image_id": "image-test",
        "subnet_id": "subnet-test",
        "network_id": "network-test",
        "worker_security_group_id": "group-test",
    }


def _record(root, cloud):
    for folder in ("workers", "retired"):
        (root / folder).mkdir(exist_ok=True)
    record = {
        "name": "worker-test",
        "runner_id": 12,
        "instance_id": "instance-test",
        "phase": "provisioned",
    }
    cloud._save(root / "workers/worker-test.json", record)
    return record


def test_worker_gets_single_job_credentials_and_a_disposable_disk(modules, config):
    cloud, _ = modules
    request = cloud._instance_request(config, "worker-test", "single-job-config")
    spec = request["spec"]
    assert "service_account_id" not in spec
    assert spec["recovery_policy"] == "FAIL"
    assert spec["resources"] == {"platform": "cpu-d3", "preset": "4vcpu-16gb"}
    assert "managed_disk" in spec["boot_disk"]
    assert spec["network_interfaces"][0]["security_groups"] == [{"id": "group-test"}]
    init = json.loads(spec["cloud_init_user_data"].split("\n", 1)[1])
    files = {entry["path"]: entry for entry in init["write_files"]}
    assert files["/run/npa-runner-jit"]["permissions"] == "0600"
    assert "--jitconfig" in files["/usr/local/bin/npa-runner-job"]["content"]
    assert "Restart=no" in files["/etc/systemd/system/npa-runner.service"]["content"]
    assert "ssh_authorized_keys" not in init


def test_capacity_includes_public_addresses_and_disk_bytes(modules, config):
    cloud, _ = modules
    requirements = cloud._capacity_requirements(config, 3)
    assert requirements["compute.instance.non-gpu.vcpu"] == 12
    assert requirements["compute.disk.size.network-ssd"] == 192 * 1024**3
    assert requirements["vpc.ipv4-address.public.count"] == 3
    assert requirements["vpc.allocation.count"] == 6


def test_public_address_shortage_blocks_pool_allocation(
    modules, config, tmp_path, monkeypatch
):
    cloud, _ = modules
    quotas = []
    for name in cloud._capacity_requirements(config, 4):
        quotas.append(
            {
                "metadata": {"name": name},
                "spec": {"region": config["region"], "limit": str(1024**5)},
                "status": {
                    "usage": "0",
                    "unit": "byte" if name.endswith("network-ssd") else "count",
                },
            }
        )
    public = next(
        q for q in quotas if q["metadata"]["name"] == "vpc.ipv4-address.public.count"
    )
    public["spec"]["limit"] = "20"
    public["status"]["usage"] = "17"
    monkeypatch.setattr(cloud, "_nebius_items", lambda *a: quotas)
    with pytest.raises(RuntimeError, match="public.count: need 4, available 3"):
        cloud._check_capacity(tmp_path, config, 4)


def test_unknown_disk_usage_blocks_capacity_planning(modules):
    cloud, _ = modules
    quota = {"spec": {"limit": "1000"}, "status": {"unit": "byte"}}
    with pytest.raises(ValueError, match="Unverified quota"):
        cloud._quota_headroom(quota, "compute.disk.size.network-ssd", inherited=False)


@pytest.mark.parametrize(
    "message",
    ["(HTTP 403)", "rpc error: code = PermissionDenied", "connection refused"],
)
def test_provider_errors_never_become_absence(modules, tmp_path, monkeypatch, message):
    cloud, _ = modules
    monkeypatch.setattr(
        cloud.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", message),
    )
    with pytest.raises(RuntimeError, match="Provider request failed"):
        cloud._command(tmp_path, ["provider"], absent=True)
    assert len(list(tmp_path.glob("error-*.json"))) == 1


def test_private_state_is_atomic_and_not_world_readable(modules, tmp_path):
    cloud, _ = modules
    path = tmp_path / "state.json"
    cloud._save(path, {"phase": "creating"})
    cloud._save(path, {"phase": "provisioned"})
    assert cloud._load(path) == {"phase": "provisioned"}
    assert path.stat().st_mode & 0o077 == 0
    assert not list(tmp_path.glob(".pending-*"))


def test_wrong_project_blocks_worker_removal(modules, config, tmp_path, monkeypatch):
    cloud, _ = modules
    record = _record(tmp_path, cloud)
    instance = cloud._instance_request(config, record["name"], "jit")
    instance["metadata"]["parent_id"] = "project-other"
    monkeypatch.setattr(cloud, "_nebius", lambda *a, **k: instance)
    monkeypatch.setattr(
        cloud,
        "_github",
        lambda *a, **k: pytest.fail(
            "GitHub changed before cloud ownership was verified"
        ),
    )
    with pytest.raises(ValueError, match="different project"):
        cloud._delete_worker(tmp_path, config, record)


def test_busy_runner_cannot_be_deleted(modules, config, tmp_path, monkeypatch):
    cloud, _ = modules
    record = _record(tmp_path, cloud)
    calls = []
    instance = cloud._instance_request(config, record["name"], "jit")
    instance["metadata"]["id"] = record["instance_id"]
    monkeypatch.setattr(
        cloud, "_nebius", lambda *a, **k: calls.append(a[3]) or instance
    )
    monkeypatch.setattr(
        cloud,
        "_pages",
        lambda *a, **k: [
            {"id": record["runner_id"], "name": record["name"], "busy": True}
        ],
    )
    with pytest.raises(RuntimeError, match="still busy"):
        cloud._delete_worker(tmp_path, config, record)
    assert calls == ["get"]
    assert (tmp_path / "workers/worker-test.json").exists()


def test_completed_worker_is_removed_and_deletion_is_verified(
    modules, config, tmp_path, monkeypatch
):
    cloud, _ = modules
    record = _record(tmp_path, cloud)
    instance = cloud._instance_request(config, record["name"], "jit")
    instance["metadata"]["id"] = record["instance_id"]
    responses = iter([instance, None, None])
    operations = []
    monkeypatch.setattr(
        cloud, "_nebius", lambda *a, **k: operations.append(a[3]) or next(responses)
    )
    monkeypatch.setattr(cloud, "_github", lambda *a, **k: None)
    monkeypatch.setattr(cloud, "_pages", lambda *a, **k: [])
    cloud._delete_worker(tmp_path, config, record)
    assert operations == ["get", "delete", "get"]
    assert not (tmp_path / "workers/worker-test.json").exists()
    assert cloud._load(tmp_path / "retired/worker-test.json")["phase"] == "deleted"


@pytest.mark.parametrize("state", ["STOPPED", "RUNNING"])
def test_offline_workers_only_retire_after_confirmed_shutdown(
    modules, config, tmp_path, monkeypatch, state
):
    cloud, pool = modules
    record = _record(tmp_path, cloud)
    monkeypatch.setattr(pool, "_owned_worker", lambda *a: {"status": {"state": state}})
    retired = pool._reconcile(
        tmp_path,
        config,
        {record["runner_id"]: {"status": "offline", "busy": False}},
        {},
    )
    assert retired == ([record] if state == "STOPPED" else [])


def test_slow_deletion_does_not_block_other_worker_retirements(modules):
    _, pool = modules
    slow, complete = Future(), Future()
    complete.set_result(None)
    retiring = {"slow": slow, "complete": complete}
    pool._finish_retirements(retiring)
    assert retiring == {"slow": slow}


def test_deletion_failure_is_not_silently_discarded(modules):
    _, pool = modules
    failed = Future()
    failed.set_exception(RuntimeError("deletion not verified"))
    retiring = {"failed": failed}
    with pytest.raises(RuntimeError, match="deletion not verified"):
        pool._finish_retirements(retiring)
    assert "failed" in retiring


def test_pending_deletion_is_not_started_twice(modules, config, tmp_path):
    cloud, pool = modules
    record = _record(tmp_path, cloud)
    assert pool._reconcile(tmp_path, config, {}, {record["name"]: Future()}) == []


def test_inventory_tolerates_a_concurrent_completed_retirement(
    modules, tmp_path, monkeypatch
):
    cloud, pool = modules
    _record(tmp_path, cloud)

    def retire_before_read(path):
        path.unlink()
        return cloud._load(path)

    monkeypatch.setattr(pool, "_load", retire_before_read)
    assert pool._records(tmp_path) == []


@pytest.mark.parametrize("previous", [None, {"value": "ubuntu-latest"}])
def test_disabling_restores_original_routing(
    modules, config, tmp_path, monkeypatch, previous
):
    cloud, pool = modules
    cloud._save(tmp_path / "routing.json", {"previous": previous})
    monkeypatch.setattr(pool, "_routing", lambda *a: {"value": config["label"]})
    calls = []
    monkeypatch.setattr(pool, "_github", lambda *a, **k: calls.append(k))
    pool._disable(tmp_path, config)
    assert calls[0]["method"] == ("PATCH" if previous else "DELETE")
    if previous:
        assert calls[0]["payload"]["value"] == previous["value"]


def test_disabling_preserves_an_operator_routing_change(
    modules, config, tmp_path, monkeypatch
):
    _, pool = modules
    monkeypatch.setattr(pool, "_routing", lambda *a: {"value": "another-pool"})
    monkeypatch.setattr(
        pool, "_github", lambda *a, **k: pytest.fail("Operator routing overwritten")
    )
    pool._disable(tmp_path, config)


def test_drain_captures_runs_before_and_after_routing_change(
    modules, config, tmp_path, monkeypatch
):
    cloud, pool = modules
    snapshots = iter([{1, 2}, {2, 3}])
    monkeypatch.setattr(pool, "_routing", lambda *a: {"value": config["label"]})
    order = []
    monkeypatch.setattr(
        pool, "_active_runs", lambda *a: order.append("inventory") or next(snapshots)
    )
    monkeypatch.setattr(pool, "_disable", lambda *a: order.append("restore"))
    pool._begin_drain(tmp_path, config)
    assert order == ["inventory", "restore", "inventory"]
    assert cloud._load(tmp_path / "drain.json")["runs"] == [1, 2, 3]


def test_drain_only_tracks_workflows_that_can_use_cpu_routing(
    modules, config, tmp_path, monkeypatch
):
    _, pool = modules
    admission = ".github/workflows/security-regression.yml"
    runs = [
        {"id": 1, "event": "pull_request", "path": admission},
        {"id": 2, "event": "merge_group", "path": admission + "@refs/heads/main"},
        {"id": 3, "event": "push", "path": admission},
        {"id": 4, "event": "push", "path": ".github/workflows/publish-images.yml"},
        {"id": 5, "event": "pull_request", "path": ".github/workflows/gitleaks.yml"},
    ]
    monkeypatch.setattr(pool, "_pages", lambda *a: runs)
    assert pool._active_runs(tmp_path, config) == {1, 2}


def test_drain_waits_for_dependent_jobs_and_busy_workers(
    modules, config, tmp_path, monkeypatch
):
    cloud, pool = modules
    record = _record(tmp_path, cloud)
    cloud._save(tmp_path / "drain.json", {"runs": [1]})
    monkeypatch.setattr(pool, "_github", lambda *a: {"status": "in_progress"})
    assert not pool._drained(tmp_path, config, {})
    monkeypatch.setattr(pool, "_github", lambda *a: {"status": "completed"})
    assert not pool._drained(tmp_path, config, {record["runner_id"]: {"busy": True}})
    assert pool._drained(tmp_path, config, {record["runner_id"]: {"busy": False}})


def test_routing_requires_the_whole_pool_online(modules, config, tmp_path, monkeypatch):
    cloud, pool = modules
    record = _record(tmp_path, cloud)
    monkeypatch.setattr(pool, "_verify_project", lambda *a: None)
    monkeypatch.setattr(pool, "_running", lambda *a: True)
    monkeypatch.setattr(
        pool, "_runners", lambda *a: {record["runner_id"]: {"status": "online"}}
    )
    monkeypatch.setattr(
        pool,
        "_github",
        lambda *a, **k: pytest.fail("Routing changed before pool readiness"),
    )
    with pytest.raises(RuntimeError, match="Every configured worker"):
        pool._enable(tmp_path, config)


def test_interrupted_creation_can_drain_without_creating_a_vm(
    modules, config, tmp_path, monkeypatch
):
    cloud, pool = modules
    _record(tmp_path, cloud)
    record = {"name": "worker-test", "phase": "creating"}
    cloud._save(tmp_path / "workers/worker-test.json", record)
    cloud._save(tmp_path / "drain.json", {"runs": []})
    operations = []
    monkeypatch.setattr(cloud, "_nebius", lambda *a, **k: operations.append(a[3]))
    monkeypatch.setattr(cloud, "_pages", lambda *a, **k: [])
    monkeypatch.setattr(pool, "_verify_project", lambda *a: None)
    monkeypatch.setattr(pool, "_runners", lambda *a: {})
    monkeypatch.setattr(
        pool, "_fill", lambda *a: pytest.fail("Drain provisioned a new worker")
    )
    pool._serve(tmp_path, config)
    assert operations == ["get-by-name", "get-by-name"]
    assert cloud._load(tmp_path / "status.json")["phase"] == "stopped"


def test_routing_can_be_reenabled_after_restoring_previous_value(
    modules, config, tmp_path
):
    cloud, pool = modules
    current = {"value": "ubuntu-latest"}
    pool._record_routing(tmp_path, config, current)
    pool._record_routing(tmp_path, config, {"value": config["label"]})
    pool._record_routing(tmp_path, config, current)
    assert cloud._load(tmp_path / "routing.json")["previous"] == current
    with pytest.raises(RuntimeError, match="changed outside"):
        pool._record_routing(tmp_path, config, {"value": "another-pool"})


@pytest.mark.parametrize("access", ["ALLOW", "DENY"])
def test_worker_firewall_rejects_inbound_access(
    modules, config, tmp_path, monkeypatch, access
):
    _, pool = modules
    rule = {
        "spec": {
            "access": access,
            "protocol": "ANY",
            "ingress": {"source_cidrs": ["0.0.0.0/0"]},
        },
        "status": {"state": "READY"},
    }
    monkeypatch.setattr(pool, "_nebius_items", lambda *a: [rule])
    if access == "ALLOW":
        with pytest.raises(ValueError, match="inbound connections"):
            pool._verify_firewall(tmp_path, config)
    else:
        pool._verify_firewall(tmp_path, config)
