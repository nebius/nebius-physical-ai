"""Verify allocation-mode quota gates without contacting or mutating infrastructure."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from npa.cluster_backends.quotas import preflight_region, required_quotas
from npa.fleet.spec import ClusterSpec, NodePoolSpec


def _cluster(*, preemptible=True, gpu_count=1, gpus_per_node=1, cpu_count=0):
    return ClusterSpec(
        name="quota-fixture",
        cpu_nodes=NodePoolSpec(count=cpu_count, preset="16vcpu-64gb"),
        gpu_nodes=NodePoolSpec(
            count=gpu_count,
            platform="gpu-rtx6000",
            preset=f"{gpus_per_node}gpu-24vcpu-218gb",
            preemptible=preemptible,
        ),
        enable_gpu_cluster=False,
        enable_filestore=False,
    )


def _allowances():
    limits = {
        "mk8s.cluster.count": 10,
        "compute.instance.count": 10,
        "compute.instance.preemptible.count": 128,
        "compute.instance.non-gpu.vcpu": 100,
        "compute.instance.gpu.rtx6000": 0,
        "compute.disk.count": 10,
        "compute.disk.size.network-ssd": 10 * 1024**4,
        "vpc.allocation.count": 100,
    }
    return {
        name: {
            "metadata": {"name": name, "parent_id": "tenant-fixture"},
            "spec": {"region": "region-fixture", "limit": str(limit)},
            "status": {
                "state": "STATE_ACTIVE",
                "unit": "byte" if "disk.size" in name else "count",
                "usage": "0",
            },
        }
        for name, limit in limits.items()
    }


def _preflight(clusters, allowances):
    calls = []

    def capture(argv, **kwargs):
        calls.append(argv)
        assert argv[:3] == ["nebius-fixture", "--profile", "profile-fixture"]
        assert argv[3:6] == ["quotas", "quota-allowance", "list"]
        assert argv[6:] == [
            "--parent-id",
            "tenant-fixture",
            "--all",
            "--format",
            "json",
        ]
        assert kwargs["check"] is False
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": list(allowances.values())})
        )

    shortfalls = preflight_region(
        nebius_bin="nebius-fixture",
        tenant_id="tenant-fixture",
        region="region-fixture",
        clusters=clusters,
        env={},
        profile="profile-fixture",
        run_capture=capture,
        nebius_argv=lambda binary, profile: [binary, "--profile", profile],
    )
    assert len(calls) == 1
    return shortfalls


@pytest.mark.parametrize("gpus_per_node", [1, 8])
def test_preemptible_workers_count_vms_and_keep_boot_disk_requirements(gpus_per_node):
    cluster = _cluster(gpu_count=2, gpus_per_node=gpus_per_node, cpu_count=1)
    needed = required_quotas([cluster])
    assert needed["compute.instance.preemptible.count"] == 2
    assert needed["compute.instance.count"] == 1
    assert "compute.instance.gpu.rtx6000" not in needed
    assert needed["compute.instance.non-gpu.vcpu"] == 16
    assert needed["compute.disk.count"] == 3
    assert needed["compute.disk.size.network-ssd"] == (128 + 2 * 1023) * 1024**3


def test_mixed_regular_and_preemptible_pools_keep_independent_limits():
    regular = _cluster(preemptible=False, cpu_count=1)
    preemptible = _cluster(gpu_count=2, gpus_per_node=8)
    needed = required_quotas([regular, preemptible])
    assert needed["compute.instance.count"] == 2
    assert needed["compute.instance.preemptible.count"] == 2
    assert needed["compute.instance.gpu.rtx6000"] == 1
    assert needed["compute.disk.count"] == 4


@pytest.mark.parametrize("gpus_per_node", [1, 8])
def test_preemptible_gpu_preflight_accepts_zero_regular_quotas(gpus_per_node):
    allowances = _allowances()
    allowances["compute.instance.count"]["spec"]["limit"] = "0"
    allowances["compute.instance.preemptible.count"]["spec"]["limit"] = "1"
    assert _preflight([_cluster(gpus_per_node=gpus_per_node)], allowances) == []


def test_regular_gpu_workers_still_require_regular_family_quota():
    shortfalls = _preflight([_cluster(preemptible=False)], _allowances())
    assert [(item.name, item.required, item.available) for item in shortfalls] == [
        ("compute.instance.gpu.rtx6000", 1, 0)
    ]


def test_preemptible_gpu_pool_does_not_exempt_regular_cpu_workers():
    allowances = _allowances()
    allowances["compute.instance.count"]["spec"]["limit"] = "0"
    shortfalls = _preflight([_cluster(cpu_count=1)], allowances)
    assert [(item.name, item.required) for item in shortfalls] == [
        ("compute.instance.count", 1)
    ]


def test_exhausted_preemptible_vm_quota_blocks_despite_regular_gpu_capacity():
    allowances = _allowances()
    allowances["compute.instance.gpu.rtx6000"]["spec"]["limit"] = "16"
    allowances["compute.instance.preemptible.count"]["status"]["usage"] = "128"
    shortfalls = _preflight([_cluster(gpus_per_node=8)], allowances)
    assert [(item.name, item.required, item.available) for item in shortfalls] == [
        ("compute.instance.preemptible.count", 1, 0)
    ]


def test_missing_preemptible_vm_allowance_fails_closed():
    allowances = _allowances()
    del allowances["compute.instance.preemptible.count"]
    with pytest.raises(ValueError, match="omitted.*compute.instance.preemptible.count"):
        _preflight([_cluster()], allowances)


@pytest.mark.parametrize("malformation", ["missing-usage", "wrong-unit", "inactive"])
def test_malformed_preemptible_vm_allowance_fails_closed(malformation):
    allowances = _allowances()
    status = allowances["compute.instance.preemptible.count"]["status"]
    if malformation == "missing-usage":
        del status["usage"]
    elif malformation == "wrong-unit":
        status["unit"] = "gpu"
    else:
        status["state"] = "STATE_FROZEN"
    with pytest.raises(ValueError, match="compute.instance.preemptible.count"):
        _preflight([_cluster()], allowances)


@pytest.mark.parametrize(
    "quota", ["compute.disk.count", "compute.disk.size.network-ssd"]
)
def test_preemptible_capacity_never_bypasses_boot_disk_quota(quota):
    allowances = _allowances()
    allowances[quota]["spec"]["limit"] = "0"
    shortfalls = _preflight([_cluster()], allowances)
    assert [item.name for item in shortfalls] == [quota]


def test_switching_allocation_mode_preserves_all_other_requirements():
    regular = _cluster(preemptible=False, gpu_count=2, gpus_per_node=8, cpu_count=1)
    preemptible = replace(
        regular, gpu_nodes=replace(regular.gpu_nodes, preemptible=True)
    )
    instance_names = {
        "compute.instance.count",
        "compute.instance.preemptible.count",
        "compute.instance.gpu.rtx6000",
    }

    def other(cluster):
        return {
            name: amount
            for name, amount in required_quotas([cluster]).items()
            if name not in instance_names
        }

    assert other(regular) == other(preemptible)
