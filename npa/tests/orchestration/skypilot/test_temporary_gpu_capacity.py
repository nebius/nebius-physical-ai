"""Prove occupied capacity is retryable without weakening single-node fit."""

from dataclasses import replace

import pytest

from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuCatalogError,
    KubernetesGpuInventory,
    KubernetesGpuNode,
    PermanentlyUnsatisfiableAcceleratorError,
    TemporarilyUnavailableAcceleratorError,
    preflight_kubernetes_gpu_gang,
)


def inventory(*free):
    nodes = tuple(
        KubernetesGpuNode(
            name=f"node-{i}",
            ready=True,
            schedulable=True,
            products=("H100",),
            capacity=8,
            allocatable=8,
            committed=8 - count,
            free=count,
            allocatable_cpu_millis=128000,
            free_cpu_millis=128000,
            allocatable_memory_bytes=1024**4,
            free_memory_bytes=1024**4,
            allocatable_pods=110,
            free_pod_slots=100,
            allocatable_ephemeral_storage_bytes=950 * 10**9,
            free_ephemeral_storage_bytes=950 * 10**9,
        )
        for i, count in enumerate(free)
    )
    return KubernetesGpuInventory(
        context="test-context",
        ready_nodes=len(nodes),
        eligible_gpu_nodes=len(nodes),
        capacity=8 * len(nodes),
        allocatable=8 * len(nodes),
        products=("H100",),
        node_labels={},
        nodes=nodes,
    )


@pytest.mark.parametrize("free", [(3, 3, 0), (2, 2, 2), (0, 0, 0)])
def test_fragmentation_waits_then_fits_one_node(free):
    current = inventory(*free)
    with pytest.raises(TemporarilyUnavailableAcceleratorError):
        preflight_kubernetes_gpu_gang(current, accelerator="H100:6", node_count=1)
    released = replace(current.nodes[1], free=8, committed=0)
    current = replace(current, nodes=(current.nodes[0], released, current.nodes[2]))
    result = preflight_kubernetes_gpu_gang(current, accelerator="H100:6", node_count=1)
    assert result["selected_nodes"] == ["node-1"]


@pytest.mark.parametrize("shape", ["H100:9", "A100:1"])
def test_impossible_shape_does_not_wait_even_with_pending_jobs(shape):
    current = replace(inventory(8, 8), unbound_pending_gpu_pods=1)
    with pytest.raises(PermanentlyUnsatisfiableAcceleratorError):
        preflight_kubernetes_gpu_gang(current, accelerator=shape, node_count=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("free_cpu_millis", 0),
        ("free_memory_bytes", 0),
        ("free_pod_slots", 0),
        ("ready", False),
        ("schedulable", False),
    ],
)
def test_temporarily_unavailable_node_resources_wait(field, value):
    current = inventory(8)
    current = replace(current, nodes=(replace(current.nodes[0], **{field: value}),))
    with pytest.raises(TemporarilyUnavailableAcceleratorError):
        preflight_kubernetes_gpu_gang(
            current, accelerator="H100:6", node_count=1, cpus=48, memory="576Gi"
        )


@pytest.mark.parametrize(
    "options",
    [
        {"cpus": 256},
        {"memory": "2048Gi"},
        {"allowed_nodes": ["missing"]},
        {"pod_spec": {"nodeSelector": {"pool": "missing"}}},
    ],
)
def test_unsupported_node_constraints_do_not_wait(options):
    with pytest.raises(PermanentlyUnsatisfiableAcceleratorError):
        preflight_kubernetes_gpu_gang(
            inventory(8), accelerator="H100:6", node_count=1, **options
        )


def test_inventory_failure_never_becomes_capacity_wait():
    with pytest.raises(KubernetesGpuCatalogError, match="Forbidden"):
        preflight_kubernetes_gpu_gang(
            replace(inventory(8), error="Forbidden"), accelerator="H100:6", node_count=1
        )


def test_two_jobs_can_share_disjoint_capacity_on_one_node():
    result = preflight_kubernetes_gpu_gang(
        inventory(2), accelerator="H100:2", node_count=1
    )
    assert result["selected_nodes"] == ["node-0"]


def test_storage_commitments_wait_even_when_gpus_are_free():
    current = inventory(2)
    current = replace(
        current,
        nodes=(
            replace(
                current.nodes[0],
                committed_ephemeral_storage_bytes=500 * 10**9,
                free_ephemeral_storage_bytes=450 * 10**9,
            ),
        ),
    )
    with pytest.raises(
        TemporarilyUnavailableAcceleratorError, match="ephemeral-storage"
    ):
        preflight_kubernetes_gpu_gang(
            current, accelerator="H100:2", node_count=1, ephemeral_storage="500G"
        )
    assert (
        preflight_kubernetes_gpu_gang(
            current, accelerator="H100:2", node_count=1, ephemeral_storage="100G"
        )["ephemeral_storage_bytes_per_node"]
        == 100 * 10**9
    )
    with pytest.raises(PermanentlyUnsatisfiableAcceleratorError):
        preflight_kubernetes_gpu_gang(
            current, accelerator="H100:2", node_count=1, ephemeral_storage="1000G"
        )
