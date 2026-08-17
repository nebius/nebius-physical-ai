from __future__ import annotations

import json

from npa.orchestration.npa_workflow.kubernetes_prerequisites import (
    cpu_millicores,
    integer_resource,
    memory_bytes,
    ready_schedulable_cpu_nodes,
)


def _node_document(*, spec=None, conditions=None) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "worker"},
                    "spec": spec or {},
                    "status": {
                        "allocatable": {
                            "cpu": "6",
                            "memory": "129e6",
                            "nvidia.com/gpu": "4",
                        },
                        "conditions": conditions
                        if conditions is not None
                        else [{"type": "Ready", "status": "True"}],
                    },
                }
            ]
        }
    )


def test_kubernetes_quantities_keep_resource_types_distinct() -> None:
    assert cpu_millicores("5900m") == 5900
    assert memory_bytes("129e6") == 129_000_000
    assert memory_bytes("24Gi") == 24 * 1024**3
    assert integer_resource("4") == 4
    assert integer_resource("4e0") == 4
    assert integer_resource("1500m") == 0


def test_shared_placement_accepts_gpu_nodes_and_rejects_scheduler_blockers() -> None:
    required = dict(
        minimum_cpu_millicores=6000,
        minimum_memory_bytes=129_000_000,
    )
    assert ready_schedulable_cpu_nodes(_node_document(), **required) == ["worker"]
    assert (
        ready_schedulable_cpu_nodes(
            _node_document(spec={"unschedulable": True}), **required
        )
        == []
    )
    assert (
        ready_schedulable_cpu_nodes(
            _node_document(spec={"taints": [{"effect": "NoExecute"}]}),
            **required,
        )
        == []
    )
    assert (
        ready_schedulable_cpu_nodes(
            _node_document(conditions=[{"type": "Ready", "status": "False"}]),
            **required,
        )
        == []
    )
