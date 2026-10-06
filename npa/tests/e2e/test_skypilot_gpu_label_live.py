"""Read-only live bridge-label checks; private config selects the exact cluster.

NPA_SKYPILOT_GPU_LABEL_LIVE_CONFIG supplies context, kubeconfig, accelerator,
cpus, memory, and evidence_path. Label readiness and currently free gang fit
are separate proofs. This test never labels nodes or launches a workload.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

from npa.orchestration.skypilot import k8s_gpu_catalog as catalog

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or not os.environ.get("NPA_SKYPILOT_GPU_LABEL_LIVE_CONFIG"),
        reason="requires an authorized owner-only exact GPU target",
    ),
]


def _read_live_inventory():
    path = Path(os.environ["NPA_SKYPILOT_GPU_LABEL_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0
    config = json.loads(path.read_text())
    assert config["context"] and config["kubeconfig"]
    inventory = catalog.discover_kubernetes_gpu_inventory(
        context=config["context"],
        kubeconfig=config["kubeconfig"],
    )
    assert not inventory.error and not inventory.diagnostics
    return config, inventory


def test_live_reviewed_gpu_labels_match_effective_task():
    config, inventory = _read_live_inventory()
    requested = catalog.parse_accelerator_request(config["accelerator"])
    candidates = [
        node
        for node in inventory.nodes
        if node.ready
        and node.schedulable
        and node.allocatable >= requested.quantity
        and catalog._known_skypilot_label(dict(node.labels)) == requested.name.lower()
    ]
    assert candidates, "exact reviewed GPU nodes must be Ready and allocatable"
    assert all(
        catalog._skypilot_node_label_ready(node, config["accelerator"])
        for node in candidates
    )
    source = Path(inspect.getfile(catalog))
    evidence = {
        "scope": "actual read-only node label readiness; no free-placement or workload proof",
        "reviewed_ready_node_count": len(candidates),
        "effective_task_label_verified": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inventory": inventory.to_dict(),
    }
    output = Path(config["evidence_path"])
    assert output.parent.stat().st_mode & 0o077 == 0
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(evidence, stream, indent=2)


def test_live_free_gpu_gang_passes_skypilot_label_gate():
    config, inventory = _read_live_inventory()
    requirements = {
        "accelerator": config["accelerator"],
        "node_count": 1,
        "cpus": config["cpus"],
        "memory": config["memory"],
    }
    try:
        catalog.preflight_kubernetes_gpu_gang(inventory, **requirements)
    except catalog.TemporarilyUnavailableAcceleratorError:
        pytest.skip(
            "actual GPU capacity is occupied; label readiness is checked separately"
        )
    result = catalog.preflight_skypilot_gpu_gang(inventory, **requirements)
    assert result["compatible_free_nodes"] >= 1
