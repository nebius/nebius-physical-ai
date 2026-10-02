"""Exercise actual Isaac placement constraints before driver rejection."""

import json
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow.sim2real_driver_preflight import (
    isaac_render_placements,
    node_can_host_isaac,
)
from npa.orchestration.npa_workflow.sim2real_preflight import (
    _managed_driver_isaac_nodes,
)


def _node():
    return {
        "metadata": {
            "name": "managed-pool",
            "labels": {
                "nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
                "kubernetes.io/os": "linux",
                "pool": "compute",
                "nebius.com/driverful": "true",
                "nvidia.com/gpu.deploy.operands": "false",
            },
        },
        "spec": {},
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "allocatable": {"nvidia.com/gpu": "1", "cpu": "24", "memory": "218Gi"},
        },
    }


def _workflow(pod=None):
    return SimpleNamespace(
        config={},
        states={
            name: SimpleNamespace(resources="isaac")
            for name in (
                "stage-07-rollouts",
                "stage-09-ppo",
                "stage-10-gold",
            )
        },
        resources={
            "isaac": {
                "cloud": "kubernetes",
                "accelerators": "RTXPRO6000:1",
                "cpus": 16,
                "memory": "128Gi",
                "kubernetes": {"pod_config": {"spec": pod or {}}},
            }
        },
    )


@pytest.mark.parametrize(
    "excluded",
    ["cordoned", "unready", "tainted", "selector", "allowlist", "memory", "gpu"],
)
def test_excluded_managed_pool_does_not_block_operator_pool(excluded):
    node = _node()
    placements = isaac_render_placements(_workflow(), context="synthetic")
    if excluded == "cordoned":
        node["spec"]["unschedulable"] = True
    elif excluded == "unready":
        node["status"]["conditions"][0]["status"] = "False"
    elif excluded == "tainted":
        node["spec"]["taints"] = [
            {"key": "dedicated", "value": "compute", "effect": "NoSchedule"}
        ]
    elif excluded == "selector":
        for placement in placements:
            placement["pod_spec"]["nodeSelector"]["pool"] = "render"
    elif excluded == "allowlist":
        for placement in placements:
            placement["allowed_nodes"] = ("operator-pool",)
    elif excluded == "memory":
        node["status"]["allocatable"]["memory"] = "64Gi"
    else:
        node["status"]["allocatable"]["nvidia.com/gpu"] = "0"
    operator = _node()
    operator["metadata"]["name"] = "operator-pool"
    operator["metadata"]["labels"].update(
        {"pool": "render", "nvidia.com/gpu.deploy.operands": "true"}
    )
    assert (
        _managed_driver_isaac_nodes(
            json.dumps({"items": [node, operator]}), placements=placements
        )
        == []
    )


@pytest.mark.parametrize("effect", ["NoSchedule", "NoExecute"])
def test_matching_toleration_keeps_managed_node_in_driver_check(effect):
    node = _node()
    node["spec"]["taints"] = [
        {"key": "dedicated", "value": "compute", "effect": effect}
    ]
    placements = isaac_render_placements(
        _workflow(
            {
                "tolerations": [
                    {
                        "key": "dedicated",
                        "operator": "Equal",
                        "value": "compute",
                        "effect": effect,
                    },
                ]
            }
        ),
        context="synthetic",
    )
    assert _managed_driver_isaac_nodes(
        json.dumps({"items": [node]}), placements=placements
    ) == ["managed-pool"]


def test_default_gpu_noschedule_toleration_is_preserved():
    node = _node()
    node["spec"]["taints"] = [{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]
    assert node_can_host_isaac(node)
    node["spec"]["taints"][0]["effect"] = "NoExecute"
    assert not node_can_host_isaac(node)


def test_global_context_and_task_placement_follow_skypilot_merge(tmp_path):
    config = tmp_path / "sky.yaml"
    _write_layered_config(config)
    placements = isaac_render_placements(
        _workflow({"nodeSelector": {"zone": "task", "pool": "compute"}}),
        context="synthetic",
        global_config_path=config,
        allowed_nodes=("managed-pool",),
    )
    assert len(placements) == 3
    assert placements[0]["pod_spec"]["nodeSelector"] == {
        "kubernetes.io/os": "linux",
        "zone": "task",
        "pool": "compute",
        "global": "yes",
        "context": "yes",
    }
    assert len(placements[0]["pod_spec"]["tolerations"]) == 2
    node = _node()
    assert not node_can_host_isaac(node, placements)
    node["metadata"]["labels"].update(
        {"zone": "task", "global": "yes", "context": "yes"}
    )
    assert node_can_host_isaac(node, placements)


def test_required_node_affinity_scopes_the_driver_check():
    placements = isaac_render_placements(
        _workflow(
            {
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "pool",
                                            "operator": "In",
                                            "values": ["render"],
                                        }
                                    ]
                                },
                            ]
                        },
                    }
                }
            }
        ),
        context="synthetic",
    )
    assert not node_can_host_isaac(_node(), placements)
    node = _node()
    node["metadata"]["labels"]["pool"] = "render"
    assert node_can_host_isaac(node, placements)


@pytest.mark.parametrize("missing", ["conditions", "allocatable"])
def test_missing_node_evidence_cannot_be_reported_as_safe(missing):
    node = _node()
    del node["status"][missing]
    with pytest.raises(ValueError, match="evidence is missing"):
        node_can_host_isaac(node)


def test_unknown_affinity_fails_closed():
    placements = isaac_render_placements(
        _workflow(
            {
                "affinity": {
                    "podAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": [
                            {"topologyKey": "zone"}
                        ],
                    }
                }
            }
        ),
        context="synthetic",
    )
    with pytest.raises(RuntimeError, match="cannot authoritatively evaluate"):
        node_can_host_isaac(_node(), placements)


def test_effective_gpu_override_changes_candidate_membership(monkeypatch):
    monkeypatch.setenv("NPA_WORKFLOW_GPU_ACCELERATOR", "B200:1")
    placements = isaac_render_placements(_workflow(), context="synthetic")
    assert placements[0]["accelerator"] == "B200:1"
    assert not node_can_host_isaac(_node(), placements)


@pytest.mark.parametrize(
    "document", [[], {"kubernetes": []}, {"kubernetes": {"context_configs": ["bad"]}}]
)
def test_malformed_sky_placement_is_rejected(tmp_path, document):
    config = tmp_path / "sky.yaml"
    config.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        isaac_render_placements(
            _workflow(), context="synthetic", global_config_path=config
        )


def _write_layered_config(config):
    config.write_text(
        yaml.safe_dump(
            {
                "kubernetes": {
                    "pod_config": {
                        "spec": {"nodeSelector": {"zone": "global", "global": "yes"}}
                    },
                    "context_configs": {
                        "synthetic": {
                            "pod_config": {
                                "spec": {
                                    "nodeSelector": {
                                        "zone": "context",
                                        "context": "yes",
                                    },
                                    "tolerations": [
                                        {"key": "dedicated", "operator": "Exists"}
                                    ],
                                }
                            }
                        }
                    },
                }
            }
        )
    )
