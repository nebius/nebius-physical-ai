"""Read-only matching against an owned, GPU-cluster-attached live node group.

Set NPA_MK8S_MATCH_LIVE_CONFIG to private JSON containing desired cluster/pool
inputs, Terraform state, exact node-group ID, profile, and evidence directory.
Provision and GPU runtime validation remain the operator's supported NPA flow.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from npa.clients.nebius import nebius_cli_env
from npa.cluster_backends.mk8s_execution import (
    _decode_v1_node_group_preemptibility,
    _tainted_node_group_matches_desired,
)
from npa.cluster_backends.mk8s_model import MK8sDesired, MK8sNodePool

pytestmark = pytest.mark.e2e


def _live_inputs():
    selected = os.environ.get("NPA_MK8S_MATCH_LIVE_CONFIG")
    if not selected:
        pytest.skip("requires private inputs for an owned GPU-attached pool")
    config = json.loads(Path(selected).read_text())
    state = json.loads(Path(config["terraform_state"]).read_text())
    attributes = [
        instance["attributes"]
        for resource in state["resources"]
        if resource["type"] == "nebius_mk8s_v1_node_group"
        for instance in resource["instances"]
        if instance["attributes"]["id"] == config["node_group_id"]
    ]
    assert len(attributes) == 1
    pool = MK8sNodePool(**config["pool"])
    cluster = MK8sDesired(gpu_nodes=pool, **config["cluster"])
    assert pool.is_gpu() and cluster.resolved_enable_gpu_cluster()
    assert attributes[0]["template"]["gpu_cluster"]["id"]
    return config, attributes[0], pool, cluster


def _read_live_node_group(config):
    evidence = Path(config["evidence_dir"])
    assert evidence.is_dir() and evidence.stat().st_mode & 0o077 == 0
    result = subprocess.run(
        [
            "nebius",
            "--no-browser",
            "--no-check-update",
            "--profile",
            config["profile"],
            "mk8s",
            "node-group",
            "get",
            "--id",
            config["node_group_id"],
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        env=nebius_cli_env(),
        check=True,
    )
    (evidence / "live-node-group.json").write_text(result.stdout)
    return json.loads(result.stdout)


def test_tainted_match_live_gpu_attachment():
    config, attributes, pool, cluster = _live_inputs()
    payload = _read_live_node_group(config)
    assert payload["status"]["state"] == "RUNNING"
    kwargs = {
        "provider_payload": _decode_v1_node_group_preemptibility(payload),
        "state_attributes": attributes,
        "pool": pool,
        "cluster": cluster,
        "cluster_id": config["cluster_id"],
        "subnet_id": config["subnet_id"],
    }
    assert _tainted_node_group_matches_desired(**kwargs)
    conflicting = deepcopy(attributes)
    conflicting["template"]["gpu_cluster"]["id"] += "-different"
    assert not _tainted_node_group_matches_desired(
        **{**kwargs, "state_attributes": conflicting}
    )
