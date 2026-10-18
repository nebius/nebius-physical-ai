"""Read-only verification of Nebius default project quota responses.

Set NPA_INTEGRATION_E2E=1 and NPA_PROJECT_QUOTA_LIVE_CONFIG to an owner-only
JSON file with project_id, tenant_id, region, and profile. Select a project with
no explicit caps on the required reserved five-GPU topology quotas. No resources
are created.
"""

import json
import os
from pathlib import Path

import pytest

from npa.provisioning_preflight import (
    build_whole_path_plan,
    read_project_quota_observations,
    resolve_topology,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or not os.environ.get("NPA_PROJECT_QUOTA_LIVE_CONFIG"),
        reason="requires an explicit private project quota target",
    ),
]


def test_live_default_project_quotas_allow_reserved_five_gpu_plan(monkeypatch):
    path = Path(os.environ["NPA_PROJECT_QUOTA_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "live config must be owner-only"
    config = json.loads(path.read_text())
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", config["profile"])
    topology = resolve_topology(
        gpu_nodes=5,
        gpu_platform="gpu-rtx6000",
        gpu_preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-example",
    )
    observations = read_project_quota_observations(
        config["project_id"], config["region"], tuple(topology.quota_requirements())
    )
    assert all(item.state == "unbounded" for item in observations.values())
    plan = build_whole_path_plan(
        project_alias="live",
        project_id=config["project_id"],
        tenant_id=config["tenant_id"],
        region=config["region"],
        topology=topology,
        quota_reader=lambda *_args: observations,
        mutation=True,
    )
    plan.assert_mutation_ready()
    assert plan.decision == "ready"
