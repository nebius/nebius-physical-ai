"""Verify project-only setup with real IAM denial and an optional owned lifecycle.

Set NPA_INTEGRATION_E2E=1 and NPA_PROJECT_SETUP_LIVE_CONFIG to an owner-only
JSON file containing project_id, tenant_id, region, profile, project_alias,
cluster_name and evidence_dir. Use isolated NPA_CONFIG_DIR and operation journals.
The principal must really be denied tenant quota reads. Setting create_cluster
to true additionally creates, validates and destroys one CPU-only test cluster.
An optional reservation_id and reserved_gpu_nodes verify project entitlement
without allocating GPUs or claiming tenant inventory is available.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import resolve_environment
from npa.clients.nebius import NebiusError, get_project_identity, is_permission_denied
from npa.cluster_backends import get_backend
from npa.cluster_backends.mk8s import MK8sApplyRequest
from npa.fleet.spec import ClusterSpec, NodePoolSpec
from npa.provisioning_preflight import (
    build_whole_path_plan,
    read_provider_quotas,
    resolve_topology,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or not os.environ.get("NPA_PROJECT_SETUP_LIVE_CONFIG"),
        reason="requires an explicit isolated project-only live test principal",
    ),
]


@pytest.fixture
def target(monkeypatch):
    path = Path(os.environ["NPA_PROJECT_SETUP_LIVE_CONFIG"])
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_mode & 0o077 == 0, "live target must be owner-only"
    config = json.loads(path.read_text())
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", config["profile"])
    identity = get_project_identity(config["project_id"], tenant_id=config["tenant_id"])
    assert identity and identity.region == config["region"]
    with pytest.raises(NebiusError) as denied:
        read_provider_quotas(
            config["tenant_id"], config["region"], ["compute.instance.count"]
        )
    assert is_permission_denied(str(denied.value)), "tenant IAM denial must be genuine"
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not evidence.is_symlink() and evidence.stat().st_mode & 0o077 == 0
    return config, evidence


def _preflight(config, *, reserved=False):
    gpu = (
        NodePoolSpec(
            count=int(config.get("reserved_gpu_nodes", 5)),
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            capacity_block_group=config["reservation_id"],
        )
        if reserved
        else None
    )
    cluster = ClusterSpec(
        name=config["cluster_name"],
        gpu_nodes=gpu,
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="8vcpu-32gb"),
    )
    messages = []
    report = get_backend("mk8s").preflight(
        cluster,
        MK8sApplyRequest(
            project_id=config["project_id"],
            tenant_id=config["tenant_id"],
            region=config["region"],
            profile=config["profile"],
            nebius_bin="nebius",
            provider_env=os.environ.copy(),
            provider_preflight=True,
            on_status=messages.append,
        ),
    )
    return report, messages


def test_live_project_only_whole_path_and_backend_preflight(target):
    config, evidence = target
    plan = build_whole_path_plan(
        project_alias=config["project_alias"],
        project_id=config["project_id"],
        tenant_id=config["tenant_id"],
        region=config["region"],
        topology=resolve_topology(cpu_nodes=1, gpu_nodes=0),
        mutation=True,
    )
    plan.assert_mutation_ready()
    assert any("unavailable due to RBAC" in item.reason for item in plan.checks)
    report, messages = _preflight(config)
    assert report["capacity_quota"] == "passed"
    assert any("tenant aggregate" in message for message in messages)
    (evidence / "project-only-preflight.json").write_text(
        json.dumps(
            {
                "tenant_quota_read": "PermissionDenied",
                "whole_path_decision": plan.decision,
                "backend_preflight": "passed",
                "provider_responses_mocked": False,
            },
            indent=2,
        )
    )


def test_live_project_only_strict_reservation_preflight(target):
    config, evidence = target
    if not config.get("reservation_id"):
        pytest.skip("requires an exact project reservation allowance")
    report, messages = _preflight(config, reserved=True)
    assert report["capacity_quota"] == "passed"
    (evidence / "project-only-reservation-preflight.json").write_text(
        json.dumps(
            {
                "backend_preflight": "passed",
                "provider_responses_mocked": False,
                "gpu_allocation_attempted": False,
                "tenant_capacity_proven": False,
                "project_reservation_fallback_observed": any(
                    "Project reservation allowances verified" in s for s in messages
                ),
            },
            indent=2,
        )
    )


def _invoke(args, evidence, name):
    result = CliRunner().invoke(app, args)
    (evidence / (name + ".log")).write_text(result.output)
    return result


def test_live_direct_cpu_cluster_creation_and_cleanup(target):
    config, evidence = target
    if not config.get("create_cluster"):
        pytest.skip("requires explicit owned test-cluster lifecycle authorization")
    assert os.environ.get("NPA_CONFIG_DIR") and os.environ.get(
        "NPA_OPERATION_JOURNAL_DIR"
    )
    saved = resolve_environment(config["project_alias"])
    assert saved and saved.project_id == config["project_id"]
    from npa.cluster.state import load_cluster_state

    _assert_unused_cluster(config)
    args = _cluster_create_args(config)
    created = None
    try:
        created = _invoke(args, evidence, "direct-cluster-create")
        assert created.exit_code == 0, "inspect private direct-cluster-create.log"
        state = load_cluster_state(config["cluster_name"])
        assert state and state.kubeconfig_path
    finally:
        cleanup = _destroy_cluster(config, evidence)
    (evidence / "direct-cluster-lifecycle.json").write_text(
        json.dumps(
            {
                "create_exit_code": created.exit_code,
                "destroy_exit_code": cleanup.exit_code,
                "cpu_nodes": 1,
                "gpu_nodes": 0,
                "provider_responses_mocked": False,
            },
            indent=2,
        )
    )


def _cluster_create_args(config):
    return [
        "cluster",
        "up",
        "--project",
        config["project_alias"],
        "--context",
        config["cluster_name"],
        "--cpu-nodes",
        "1",
        "--gpu-nodes",
        "0",
        "--skip-sky-smoke",
    ]


def _destroy_cluster(config, evidence):
    cleanup = _invoke(
        [
            "cluster",
            "down",
            "--project",
            config["project_alias"],
            "--context",
            config["cluster_name"],
            "--force",
        ],
        evidence,
        "direct-cluster-destroy",
    )
    assert cleanup.exit_code == 0, "inspect private direct-cluster-destroy.log"
    from npa.cluster.api import MK8sClient
    from npa.cluster.exceptions import ClusterNotFoundError

    with pytest.raises(ClusterNotFoundError):
        MK8sClient().get_cluster(
            config["cluster_name"], project_id=config["project_id"]
        )
    return cleanup


def _assert_unused_cluster(config):
    from npa.cluster.api import MK8sClient
    from npa.cluster.exceptions import ClusterNotFoundError
    from npa.cluster.state import load_cluster_state

    assert load_cluster_state(config["cluster_name"]) is None, (
        "choose an unused cluster identity"
    )
    with pytest.raises(ClusterNotFoundError):
        MK8sClient().get_cluster(
            config["cluster_name"], project_id=config["project_id"]
        )
