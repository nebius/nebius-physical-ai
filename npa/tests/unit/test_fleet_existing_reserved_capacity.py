"""A repair reuses only exact, authoritative capacity already provisioned."""

from copy import deepcopy
import json
from subprocess import CompletedProcess

import pytest

from npa.cluster_backends import mk8s_execution as E
from npa.cluster_backends.mk8s_render import render_tfvars
from npa.fleet.spec import ClusterSpec, NodePoolSpec, ProjectSpec


@pytest.fixture
def existing(tmp_path, monkeypatch):
    cluster = ClusterSpec(
        name="render",
        gpu_workload_profile="rtx-rendering",
        cpu_nodes=NodePoolSpec(count=3, platform="cpu-d3", preset="48vcpu-192gb"),
        gpu_nodes=NodePoolSpec(
            count=2,
            platform="gpu-rtx6000-a",
            preset="8gpu-192vcpu-1744gb",
            capacity_block_group="capacityblockgroup-test",
        ),
    )
    project = ProjectSpec(
        name="public-alias", project_id="project-test", clusters=[cluster]
    )
    install = tmp_path / project.key() / cluster.name
    workdir = install / "k8s-training"
    workdir.mkdir(parents=True)
    (workdir / "terraform.tfvars").write_text(
        "\n".join(
            line
            for line in render_tfvars(cluster).splitlines()
            if not line.startswith("gpu_operator_rtx_driver_profile")
        )
    )
    saved = {
        "status": "validating-gpu-health",
        "tenant_id": "tenant-test",
        "project_id": "project-test",
        "region": "uk-south2",
        "cluster_name": "render",
        "cluster_id": "cluster-test",
    }
    groups = []
    for pool, count in [
        (cluster.cpu_nodes, 3),
        (cluster.gpu_nodes, 1),
        (cluster.gpu_nodes, 1),
    ]:
        template = {
            "resources": {"platform": pool.platform, "preset": pool.preset},
            "preemptible": False,
        }
        if pool.capacity_block_group:
            template["reservation_policy"] = {
                "policy": "STRICT",
                "reservation_ids": [pool.capacity_block_group],
            }
        groups.append(
            {
                "spec": {"fixed_node_count": count, "template": template},
                "status": {"state": "RUNNING"},
            }
        )
    provider_cluster = {
        "metadata": {
            "id": "cluster-test",
            "parent_id": "project-test",
            "name": "render",
        },
        "status": {"state": "RUNNING"},
    }
    monkeypatch.setattr(E, "_load_env_sidecar", lambda _: saved)
    monkeypatch.setattr(
        E,
        "_get_project",
        lambda *args: {
            "metadata": {
                "id": "project-test",
                "parent_id": "tenant-test",
                "name": "different-private-name",
            },
            "spec": {"region": "uk-south2"},
        },
    )
    monkeypatch.setattr(
        E,
        "_run_capture",
        lambda cmd, **kwargs: CompletedProcess(
            cmd,
            0,
            json.dumps({"items": groups} if "node-group" in cmd else provider_cluster),
            "",
        ),
    )
    kwargs = dict(
        project=project,
        cluster=cluster,
        prefix="example-",
        tenant_id="tenant-test",
        region="uk-south2",
        ssh_public_key="",
        fleet_root=tmp_path,
        nebius_bin="nebius",
        profile="selected",
        env={},
    )
    return kwargs, saved, groups, provider_cluster, workdir


def test_repair_verifies_split_reserved_pools_without_charging_them_again(existing):
    kwargs, _, groups, *_ = existing
    # The actual v1 API represents non-preemption by omitting an Empty marker.
    groups[0]["spec"]["template"].pop("preemptible")
    assert E._is_verified_unchanged_target(**kwargs)


def test_v1_preemptible_marker_cannot_satisfy_strict_reserved_pool(existing):
    kwargs, _, groups, *_ = existing
    groups[-1]["spec"]["template"]["preemptible"] = {}
    assert not E._is_verified_unchanged_target(**kwargs)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "extra",
        "wrong-reservation",
        "auto",
        "not-running",
        "wrong-project",
        "unknown-status",
    ],
)
def test_uncertain_or_changed_capacity_remains_fail_closed(existing, failure):
    kwargs, saved, groups, cluster, _ = existing
    if failure == "missing":
        groups.pop()
    if failure == "extra":
        groups.append(deepcopy(groups[-1]))
    if failure == "wrong-reservation":
        groups[-1]["spec"]["template"]["reservation_policy"]["reservation_ids"] = [
            "capacityblockgroup-other"
        ]
    if failure == "auto":
        groups[-1]["spec"]["template"]["reservation_policy"]["policy"] = "AUTO"
    if failure == "not-running":
        groups[-1]["status"]["state"] = "PROVISIONING"
    if failure == "wrong-project":
        cluster["metadata"]["parent_id"] = "project-other"
    if failure == "unknown-status":
        saved["status"] = "destroy-incomplete"
    assert not E._is_verified_unchanged_target(**kwargs)


@pytest.mark.parametrize("valid_identity", [True, False])
def test_partial_apply_requires_exact_local_and_live_cluster_identity(
    existing, valid_identity
):
    kwargs, saved, _, _, workdir = existing
    saved.pop("cluster_id")
    saved["status"] = "provisioning"
    (workdir / "terraform.tfstate").write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "mode": "managed",
                        "type": "nebius_mk8s_v1_cluster",
                        "instances": [
                            {
                                "attributes": {
                                    "id": "cluster-test",
                                    "name": "render",
                                    "parent_id": "project-test"
                                    if valid_identity
                                    else "project-other",
                                }
                            }
                        ],
                    }
                ]
            }
        )
    )
    assert E._is_verified_unchanged_target(**kwargs) is valid_identity
