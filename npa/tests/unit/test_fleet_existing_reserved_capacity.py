"""A repair reuses only exact, authoritative capacity already provisioned."""

from copy import deepcopy
from dataclasses import replace
import json
from subprocess import CompletedProcess
from unittest.mock import Mock

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
        "",
        "missing-state",
        "wrong-parent",
        "wrong-id",
        "wrong-shape",
        "reserved-cpu",
        "extra",
        "gpu-growth",
    ],
)
@pytest.mark.parametrize("retry", [False, True])
def test_cpu_pool_removal_reuses_only_proven_unchanged_gpu_capacity(
    existing, failure, retry
):
    kwargs, groups, workdir = _cpu_removal(existing)
    cpu = groups[0]
    if retry:
        (workdir / "terraform.tfvars").write_text(render_tfvars(kwargs["cluster"]))
    if failure == "missing-state":
        (workdir / "terraform.tfstate").unlink()
    elif failure == "wrong-parent":
        cpu["metadata"]["parent_id"] = "cluster-other"
    elif failure == "wrong-id":
        cpu["metadata"]["id"] = "nodegroup-other"
    elif failure == "wrong-shape":
        cpu["spec"]["template"]["resources"]["preset"] = "32vcpu-128gb"
    elif failure == "reserved-cpu":
        cpu["spec"]["template"]["reservation_policy"] = {"policy": "STRICT"}
    elif failure == "extra":
        groups.append(deepcopy(groups[-1]))
    elif failure == "gpu-growth":
        kwargs["cluster"] = replace(
            kwargs["cluster"], gpu_nodes=replace(kwargs["cluster"].gpu_nodes, count=4)
        )
    assert E._is_verified_unchanged_target(**kwargs) is (not failure)


def _cpu_removal(existing):
    kwargs, _, groups, _, workdir = existing
    original = kwargs["cluster"]
    kwargs["cluster"] = replace(
        original, cpu_nodes=replace(original.cpu_nodes, count=0)
    )
    cpu = groups[0]
    cpu["metadata"] = {
        "id": "nodegroup-cpu",
        "parent_id": "cluster-test",
        "name": "render-ng-cpu",
    }
    cpu["status"]["state"] = "PROVISIONING"
    attributes = {**cpu["metadata"], "template": deepcopy(cpu["spec"]["template"])}
    state = {
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_node_group",
                "name": "cpu-only",
                "instances": [{"attributes": attributes}],
            }
        ]
    }
    (workdir / "terraform.tfstate").write_text(json.dumps(state))
    return kwargs, groups, workdir


@pytest.mark.parametrize("wrong_parent", [False, True])
def test_removed_cpu_taint_is_never_adopted_or_replaced(
    existing, monkeypatch, wrong_parent
):
    kwargs, _, workdir = _cpu_removal(existing)
    state = json.loads((workdir / "terraform.tfstate").read_text())
    instance = state["resources"][0]["instances"][0]
    instance["status"] = "tainted"
    if wrong_parent:
        instance["attributes"]["parent_id"] = "cluster-other"
    state["resources"].append(
        {
            "mode": "managed",
            "type": "nebius_mk8s_v1_cluster",
            "instances": [{"attributes": {"id": "cluster-test"}}],
        }
    )
    commands = []

    def capture(command, **options):
        commands.append(command)
        assert command == ["terraform", "state", "pull"]
        return CompletedProcess(command, 0, json.dumps(state), "")

    monkeypatch.setattr(E, "_run_capture", capture)
    args = dict(
        terraform_bin="terraform",
        workdir=workdir,
        env={},
        cluster=kwargs["cluster"],
        project_id="project-test",
        subnet_id="vpcsubnet-test",
        nebius_bin="nebius",
        profile="selected",
        on_status=None,
    )
    if wrong_parent:
        with pytest.raises(RuntimeError, match="no exact desired pool"):
            E._reconcile_tainted_node_groups(**args)
    else:
        assert E._reconcile_tainted_node_groups(**args) == {}
    assert len(commands) == 1


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


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "node-count",
        "target-count",
        "stopped",
        "missing-binding",
        "missing-disk",
        "wrong-template",
        "wrong-project",
        "wrong-cluster",
        "wrong-group",
        "duplicate",
        "unreadable",
    ],
)
@pytest.mark.parametrize("gpu_groups_first", [False, True])
def test_readiness_repair_requires_allocated_instance_evidence(
    existing,
    monkeypatch,
    failure,
    gpu_groups_first,
):
    kwargs, _, groups, _, _ = existing
    instances = []
    for index, group in enumerate(groups[1:]):
        group_id = f"node-group-test-{index}"
        group["metadata"] = {"id": group_id, "parent_id": "cluster-test"}
        group["status"] = {
            "state": "PROVISIONING",
            "node_count": "1",
            "target_node_count": "1",
        }
        template = group["spec"]["template"]
        template.update(
            network_interfaces=[{"subnet_id": "subnet-test"}],
            boot_disk={"type": "NETWORK_SSD", "size_gibibytes": 128},
        )
        instance_spec = deepcopy(template)
        instance_spec["boot_disk"] = {
            "managed_disk": {"spec": deepcopy(template["boot_disk"])}
        }
        instance_spec["gpu_cluster"] = {}
        instances.append(
            {
                "metadata": {
                    "id": f"instance-test-{index}",
                    "parent_id": "project-test",
                    "labels": {
                        "mk8s-cluster-id": "cluster-test",
                        "mk8s-node-group-id": group_id,
                    },
                },
                "spec": instance_spec,
                "status": {
                    "state": "RUNNING",
                    "reservation_id": "reservation-test",
                    "disk_attachments": [{}],
                },
            }
        )
    if failure == "node-count":
        groups[-1]["status"]["node_count"] = "0"
    if failure == "target-count":
        groups[-1]["status"]["target_node_count"] = "2"
    if failure == "stopped":
        instances[-1]["status"]["state"] = "STOPPED"
    if failure == "missing-binding":
        instances[-1]["status"].pop("reservation_id")
    if failure == "missing-disk":
        instances[-1]["status"]["disk_attachments"] = []
    if failure == "wrong-template":
        instances[-1]["spec"]["resources"]["preset"] = "1gpu-24vcpu-218gb"
    if failure == "wrong-project":
        instances[-1]["metadata"]["parent_id"] = "project-other"
    if failure == "wrong-cluster":
        instances[-1]["metadata"]["labels"]["mk8s-cluster-id"] = "cluster-other"
    if failure == "wrong-group":
        instances[-1]["metadata"]["labels"]["mk8s-node-group-id"] = "node-group-other"
    if failure == "duplicate":
        instances.append(deepcopy(instances[-1]))
    if gpu_groups_first:
        # Each GPU group is considered for the CPU pool before its own pool.
        groups.append(groups.pop(0))
    capture = E._run_capture
    allocation_probe = Mock(wraps=E._node_group_has_allocated_workers)
    monkeypatch.setattr(E, "_node_group_has_allocated_workers", allocation_probe)

    def run(cmd, **options):
        if "compute" in cmd:
            assert cmd[:3] == ["nebius", "--profile", "selected"]
            assert "--all" in cmd
            return CompletedProcess(
                cmd,
                1 if failure == "unreadable" else 0,
                json.dumps({"items": instances}),
                "",
            )
        return capture(cmd, **options)

    monkeypatch.setattr(E, "_run_capture", run)
    assert E._is_verified_unchanged_target(**kwargs) is (not failure)
    if failure == "unreadable":
        allocation_probe.assert_not_called()
    else:
        # Allocation evidence depends only on the provider item, regardless of
        # which desired pool is being compared or which earlier matches pop.
        assert [call.args[0] for call in allocation_probe.call_args_list] == groups
