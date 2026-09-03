from __future__ import annotations

import pytest

from npa.provisioning_preflight import (
    DISK_QUOTA,
    GIB,
    INSTANCE_QUOTA,
    NETWORK_SSD_BYTES_QUOTA,
    PUBLIC_IP_QUOTA,
    PreflightBlockedError,
    QuotaObservation,
    assess_quota,
    build_whole_path_plan,
    discover_existing_capacity,
    parse_quota_allowances,
    resolve_topology,
)


def _reader(values):
    def read(_tenant, _region, names):
        return {
            name: QuotaObservation(
                name=name,
                used=values.get(
                    name, (0, 20 * 1024 * GIB if name == NETWORK_SSD_BYTES_QUOTA else 20)
                )[0],
                limit=values.get(
                    name, (0, 20 * 1024 * GIB if name == NETWORK_SSD_BYTES_QUOTA else 20)
                )[1],
                state="known",
            )
            for name in names
        }

    return read


def _plan(*, topology=None, values=None, mutation=True):
    return build_whole_path_plan(
        project_alias="p",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        topology=topology or resolve_topology(),
        quota_reader=_reader(values or {}),
        mutation=mutation,
    )


def test_missing_usage_with_limit_is_zero() -> None:
    parsed = parse_quota_allowances(
        {
            "items": [
                {
                    "metadata": {"name": INSTANCE_QUOTA},
                    "spec": {"region": "eu-north1", "limit": "4"},
                    "status": {},
                }
            ]
        },
        region="eu-north1",
        names=[INSTANCE_QUOTA],
    )
    assert parsed[INSTANCE_QUOTA].used == 0
    assert parsed[INSTANCE_QUOTA].limit == 4


@pytest.mark.parametrize(
    "name", [INSTANCE_QUOTA, DISK_QUOTA, NETWORK_SSD_BYTES_QUOTA, PUBLIC_IP_QUOTA]
)
def test_each_hard_quota_can_block(name: str) -> None:
    topology = resolve_topology(public_node_ips=name == PUBLIC_IP_QUOTA)
    plan = _plan(topology=topology, values={name: (10, 10)})
    assert plan.decision == "blocked"
    assert {item.name for item in plan.quotas if item.status == "blocked"} == {name}


def test_multiple_simultaneous_deficits_are_reported() -> None:
    plan = _plan(
        values={
            INSTANCE_QUOTA: (10, 10),
            DISK_QUOTA: (9, 10),
            PUBLIC_IP_QUOTA: (0, 20),
            "compute.instance.gpu.rtx6000": (1, 1),
        }
    )
    blocked = {item.name for item in plan.quotas if item.status == "blocked"}
    assert blocked == {
        INSTANCE_QUOTA,
        DISK_QUOTA,
        "compute.instance.gpu.rtx6000",
    }
    assert all(
        "required new limit=" in item.reason
        for item in plan.quotas
        if item.status == "blocked"
    )


def test_resume_counts_only_missing_resources() -> None:
    topology = resolve_topology(
        cpu_nodes=2,
        existing_cpu_nodes=1,
        gpu_nodes=2,
        existing_gpu_nodes=2,
    )
    plan = _plan(topology=topology)
    assert topology.required_instances == 1
    assert topology.required_disks == 1
    assert topology.required_gpus == 0
    assert INSTANCE_QUOTA in {item.name for item in plan.quotas}


def test_provider_resume_counts_one_matching_node_group_and_no_kubeconfig() -> None:
    from types import SimpleNamespace

    class Provider:
        def get_cluster(self, name, *, project_id):  # noqa: ANN001, ANN201
            assert name == "partial-cluster"
            assert project_id == "project-1"
            return SimpleNamespace(id="cluster-1")

        def list_node_groups(self, cluster_id):  # noqa: ANN001, ANN201
            assert cluster_id == "cluster-1"
            return [
                SimpleNamespace(node_count=1, platform="cpu-d3", preset="8vcpu-32gb"),
                SimpleNamespace(
                    node_count=0,
                    platform="gpu-rtx6000",
                    preset="1gpu-24vcpu-218gb",
                ),
            ]

    existing = discover_existing_capacity(
        project_id="project-1",
        cluster_name="partial-cluster",
        cpu_platform="cpu-d3",
        cpu_preset="8vcpu-32gb",
        gpu_platform="gpu-rtx6000",
        gpu_preset="1gpu-24vcpu-218gb",
        client=Provider(),
    )
    topology = resolve_topology(
        existing_cpu_nodes=existing.cpu_nodes,
        existing_gpu_nodes=existing.gpu_nodes,
    )

    assert existing.check.status == "ready"
    assert topology.required_instances == 1
    assert topology.required_disks == 1
    assert topology.required_gpus == 1


def test_unbounded_quota_is_explicitly_ready() -> None:
    decision = assess_quota(
        QuotaObservation(name=DISK_QUOTA, state="unbounded"), required=2
    )
    assert decision.status == "unbounded"
    assert decision.shortfall == 0


def test_unsupported_and_malformed_responses_are_unknown() -> None:
    unsupported = parse_quota_allowances(
        {"items": []}, region="eu-north1", names=[DISK_QUOTA]
    )[DISK_QUOTA]
    malformed = parse_quota_allowances(
        {"items": "denied"}, region="eu-north1", names=[DISK_QUOTA]
    )[DISK_QUOTA]
    assert unsupported.state == "unsupported"
    assert malformed.state == "unknown"


def test_rbac_failure_is_unknown_for_plan_and_closed_for_mutation() -> None:
    def denied(_tenant, _region, _names):
        raise PermissionError("RBAC denied")

    kwargs = dict(
        project_alias="p",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        topology=resolve_topology(),
        quota_reader=denied,
    )
    readonly = build_whole_path_plan(**kwargs, mutation=False)
    mutation = build_whole_path_plan(**kwargs, mutation=True)
    assert readonly.decision == "unknown"
    assert mutation.decision == "blocked"
    assert all(
        item.status == "unknown" for item in mutation.quotas if item.required > 0
    )
    with pytest.raises(PreflightBlockedError, match="no resources were created"):
        mutation.assert_mutation_ready()


def test_strict_reservation_backed_pool_skips_ordinary_gpu_quota() -> None:
    topology = resolve_topology(
        capacity_block_group="capacityblockgroup-1", gpu_nodes=2
    )
    requirements = topology.quota_requirements()
    # Hard instance/disk/IP arithmetic is unchanged.
    assert requirements[INSTANCE_QUOTA] == 3  # 2 GPU + 1 CPU node
    assert requirements[DISK_QUOTA] == 3
    # The bound STRICT reservation replaces the ordinary GPU-family allowance.
    assert "compute.instance.gpu.rtx6000" not in requirements


def test_on_demand_non_preemptible_pool_keeps_gpu_quota() -> None:
    topology = resolve_topology(gpu_nodes=2, gpu_preset="1gpu-24vcpu-218gb")
    requirements = topology.quota_requirements()
    assert requirements.get("compute.instance.gpu.rtx6000") == 2


def test_preemptible_nodes_still_consume_hard_quotas() -> None:
    topology = resolve_topology(preemptible=True)
    requirements = topology.quota_requirements()
    assert requirements[INSTANCE_QUOTA] == 2
    assert requirements[DISK_QUOTA] == 2
    assert requirements[NETWORK_SSD_BYTES_QUOTA] == 1151 * GIB
    assert "compute.instance.gpu.rtx6000" not in requirements


def test_whole_path_network_ssd_reproduced_shortfall_is_exact() -> None:
    topology = resolve_topology(agent_requested=True)
    plan = _plan(
        topology=topology,
        values={NETWORK_SSD_BYTES_QUOTA: (0, 21 * GIB)},
    )
    decision = next(
        item for item in plan.quotas if item.name == NETWORK_SSD_BYTES_QUOTA
    )

    assert topology.required_network_ssd_bytes == 1251 * GIB
    assert decision.required == 1251 * GIB
    assert decision.available == 21 * GIB
    assert decision.shortfall == 1230 * GIB
    assert decision.to_dict()["required_gib"] == "1251"
    assert decision.to_dict()["available_gib"] == "21"
    assert decision.to_dict()["shortfall_gib"] == "1230"
    assert plan.decision == "blocked"


@pytest.mark.parametrize("available_gib", [1251, 1252])
def test_network_ssd_exact_boundary_and_sufficient_quota(available_gib: int) -> None:
    plan = _plan(
        topology=resolve_topology(agent_requested=True),
        values={NETWORK_SSD_BYTES_QUOTA: (7 * GIB, (7 + available_gib) * GIB)},
    )
    decision = next(
        item for item in plan.quotas if item.name == NETWORK_SSD_BYTES_QUOTA
    )
    assert decision.status == "ready"
    assert decision.shortfall == 0


def test_multiple_node_counts_and_existing_delta_use_exact_disk_bytes() -> None:
    topology = resolve_topology(
        agent_requested=True,
        agent_exists=True,
        cpu_nodes=3,
        existing_cpu_nodes=1,
        gpu_nodes=4,
        existing_gpu_nodes=2,
        cpu_disk_gib=64,
        gpu_disk_gib=512,
    )
    assert topology.required_disks == 4
    assert topology.required_network_ssd_bytes == (2 * 64 + 2 * 512) * GIB


def test_preemptible_does_not_change_disk_byte_requirement() -> None:
    on_demand = resolve_topology(cpu_nodes=2, gpu_nodes=3, preemptible=False)
    preemptible = resolve_topology(cpu_nodes=2, gpu_nodes=3, preemptible=True)
    assert (
        preemptible.required_network_ssd_bytes
        == on_demand.required_network_ssd_bytes
    )


def test_disk_count_and_disk_bytes_are_independent_quota_decisions() -> None:
    plan = _plan(
        values={
            DISK_QUOTA: (0, 2),
            NETWORK_SSD_BYTES_QUOTA: (0, 100 * GIB),
        }
    )
    decisions = {item.name: item for item in plan.quotas}
    assert decisions[DISK_QUOTA].status == "ready"
    assert decisions[NETWORK_SSD_BYTES_QUOTA].status == "blocked"


def test_missing_and_contradictory_disk_byte_allowance_fail_closed() -> None:
    payload = {
        "items": [
            {
                "metadata": {"name": NETWORK_SSD_BYTES_QUOTA},
                "spec": {"region": "eu-north1", "limit": str(2 * GIB)},
                "status": {"usage": "0"},
            },
            {
                "metadata": {"name": NETWORK_SSD_BYTES_QUOTA},
                "spec": {"region": "eu-north1", "limit": str(3 * GIB)},
                "status": {"usage": "0"},
            },
        ]
    }
    parsed = parse_quota_allowances(
        payload, region="eu-north1", names=[NETWORK_SSD_BYTES_QUOTA]
    )
    assert parsed[NETWORK_SSD_BYTES_QUOTA].state == "unknown"
    assert "contradictory" in parsed[NETWORK_SSD_BYTES_QUOTA].reason

    def reader(_tenant, _region, names):
        return {name: parsed.get(name, QuotaObservation(name=name)) for name in names}

    plan = build_whole_path_plan(
        project_alias="p",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        topology=resolve_topology(),
        quota_reader=reader,
        mutation=True,
    )
    assert plan.decision == "blocked"


def test_provider_available_must_equal_limit_minus_usage() -> None:
    parsed = parse_quota_allowances(
        {
            "items": [
                {
                    "metadata": {"name": NETWORK_SSD_BYTES_QUOTA},
                    "spec": {"region": "eu-north1", "limit": str(100 * GIB)},
                    "status": {
                        "usage": str(20 * GIB),
                        "available": str(79 * GIB),
                    },
                }
            ]
        },
        region="eu-north1",
        names=[NETWORK_SSD_BYTES_QUOTA],
    )
    assert parsed[NETWORK_SSD_BYTES_QUOTA].state == "unknown"
    assert "limit - status.usage" in parsed[NETWORK_SSD_BYTES_QUOTA].reason


def test_canonical_paidf_shape_is_exact() -> None:
    topology = resolve_topology(accelerator="RTXPRO6000:1")
    assert topology.cpu_nodes == 1
    assert topology.cpu_platform == "cpu-d3"
    assert topology.cpu_preset == "8vcpu-32gb"
    assert topology.gpu_nodes == 1
    assert topology.gpu_platform == "gpu-rtx6000"
    assert topology.gpu_preset == "1gpu-24vcpu-218gb"
    assert topology.gpu_preemptible is False
