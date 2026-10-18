"""Prove project-admin capacity paths and retained failure boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
from subprocess import CompletedProcess

import pytest

from npa.cli.cluster import terraform_lifecycle as legacy
from npa.cluster_backends.project_preflight import TenantCapacityAccessDenied
from npa.cluster_backends.quotas import required_quotas
from npa.fleet import lifecycle as fleet
from npa.fleet.capacity_preflight import ProjectDemand, group_project_demands
from npa.fleet.spec import ClusterSpec, NodePoolSpec, ObjectStorageSpec, ProjectSpec
from npa.soperator import lifecycle as slurm
from npa.soperator.spec import SoperatorSpec, WorkerPoolSpec

REGION = "us-central1"
DENIED = "rpc error: code = PermissionDenied desc = access denied"
CLUSTER = ClusterSpec(name="test", cpu_nodes=NodePoolSpec(count=1))


def _catalog(parent, limit=None):
    names = set(required_quotas([CLUSTER])) | {"compute.filesystem.size.network-ssd"}
    rows = []
    for name in names:
        spec = {"region": REGION}
        if limit is not None:
            spec["limit"] = str(limit)
        rows.append(
            {
                "metadata": {"name": name, "parent_id": parent},
                "spec": spec,
                "status": {
                    "unit": "byte" if ".size." in name else "count",
                    "usage_state": "USAGE_STATE_NOT_USED",
                },
            }
        )
    return {"items": rows}


class CapacityProvider:
    def __init__(self):
        self.calls = []
        self.tenant_error = DENIED
        self.project_catalog = _catalog("project-test")
        self.tenant_catalog = _catalog("tenant-test", 10**15)
        self.capacity_limit = "40"
        self.capacity_status = {
            "state": "STATE_ACTIVE",
            "unit": "gpu",
            "usage_state": "USAGE_STATE_NOT_USED",
        }

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if "quota-allowance" in argv:
            parent = argv[argv.index("--parent-id") + 1]
            if parent == "tenant-test" and self.tenant_error:
                return CompletedProcess(argv, 7, "", self.tenant_error)
            payload = (
                self.tenant_catalog if parent == "tenant-test" else self.project_catalog
            )
        elif "capacity-allowance" in argv:
            project = argv[argv.index("--parent-id") + 1]
            payload = {
                "items": [
                    {
                        "metadata": {"parent_id": project},
                        "spec": {
                            "capacity_block_group_id": "capacityblockgroup-test",
                            "limit": self.capacity_limit,
                        },
                        "status": self.capacity_status,
                    }
                ]
            }
        elif argv[1:4] == ["iam", "project", "get"]:
            payload = {
                "metadata": {"id": "project-test", "parent_id": "tenant-test"},
                "status": {"region": REGION},
            }
        elif "capacity-block-group" in argv:
            return CompletedProcess(argv, 7, "", DENIED)
        else:
            raise AssertionError(argv)
        return CompletedProcess(argv, 0, json.dumps(payload), "")


def _fleet_preflight(monkeypatch, provider, clusters, **overrides):
    monkeypatch.setattr(fleet, "_run_capture", provider)
    request = dict(
        tenant_id="tenant-test",
        by_region={REGION: clusters},
        new_projects_by_region={},
        env={"IDENTITY": "selected"},
        profile="selected",
        on_status=None,
        project_demands={REGION: [ProjectDemand("project-test", clusters)]},
    )
    fleet._preflight_quotas("nebius", **{**request, **overrides})


def test_fleet_project_fallback_preserves_profile_environment_and_demand(monkeypatch):
    provider = CapacityProvider()
    _fleet_preflight(monkeypatch, provider, [CLUSTER])
    assert any("project-test" in argv for argv, _ in provider.calls)
    assert all(argv[1:3] == ["--profile", "selected"] for argv, _ in provider.calls)
    assert all(
        kwargs["env"] == {"IDENTITY": "selected"} for _, kwargs in provider.calls
    )
    assert all("--all" in argv for argv, _ in provider.calls)


def test_fleet_combines_all_clusters_before_checking_project_limit(monkeypatch):
    provider = CapacityProvider()
    for row in provider.project_catalog["items"]:
        if row["metadata"]["name"] == "compute.instance.count":
            row["spec"]["limit"] = "1"
    with pytest.raises(ValueError, match="project quota allowances"):
        _fleet_preflight(
            monkeypatch, provider, [CLUSTER, replace(CLUSTER, name="second")]
        )


@pytest.mark.parametrize(
    "reason", ["Unauthenticated", "DeadlineExceeded", "Unavailable"]
)
def test_fleet_does_not_fallback_for_non_iam_failures(monkeypatch, reason):
    provider = CapacityProvider()
    provider.tenant_error = f"rpc error: code = {reason} desc = provider unavailable"
    with pytest.raises(ValueError):
        _fleet_preflight(monkeypatch, provider, [CLUSTER])
    assert not any("project-test" in argv for argv, _ in provider.calls)


def test_fleet_creating_projects_still_requires_tenant_evidence(monkeypatch):
    provider = CapacityProvider()
    with pytest.raises(TenantCapacityAccessDenied):
        _fleet_preflight(
            monkeypatch, provider, [CLUSTER], new_projects_by_region={REGION: 1}
        )
    assert not any("project-test" in argv for argv, _ in provider.calls)


def test_project_grouping_keeps_bucket_ownership_and_combines_aliases():
    first, second = ProjectSpec(name="first"), ProjectSpec(name="second")
    storage = ObjectStorageSpec(enabled=True, size_gibibytes=1)
    result = group_project_demands(
        [(first, REGION), (second, REGION)],
        {"first": "project-test", "second": "project-test"},
        {"first": [CLUSTER], "second": [replace(CLUSTER, name="second")]},
        {"second": [storage]},
    )
    assert len(result[REGION]) == 1
    assert len(result[REGION][0].clusters) == 2
    assert result[REGION][0].object_storage == [storage]


def _reserved_spec(size=2):
    worker = WorkerPoolSpec(
        name="worker",
        platform="gpu-b200-sxm",
        size=size,
        preset="8gpu-160vcpu-1792gb",
        fabric="us-central1-b",
        capacity_block_group="capacityblockgroup-test",
    )
    return SoperatorSpec(name="test", workers=[worker])


def _resolve_reserved(monkeypatch, tmp_path, provider, spec=None, env=None):
    monkeypatch.setattr(slurm, "_run_capture", provider)
    return slurm._resolve_reserved_worker_capacity(
        spec or _reserved_spec(),
        install_dir=tmp_path,
        nebius_bin="nebius",
        tenant_id="tenant-test",
        project_id="project-test",
        region=REGION,
        env=env or {},
    )


def test_reserved_soperator_uses_exact_project_allowance_and_retains_strict_ids(
    monkeypatch, tmp_path
):
    provider = CapacityProvider()
    original = _reserved_spec()
    resolved, summary = _resolve_reserved(monkeypatch, tmp_path, provider, original)
    assert (
        resolved.workers[0].resolved_capacity_block_group_id
        == "capacityblockgroup-test"
    )
    assert original.workers[0].resolved_capacity_block_group_id == ""
    assert summary[0]["reservation_evidence_scope"] == "project"
    assert summary[0]["capacity_mode"] == "reserved"
    assert any("capacity-allowance" in argv for argv, _ in provider.calls)


@pytest.mark.parametrize("source", ["explicit", "npa", "nebius", "both"])
def test_soperator_fallback_preserves_resolved_profile_and_environment(
    monkeypatch, tmp_path, source
):
    monkeypatch.delenv("NPA_NEBIUS_PROFILE", raising=False)
    monkeypatch.delenv("NEBIUS_PROFILE", raising=False)
    if source in {"npa", "both"}:
        monkeypatch.setenv("NPA_NEBIUS_PROFILE", "selected-profile")
    if source in {"explicit", "both"}:
        monkeypatch.setenv("NEBIUS_PROFILE", "other-profile")
    if source == "nebius":
        monkeypatch.setenv("NEBIUS_PROFILE", "selected-profile")
    if source == "explicit":
        monkeypatch.setenv("NPA_NEBIUS_PROFILE", "ambient-profile")
    env = slurm._nebius_cli_env("selected-profile" if source == "explicit" else "")
    provider, calls = CapacityProvider(), []

    def capture(argv, **kwargs):
        calls.append((argv, kwargs))
        normalized = [argv[0], *argv[3:]] if argv[1:2] == ["--profile"] else argv
        return provider(normalized, **kwargs)

    _resolve_reserved(monkeypatch, tmp_path, capture, env=env)
    assert all(kwargs["env"] == env for _, kwargs in calls)
    assert env["NEBIUS_PROFILE"] == "selected-profile"
    allowance_reads = [argv for argv, _ in calls if "capacity-allowance" in argv]
    assert allowance_reads and all(
        argv[1:3] == ["--profile", "selected-profile"] for argv in allowance_reads
    )


def test_reserved_soperator_aggregates_pools_sharing_one_allowance(
    monkeypatch, tmp_path
):
    provider = CapacityProvider()
    spec = _reserved_spec(size=3)
    spec.workers.append(replace(spec.workers[0], name="second"))
    with pytest.raises(ValueError, match="insufficient GPU headroom"):
        _resolve_reserved(monkeypatch, tmp_path, provider, spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "STATE_FROZEN"),
        ("unit", "count"),
        ("usage_state", "USAGE_STATE_UNKNOWN"),
    ],
)
def test_reserved_soperator_rejects_unusable_project_entitlement(
    monkeypatch, tmp_path, field, value
):
    provider = CapacityProvider()
    provider.capacity_status[field] = value
    with pytest.raises(ValueError):
        _resolve_reserved(monkeypatch, tmp_path, provider)


def test_reserved_soperator_requires_immutable_id_when_tenant_names_unreadable(
    monkeypatch, tmp_path
):
    spec = _reserved_spec()
    spec.workers[0] = replace(
        spec.workers[0],
        capacity_block_group="",
        capacity_block_group_name="private-reservation-name",
    )
    with pytest.raises(ValueError, match="immutable capacity_block_group ID"):
        _resolve_reserved(monkeypatch, tmp_path, CapacityProvider(), spec)


@pytest.mark.parametrize("limit", ["0", "1", "invalid", "-1"])
def test_reserved_soperator_rejects_small_or_invalid_allowance(
    monkeypatch, tmp_path, limit
):
    provider = CapacityProvider()
    provider.capacity_limit = limit
    with pytest.raises(ValueError):
        _resolve_reserved(monkeypatch, tmp_path, provider)


def _filestore_request():
    return {
        "enable_filestore": True,
        "filestore_disk_size_gibibytes": 1,
        "filesystem_csi_chart_repository": "https://example.invalid/charts",
        "tenant_id": "tenant-test",
        "parent_id": "project-test",
        "region": REGION,
    }


def test_legacy_filestore_accepts_omitted_project_limit_after_tenant_denial(
    monkeypatch,
):
    provider = CapacityProvider()
    original = deepcopy(provider.project_catalog)
    monkeypatch.setattr(legacy, "_run_capture", provider)
    legacy._preflight_filestore_quota("nebius", _filestore_request(), {})
    assert provider.project_catalog == original
    assert any("project-test" in argv for argv, _ in provider.calls)


@pytest.mark.parametrize("limit", ["0", "1", "invalid", "-1", None, False])
def test_legacy_filestore_rejects_explicit_small_or_invalid_limit(monkeypatch, limit):
    provider = CapacityProvider()
    for row in provider.project_catalog["items"]:
        if row["metadata"]["name"] == "compute.filesystem.size.network-ssd":
            row["spec"]["limit"] = limit
    monkeypatch.setattr(legacy, "_run_capture", provider)
    with pytest.raises(legacy.typer.BadParameter):
        legacy._preflight_filestore_quota("nebius", _filestore_request(), {})


@pytest.mark.parametrize(
    "payload",
    [{"items": []}, {"items": None}, {"items": [], "next_page_token": "more"}],
)
def test_legacy_filestore_rejects_missing_or_incomplete_project_evidence(
    monkeypatch, payload
):
    provider = CapacityProvider()
    provider.project_catalog = payload
    monkeypatch.setattr(legacy, "_run_capture", provider)
    with pytest.raises(legacy.typer.BadParameter):
        legacy._preflight_filestore_quota("nebius", _filestore_request(), {})


def test_denied_reservation_inventory_keeps_readable_aggregate_cpu_deficit(monkeypatch):
    provider = CapacityProvider()
    provider.tenant_error = ""
    for row in provider.tenant_catalog["items"]:
        if row["metadata"]["name"] == "compute.instance.non-gpu.vcpu":
            row["spec"]["limit"] = str(
                required_quotas([CLUSTER])["compute.instance.non-gpu.vcpu"]
            )
    gpu = NodePoolSpec(
        count=1,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-test",
    )
    first = replace(CLUSTER, gpu_nodes=gpu)
    second = replace(first, name="second")
    demands = [
        ProjectDemand("project-test", [first]),
        ProjectDemand("project-other", [second]),
    ]
    with pytest.raises(ValueError, match="compute.instance.non-gpu.vcpu"):
        _fleet_preflight(
            monkeypatch, provider, [first, second], project_demands={REGION: demands}
        )


@pytest.mark.parametrize(
    "platform,passes", [("gpu-b200-sxm", True), ("gpu-h200-sxm", False)]
)
@pytest.mark.parametrize("selector_kind", ["id", "name"])
def test_soperator_advice_denial_keeps_visible_reservation_constraints(
    monkeypatch, tmp_path, platform, passes, selector_kind
):
    provider = CapacityProvider()
    inventory = {
        "items": [
            {
                "metadata": {
                    "id": "capacityblockgroup-test",
                    "parent_id": "tenant-test",
                    "name": "reserved-test",
                },
                "status": {
                    "state": "STATE_ACTIVE",
                    "region": REGION,
                    "current_limit": "16",
                    "usage": "0",
                    "resource_affinity": {
                        "compute_v1": {"platform": platform, "fabric": "us-central1-b"}
                    },
                },
            }
        ]
    }

    def capture(argv, **kwargs):
        if "capacity-block-group" in argv:
            return CompletedProcess(argv, 0, json.dumps(inventory), "")
        if "resource-advice" in argv:
            return CompletedProcess(argv, 7, "", DENIED)
        return provider(argv, **kwargs)

    spec = _reserved_spec()
    if selector_kind == "name":
        spec.workers[0] = replace(
            spec.workers[0],
            capacity_block_group="",
            capacity_block_group_name="reserved-test",
        )
    if passes:
        resolved, _ = _resolve_reserved(monkeypatch, tmp_path, capture, spec)
        assert resolved.workers[0].resolved_capacity_block_group_id
    else:
        with pytest.raises(ValueError, match="platform"):
            _resolve_reserved(monkeypatch, tmp_path, capture, spec)


@pytest.mark.parametrize(
    "error", ["Unauthenticated", "DeadlineExceeded", "Unavailable"]
)
def test_soperator_does_not_fallback_on_other_tenant_failures(
    monkeypatch, tmp_path, error
):
    provider = CapacityProvider()

    def capture(argv, **kwargs):
        if "capacity-block-group" in argv:
            return CompletedProcess(argv, 7, "", f"rpc error: code = {error}")
        return provider(argv, **kwargs)

    with pytest.raises(ValueError, match="inventory preflight failed"):
        _resolve_reserved(monkeypatch, tmp_path, capture)
    assert not any("capacity-allowance" in argv for argv, _ in provider.calls)


@pytest.mark.parametrize("size,passes", [(3, True), (4, False)])
def test_soperator_project_headroom_credits_existing_strict_workers(
    monkeypatch, tmp_path, size, passes
):
    provider = CapacityProvider()
    provider.capacity_limit = "24"
    provider.capacity_status.update(usage="16", usage_state="USAGE_STATE_USED")
    (tmp_path / "terraform.tfstate").write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "type": "nebius_mk8s_v1_node_group",
                        "name": "worker_v2",
                        "instances": [
                            {
                                "attributes": {
                                    "fixed_node_count": 2,
                                    "template": {
                                        "resources": {"preset": "8gpu-160vcpu-1792gb"},
                                        "reservation_policy": {
                                            "policy": "STRICT",
                                            "reservation_ids": [
                                                "capacityblockgroup-test"
                                            ],
                                        },
                                    },
                                }
                            }
                        ],
                    }
                ]
            }
        )
    )
    if passes:
        resolved, _ = _resolve_reserved(
            monkeypatch, tmp_path, provider, _reserved_spec(size)
        )
        assert resolved.workers[0].size == size
    else:
        with pytest.raises(ValueError, match="insufficient GPU headroom"):
            _resolve_reserved(monkeypatch, tmp_path, provider, _reserved_spec(size))
