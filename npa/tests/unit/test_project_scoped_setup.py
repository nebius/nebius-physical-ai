"""Exercise project-admin quota and strict-reservation checks at setup boundaries."""

from copy import deepcopy
import json
from subprocess import CompletedProcess

import pytest

from npa.clients.nebius import NebiusError
from npa.cluster_backends import get_backend
from npa.cluster_backends.mk8s import MK8sApplyRequest
from npa.cluster_backends.quotas import required_quotas, shortfall_message
from npa.fleet.spec import ClusterSpec, NodePoolSpec
from npa.provisioning_preflight import (
    ExistingCapacity,
    PreflightBlockedError,
    build_whole_path_plan,
    resolve_topology,
)

REGION = "region-test"
PROJECT = "project-test"
TENANT = "tenant-test"


def _cluster(reserved=True):
    return ClusterSpec(
        name="project-cluster",
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="8vcpu-32gb"),
        gpu_nodes=NodePoolSpec(
            count=5,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            capacity_block_group="reservation-test" if reserved else "",
        ),
    )


def _catalog(names):
    return {
        "items": [
            {
                "metadata": {"parent_id": PROJECT, "name": name},
                "spec": {"region": REGION},
                "status": {
                    "state": "STATE_ACTIVE",
                    "usage_state": "USAGE_STATE_NOT_USED",
                    "unit": "byte" if ".size." in name else "count",
                },
            }
            for name in names
        ]
    }


def _entitlement():
    return {
        "items": [
            {
                "metadata": {"parent_id": PROJECT, "id": "allowance-test"},
                "spec": {"capacity_block_group_id": "reservation-test", "limit": "5"},
                "status": {
                    "state": "STATE_ACTIVE",
                    "usage_state": "USAGE_STATE_NOT_USED",
                    "unit": "gpu",
                },
            }
        ]
    }


def _capture(
    monkeypatch, cluster, *, quotas=None, entitlement=None, denial="PermissionDenied"
):
    calls = []
    payload = quotas if quotas is not None else _catalog(required_quotas([cluster]))
    grant = entitlement if entitlement is not None else _entitlement()

    def run(argv, *, env, check, timeout):
        assert argv[:3] == ["nebius", "--profile", "project-operator"]
        assert env == {"private-runtime": "selected"}
        assert check is False
        parent = argv[argv.index("--parent-id") + 1]
        calls.append((argv[3:5], parent))
        if parent == TENANT:
            return CompletedProcess(argv, 15, "", denial)
        assert parent == PROJECT
        data = grant if argv[3:5] == ["capacity", "capacity-allowance"] else payload
        return CompletedProcess(argv, 0, json.dumps(data), "")

    monkeypatch.setattr("npa.cluster_backends.process.run_capture", run)
    return calls


def _request(messages):
    return MK8sApplyRequest(
        nebius_bin="nebius",
        tenant_id=TENANT,
        project_id=PROJECT,
        region=REGION,
        profile="project-operator",
        provider_preflight=True,
        provider_env={"private-runtime": "selected"},
        on_status=messages.append,
    )


@pytest.mark.parametrize("reserved", [False, True])
def test_backend_accepts_exact_project_defaults_after_tenant_scope_denial(
    monkeypatch, reserved
):
    cluster = _cluster(reserved)
    calls = _capture(monkeypatch, cluster)
    messages = []

    result = get_backend("mk8s").preflight(cluster, _request(messages))

    assert result["capacity_quota"] == "passed"
    assert (["quotas", "quota-allowance"], TENANT) in calls
    assert (["quotas", "quota-allowance"], PROJECT) in calls
    assert any("tenant aggregate" in message for message in messages)
    if reserved:
        assert (["capacity", "capacity-allowance"], PROJECT) in calls
        assert any(
            "actual reserved capacity and compatibility" in message
            for message in messages
        )


def test_explicit_project_shortfall_keeps_project_scope_in_remedy(monkeypatch):
    from npa.cluster_backends.quotas import preflight_region

    cluster = _cluster(False)
    payload = _catalog(required_quotas([cluster]))
    payload["items"][0]["spec"]["limit"] = "0"
    _capture(monkeypatch, cluster, quotas=payload)
    import npa.cluster_backends.process as process

    deficits = preflight_region(
        nebius_bin="nebius",
        tenant_id=TENANT,
        project_id=PROJECT,
        region=REGION,
        clusters=[cluster],
        env={"private-runtime": "selected"},
        profile="project-operator",
        run_capture=process.run_capture,
        nebius_argv=lambda binary, profile: [binary, "--profile", profile],
    )

    assert deficits and all(item.scope == "project" for item in deficits)
    message = shortfall_message(deficits, TENANT)
    assert "project quota allowances" in message
    assert "project administrator" in message
    assert "tenant (root) operation" not in message


@pytest.mark.parametrize(
    "failure", ["foreign", "missing", "partial", "unknown-default"]
)
def test_backend_project_catalog_cannot_weaken_unknown_evidence(monkeypatch, failure):
    cluster = _cluster(False)
    payload = _catalog(required_quotas([cluster]))
    if failure == "foreign":
        payload["items"][0]["metadata"]["parent_id"] = "project-other"
    elif failure == "missing":
        payload["items"].pop(0)
    elif failure == "partial":
        payload["next_page_token"] = "more"
    else:
        payload["items"][0]["status"]["usage_state"] = "USAGE_STATE_UNKNOWN"
    _capture(monkeypatch, cluster, quotas=payload)

    with pytest.raises(ValueError):
        get_backend("mk8s").preflight(cluster, _request([]))


@pytest.mark.parametrize(
    "failure",
    ["missing", "foreign", "small", "used", "frozen", "unit", "duplicate", "partial"],
)
def test_strict_project_reservation_requires_exact_usable_entitlement(
    monkeypatch, failure
):
    grant = _entitlement()
    row = grant["items"][0]
    if failure == "missing":
        row["spec"]["capacity_block_group_id"] = "reservation-other"
    elif failure == "foreign":
        row["metadata"]["parent_id"] = "project-other"
    elif failure == "small":
        row["spec"]["limit"] = "4"
    elif failure == "used":
        row["status"].update(usage="1", usage_state="USAGE_STATE_USED")
    elif failure == "frozen":
        row["status"]["state"] = "STATE_FROZEN"
    elif failure == "unit":
        row["status"]["unit"] = "count"
    elif failure == "duplicate":
        grant["items"].append(deepcopy(row))
    else:
        grant["next_page_token"] = "more"
    cluster = _cluster()
    _capture(monkeypatch, cluster, entitlement=grant)

    with pytest.raises(ValueError):
        get_backend("mk8s").preflight(cluster, _request([]))


@pytest.mark.parametrize("reserved", [False, True])
@pytest.mark.parametrize(
    "error", ["ResourceExhausted", "Unavailable", "Unauthenticated"]
)
def test_backend_does_not_fallback_for_non_scope_failures(monkeypatch, error, reserved):
    cluster = _cluster(reserved)
    calls = _capture(monkeypatch, cluster, denial=error + " private-details")

    with pytest.raises(ValueError) as caught:
        get_backend("mk8s").preflight(cluster, _request([]))

    assert all(parent == TENANT for _command, parent in calls)
    assert "private-details" not in str(caught.value)


def test_direct_cluster_whole_path_uses_same_project_quota_fallback(
    monkeypatch, capsys
):
    from npa.cli.cluster.terraform_lifecycle import _preflight_whole_path_capacity

    topology = resolve_topology(
        cpu_nodes=1, gpu_nodes=5, capacity_block_group="reservation-test"
    )
    payload = _catalog(topology.quota_requirements())
    calls = []

    def read(parent, **_kwargs):
        calls.append(parent)
        if parent == TENANT:
            raise NebiusError("PermissionDenied")
        assert parent == PROJECT
        return payload

    monkeypatch.setattr("npa.clients.nebius.list_quota_allowances", read)
    monkeypatch.setattr(
        "npa.provisioning_preflight.discover_existing_capacity",
        lambda **_kwargs: ExistingCapacity(),
    )
    _preflight_whole_path_capacity(
        _direct_tfvars(),
        {},
        context="project-cluster",
        project_alias="test",
    )

    assert calls == [TENANT, PROJECT]
    output = capsys.readouterr()
    assert "Whole-path preflight ready" in output.out
    assert "tenant aggregate" in output.err


def test_default_shared_plan_does_not_hide_project_zero_cap(monkeypatch):
    topology = resolve_topology(cpu_nodes=1, gpu_nodes=0)
    payload = _catalog(topology.quota_requirements())
    payload["items"][0]["spec"]["limit"] = "0"

    def read(parent, **_kwargs):
        if parent == TENANT:
            raise NebiusError("PermissionDenied")
        return payload

    monkeypatch.setattr("npa.clients.nebius.list_quota_allowances", read)
    plan = build_whole_path_plan(
        project_alias="test",
        project_id=PROJECT,
        tenant_id=TENANT,
        region=REGION,
        topology=topology,
        mutation=True,
    )

    assert plan.decision == "blocked"
    with pytest.raises(PreflightBlockedError):
        plan.assert_mutation_ready()


def _direct_tfvars():
    return {
        "parent_id": PROJECT,
        "tenant_id": TENANT,
        "region": REGION,
        "cluster_name": "project-cluster",
        "cpu_nodes_count": 1,
        "gpu_nodes_count": 5,
        "capacity_block_group": "reservation-test",
    }


def test_legacy_strict_reservation_uses_project_allowance_after_get_denial(monkeypatch):
    from npa.cli.cluster import terraform_lifecycle as lifecycle

    calls = []
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "project-operator")

    def capture(argv, **_kwargs):
        calls.append(argv)
        if "capacity-allowance" in argv:
            assert argv[argv.index("--parent-id") + 1] == PROJECT
            return CompletedProcess(argv, 0, json.dumps(_entitlement()), "")
        return CompletedProcess(argv, 15, "", "PermissionDenied")

    monkeypatch.setattr(lifecycle, "_run_capture", capture)
    tfvars = _direct_tfvars()
    tfvars.update(
        gpu_nodes_platform="gpu-rtx6000", gpu_nodes_preset="1gpu-24vcpu-218gb"
    )
    lifecycle._preflight_gpu_capacity("nebius", tfvars, {})

    assert any("capacity-allowance" in argv for argv in calls)
    assert not any("quota-allowance" in argv for argv in calls)


def test_backend_rejects_conflicting_project_identities_before_reading():
    from dataclasses import replace
    from npa.cluster_backends.mk8s import MK8sProjectIdentity

    request = replace(
        _request([]),
        project=MK8sProjectIdentity(
            project_key="other",
            project_id="project-other",
            project_name="",
            expected_provider_name="",
        ),
    )
    with pytest.raises(ValueError, match="identities disagree"):
        get_backend("mk8s").preflight(_cluster(), request)


def test_non_provider_exception_cannot_trigger_scope_fallback():
    from npa.quota_scope import read_scoped_quota_catalog

    calls = []

    def read(parent):
        calls.append(parent)
        raise ValueError("malformed response containing PermissionDenied")

    with pytest.raises(ValueError, match="reason other than an RBAC"):
        read_scoped_quota_catalog(TENANT, PROJECT, reader=read)
    assert calls == [TENANT]


@pytest.mark.parametrize("limit", [None, False, "bad", -1, "1.5"])
def test_backend_invalid_explicit_project_limit_cannot_become_unlimited(
    monkeypatch, limit
):
    cluster = _cluster(False)
    payload = _catalog(required_quotas([cluster]))
    payload["items"][0]["spec"]["limit"] = limit
    _capture(monkeypatch, cluster, quotas=payload)

    with pytest.raises(ValueError):
        get_backend("mk8s").preflight(cluster, _request([]))
