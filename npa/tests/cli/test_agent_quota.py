from __future__ import annotations

import pytest

from npa.cli.agent_quota import (
    _agent_check_whole_path_capacity,
    _agent_whole_path_capacity_result,
)
from npa.clients.nebius import NebiusError
from npa.provisioning_preflight import GIB, PreflightBlockedError


REGION = "eu-test1"


def _allowances(*, instance_limit: int = 20) -> dict:
    limits = {
        "compute.instance.count": instance_limit,
        "compute.disk.count": 20,
        "compute.disk.size.network-ssd": 4096 * GIB,
        "vpc.ipv4-address.public.count": 20,
    }
    return {
        "items": [
            {
                "metadata": {"name": name},
                "spec": {"region": REGION, "limit": str(limit)},
                "status": {"usage": "0"},
            }
            for name, limit in limits.items()
        ]
    }


def _project_scoped_quota_setup(monkeypatch, project_payload: dict) -> list[str]:
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _project_id: REGION
    )
    calls: list[str] = []

    def list_allowances(parent_id: str, **_kwargs) -> dict:
        calls.append(parent_id)
        if parent_id == "tenant-test":
            raise NebiusError(
                "PermissionDenied: UnauthorizedSingle for compute.instance.count"
            )
        assert parent_id == "project-test"
        return project_payload

    monkeypatch.setattr(
        "npa.clients.nebius.list_quota_allowances", list_allowances
    )
    return calls


def test_agent_capacity_uses_project_quotas_when_tenant_list_is_forbidden(
    monkeypatch,
) -> None:
    calls = _project_scoped_quota_setup(monkeypatch, _allowances())

    plan = _agent_check_whole_path_capacity(
        "project-test",
        "tenant-test",
        REGION,
        include_paidf=False,
    )

    assert calls == ["tenant-test", "project-test"]
    assert plan.decision == "ready"
    scope = next(item for item in plan.checks if item.name == "quota_evidence_scope")
    assert "unavailable due to RBAC" in scope.reason
    assert all(item.status in {"ready", "unbounded"} for item in plan.quotas)


@pytest.mark.parametrize(
    "project_payload",
    [
        {"items": []},
        {
            "items": [
                {
                    "metadata": {"name": "compute.instance.count"},
                    "spec": {"region": REGION},
                    "status": {},
                }
            ]
        },
    ],
)
def test_agent_capacity_allows_unrestricted_project_quotas_with_warning(
    monkeypatch, project_payload: dict
) -> None:
    _project_scoped_quota_setup(monkeypatch, project_payload)

    result = _agent_whole_path_capacity_result(
        "project-test",
        "tenant-test",
        REGION,
        include_paidf=False,
    )

    assert result.status == "WARN"
    assert "tenant-wide quota visibility is unavailable due to RBAC" in result.summary
    assert "provider will still enforce the tenant aggregate" in result.details[0]


def test_project_quota_denial_is_not_hidden_by_tenant_rbac_fallback(
    monkeypatch,
) -> None:
    _project_scoped_quota_setup(monkeypatch, _allowances(instance_limit=0))

    with pytest.raises(
        PreflightBlockedError,
        match="Project-scoped quota evidence denies.*compute.instance.count",
    ):
        _agent_check_whole_path_capacity(
            "project-test",
            "tenant-test",
            REGION,
            include_paidf=False,
        )


def test_project_quota_query_failure_remains_fail_closed_and_sanitized(
    monkeypatch,
) -> None:
    opaque = "private-resource-token"
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _project_id: REGION
    )

    def denied(parent_id: str, **_kwargs) -> dict:
        if parent_id == "tenant-test":
            raise NebiusError("PermissionDenied: UnauthorizedSingle")
        raise NebiusError(f"PermissionDenied: {opaque}")

    monkeypatch.setattr("npa.clients.nebius.list_quota_allowances", denied)

    result = _agent_whole_path_capacity_result(
        "project-test",
        "tenant-test",
        REGION,
        include_paidf=False,
    )

    assert result.status == "FAIL"
    assert "project-scoped quota query failed (NebiusError)" in result.summary
    assert opaque not in result.summary


def test_malformed_project_quota_catalog_remains_fail_closed(monkeypatch) -> None:
    _project_scoped_quota_setup(monkeypatch, {"items": ["malformed"]})

    result = _agent_whole_path_capacity_result(
        "project-test",
        "tenant-test",
        REGION,
        include_paidf=False,
    )

    assert result.status == "FAIL"
    assert "project-scoped quota query failed (ValueError)" in result.summary


def test_non_rbac_tenant_quota_failure_does_not_use_project_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.clients.nebius.get_project_region", lambda _project_id: REGION
    )
    calls: list[str] = []

    def broken(parent_id: str, **_kwargs) -> dict:
        calls.append(parent_id)
        raise NebiusError("quota service unavailable: private-resource-token")

    monkeypatch.setattr("npa.clients.nebius.list_quota_allowances", broken)

    result = _agent_whole_path_capacity_result(
        "project-test",
        "tenant-test",
        REGION,
        include_paidf=False,
    )

    assert calls == ["tenant-test"]
    assert result.status == "FAIL"
    assert "reason other than an RBAC scope limitation" in result.summary
    assert "private-resource-token" not in result.summary
