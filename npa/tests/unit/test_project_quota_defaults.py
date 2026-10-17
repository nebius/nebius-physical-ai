"""Verify provider-default project quotas without relaxing tenant quota checks."""

from copy import deepcopy

import pytest

from npa.provisioning_preflight import (
    INSTANCE_QUOTA,
    assess_quota,
    parse_quota_allowances,
    read_project_quota_observations,
)


def _default_allowance() -> dict:
    return {
        "metadata": {
            "parent_id": "project-test",
            "name": INSTANCE_QUOTA,
            "created_at": "1970-01-01T00:00:00Z",
        },
        "spec": {"region": "region-test"},
        "status": {"unit": "count", "usage_state": "USAGE_STATE_NOT_USED"},
    }


def _read(monkeypatch: pytest.MonkeyPatch, payload: dict):
    monkeypatch.setattr(
        "npa.clients.nebius.list_quota_allowances", lambda _project: payload
    )
    return read_project_quota_observations(
        "project-test", "region-test", [INSTANCE_QUOTA]
    )[INSTANCE_QUOTA]


def test_provider_default_project_limit_is_unbounded_without_mutating_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"items": [_default_allowance()]}
    original = deepcopy(payload)

    observation = _read(monkeypatch, payload)

    assert observation.state == "unbounded"
    assert assess_quota(observation, required=6).status == "unbounded"
    assert payload == original


@pytest.mark.parametrize("limit", [0, "0", 3, "3"])
def test_explicit_project_limits_remain_enforced(monkeypatch, limit) -> None:
    row = _default_allowance()
    row["spec"]["limit"] = limit

    observation = _read(monkeypatch, {"items": [row]})

    assert observation.state == "known"
    assert observation.limit == int(limit)
    assert assess_quota(observation, required=6).status == "blocked"


@pytest.mark.parametrize(
    "limit", [None, "", "invalid", False, "1.5", -1, "-1", -2, "unbounded", "inf"]
)
def test_invalid_explicit_limit_is_not_treated_as_an_absent_field(
    monkeypatch, limit
) -> None:
    row = _default_allowance()
    row["spec"]["limit"] = limit

    assert _read(monkeypatch, {"items": [row]}).state == "unknown"


@pytest.mark.parametrize("usage", [0, "0", 2, "2"])
def test_default_project_limit_accepts_valid_provider_usage(monkeypatch, usage) -> None:
    row = _default_allowance()
    row["status"].update(usage=usage, usage_state="USAGE_STATE_USED")

    assert _read(monkeypatch, {"items": [row]}).state == "unbounded"


@pytest.mark.parametrize("usage", [None, -1, "invalid", False, "1.5"])
def test_default_project_limit_rejects_invalid_usage(monkeypatch, usage) -> None:
    row = _default_allowance()
    row["status"]["usage"] = usage

    assert _read(monkeypatch, {"items": [row]}).state == "unknown"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", None),
        ("state", "STATE_UNSPECIFIED"),
        ("state", "STATE_FROZEN"),
        ("state", "STATE_PROVISIONING"),
        ("usage_state", "USAGE_STATE_UNKNOWN"),
        ("usage_state", None),
        ("usage_state", "USAGE_STATE_UNSPECIFIED"),
        ("available", "10"),
    ],
)
def test_default_project_limit_requires_usable_status(
    monkeypatch, field, value
) -> None:
    row = _default_allowance()
    row["status"][field] = value

    assert _read(monkeypatch, {"items": [row]}).state == "unknown"


@pytest.mark.parametrize("missing", ["parent_id", "region", "usage_state"])
def test_default_project_limit_requires_project_region_and_usage_evidence(
    monkeypatch, missing
) -> None:
    row = _default_allowance()
    section = {"parent_id": "metadata", "region": "spec", "usage_state": "status"}
    row[section[missing]].pop(missing)

    assert _read(monkeypatch, {"items": [row]}).state == "unknown"


def test_project_quota_response_from_another_project_is_rejected(monkeypatch) -> None:
    row = _default_allowance()
    row["metadata"]["parent_id"] = "project-other"

    with pytest.raises(ValueError, match="different project"):
        _read(monkeypatch, {"items": [row]})


def test_default_project_limit_from_another_region_remains_unknown(monkeypatch) -> None:
    row = _default_allowance()
    row["spec"]["region"] = "region-other"

    assert _read(monkeypatch, {"items": [row]}).state == "unknown"


@pytest.mark.parametrize(
    "payload",
    [
        {"items": None},
        {"items": [None]},
        {"items": [{"metadata": None}]},
        {"items": [{"metadata": {}, "spec": None}]},
        {"items": [{"metadata": {}, "spec": {}, "status": None}]},
        {"items": [_default_allowance()], "next_page_token": "more"},
    ],
)
def test_malformed_or_incomplete_project_catalog_is_rejected(
    monkeypatch, payload
) -> None:
    with pytest.raises(ValueError):
        _read(monkeypatch, payload)


def test_absent_project_quota_row_stays_unknown(monkeypatch) -> None:
    assert _read(monkeypatch, {"items": []}).state == "unknown"


def test_conflicting_duplicate_project_allowances_stay_unknown(monkeypatch) -> None:
    default = _default_allowance()
    restricted = deepcopy(default)
    restricted["spec"]["limit"] = "0"

    assert _read(monkeypatch, {"items": [default, restricted]}).state == "unknown"


def test_tenant_parser_does_not_infer_unbounded_from_an_omitted_limit() -> None:
    payload = {"items": [_default_allowance()]}

    observations = parse_quota_allowances(
        payload, region="region-test", names=[INSTANCE_QUOTA]
    )

    assert observations[INSTANCE_QUOTA].state == "unknown"
