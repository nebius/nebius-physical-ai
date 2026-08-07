from __future__ import annotations

import pytest
from fastapi import HTTPException

from npa.cli.agent_access import (
    ACCESS_SCHEMA,
    AccessProbeError,
    BucketProbe,
    accessible_artifact_buckets,
    artifact_bucket_projects,
    discover_agent_access,
)


NOW = "2026-08-06T23:30:00+00:00"


def _project(project_id: str, name: str) -> dict:
    return {"metadata": {"id": project_id, "name": name}}


def _bucket(resource_id: str, name: str) -> dict:
    return {"metadata": {"id": resource_id, "name": name}}


def _available_probe(_bucket_name: str) -> BucketProbe:
    return BucketProbe("available", "available", "List and read access verified.")


def _discover(**overrides):
    values = {
        "tenant_id": "tenant-test",
        "deployment_project_id": "project-a",
        "deployment_project_name": "deployment",
        "fallback_buckets": ["bucket-a"],
        "list_projects": lambda _tenant: [
            _project("project-a", "Alpha"),
            _project("project-b", "Beta"),
        ],
        "list_buckets": lambda project_id: [
            _bucket("bucket-resource-a", "bucket-a")
            if project_id == "project-a"
            else _bucket("bucket-resource-b", "bucket-b")
        ],
        "probe_bucket": _available_probe,
        "now": lambda: NOW,
    }
    values.update(overrides)
    return discover_agent_access(**values)


def test_full_tenant_access_is_explicit_and_project_owned() -> None:
    report = _discover()
    payload = report.to_dict()

    assert payload["apiVersion"] == ACCESS_SCHEMA
    assert payload["status"] == "available"
    assert payload["scope"] == "tenant"
    assert payload["refreshed_at"] == NOW
    assert [project["id"] for project in payload["projects"]] == ["project-a", "project-b"]
    assert payload["projects"][0]["deployment_project"] is True
    assert accessible_artifact_buckets(report) == ["bucket-a", "bucket-b"]
    assert artifact_bucket_projects(report) == {
        "bucket-a": "project-a",
        "bucket-b": "project-b",
    }
    assert payload["capabilities"]["artifact_delete"]["status"] == "unavailable"
    assert payload["capabilities"]["arbitrary_s3_uri"]["status"] == "unavailable"


def test_partial_access_keeps_accessible_project_and_reports_denied_project() -> None:
    def list_buckets(project_id: str):
        if project_id == "project-b":
            raise AccessProbeError("denied", "list project object storage resources")
        return [_bucket("bucket-resource-a", "bucket-a")]

    report = _discover(list_buckets=list_buckets)
    payload = report.to_dict()
    by_id = {item["id"]: item for item in payload["projects"]}

    assert payload["status"] == "partial"
    assert payload["scope"] == "partial_tenant"
    assert by_id["project-a"]["status"] == "available"
    assert by_id["project-b"]["status"] == "denied"
    assert by_id["project-b"]["capabilities"]["storage_resource_discovery"]["status"] == "denied"
    assert accessible_artifact_buckets(report) == ["bucket-a"]
    assert any(error["code"] == "permission_denied" for error in payload["errors"])


def test_denied_access_has_no_searchable_storage() -> None:
    def denied(_value: str):
        raise AccessProbeError("denied", "access resource")

    report = _discover(
        list_projects=denied,
        list_buckets=denied,
        probe_bucket=denied,
    )
    payload = report.to_dict()

    assert payload["status"] == "denied"
    assert payload["scope"] == "single_project"
    assert payload["capabilities"]["project_discovery"]["status"] == "denied"
    assert accessible_artifact_buckets(report) == []


def test_no_tenant_project_listing_permission_preserves_single_project_fallback() -> None:
    def deny_projects(_tenant: str):
        raise AccessProbeError("denied", "list tenant projects")

    def deny_bucket_inventory(_project: str):
        raise AccessProbeError("denied", "list project object storage resources")

    report = _discover(
        list_projects=deny_projects,
        list_buckets=deny_bucket_inventory,
        probe_bucket=_available_probe,
    )
    payload = report.to_dict()

    assert payload["status"] == "partial"
    assert payload["scope"] == "single_project"
    assert [item["id"] for item in payload["projects"]] == ["project-a"]
    assert payload["projects"][0]["resources"][0]["source"] == "agent_configuration"
    assert accessible_artifact_buckets(report) == ["bucket-a"]


def test_existing_single_project_behavior_stays_available() -> None:
    report = _discover(
        list_projects=lambda _tenant: [_project("project-a", "Alpha")],
        list_buckets=lambda _project: [_bucket("bucket-resource-a", "bucket-a")],
    )

    assert report.status == "available"
    assert report.scope == "single_project"
    assert accessible_artifact_buckets(report) == ["bucket-a"]
    project = report.to_dict()["projects"][0]
    assert project["capabilities"]["workflow_submission"]["status"] == "available"
    assert project["capabilities"]["artifact_write"]["status"] == "unverified"
    assert report.to_dict()["capabilities"]["artifact_write"]["status"] == "unverified"


def test_cross_project_mutations_remain_unavailable() -> None:
    payload = _discover().to_dict()
    by_id = {item["id"]: item for item in payload["projects"]}

    assert by_id["project-a"]["capabilities"]["workflow_submission"]["status"] == "available"
    assert by_id["project-b"]["capabilities"]["workflow_submission"]["status"] == "unavailable"
    for resource in by_id["project-b"]["resources"]:
        assert resource["capabilities"]["artifact_write"]["status"] == "unavailable"
        assert resource["capabilities"]["artifact_delete"]["status"] == "unavailable"


def test_access_errors_and_payload_never_copy_secret_material() -> None:
    secret = "super-secret-access-token"

    def fail_with_secret(_tenant: str):
        raise RuntimeError(f"credential={secret} path=/private/token/file")

    payload = _discover(list_projects=fail_with_secret).to_dict()
    rendered = repr(payload)

    assert secret not in rendered
    assert "/private/token/file" not in rendered
    assert "Unable to list tenant projects." in rendered
    assert payload["capabilities"]["project_discovery"]["status"] == "unavailable"


def test_access_model_is_embedded_with_api_ui_and_read_boundary() -> None:
    from pathlib import Path

    from npa.cli import agent

    embedded = agent._embedded_agent_access_source()
    runtime = agent._embedded_agent_access_runtime_source()
    backend_source = Path(agent.__file__).read_text(encoding="utf-8")
    ui_source = Path(agent.__file__).with_name("agent_ui.html").read_text(encoding="utf-8")

    assert "def discover_agent_access(" in embedded
    assert '@app.get("/access")' in backend_source
    assert "accessible_artifact_buckets(_agent_access_report())" in runtime
    assert "def _resolve_accessible_run_artifact(" in runtime
    assert "cross-project s3_uri requires a run_id and exact discovered artifact" in runtime
    assert 'id="agentAccessPanel"' in ui_source
    assert 'apiJson("/api/access"' in ui_source


def test_cross_project_object_read_requires_exact_run_membership(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    class FakeS3:
        calls: list[tuple[str, str]] = []

        def head_object(self, *, Bucket, Key):  # noqa: N803
            self.calls.append((Bucket, Key))
            if Bucket != "accessible-bucket" or Key != "category/run-a/report.json":
                raise RuntimeError("not found")

    s3 = FakeS3()
    monkeypatch.setattr(runtime, "validate_run_id", lambda value: value, raising=False)
    monkeypatch.setattr(runtime, "_safe_artifact_key", lambda value: value, raising=False)
    monkeypatch.setattr(
        runtime,
        "_agent_s3_buckets",
        lambda _s3, _settings: ["accessible-bucket"],
        raising=False,
    )
    monkeypatch.setattr(runtime, "HTTPException", HTTPException, raising=False)

    assert runtime._resolve_accessible_run_artifact(
        s3=s3,
        settings={},
        run_id="run-a",
        key="category/run-a/report.json",
        bucket="accessible-bucket",
    ) == ("accessible-bucket", "category/run-a/report.json", "run-a")

    with pytest.raises(HTTPException, match="not a discovered object"):
        runtime._resolve_accessible_run_artifact(
            s3=s3,
            settings={},
            run_id="run-b",
            key="category/run-a/report.json",
            bucket="accessible-bucket",
        )
    with pytest.raises(HTTPException, match="outside effective agent access"):
        runtime._resolve_accessible_run_artifact(
            s3=s3,
            settings={},
            run_id="run-a",
            key="category/run-a/report.json",
            bucket="other-bucket",
        )
