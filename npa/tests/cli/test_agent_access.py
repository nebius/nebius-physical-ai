from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import secrets
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from npa.cli.agent_access import (
    ACCESS_SCHEMA,
    AccessProbeError,
    BucketProbe,
    accessible_artifact_buckets,
    artifact_bucket_projects,
    discover_agent_access,
    scoped_artifact_buckets,
)
from npa.workflows.artifacts import encode_run_ref


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
    assert [project["id"] for project in payload["projects"]] == [
        "project-a",
        "project-b",
    ]
    assert payload["projects"][0]["deployment_project"] is True
    assert accessible_artifact_buckets(report) == ["bucket-a", "bucket-b"]
    assert artifact_bucket_projects(report) == {
        "bucket-a": "project-a",
        "bucket-b": "project-b",
    }
    assert payload["capabilities"]["artifact_delete"]["status"] == "unavailable"
    assert payload["capabilities"]["arbitrary_s3_uri"]["status"] == "unavailable"


def test_selected_artifact_scope_is_verified_against_project_ownership() -> None:
    report = _discover()

    assert scoped_artifact_buckets(report) == ["bucket-a", "bucket-b"]
    assert scoped_artifact_buckets(report, project_id="project-b") == ["bucket-b"]
    assert scoped_artifact_buckets(
        report,
        project_id="project-b",
        resource_bucket="bucket-b",
    ) == ["bucket-b"]

    with pytest.raises(ValueError, match="does not belong to the selected project"):
        scoped_artifact_buckets(
            report,
            project_id="project-a",
            resource_bucket="bucket-b",
        )
    with pytest.raises(ValueError, match="outside effective agent access"):
        scoped_artifact_buckets(report, resource_bucket="caller-controlled-bucket")


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
    assert (
        by_id["project-b"]["capabilities"]["storage_resource_discovery"]["status"]
        == "denied"
    )
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
    assert payload["scope"] == "partial_tenant"
    assert payload["capabilities"]["project_discovery"]["status"] == "denied"
    assert accessible_artifact_buckets(report) == []


def test_no_tenant_project_listing_permission_never_claims_single_project_success() -> (
    None
):
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
    assert payload["scope"] == "partial_tenant"
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


def test_empty_bucket_read_probe_does_not_shrink_complete_tenant_scope() -> None:
    report = _discover(
        probe_bucket=lambda _bucket: BucketProbe(
            "available",
            "unverified",
            "Object listing succeeded; empty bucket has no read probe object.",
        )
    )

    assert report.status == "available"
    assert report.scope == "tenant"
    assert len(report.projects) == 2
    assert accessible_artifact_buckets(report) == ["bucket-a", "bucket-b"]
    assert (
        report.to_dict()["projects"][0]["resources"][0]["capabilities"][
            "artifact_read"
        ]["status"]
        == "unverified"
    )


def test_access_identity_reports_non_secret_credential_provenance() -> None:
    payload = _discover(
        service_account_id="serviceaccount-test",
        credential_source="instance_metadata",
        credential_profile="cursor-sa",
        credential_config="/root/.nebius/config.yaml",
    ).to_dict()

    assert payload["identity"] == {
        "tenant_id": "tenant-test",
        "deployment_project_id": "project-a",
        "deployment_project_name": "deployment",
        "service_account_id": "serviceaccount-test",
        "credential_source": "instance_metadata",
        "credential_profile": "cursor-sa",
        "credential_config": "/root/.nebius/config.yaml",
    }


def test_cross_project_mutations_remain_unavailable() -> None:
    payload = _discover().to_dict()
    by_id = {item["id"]: item for item in payload["projects"]}

    assert (
        by_id["project-a"]["capabilities"]["workflow_submission"]["status"]
        == "available"
    )
    assert (
        by_id["project-b"]["capabilities"]["workflow_submission"]["status"]
        == "unavailable"
    )
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
    ui_source = (
        Path(agent.__file__).with_name("agent_ui.html").read_text(encoding="utf-8")
    )

    assert "def discover_agent_access(" in embedded
    assert '@app.get("/access")' in backend_source
    assert "accessible_artifact_buckets(_agent_access_report())" in runtime
    assert "def _resolve_accessible_run_artifact(" in runtime
    assert (
        "cross-project s3_uri requires a run_id and exact discovered artifact"
        in runtime
    )
    assert 'id="agentAccessPanel"' in ui_source
    assert 'id="agentAccessProjectSelect"' in ui_source
    assert 'for="agentAccessProjectSelect"' in ui_source
    assert 'id="agentAccessBucketSelect"' in ui_source
    assert 'for="agentAccessBucketSelect"' in ui_source
    assert 'apiJson("/api/access"' in ui_source
    assert 'data-access-action="' in ui_source
    assert 'data-capability-status="' in ui_source
    assert "async function listAccessResource" in ui_source
    assert "async function readAccessResource" in ui_source
    assert "resource_bucket=" in ui_source
    assert "project_id=" in ui_source
    assert "accessActionController.abort()" in ui_source
    assert 'event.key !== "Enter" && event.key !== " "' in ui_source
    assert 'id="agentAccessActionResult"' in ui_source
    assert 'if (status === "partial") return "Limited"' in ui_source
    assert "picker.replaceChildren()" in ui_source
    assert 'class="access-project access-project-detail"' in ui_source
    assert (
        "window.sessionStorage.setItem(ACCESS_PROJECT_STORAGE_KEY, projectId)"
        in ui_source
    )
    assert "const nextSelection = retained || deployment" in ui_source
    assert "populateAccessBucketPicker(selected, !projectChanged)" in ui_source
    assert "function selectAccessBucket(bucketName)" in ui_source
    assert 'data-selected-bucket="' in ui_source
    assert "No searchable artifact bucket." in ui_source


def test_cross_project_object_read_requires_exact_run_membership(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    class FakeS3:
        calls: list[tuple[str, str]] = []

        def head_object(self, *, Bucket, Key):  # noqa: N803
            self.calls.append((Bucket, Key))
            if Bucket != "accessible-bucket":
                raise RuntimeError("not found")

    s3 = FakeS3()
    monkeypatch.setattr(runtime, "validate_run_id", lambda value: value, raising=False)
    monkeypatch.setattr(
        runtime, "_safe_artifact_key", lambda value: value, raising=False
    )

    def find_sources(buckets, *, run_id, **_kwargs):
        assert buckets == ["accessible-bucket"]
        if run_id == "run-a":
            return (
                [
                    SimpleNamespace(resolved_prefix="category"),
                    SimpleNamespace(resolved_prefix="nested"),
                ],
                (),
                True,
            )
        return [], (), True

    monkeypatch.setattr(runtime, "find_run_sources_across_buckets", find_sources)
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
    assert runtime._resolve_accessible_run_artifact(
        s3=s3,
        settings={},
        run_id="run-a",
        key="nested/run-a/report.json",
        bucket="accessible-bucket",
    ) == ("accessible-bucket", "nested/run-a/report.json", "run-a")

    with pytest.raises(HTTPException, match="not a discovered object"):
        runtime._resolve_accessible_run_artifact(
            s3=s3,
            settings={},
            run_id="run-b",
            key="category/run-a/report.json",
            bucket="accessible-bucket",
        )
    with pytest.raises(HTTPException, match="not a discovered object"):
        runtime._resolve_accessible_run_artifact(
            s3=s3,
            settings={},
            run_id="run-a",
            key="category/run-ab/report.json",
            bucket="accessible-bucket",
        )
    # A valid-looking substring/path pair is not membership. Even though HEAD
    # would succeed in this tenant-accessible bucket, the server did not discover
    # a run named "config", so this exact malicious review shape is rejected.
    with pytest.raises(HTTPException, match="not a discovered object"):
        runtime._resolve_accessible_run_artifact(
            s3=s3,
            settings={},
            run_id="config",
            key="prod/config/database.json",
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


def test_selected_run_source_requires_complete_discovery_when_unqualified(
    monkeypatch,
) -> None:
    from npa.cli import agent_access_runtime as runtime

    source = SimpleNamespace(
        bucket="accessible-bucket",
        project_id="project-a",
        resolved_prefix="category",
    )
    monkeypatch.setattr(runtime, "HTTPException", HTTPException, raising=False)
    monkeypatch.setattr(runtime, "validate_run_id", lambda value: value, raising=False)
    monkeypatch.setattr(runtime, "_validated_resolved_prefix", lambda value: value)
    monkeypatch.setattr(
        runtime, "_agent_access_report", lambda: object(), raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_agent_artifact_list_scope",
        lambda *_args, **_kwargs: (["accessible-bucket"], {}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "find_run_sources_across_buckets",
        lambda *_args, **_kwargs: ([source], (), False),
    )

    with pytest.raises(HTTPException, match="discovery was incomplete") as exc_info:
        runtime._resolve_selected_run_source(
            s3=object(),
            settings={},
            run_id="run-a",
            resource_bucket="accessible-bucket",
        )
    assert exc_info.value.status_code == 503

    assert runtime._resolve_selected_run_source(
        s3=object(),
        settings={},
        run_id="run-a",
        resource_bucket="accessible-bucket",
        resolved_prefix="category",
        source_selected=True,
    ) == ("accessible-bucket", "project-a", "category")


def test_authorized_artifact_source_metadata_is_derived_from_key() -> None:
    from npa.cli import agent_access_runtime as runtime

    report = _discover()

    assert runtime._artifact_source_metadata(
        report,
        "bucket-b",
        "groot-1-7-finetune/duplicate-run/reports/sim2real.rrd",
        "duplicate-run",
    ) == ("bucket-b", "project-b", "groot-1-7-finetune")
    assert runtime._artifact_source_metadata(
        report,
        "bucket-a",
        "root-run/reports/sim2real.rrd",
        "root-run",
    ) == ("bucket-a", "project-a", "")


def test_cross_project_membership_discovery_has_a_bucket_cap(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    attempted: list[str] = []
    buckets = [f"accessible-{index:03d}" for index in range(100)]

    monkeypatch.setattr(runtime, "validate_run_id", lambda value: value, raising=False)
    monkeypatch.setattr(
        runtime, "_safe_artifact_key", lambda value: value, raising=False
    )
    monkeypatch.setattr(runtime, "_agent_s3_buckets", lambda _s3, _settings: buckets)

    def no_run(bucket, **_kwargs):
        attempted.extend(bucket)
        return [], (), True

    monkeypatch.setattr(runtime, "find_run_sources_across_buckets", no_run)

    with pytest.raises(HTTPException, match="not a discovered object"):
        runtime._resolve_accessible_run_artifact(
            s3=object(),
            settings={},
            run_id="missing-run",
            key="category/missing-run/report.json",
        )

    assert attempted == buckets[: runtime._MAX_ARTIFACT_MEMBERSHIP_BUCKETS]


def test_access_discovery_probes_projects_and_buckets_with_bounded_concurrency() -> (
    None
):
    active = 0
    maximum = 0
    lock = threading.Lock()

    def tracked_pause() -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with lock:
            active -= 1

    projects = [_project(f"project-{index}", f"Project {index}") for index in range(12)]

    def list_buckets(project_id: str):
        tracked_pause()
        return [_bucket(f"resource-{project_id}", f"bucket-{project_id}")]

    def probe(_bucket_name: str):
        tracked_pause()
        return BucketProbe("available", "available")

    report = discover_agent_access(
        tenant_id="tenant-test",
        deployment_project_id="project-0",
        list_projects=lambda _tenant: projects,
        list_buckets=list_buckets,
        probe_bucket=probe,
        now=lambda: NOW,
    )

    assert 1 < maximum <= 8
    assert [item.project_id for item in report.projects] == [
        "project-0",
        "project-1",
        "project-10",
        "project-11",
        *[f"project-{index}" for index in range(2, 10)],
    ]


def test_configured_project_metadata_is_unverified_when_tenant_lookup_fails() -> None:
    report = _discover(
        list_projects=lambda _tenant: (_ for _ in ()).throw(
            AccessProbeError("unavailable", "list tenant projects")
        ),
        list_buckets=lambda _project: [_bucket("resource-a", "bucket-a")],
    )

    project = report.to_dict()["projects"][0]
    assert project["capabilities"]["project_metadata"]["status"] == "unverified"
    assert report.status == "partial"


def test_agent_nebius_timeout_is_public_safe_and_bounded(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    canary = secrets.token_urlsafe(32)
    seen: dict[str, object] = {}

    def timeout_run(command, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=canary)

    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/bin/true")
    monkeypatch.setattr(runtime, "_agent_command_env", lambda: {})
    monkeypatch.setattr(runtime.subprocess, "run", timeout_run)

    with pytest.raises(AccessProbeError) as exc_info:
        runtime._agent_nebius_json(
            ["iam", "project", "list", "--parent-id", "tenant-test"],
            operation="list tenant projects",
        )
    assert exc_info.value.status == "unavailable"
    assert canary not in str(exc_info.value)
    assert seen["timeout"] == runtime._AGENT_NEBIUS_TIMEOUT_SECONDS


def test_agent_nebius_inventory_scrubs_tokens_and_pins_profile_config(
    monkeypatch, tmp_path
) -> None:
    from npa.cli import agent_access_runtime as runtime

    config = tmp_path / ".nebius" / "config.yaml"
    config.parent.mkdir()
    config.write_text("profiles: {}\n", encoding="utf-8")
    canary = secrets.token_urlsafe(24)
    seen: dict[str, object] = {}

    class Result:
        returncode = 0
        stdout = '{"items": []}'
        stderr = ""

    def run(command, **kwargs):
        seen["command"] = list(command)
        seen["env"] = dict(kwargs["env"])
        return Result()

    monkeypatch.setenv("NPA_NEBIUS_CONFIG", str(config))
    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "cursor-sa")
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: "/bin/true")
    monkeypatch.setattr(
        runtime,
        "_agent_command_env",
        lambda: {
            "NEBIUS_IAM_TOKEN": canary,
            "NPA_NEBIUS_IAM_TOKEN": canary,
            "TF_VAR_iam_token": canary,
            "NPA_REUSE_IAM_TOKEN": "1",
            "NEBIUS_PROFILE": "stale-profile",
        },
    )
    monkeypatch.setattr(runtime.subprocess, "run", run)

    assert runtime._agent_nebius_json(
        ["iam", "project", "list", "--parent-id", "tenant-test"],
        operation="list tenant projects",
    ) == {"items": []}
    command = seen["command"]
    env = seen["env"]
    assert command[:5] == [
        "/bin/true",
        "--config",
        str(config),
        "--profile",
        "cursor-sa",
    ]
    assert env["NEBIUS_PROFILE"] == "cursor-sa"
    assert env["HOME"] == str(tmp_path)
    assert not (runtime._AMBIENT_NEBIUS_TOKEN_KEYS & set(env))
    assert canary not in repr(command)
    assert canary not in repr(env)


def test_access_cache_refresh_is_singleflight_after_expiry(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    report = _discover()
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    def discover(**_kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return report

    monkeypatch.setattr(
        runtime, "_agent_s3_client_optional", lambda: (object(), {"bucket": ""})
    )
    monkeypatch.setattr(runtime, "discover_agent_access", discover)
    monkeypatch.setattr(runtime, "NPA_PROJECT_ALIAS", "test")
    with runtime._AGENT_ACCESS_CONDITION:
        runtime._AGENT_ACCESS_CACHE.update(
            report=report,
            expires_at=0.0,
            refreshing=False,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(runtime._agent_access_report) for _ in range(8)]
        assert entered.wait(timeout=2)
        release.set()
        results = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert all(result is report for result in results)


def test_exact_run_ref_authorization_checks_only_selected_project_and_bucket(
    monkeypatch,
) -> None:
    from npa.cli import agent_access_runtime as runtime

    runtime._clear_exact_run_ref_source_authorizations()
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv("NEBIUS_TENANT_ID", "tenant-test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    monkeypatch.setattr(
        runtime,
        "_agent_list_tenant_projects",
        lambda tenant: (
            calls.append(("tenant", tenant))
            or [_project("selected-project", "Selected")]
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_list_project_buckets",
        lambda project: (
            calls.append(("project", project))
            or [_bucket("resource-selected", "selected-bucket")]
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_probe_bucket",
        lambda _s3, bucket: (
            calls.append(("probe", bucket)) or BucketProbe("available", "available")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_access_report",
        lambda **_kwargs: pytest.fail("exact authorization must not scan all projects"),
    )

    run_ref = encode_run_ref("selected-bucket", "nested/source", "run-one")
    assert runtime._authorize_exact_run_ref_source(
        s3=object(),
        settings={"bucket": "deployment-bucket"},
        run_id="run-one",
        run_ref=run_ref,
        resource_bucket="selected-bucket",
        project_id="selected-project",
        resolved_prefix="nested/source",
    ) == ("selected-bucket", "selected-project", "nested/source")
    assert calls == [
        ("tenant", "tenant-test"),
        ("project", "selected-project"),
        ("probe", "selected-bucket"),
    ]
    monkeypatch.setattr(
        runtime,
        "_agent_list_tenant_projects",
        lambda _tenant: pytest.fail("fresh exact-source proof must be reused"),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_list_project_buckets",
        lambda _project: pytest.fail("fresh exact-source proof must be reused"),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_probe_bucket",
        lambda _s3, _bucket: pytest.fail("fresh exact-source proof must be reused"),
    )
    assert runtime._authorize_exact_run_ref_source(
        s3=object(),
        settings={"bucket": "deployment-bucket"},
        run_id="run-one",
        run_ref=run_ref,
        resource_bucket="selected-bucket",
        project_id="selected-project",
        resolved_prefix="nested/source",
    ) == ("selected-bucket", "selected-project", "nested/source")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "run-two", "run_ref does not identify run_id"),
        ("resource_bucket", "other-bucket", "bucket does not match run_ref"),
        ("resolved_prefix", "other/source", "prefix does not match run_ref"),
    ],
)
def test_exact_run_ref_authorization_rejects_mismatched_provenance_before_cloud_calls(
    monkeypatch, field, value, message
) -> None:
    from npa.cli import agent_access_runtime as runtime

    runtime._clear_exact_run_ref_source_authorizations()
    monkeypatch.setenv("NEBIUS_TENANT_ID", "tenant-test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    monkeypatch.setattr(
        runtime,
        "_agent_list_tenant_projects",
        lambda _tenant: pytest.fail("mismatched provenance must fail before inventory"),
    )
    values = {
        "run_id": "run-one",
        "run_ref": encode_run_ref("selected-bucket", "nested/source", "run-one"),
        "resource_bucket": "selected-bucket",
        "project_id": "selected-project",
        "resolved_prefix": "nested/source",
    }
    values[field] = value
    with pytest.raises(HTTPException, match=message):
        runtime._authorize_exact_run_ref_source(s3=object(), settings={}, **values)


def test_exact_run_ref_authorization_allows_configured_deployment_bucket_fallback(
    monkeypatch,
) -> None:
    from npa.cli import agent_access_runtime as runtime

    runtime._clear_exact_run_ref_source_authorizations()
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    monkeypatch.delenv("NEBIUS_TENANT_ID", raising=False)
    monkeypatch.setattr(
        runtime,
        "_agent_list_project_buckets",
        lambda _project: (_ for _ in ()).throw(
            AccessProbeError("unavailable", "list project buckets")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_agent_probe_bucket",
        lambda _s3, _bucket: BucketProbe("available", "available"),
    )

    run_ref = encode_run_ref("deployment-bucket", "", "run-one")
    assert runtime._authorize_exact_run_ref_source(
        s3=object(),
        settings={"bucket": "deployment-bucket"},
        run_id="run-one",
        run_ref=run_ref,
        resource_bucket="deployment-bucket",
        project_id="deployment-project",
        resolved_prefix="",
    ) == ("deployment-bucket", "deployment-project", "")


def test_exact_run_ref_authorization_fails_closed_on_wrong_project_bucket(
    monkeypatch,
) -> None:
    from npa.cli import agent_access_runtime as runtime

    runtime._clear_exact_run_ref_source_authorizations()
    monkeypatch.setenv("NEBIUS_TENANT_ID", "tenant-test")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "deployment-project")
    monkeypatch.setattr(
        runtime,
        "_agent_list_tenant_projects",
        lambda _tenant: [_project("selected-project", "Selected")],
    )
    monkeypatch.setattr(
        runtime,
        "_agent_list_project_buckets",
        lambda _project: [_bucket("resource-other", "other-bucket")],
    )
    monkeypatch.setattr(
        runtime,
        "_agent_probe_bucket",
        lambda _s3, _bucket: pytest.fail("unowned bucket must not be probed"),
    )

    with pytest.raises(HTTPException, match="does not belong"):
        runtime._authorize_exact_run_ref_source(
            s3=object(),
            settings={},
            run_id="run-one",
            run_ref=encode_run_ref("selected-bucket", "nested/source", "run-one"),
            resource_bucket="selected-bucket",
            project_id="selected-project",
            resolved_prefix="nested/source",
        )


def test_expired_access_cache_is_served_while_single_refresh_runs(monkeypatch) -> None:
    from npa.cli import agent_access_runtime as runtime

    stale = _discover()
    fresh = _discover()
    entered = threading.Event()
    release = threading.Event()

    def discover():
        entered.set()
        assert release.wait(timeout=2)
        return fresh

    monkeypatch.setattr(runtime, "_discover_agent_access_report", discover)
    with runtime._AGENT_ACCESS_CONDITION:
        runtime._AGENT_ACCESS_CACHE.update(
            report=stale,
            expires_at=0.0,
            refreshing=False,
        )

    assert runtime._agent_access_report() is stale
    assert entered.wait(timeout=2)
    assert runtime._agent_access_report() is stale
    release.set()
    with runtime._AGENT_ACCESS_CONDITION:
        assert runtime._AGENT_ACCESS_CONDITION.wait_for(
            lambda: not bool(runtime._AGENT_ACCESS_CACHE["refreshing"]),
            timeout=2,
        )
    assert runtime._agent_access_report() is fresh
