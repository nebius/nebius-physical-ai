"""Live proof of tenant-wide and project-only Nebius storage IAM behavior.

This test first runs the real NPA storage bootstrap through a tenant-wide setup
identity, then creates a disposable project-only operator identity and repeats
the same bootstrap. Each scope verifies all declared S3 actions and removes
every resource it creates before the test advances or exits.

Run only with explicit live target variables::

    NPA_INTEGRATION_E2E=1 \
    NPA_PROJECT_SCOPED_STORAGE_E2E=1 \
    NPA_PROJECT_SCOPED_STORAGE_E2E_ADMIN_PROFILE=<tenant-admin-profile> \
    NPA_PROJECT_SCOPED_STORAGE_E2E_PROJECT_ID=<project-id> \
    NPA_PROJECT_SCOPED_STORAGE_E2E_TENANT_ID=<tenant-id> \
      npa/.venv/bin/python -m pytest \
      npa/tests/e2e/test_project_scoped_storage_live_e2e.py -q -s

The admin profile must be able to create project IAM identities, list tenant
projects, and read the tenant ``editors`` group. Live identifiers and
credentials are never printed by the test.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.client import Config
from botocore.exceptions import ClientError

from npa.clients import nebius
from npa.clients.storage_validation import (
    StorageCredentialContext,
    probe_storage_write,
)

pytestmark = pytest.mark.e2e


def _error_category(result: subprocess.CompletedProcess[str]) -> str:
    text = f"{result.stdout}\n{result.stderr}".lower()
    if "permissiondenied" in text or "permission denied" in text:
        return "permission_denied"
    if "unauthenticated" in text or "authentication" in text:
        return "unauthenticated"
    if "notfound" in text or "not found" in text:
        return "not_found"
    if "alreadyexists" in text or "already exists" in text:
        return "already_exists"
    if "bucketnotempty" in text or "bucket not empty" in text:
        return "bucket_not_empty"
    if "invalidargument" in text or "invalid argument" in text:
        return "invalid_argument"
    return "other_error"


def _run_cli(
    args: list[str],
    *,
    profile: str = "",
    check: bool = True,
    json_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["nebius"]
    if profile:
        command.extend(["--profile", profile])
    command.extend(args)
    if json_output:
        command.extend(["--format", "json"])
    env = nebius.nebius_cli_env()
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=env,
    )
    if check and result.returncode != 0:
        action = " ".join(part for part in args[:3] if not part.startswith("-"))
        raise AssertionError(
            f"Nebius CLI {action} failed with {_error_category(result)}"
        )
    return result


def _run_json(args: list[str], *, profile: str) -> dict[str, Any]:
    result = _run_cli(args, profile=profile, json_output=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError("Nebius CLI returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AssertionError("Nebius CLI returned a non-object JSON document")
    return payload


def _resource_id(payload: dict[str, Any], kind: str) -> str:
    resource_id = str((payload.get("metadata") or {}).get("id") or "")
    if not resource_id:
        raise AssertionError(f"Nebius {kind} creation returned no resource ID")
    return resource_id


def _wait_until(
    action: Callable[[], subprocess.CompletedProcess[str]],
    *,
    expected: str,
    attempts: int = 18,
) -> subprocess.CompletedProcess[str]:
    last: subprocess.CompletedProcess[str] | None = None
    for _ in range(attempts):
        last = action()
        category = "ok" if last.returncode == 0 else _error_category(last)
        if category == expected:
            return last
        time.sleep(5)
    if last is None:
        raise AssertionError("provider convergence check did not execute")
    actual = "ok" if last.returncode == 0 else _error_category(last)
    raise AssertionError(
        f"provider convergence expected {expected}, received {actual}"
    )


def _profile_parent(profile: str) -> str:
    return _run_cli(["config", "get", "parent-id"], profile=profile).stdout.strip()


def _bucket_by_name(
    *, project_id: str, bucket_name: str, admin_profile: str
) -> dict[str, Any] | None:
    payload = _run_json(
        ["storage", "bucket", "list", "--parent-id", project_id, "--all"],
        profile=admin_profile,
    )
    for item in payload.get("items", []):
        if str((item.get("metadata") or {}).get("name") or "") == bucket_name:
            return item
    return None


def _purge_bucket_versions(client: Any, bucket_name: str) -> None:
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket_name):
        entries = list(page.get("Versions", [])) + list(page.get("DeleteMarkers", []))
        for entry in entries:
            client.delete_object(
                Bucket=bucket_name,
                Key=entry["Key"],
                VersionId=entry["VersionId"],
            )


def _assert_all_s3_actions(
    *, client: Any, bucket_name: str, key: str
) -> tuple[str, ...]:
    body = b"npa-project-scoped-storage-live-e2e"
    client.put_object(Bucket=bucket_name, Key=key, Body=body)
    head = client.head_object(Bucket=bucket_name, Key=key)
    assert int(head.get("ContentLength", -1)) == len(body)
    fetched = client.get_object(Bucket=bucket_name, Key=key)
    assert fetched["Body"].read() == body
    listed = client.list_objects_v2(Bucket=bucket_name, Prefix=key, MaxKeys=1)
    assert key in {
        str(item.get("Key") or "") for item in listed.get("Contents", [])
    }
    client.delete_object(Bucket=bucket_name, Key=key)
    with pytest.raises(ClientError) as deleted:
        client.head_object(Bucket=bucket_name, Key=key)
    assert str(deleted.value.response.get("Error", {}).get("Code", "")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }
    return (
        "GetObject",
        "HeadObject",
        "PutObject",
        "DeleteObject",
        "ListObjectsV2",
    )


def _exercise_storage_scope(
    *,
    scope: str,
    active_profile: str,
    admin_profile: str,
    project_id: str,
    tenant_id: str,
    region: str,
) -> tuple[str, ...]:
    """Run and clean one complete production storage bootstrap."""
    assert scope in {"tenant", "project"}
    suffix = uuid.uuid4().hex[:10]
    bucket_name = f"npa-{scope}-scope-{uuid.uuid4().hex[:14]}"
    storage_sa_name = f"npa-{scope}-scope-storage-{suffix}"
    access_key_name = f"npa-{scope}-scope-storage-key-{suffix}"
    fixed_storage_group = nebius.storage_binding_group_name(project_id)

    storage_bucket_id = ""
    storage_access_key_id = ""
    storage_permit_id = ""
    storage_group_id = ""
    storage_sa_id = ""
    s3_client: Any | None = None
    statuses: list[str] = []
    actions: tuple[str, ...] = ()
    cleanup_errors: list[str] = []

    def cleanup_cli(args: list[str], label: str) -> None:
        result = _run_cli(args, profile=admin_profile, check=False)
        if result.returncode != 0 and _error_category(result) != "not_found":
            cleanup_errors.append(f"{label}:{_error_category(result)}")

    def record_created(kind: str, payload: dict[str, str]) -> None:
        nonlocal storage_access_key_id
        nonlocal storage_group_id
        nonlocal storage_permit_id
        nonlocal storage_sa_id
        if kind == "service_account":
            storage_sa_id = payload.get("id", "")
        elif kind == "access_key":
            storage_access_key_id = payload.get("id", "")
        elif kind == "iam_permit":
            storage_permit_id = payload.get("id", "")
        elif kind == "iam_group":
            storage_group_id = payload.get("id", "")

    try:
        _run_cli(["profile", "activate", active_profile])
        assert _run_cli(["profile", "current"]).stdout.strip() == active_profile

        legacy_editors = _run_cli(
            [
                "iam",
                "group",
                "get-by-name",
                "--parent-id",
                tenant_id,
                "--name",
                "editors",
            ],
            profile=active_profile,
            check=False,
            json_output=True,
        )
        if scope == "tenant":
            assert legacy_editors.returncode == 0
        else:
            assert _error_category(legacy_editors) == "permission_denied"

        bootstrap = nebius.bootstrap_environment(
            project_id,
            tenant_id,
            region,
            bucket_name=bucket_name,
            service_account_name=storage_sa_name,
            access_key_name=access_key_name,
            service_account_description=f"Disposable {scope}-scope storage live E2E",
            access_key_description=f"Disposable {scope}-scope storage live E2E",
            on_status=statuses.append,
            on_resource_created=record_created,
        )

        bucket = _bucket_by_name(
            project_id=project_id,
            bucket_name=bucket_name,
            admin_profile=admin_profile,
        )
        assert bucket is not None
        storage_bucket_id = str((bucket.get("metadata") or {}).get("id") or "")
        assert all(
            (
                storage_bucket_id,
                storage_access_key_id,
                storage_permit_id,
                storage_group_id,
                storage_sa_id,
            )
        )
        assert bootstrap["iam_binding_role"] == nebius.STORAGE_RUNTIME_ROLE
        assert bootstrap["iam_binding_compatibility_fallback"] == "false"
        unreadable_status = any(
            "Tenant-wide editors membership is not readable" in message
            for message in statuses
        )
        assert unreadable_status is (scope == "project")

        s3_client = boto3.client(
            "s3",
            endpoint_url=bootstrap["s3_endpoint"],
            aws_access_key_id=bootstrap["nebius_api_key"],
            aws_secret_access_key=bootstrap["nebius_secret_key"],
            region_name=region,
            config=Config(s3={"addressing_style": "path"}),
        )
        last_s3_error: Exception | None = None
        for _ in range(18):
            try:
                actions = _assert_all_s3_actions(
                    client=s3_client,
                    bucket_name=bucket_name,
                    key=f"live-e2e/{uuid.uuid4().hex}.txt",
                )
                last_s3_error = None
                break
            except ClientError as exc:
                last_s3_error = exc
                if str(exc.response.get("Error", {}).get("Code", "")) not in {
                    "403",
                    "AccessDenied",
                    "SlowDown",
                }:
                    raise
                time.sleep(5)
        if last_s3_error is not None:
            raise AssertionError("S3 IAM binding did not converge") from last_s3_error
        assert actions == nebius.STORAGE_REQUIRED_S3_ACTIONS

        product_probe = probe_storage_write(
            bucket=bucket_name,
            endpoint_url=bootstrap["s3_endpoint"],
            access_key_id=bootstrap["nebius_api_key"],
            secret_access_key=bootstrap["nebius_secret_key"],
            region=region,
            client=s3_client,
            credential_context=StorageCredentialContext.NEWLY_CREATED,
        )
        assert product_probe.ok, product_probe.summary
        print(f"live_{scope}_scope_storage=passed")
        print(f"live_{scope}_scope_s3_actions=" + ",".join(actions))
        return actions
    finally:
        if not storage_bucket_id:
            try:
                partial_bucket = _bucket_by_name(
                    project_id=project_id,
                    bucket_name=bucket_name,
                    admin_profile=admin_profile,
                )
                storage_bucket_id = str(
                    ((partial_bucket or {}).get("metadata") or {}).get("id") or ""
                )
            except Exception:  # noqa: BLE001 - preserve cleanup of other exact IDs
                cleanup_errors.append("bucket_inventory:failed")
        if s3_client is not None:
            try:
                _purge_bucket_versions(s3_client, bucket_name)
            except Exception:  # noqa: BLE001 - cleanup continues through provider path
                cleanup_errors.append("bucket_object_purge:failed")

        if storage_bucket_id:
            delete_bucket = _run_cli(
                ["storage", "bucket", "delete", "--id", storage_bucket_id],
                profile=admin_profile,
                check=False,
            )
            if delete_bucket.returncode != 0:
                scheduled = _run_cli(
                    [
                        "storage",
                        "bucket",
                        "delete",
                        "--id",
                        storage_bucket_id,
                        "--ttl",
                        "1m",
                    ],
                    profile=admin_profile,
                    check=False,
                )
                if scheduled.returncode != 0 and _error_category(scheduled) != "not_found":
                    cleanup_errors.append(
                        f"bucket_delete:{_error_category(scheduled)}"
                    )

        if storage_access_key_id:
            cleanup_cli(
                ["iam", "v2", "access-key", "delete", "--id", storage_access_key_id],
                "storage_access_key",
            )
        if storage_permit_id:
            cleanup_cli(
                ["iam", "access-permit", "delete", "--id", storage_permit_id],
                "storage_permit",
            )
        if storage_group_id:
            cleanup_cli(
                ["iam", "group", "delete", "--id", storage_group_id],
                "storage_group",
            )
        if storage_sa_id:
            cleanup_cli(
                ["iam", "service-account", "delete", "--id", storage_sa_id],
                "storage_service_account",
            )

        audit_targets = [
            (["storage", "bucket", "get", "--id", storage_bucket_id], storage_bucket_id),
            (
                ["iam", "v2", "access-key", "get", "--id", storage_access_key_id],
                storage_access_key_id,
            ),
            (
                ["iam", "access-permit", "get", "--id", storage_permit_id],
                storage_permit_id,
            ),
            (["iam", "group", "get", "--id", storage_group_id], storage_group_id),
            (
                ["iam", "service-account", "get", "--id", storage_sa_id],
                storage_sa_id,
            ),
        ]
        for audit_args, resource_id in audit_targets:
            if not resource_id:
                continue
            absent = _wait_until(
                lambda args=audit_args: _run_cli(
                    args,
                    profile=admin_profile,
                    check=False,
                    json_output=True,
                ),
                expected="not_found",
            )
            assert _error_category(absent) == "not_found"
        named_bucket = _bucket_by_name(
            project_id=project_id,
            bucket_name=bucket_name,
            admin_profile=admin_profile,
        )
        assert named_bucket is None
        group_absent = _wait_until(
            lambda: _run_cli(
                [
                    "iam",
                    "group",
                    "get-by-name",
                    "--parent-id",
                    project_id,
                    "--name",
                    fixed_storage_group,
                ],
                profile=admin_profile,
                check=False,
                json_output=True,
            ),
            expected="not_found",
        )
        assert _error_category(group_absent) == "not_found"
        assert not cleanup_errors, ", ".join(cleanup_errors)
        print(f"live_{scope}_scope_cleanup_audit=passed")


def test_tenant_and_project_scoped_storage_binding_and_actions(
    tmp_path: Path,
) -> None:
    if os.environ.get("NPA_PROJECT_SCOPED_STORAGE_E2E") != "1":
        pytest.skip("set NPA_PROJECT_SCOPED_STORAGE_E2E=1 to run live")

    admin_profile = os.environ.get(
        "NPA_PROJECT_SCOPED_STORAGE_E2E_ADMIN_PROFILE", ""
    ).strip()
    project_id = os.environ.get(
        "NPA_PROJECT_SCOPED_STORAGE_E2E_PROJECT_ID", ""
    ).strip()
    tenant_id = os.environ.get(
        "NPA_PROJECT_SCOPED_STORAGE_E2E_TENANT_ID", ""
    ).strip()
    if not all((admin_profile, project_id, tenant_id)):
        pytest.skip("explicit admin profile, project ID, and tenant ID are required")

    project = _run_json(
        ["iam", "project", "get", "--id", project_id], profile=admin_profile
    )
    metadata = project.get("metadata") or {}
    status = project.get("status") or {}
    spec = project.get("spec") or {}
    assert str(metadata.get("parent_id") or metadata.get("parentId") or "") == tenant_id
    region = str(status.get("region") or spec.get("region") or "")
    assert region

    # Prove the setup identity has tenant-wide IAM visibility, not merely project access.
    _run_json(
        [
            "iam",
            "group",
            "get-by-name",
            "--parent-id",
            tenant_id,
            "--name",
            "editors",
        ],
        profile=admin_profile,
    )
    _run_json(
        ["iam", "project", "list", "--parent-id", tenant_id, "--all"],
        profile=admin_profile,
    )

    fixed_storage_group = nebius.storage_binding_group_name(project_id)
    existing_group = _run_cli(
        [
            "iam",
            "group",
            "get-by-name",
            "--parent-id",
            project_id,
            "--name",
            fixed_storage_group,
        ],
        profile=admin_profile,
        check=False,
        json_output=True,
    )
    if existing_group.returncode == 0:
        pytest.skip("target already has the NPA storage IAM group")
    assert _error_category(existing_group) == "not_found"

    suffix = uuid.uuid4().hex[:10]
    operator_name = f"npa-project-scope-operator-{suffix}"
    admin_group_name = f"npa-project-scope-admins-{suffix}"
    temporary_profile = f"npa-project-scope-{suffix}"
    credentials_file = tmp_path / "operator-service-account.json"
    original_profile = _run_cli(["profile", "current"], check=False).stdout.strip()

    operator_sa_id = ""
    operator_public_key_id = ""
    admin_group_id = ""
    admin_permit_id = ""
    temporary_profile_created = False
    cleanup_errors: list[str] = []

    def cleanup_cli(args: list[str], label: str) -> None:
        result = _run_cli(args, profile=admin_profile, check=False)
        if result.returncode != 0 and _error_category(result) != "not_found":
            cleanup_errors.append(f"{label}:{_error_category(result)}")

    try:
        tenant_actions = _exercise_storage_scope(
            scope="tenant",
            active_profile=admin_profile,
            admin_profile=admin_profile,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
        )

        operator_sa_id = _resource_id(
            _run_json(
                [
                    "iam",
                    "service-account",
                    "create",
                    "--parent-id",
                    project_id,
                    "--name",
                    operator_name,
                ],
                profile=admin_profile,
            ),
            "operator service account",
        )
        admin_group_id = _resource_id(
            _run_json(
                [
                    "iam",
                    "group",
                    "create",
                    "--parent-id",
                    project_id,
                    "--name",
                    admin_group_name,
                ],
                profile=admin_profile,
            ),
            "operator group",
        )
        admin_permit_id = _resource_id(
            _run_json(
                [
                    "iam",
                    "access-permit",
                    "create",
                    "--parent-id",
                    admin_group_id,
                    "--resource-id",
                    project_id,
                    "--role",
                    "admin",
                ],
                profile=admin_profile,
            ),
            "operator access permit",
        )
        _run_cli(
            [
                "iam",
                "group-membership",
                "create",
                "--parent-id",
                admin_group_id,
                "--member-id",
                operator_sa_id,
            ],
            profile=admin_profile,
        )
        _run_cli(
            [
                "iam",
                "auth-public-key",
                "generate",
                "--service-account-id",
                operator_sa_id,
                "--parent-id",
                project_id,
                "--output",
                str(credentials_file),
                "--output-format",
                "service-account-json",
            ],
            profile=admin_profile,
        )
        credentials_payload = json.loads(credentials_file.read_text())
        operator_public_key_id = str(
            credentials_payload.get("public_key_id")
            or credentials_payload.get("publicKeyId")
            or ""
        )

        endpoint = _run_cli(
            ["config", "get", "endpoint"], profile=admin_profile
        ).stdout.strip()
        assert endpoint
        profile_create = _run_cli(
            [
                "profile",
                "create",
                "--parent-id",
                project_id,
                "--profile",
                temporary_profile,
                "--endpoint",
                endpoint,
                "--service-account-file",
                str(credentials_file),
            ],
            check=False,
        )
        assert profile_create.returncode == 0, (
            "real profile create --parent-id failed with "
            f"{_error_category(profile_create)}"
        )
        temporary_profile_created = True
        assert _profile_parent(temporary_profile) == project_id

        _wait_until(
            lambda: _run_cli(
                ["iam", "project", "get", "--id", project_id],
                profile=temporary_profile,
                check=False,
                json_output=True,
            ),
            expected="ok",
        )
        tenant_projects = _run_cli(
            ["iam", "project", "list", "--parent-id", tenant_id, "--all"],
            profile=temporary_profile,
            check=False,
            json_output=True,
        )
        assert _error_category(tenant_projects) == "permission_denied"

        project_actions = _exercise_storage_scope(
            scope="project",
            active_profile=temporary_profile,
            admin_profile=admin_profile,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
        )
        assert tenant_actions == project_actions == nebius.STORAGE_REQUIRED_S3_ACTIONS
        print("live_tenant_and_project_scope_matrix=passed")
    finally:
        if original_profile:
            restored = _run_cli(
                ["profile", "activate", original_profile], check=False
            )
            if restored.returncode != 0:
                cleanup_errors.append("profile_restore:failed")
        else:
            restored = _run_cli(["profile", "activate", admin_profile], check=False)
            if restored.returncode != 0:
                cleanup_errors.append("profile_restore:failed")
        local_profiles = {
            line.strip() for line in _run_cli(["profile", "list"]).stdout.splitlines()
        }
        if temporary_profile_created or temporary_profile in local_profiles:
            deleted_profile = _run_cli(
                ["profile", "delete", temporary_profile], check=False
            )
            if deleted_profile.returncode != 0:
                cleanup_errors.append("temporary_profile_delete:failed")

        if operator_public_key_id:
            cleanup_cli(
                [
                    "iam",
                    "auth-public-key",
                    "delete",
                    "--id",
                    operator_public_key_id,
                ],
                "operator_public_key",
            )
        if admin_permit_id:
            cleanup_cli(
                ["iam", "access-permit", "delete", "--id", admin_permit_id],
                "operator_permit",
            )
        if admin_group_id:
            cleanup_cli(
                ["iam", "group", "delete", "--id", admin_group_id],
                "operator_group",
            )
        if operator_sa_id:
            cleanup_cli(
                ["iam", "service-account", "delete", "--id", operator_sa_id],
                "operator_service_account",
            )

        audit_targets = [
            (
                ["iam", "auth-public-key", "get", "--id", operator_public_key_id],
                operator_public_key_id,
            ),
            (
                ["iam", "access-permit", "get", "--id", admin_permit_id],
                admin_permit_id,
            ),
            (["iam", "group", "get", "--id", admin_group_id], admin_group_id),
            (
                ["iam", "service-account", "get", "--id", operator_sa_id],
                operator_sa_id,
            ),
        ]
        for audit_args, resource_id in audit_targets:
            if not resource_id:
                continue
            absent = _wait_until(
                lambda args=audit_args: _run_cli(
                    args,
                    profile=admin_profile,
                    check=False,
                    json_output=True,
                ),
                expected="not_found",
            )
            assert _error_category(absent) == "not_found"
        profiles = {
            line.strip() for line in _run_cli(["profile", "list"]).stdout.splitlines()
        }
        assert temporary_profile not in profiles
        if original_profile:
            assert _run_cli(["profile", "current"]).stdout.strip() == original_profile
        assert not cleanup_errors, ", ".join(cleanup_errors)
        print("live_operator_cleanup_audit=passed")
