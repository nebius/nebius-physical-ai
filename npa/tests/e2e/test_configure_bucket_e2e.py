"""Live Nebius validation for configure bucket provisioning.

Run only when explicitly enabled:

    NPA_CONFIGURE_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_configure_bucket_e2e.py -q

Requires an authenticated Nebius CLI profile and permission to create/delete
object-storage buckets in the target project.
"""

from __future__ import annotations

import os
import uuid

import pytest

from npa.clients import nebius


def _require_live_configure_env() -> tuple[str, str, str]:
    if os.environ.get("NPA_CONFIGURE_E2E") != "1":
        pytest.skip("Set NPA_CONFIGURE_E2E=1 to run live configure bucket e2e")

    project_id = (
        os.environ.get("NPA_CONFIGURE_E2E_PROJECT_ID", "").strip()
        or nebius.current_project_id()
    )
    tenant_id = (
        os.environ.get("NPA_CONFIGURE_E2E_TENANT_ID", "").strip()
        or nebius.current_tenant_id()
    )
    region = os.environ.get("NPA_CONFIGURE_E2E_REGION", "eu-north1").strip()
    if not project_id or not tenant_id:
        pytest.skip(
            "NPA_CONFIGURE_E2E_PROJECT_ID and NPA_CONFIGURE_E2E_TENANT_ID "
            "(or an authenticated Nebius CLI profile) are required"
        )
    try:
        nebius.get_iam_token()
    except nebius.NebiusError as exc:
        pytest.skip(f"Nebius CLI profile is not authenticated: {exc}")
    return project_id, tenant_id, region


def test_configure_bucket_provision_standard_with_size_cap() -> None:
    """Create a fresh bucket with standard storage and a size cap, then clean up."""

    project_id, tenant_id, region = _require_live_configure_env()
    bucket_name = f"npa-e2e-{uuid.uuid4().hex[:12]}"
    size_bytes = 50 * 1024**3

    assert nebius.bucket_exists(project_id, bucket_name) is False

    try:
        nebius.ensure_bucket(
            project_id,
            bucket_name,
            max_size_bytes=size_bytes,
            default_storage_class="standard",
        )
        item = nebius.get_bucket_by_name(project_id, bucket_name)
        assert item is not None, f"bucket {bucket_name} was not listed after create"
        spec = item.get("spec", {})
        assert nebius.normalize_bucket_storage_class(
            str(spec.get("default_storage_class", ""))
        ) == "standard"
        assert int(spec.get("max_size_bytes", 0) or 0) == size_bytes

        creds = nebius.bootstrap_environment(
            project_id,
            tenant_id,
            region,
            bucket_name=bucket_name,
            bucket_max_size_bytes=size_bytes,
            bucket_storage_class="standard",
        )
        assert creds["s3_bucket"] == bucket_name
        assert creds["nebius_api_key"]
        assert creds["nebius_secret_key"]
    finally:
        item = nebius.get_bucket_by_name(project_id, bucket_name)
        if item is None:
            return
        bucket_id = item.get("metadata", {}).get("id", "")
        if bucket_id:
            nebius.delete_bucket(bucket_id)


def test_default_bucket_name_uses_npa_bucket_prefix() -> None:
    """Default bucket naming stays deterministic and uses the npa-bucket prefix."""

    name = nebius.bucket_name_for("tenant-abc", "project-xyz")
    assert name.startswith("npa-bucket-")
    assert name == nebius.bucket_name_for("tenant-abc", "project-xyz")
