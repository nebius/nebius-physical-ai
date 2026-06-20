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
import yaml
from typer.testing import CliRunner

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients import nebius

runner = CliRunner()


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


def _delete_bucket_by_name(project_id: str, bucket_name: str) -> None:
    item = nebius.get_bucket_by_name(project_id, bucket_name)
    if item is None:
        return
    bucket_id = item.get("metadata", {}).get("id", "")
    if bucket_id:
        nebius.delete_bucket(bucket_id)


def _assert_bucket_standard_with_size(
    project_id: str,
    bucket_name: str,
    *,
    size_bytes: int,
) -> None:
    item = nebius.get_bucket_by_name(project_id, bucket_name)
    assert item is not None, f"bucket {bucket_name} was not listed after configure"
    spec = item.get("spec", {})
    assert nebius.normalize_bucket_storage_class(
        str(spec.get("default_storage_class", ""))
    ) == "standard"
    assert int(spec.get("max_size_bytes", 0) or 0) == size_bytes


def _interactive_configure_answers(
    project_id: str,
    tenant_id: str,
    region: str,
    *,
    bucket_name: str = "",
) -> str:
    return "\n".join(
        [
            project_id,
            tenant_id,
            region,
            "",  # container registry (default)
            bucket_name,
            "",  # storage class -> standard (default)
            "",  # size GB -> 50 (default)
            "",  # HF token
            "",  # Token Factory key
            "",  # NGC key
        ]
    ) + "\n"


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
        _assert_bucket_standard_with_size(
            project_id,
            bucket_name,
            size_bytes=size_bytes,
        )

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
        _delete_bucket_by_name(project_id, bucket_name)


def test_configure_interactive_full_flow_creates_standard_bucket(
    monkeypatch,
    tmp_path,
) -> None:
    """Run the full interactive `npa configure` path against live Nebius APIs."""

    project_id, tenant_id, region = _require_live_configure_env()
    bucket_name = f"npa-e2e-interactive-{uuid.uuid4().hex[:12]}"
    size_bytes = 50 * 1024**3

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)

    _delete_bucket_by_name(project_id, bucket_name)
    assert nebius.bucket_exists(project_id, bucket_name) is False

    try:
        result = runner.invoke(
            app,
            ["configure", "--interactive"],
            input=_interactive_configure_answers(
                project_id,
                tenant_id,
                region,
                bucket_name=bucket_name,
            ),
        )

        assert result.exit_code == 0, result.output
        assert "Using standard storage (default)." in result.output
        assert "Provisioned bucket" in result.output

        creds = yaml.safe_load(creds_path.read_text())
        assert creds["storage"]["aws_access_key_id"]
        assert creds["storage"]["aws_secret_access_key"]
        assert creds["storage"]["bucket"] == f"s3://{bucket_name}/"
        assert creds["storage"]["endpoint_url"] == f"https://storage.{region}.nebius.cloud"

        config = yaml.safe_load(config_path.read_text())
        project = config["projects"][region]
        assert project["project_id"] == project_id
        assert project["tenant_id"] == tenant_id

        _assert_bucket_standard_with_size(
            project_id,
            bucket_name,
            size_bytes=size_bytes,
        )
    finally:
        _delete_bucket_by_name(project_id, bucket_name)


def test_configure_interactive_default_npa_bucket_uses_standard_storage(
    monkeypatch,
    tmp_path,
) -> None:
    """Press Enter for bucket/class/size defaults: npa-bucket + standard + 50 GB."""

    project_id, tenant_id, region = _require_live_configure_env()
    bucket_name = nebius.bucket_name_for(tenant_id, project_id)
    size_bytes = 50 * 1024**3

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)

    _delete_bucket_by_name(project_id, bucket_name)
    assert nebius.bucket_exists(project_id, bucket_name) is False

    try:
        result = runner.invoke(
            app,
            ["configure", "--interactive"],
            input=_interactive_configure_answers(project_id, tenant_id, region),
        )

        assert result.exit_code == 0, result.output
        assert "No bucket name provided" in result.output
        assert "Using standard storage (default)." in result.output

        creds = yaml.safe_load(creds_path.read_text())
        assert creds["storage"]["bucket"] == f"s3://{bucket_name}/"

        _assert_bucket_standard_with_size(
            project_id,
            bucket_name,
            size_bytes=size_bytes,
        )
    finally:
        _delete_bucket_by_name(project_id, bucket_name)


def test_default_bucket_name_uses_npa_bucket_prefix() -> None:
    """Default bucket naming stays deterministic and uses the npa-bucket prefix."""

    name = nebius.bucket_name_for("tenant-abc", "project-xyz")
    assert name.startswith("npa-bucket-")
    assert name == nebius.bucket_name_for("tenant-abc", "project-xyz")
