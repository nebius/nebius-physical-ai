"""Full live validation of interactive ``npa configure`` bucket provisioning.

Every test runs the real ``npa configure --interactive`` CLI against Nebius APIs,
writes dotfiles, verifies bucket metadata, and performs an S3 put/list/delete
round-trip with the provisioned credentials.

Run:

    NPA_CONFIGURE_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_configure_bucket_e2e.py -q

Requires an authenticated Nebius CLI profile with permission to create/delete
object-storage buckets and provision access keys in the target project.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import boto3
import pytest
import yaml
from typer.testing import CliRunner

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients import nebius

runner = CliRunner()
GB = 1024**3

# Mark as a live e2e module so the autouse HOME-isolation fixture in the root
# conftest exempts these tests: they must see the operator's real ~/.nebius
# profile to reach Nebius APIs (the npa dotfiles are still redirected to tmp by
# the configure_paths fixture). Also gated at runtime by NPA_CONFIGURE_E2E=1.
pytestmark = pytest.mark.e2e


@dataclass(frozen=True)
class LiveConfigureEnv:
    project_id: str
    tenant_id: str
    region: str


@pytest.fixture(scope="module")
def live_configure_env() -> LiveConfigureEnv:
    if os.environ.get("NPA_CONFIGURE_E2E") != "1":
        pytest.skip("Set NPA_CONFIGURE_E2E=1 to run live configure e2e")

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
    return LiveConfigureEnv(project_id=project_id, tenant_id=tenant_id, region=region)


@pytest.fixture
def configure_paths(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    # Represent a ready, authenticated profile (the live env is authenticated per
    # live_configure_env): _ensure_nebius_profile() returns True so provisioning
    # proceeds instead of aborting on the "no profile" gate.
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    return creds_path, config_path


def _delete_bucket_by_name(project_id: str, bucket_name: str) -> None:
    item = nebius.get_bucket_by_name(project_id, bucket_name)
    if item is None:
        return
    bucket_id = item.get("metadata", {}).get("id", "")
    if bucket_id:
        nebius.delete_bucket(bucket_id)


def _s3_client_from_creds(creds: dict[str, Any], region: str):
    storage = creds["storage"]
    return boto3.client(
        "s3",
        endpoint_url=storage["endpoint_url"],
        aws_access_key_id=storage["aws_access_key_id"],
        aws_secret_access_key=storage["aws_secret_access_key"],
        region_name=region,
    )


def _purge_bucket_objects(creds: dict[str, Any], *, bucket_name: str, region: str) -> None:
    client = _s3_client_from_creds(creds, region)
    paginator = client.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket_name):
        for entry in page.get("Versions", []) + page.get("DeleteMarkers", []):
            client.delete_object(
                Bucket=bucket_name,
                Key=entry["Key"],
                VersionId=entry["VersionId"],
            )
    listed = client.list_objects_v2(Bucket=bucket_name)
    for entry in listed.get("Contents", []):
        client.delete_object(Bucket=bucket_name, Key=entry["Key"])


def _cleanup_bucket(
    env: LiveConfigureEnv,
    bucket_name: str,
    *,
    creds: dict[str, Any] | None = None,
) -> None:
    if creds:
        _purge_bucket_objects(creds, bucket_name=bucket_name, region=env.region)
    _delete_bucket_by_name(env.project_id, bucket_name)


def _ensure_bucket_absent(env: LiveConfigureEnv, bucket_name: str) -> None:
    if not nebius.bucket_exists(env.project_id, bucket_name):
        return
    try:
        _delete_bucket_by_name(env.project_id, bucket_name)
        return
    except nebius.NebiusError as exc:
        if "BucketNotEmpty" not in str(exc):
            raise
    bootstrap = nebius.bootstrap_environment(
        env.project_id,
        env.tenant_id,
        env.region,
        bucket_name=bucket_name,
    )
    _cleanup_bucket(
        env,
        bucket_name,
        creds={
            "storage": {
                "endpoint_url": bootstrap["s3_endpoint"],
                "aws_access_key_id": bootstrap["nebius_api_key"],
                "aws_secret_access_key": bootstrap["nebius_secret_key"],
            }
        },
    )


def _unique_bucket(prefix: str = "npa-e2e") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _interactive_configure_answers(
    env: LiveConfigureEnv,
    *,
    bucket_name: str = "",
    new_bucket: bool = True,
    storage_class: str = "",
    size_gb: str = "",
    hf_token: str = "hf_live_e2e_token",
    ai_cloud_key: str = "",
    token_factory_key: str = "",
    ngc_api_key: str = "",
) -> str:
    """Build stdin answers for interactive configure (HF, AI Cloud, TF, NGC)."""

    lines = [
        env.project_id,
        env.tenant_id,
        env.region,
        "",  # container registry (default)
        bucket_name,
    ]
    if new_bucket:
        lines.extend([storage_class, size_gb])
    # Prompt order matches npa configure --interactive: HF → AI Cloud → Token Factory → NGC.
    lines.extend([hf_token, ai_cloud_key, token_factory_key, ngc_api_key])
    return "\n".join(lines) + "\n"


def _run_configure(input_text: str):
    return runner.invoke(app, ["configure", "--interactive"], input=input_text)


def _load_written_files(creds_path, config_path) -> tuple[dict[str, Any], dict[str, Any]]:
    creds = yaml.safe_load(creds_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    return creds, config


def _assert_bucket_spec(
    project_id: str,
    bucket_name: str,
    *,
    storage_class: str,
    size_bytes: int,
) -> dict[str, Any]:
    item = nebius.get_bucket_by_name(project_id, bucket_name)
    assert item is not None, f"bucket {bucket_name} was not listed after configure"
    spec = item.get("spec", {})
    assert nebius.normalize_bucket_storage_class(
        str(spec.get("default_storage_class", ""))
    ) == nebius.normalize_bucket_storage_class(storage_class)
    assert int(spec.get("max_size_bytes", 0) or 0) == size_bytes
    return item


def _assert_s3_credentials_work(
    creds: dict[str, Any],
    *,
    bucket_name: str,
    region: str,
    object_key: str,
) -> None:
    client = _s3_client_from_creds(creds, region)
    body = b"npa-configure-live-e2e"
    client.put_object(Bucket=bucket_name, Key=object_key, Body=body)
    listed = client.list_objects_v2(Bucket=bucket_name, Prefix=object_key)
    assert listed.get("KeyCount", 0) >= 1
    fetched = client.get_object(Bucket=bucket_name, Key=object_key)
    assert fetched["Body"].read() == body
    client.delete_object(Bucket=bucket_name, Key=object_key)


def _assert_dotfiles(
    creds_path,
    config_path,
    env: LiveConfigureEnv,
    *,
    bucket_name: str,
    expect_hf_token: str = "hf_live_e2e_token",
) -> dict[str, Any]:
    assert creds_path.exists()
    assert config_path.exists()
    assert oct(creds_path.stat().st_mode)[-3:] == "600"

    creds, config = _load_written_files(creds_path, config_path)
    storage = creds["storage"]
    assert storage["aws_access_key_id"]
    assert storage["aws_secret_access_key"]
    assert storage["bucket"] == f"s3://{bucket_name}/"
    assert storage["endpoint_url"] == f"https://storage.{env.region}.nebius.cloud"
    assert creds["tokens"]["HF_TOKEN"] == expect_hf_token

    project = config["projects"][env.region]
    assert project["project_id"] == env.project_id
    assert project["tenant_id"] == env.tenant_id
    assert project["region"] == env.region
    assert project["container_registry"]
    assert config["default_project"] == env.region
    return creds


class TestConfigureInteractiveLive:
    def test_default_npa_bucket_standard_storage_and_size(
        self,
        live_configure_env: LiveConfigureEnv,
        configure_paths,
    ) -> None:
        """Enter at bucket/class/size prompts -> npa-bucket + standard + 50 GB."""

        env = live_configure_env
        creds_path, config_path = configure_paths
        bucket_name = nebius.bucket_name_for(env.tenant_id, env.project_id)
        size_bytes = 50 * GB
        creds = None

        _ensure_bucket_absent(env, bucket_name)
        assert nebius.bucket_exists(env.project_id, bucket_name) is False

        try:
            result = _run_configure(_interactive_configure_answers(env))
            assert result.exit_code == 0, result.output
            assert "No bucket name provided" in result.output
            assert "Using standard storage (default)." in result.output
            assert "Provisioned bucket" in result.output

            creds = _assert_dotfiles(creds_path, config_path, env, bucket_name=bucket_name)
            _assert_bucket_spec(
                env.project_id,
                bucket_name,
                storage_class="standard",
                size_bytes=size_bytes,
            )
            _assert_s3_credentials_work(
                creds,
                bucket_name=bucket_name,
                region=env.region,
                object_key="configure-e2e/default-npa-bucket",
            )
        finally:
            _cleanup_bucket(env, bucket_name, creds=creds)

    def test_explicit_bucket_standard_custom_size(
        self,
        live_configure_env: LiveConfigureEnv,
        configure_paths,
    ) -> None:
        """Named bucket + standard (default) + custom 100 GB size cap."""

        env = live_configure_env
        creds_path, config_path = configure_paths
        bucket_name = _unique_bucket("npa-e2e-standard")
        size_bytes = 100 * GB
        creds = None

        _ensure_bucket_absent(env, bucket_name)
        assert nebius.bucket_exists(env.project_id, bucket_name) is False

        try:
            result = _run_configure(
                _interactive_configure_answers(
                    env,
                    bucket_name=bucket_name,
                    storage_class="",
                    size_gb="100",
                )
            )
            assert result.exit_code == 0, result.output
            assert "Using standard storage (default)." in result.output
            assert "100 GB cap" in result.output

            creds = _assert_dotfiles(creds_path, config_path, env, bucket_name=bucket_name)
            _assert_bucket_spec(
                env.project_id,
                bucket_name,
                storage_class="standard",
                size_bytes=size_bytes,
            )
            _assert_s3_credentials_work(
                creds,
                bucket_name=bucket_name,
                region=env.region,
                object_key="configure-e2e/explicit-standard-100gb",
            )
        finally:
            _cleanup_bucket(env, bucket_name, creds=creds)

    def test_explicit_bucket_enhanced_storage_default_size(
        self,
        live_configure_env: LiveConfigureEnv,
        configure_paths,
    ) -> None:
        """Named bucket + enhanced throughput + default 50 GB size cap."""

        env = live_configure_env
        creds_path, config_path = configure_paths
        bucket_name = _unique_bucket("npa-e2e-enhanced")
        size_bytes = 50 * GB
        creds = None

        _ensure_bucket_absent(env, bucket_name)
        assert nebius.bucket_exists(env.project_id, bucket_name) is False

        try:
            result = _run_configure(
                _interactive_configure_answers(
                    env,
                    bucket_name=bucket_name,
                    storage_class="enhanced",
                    size_gb="",
                )
            )
            assert result.exit_code == 0, result.output
            assert "enhanced_throughput" in result.output
            assert "Using standard storage (default)." not in result.output

            creds = _assert_dotfiles(creds_path, config_path, env, bucket_name=bucket_name)
            _assert_bucket_spec(
                env.project_id,
                bucket_name,
                storage_class="enhanced_throughput",
                size_bytes=size_bytes,
            )
            _assert_s3_credentials_work(
                creds,
                bucket_name=bucket_name,
                region=env.region,
                object_key="configure-e2e/enhanced-throughput",
            )
        finally:
            _cleanup_bucket(env, bucket_name, creds=creds)

    def test_reuses_existing_bucket_without_recreating(
        self,
        live_configure_env: LiveConfigureEnv,
        configure_paths,
    ) -> None:
        """Pre-created bucket is reused; class/size prompts are skipped."""

        env = live_configure_env
        creds_path, config_path = configure_paths
        bucket_name = _unique_bucket("npa-e2e-reuse")
        size_bytes = 75 * GB
        creds = None

        _ensure_bucket_absent(env, bucket_name)
        nebius.ensure_bucket(
            env.project_id,
            bucket_name,
            max_size_bytes=size_bytes,
            default_storage_class="standard",
        )
        before = _assert_bucket_spec(
            env.project_id,
            bucket_name,
            storage_class="standard",
            size_bytes=size_bytes,
        )

        try:
            result = _run_configure(
                _interactive_configure_answers(
                    env,
                    bucket_name=bucket_name,
                    new_bucket=False,
                )
            )
            assert result.exit_code == 0, result.output
            assert "Reusing existing object-storage bucket" in result.output
            assert "New bucket storage class" not in result.output
            assert "Using standard storage (default)." not in result.output

            creds = _assert_dotfiles(creds_path, config_path, env, bucket_name=bucket_name)
            after = _assert_bucket_spec(
                env.project_id,
                bucket_name,
                storage_class="standard",
                size_bytes=size_bytes,
            )
            assert after.get("metadata", {}).get("id") == before.get("metadata", {}).get("id")
            _assert_s3_credentials_work(
                creds,
                bucket_name=bucket_name,
                region=env.region,
                object_key="configure-e2e/reuse-existing",
            )
        finally:
            _cleanup_bucket(env, bucket_name, creds=creds)

    def test_optional_tokens_persist_when_provided(
        self,
        live_configure_env: LiveConfigureEnv,
        configure_paths,
    ) -> None:
        """HF, Token Factory, and NGC keys supplied at prompts land in credentials.yaml."""

        env = live_configure_env
        creds_path, config_path = configure_paths
        bucket_name = _unique_bucket("npa-e2e-tokens")
        creds = None

        _ensure_bucket_absent(env, bucket_name)
        assert nebius.bucket_exists(env.project_id, bucket_name) is False

        try:
            result = _run_configure(
                _interactive_configure_answers(
                    env,
                    bucket_name=bucket_name,
                    hf_token="hf_provided_live_e2e",
                    token_factory_key="v1.tokenfactory.live.e2e",
                    ngc_api_key="nvapi_provided_live_e2e",
                )
            )
            assert result.exit_code == 0, result.output

            creds = _assert_dotfiles(
                creds_path,
                config_path,
                env,
                bucket_name=bucket_name,
                expect_hf_token="hf_provided_live_e2e",
            )
            assert creds["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "v1.tokenfactory.live.e2e"
            assert creds["ngc"]["api_key"] == "nvapi_provided_live_e2e"
            _assert_s3_credentials_work(
                creds,
                bucket_name=bucket_name,
                region=env.region,
                object_key="configure-e2e/tokens",
            )
        finally:
            _cleanup_bucket(env, bucket_name, creds=creds)
