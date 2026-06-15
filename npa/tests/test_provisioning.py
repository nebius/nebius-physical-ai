from __future__ import annotations

from pathlib import Path

import yaml

from npa import provisioning
from npa.clients import config, credentials


def _write_runtime(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", credentials_path)
    for env_var in (
        "NPA_PROJECT_ID",
        "NPA_TENANT_ID",
        "NPA_REGION",
        "NPA_REGISTRY",
        "NPA_REGISTRY_ID",
        "NPA_S3_BUCKET",
        "NPA_CHECKPOINT_BUCKET",
        "NEBIUS_S3_BUCKET",
        "NPA_STORAGE_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "NEBIUS_S3_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "proj",
                "projects": {
                    "proj": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "eu-north1",
                        "registry_id": "registry-1",
                        "storage": {
                            "checkpoint_bucket": "s3://bucket/checkpoints/",
                            "endpoint_url": "https://storage.example",
                        },
                    }
                },
            }
        )
    )


def test_provision_if_absent_dry_run_reports_actions(tmp_path: Path, monkeypatch) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "missing-kubeconfig"

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        dry_run=True,
    )

    assert result.status == "ok"
    assert "s3:dry-run ensure bucket bucket" in result.actions
    assert "k8s:dry-run terraform apply deploy/cluster" in result.actions
    assert result.storage_bucket == "s3://bucket/checkpoints/"


def test_provision_if_absent_reuses_kubeconfig_and_ensures_bucket(
    tmp_path: Path,
    monkeypatch,
    mocker,
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    ensure_bucket = mocker.patch("npa.provisioning.ensure_bucket")

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
    )

    assert result.status == "ok"
    ensure_bucket.assert_called_once_with("project-1", "bucket")
    assert f"k8s:reused kubeconfig {kubeconfig}" in result.actions
