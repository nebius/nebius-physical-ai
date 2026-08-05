from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa import provisioning
from npa.clients import config, credentials

runner = CliRunner()


@pytest.fixture(autouse=True)
def _successful_storage_probe(monkeypatch):
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            True,
            "ok",
            "Writable S3 verified with a cleaned write/delete probe.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        ),
    )


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
    assert "s3:dry-run ensure writable bucket bucket" in result.actions
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
    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
    )

    assert result.status == "ok"
    assert "s3:verified writable bucket bucket" in result.actions
    assert f"k8s:reused kubeconfig {kubeconfig}" in result.actions


def test_reused_cluster_still_waits_for_skypilot_gpu_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.orchestration.skypilot import k8s_gpu_catalog

    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def wait(accelerators, **kwargs):  # noqa: ANN001, ANN202
        seen["accelerators"] = accelerators
        seen["context"] = kwargs["context"]
        seen["kubeconfig"] = __import__("os").environ["KUBECONFIG"]
        kwargs["on_status"]("Kubernetes allocatable=1; SkyPilot discovery=ready")
        return {}

    monkeypatch.setattr(k8s_gpu_catalog, "wait_for_kubernetes_accelerators", wait)

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        accelerator="RTXPRO6000:1",
        gpu_readiness_timeout=9,
        gpu_readiness_poll_interval=0.1,
    )

    assert result.status == "ok"
    assert result.gpu_readiness == "ready"
    assert seen == {
        "accelerators": ["RTXPRO6000:1"],
        "context": "npa-cluster",
        "kubeconfig": str(kubeconfig),
    }
    assert any("SkyPilot discovery=ready" in action for action in result.actions)


def test_provision_if_absent_cli_exits_non_zero_on_a_partial_run(mocker) -> None:
    """Exiting 0 with warnings made the next submit the place the failure appeared."""
    from npa.cli.main import app

    mocker.patch(
        "npa.cli.provision.provision_if_absent",
        return_value=provisioning.ProvisionIfAbsentResult(
            status="partial",
            project="proj",
            cluster_name="npa-cluster",
            warnings=["project_id and tenant_id are required to ensure Kubernetes"],
        ),
    )

    result = runner.invoke(app, ["provision-if-absent", "--project", "proj"])

    assert result.exit_code == 1
    assert "project_id and tenant_id are required" in result.output


def test_provision_if_absent_cli_prints_the_kubeconfig_export(mocker) -> None:
    """The kubeconfig is written outside ~/.kube/config, so kubectl needs the path."""
    from npa.cli.main import app

    mocker.patch(
        "npa.cli.provision.provision_if_absent",
        return_value=provisioning.ProvisionIfAbsentResult(
            status="ok",
            project="proj",
            cluster_name="npa-cluster",
            kubeconfig_path="/home/op/.npa/clusters/npa-cluster/kubeconfig",
            actions=["k8s:ensured terraform cluster npa-cluster"],
        ),
    )

    result = runner.invoke(app, ["provision-if-absent", "--project", "proj"])

    assert result.exit_code == 0
    assert "export KUBECONFIG=/home/op/.npa/clusters/npa-cluster/kubeconfig" in result.output


def test_provision_blocks_kubernetes_when_storage_reconciliation_fails(
    tmp_path, monkeypatch, mocker
) -> None:
    from npa.clients import storage_setup, storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(False, "forbidden", "write forbidden"),
    )
    monkeypatch.setattr(
        storage_setup,
        "provision_storage",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("reconcile failed")),
    )
    cluster_up = mocker.patch("npa.cli.cluster.terraform_lifecycle.up_cmd")

    result = provisioning.provision_if_absent(project="proj")

    assert result.status == "partial"
    assert any("s3:partial" in action for action in result.actions)
    assert "k8s:blocked until writable S3 is reconciled" in result.actions
    assert any("reconciliation failed" in warning for warning in result.warnings)
    cluster_up.assert_not_called()


def test_provision_dry_run_recognizes_interrupted_storage_setup(
    tmp_path, monkeypatch
) -> None:
    from npa.clients import storage_setup

    _write_runtime(tmp_path, monkeypatch)
    transaction = storage_setup.StorageSetupTransaction(
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        bucket_name="bucket",
    )
    transaction.begin()
    transaction.record_created("bucket", {"name": "bucket"})

    result = provisioning.provision_if_absent(
        project="proj", dry_run=True, skip_k8s=True
    )

    assert result.status == "ok"
    assert "s3:dry-run reconcile writable bucket bucket" in result.actions


def test_provision_resume_reconciles_storage_before_returning_ok(
    tmp_path, monkeypatch
) -> None:
    from npa.clients import storage_setup, storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            False, "missing_configuration", "credentials missing"
        ),
    )
    calls: list[dict] = []

    def reconcile(**kwargs):
        calls.append(kwargs)
        return (
            {
                "s3_bucket": "bucket",
                "s3_endpoint": "https://storage.example",
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
            },
            StorageProbeResult(
                True,
                "ok",
                "verified",
                cleanup_attempted=True,
                cleanup_succeeded=True,
            ),
        )

    monkeypatch.setattr(storage_setup, "provision_storage", reconcile)

    result = provisioning.provision_if_absent(project="proj", skip_k8s=True)

    assert result.status == "ok"
    assert calls and calls[0]["bucket_name"] == "bucket"
    assert "s3:reconciled writable bucket bucket" in result.actions


def test_empty_storage_state_first_run_uses_the_deterministic_bucket(
    tmp_path, monkeypatch
) -> None:
    from npa.clients import nebius, storage_setup, storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    _write_runtime(tmp_path, monkeypatch)
    document = yaml.safe_load(config.CONFIG_PATH.read_text())
    document["projects"]["proj"].pop("storage")
    config.CONFIG_PATH.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(
        storage_validation,
        "probe_storage_write",
        lambda **_kwargs: StorageProbeResult(
            False, "missing_configuration", "credentials missing"
        ),
    )
    monkeypatch.setattr(nebius, "bucket_name_for", lambda *_args: "npa-bucket-first")
    calls: list[dict] = []

    def provision(**kwargs):
        calls.append(kwargs)
        return (
            {
                "s3_bucket": kwargs["bucket_name"],
                "s3_endpoint": "https://storage.example",
                "nebius_api_key": "access",
                "nebius_secret_key": "secret",
            },
            StorageProbeResult(
                True,
                "ok",
                "verified",
                cleanup_attempted=True,
                cleanup_succeeded=True,
            ),
        )

    monkeypatch.setattr(storage_setup, "provision_storage", provision)

    result = provisioning.provision_if_absent(project="proj", skip_k8s=True)

    assert result.status == "ok"
    assert calls[0]["bucket_name"] == "npa-bucket-first"
    assert "s3:reconciled writable bucket npa-bucket-first" in result.actions
