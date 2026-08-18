from __future__ import annotations

import os
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
    from npa import provisioning_preflight
    from npa.provisioning_preflight import (
        GIB,
        NETWORK_SSD_BYTES_QUOTA,
        QuotaObservation,
    )

    monkeypatch.setattr(
        provisioning_preflight,
        "read_provider_quotas",
        lambda _tenant, _region, names: {
            name: QuotaObservation(
                name=name,
                used=0,
                limit=(20_000 * GIB if name == NETWORK_SSD_BYTES_QUOTA else 100),
                state="known",
            )
            for name in names
        },
    )
    from npa.provisioning_preflight import ExistingCapacity

    monkeypatch.setattr(
        provisioning_preflight,
        "discover_existing_capacity",
        lambda **_kwargs: ExistingCapacity(),
    )
    monkeypatch.setattr(
        "npa.controller_ownership.ensure_controller_owner",
        lambda *_args, **_kwargs: type(
            "Owner",
            (),
            {
                "project_alias": "proj",
                "context": "npa-cluster",
                "cluster_id": "cluster-1",
            },
        )(),
    )
    # Cached-cluster tests exercise provisioning orchestration, not a live API.
    # The shared health validator itself has hermetic topology/fabric/smoke tests.
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._validate_cluster",
        lambda *_args, **_kwargs: {
            "ready_nodes": 2,
            "total_gpus": 1,
            "default_storage_class": "compute-csi-default-sc",
        },
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


def test_provision_if_absent_dry_run_reports_actions(
    tmp_path: Path, monkeypatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "missing-kubeconfig"

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        dry_run=True,
    )

    assert result.status == "ready"
    assert "s3:dry-run ensure writable bucket bucket" in result.actions
    assert any(
        action.startswith("k8s:dry-run terraform apply deploy/cluster")
        for action in result.actions
    )
    assert result.storage_bucket == "s3://bucket/checkpoints/"


def test_provision_dry_run_from_installed_package_ignores_cached_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    from npa import cluster_backends

    real = cluster_backends.get_backend("mk8s")
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *_args: True)
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._validate_cluster",
        lambda *_args, **_kwargs: pytest.fail(
            "dry-run must not validate a cached live cluster"
        ),
    )

    class PlanOnlyBackend:
        def plan(self, desired):
            assert desired.cpu_nodes.count == 1
            assert desired.cpu_nodes.platform == "cpu-d3"
            assert desired.cpu_nodes.preset == "8vcpu-32gb"
            assert desired.gpu_nodes.count == 1
            assert desired.gpu_nodes.platform == "gpu-rtx6000"
            assert desired.gpu_nodes.preset == "1gpu-24vcpu-218gb"
            return real.plan(desired)

        def preflight(self, desired, request):
            return real.preflight(desired, request)

        def materialize(self, *_args, **_kwargs):
            pytest.fail("dry-run must not materialize a Terraform deployment")

    monkeypatch.setattr(
        cluster_backends,
        "get_backend",
        lambda name: PlanOnlyBackend() if name == "mk8s" else real,
    )
    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=tmp_path / "missing-kubeconfig",
        terraform_dir=tmp_path / "does-not-exist",
        dry_run=True,
        skip_s3=True,
    )
    assert result.status == "ready"
    assert any(
        "cpu_nodes=1 gpu_nodes=1 provider_mutation=false" in action
        for action in result.actions
    )


def test_provisioning_normalizes_uri_bucket_for_probe_and_runtime_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: dict[str, str] = {}

    from npa.clients.storage_validation import StorageProbeResult

    def probe(**kwargs):  # noqa: ANN003, ANN202
        seen["probe_bucket"] = kwargs["bucket"]
        return StorageProbeResult(
            True,
            "ok",
            "Writable S3 verified with a cleaned write/delete probe.",
            cleanup_attempted=True,
            cleanup_succeeded=True,
        )

    monkeypatch.setattr("npa.clients.storage_validation.probe_storage_write", probe)

    alias, environment, storage, registry = provisioning._resolve_project_runtime(
        "proj"
    )
    with provisioning._runtime_env(alias, environment, storage, registry):
        seen["runtime_bucket"] = os.environ["NPA_S3_BUCKET"]

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
    )

    assert result.status == "ok"
    assert seen == {"runtime_bucket": "bucket", "probe_bucket": "bucket"}


def test_quota_blocker_reaches_no_storage_or_cluster_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa import provisioning_preflight
    from npa.provisioning_preflight import (
        INSTANCE_QUOTA,
        PreflightBlockedError,
        QuotaObservation,
    )

    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        provisioning_preflight,
        "read_provider_quotas",
        lambda _tenant, _region, names: {
            name: QuotaObservation(
                name=name,
                used=10 if name == INSTANCE_QUOTA else 0,
                limit=10 if name == INSTANCE_QUOTA else 100,
                state="known",
            )
            for name in names
        },
    )
    storage_probe_calls: list[dict[str, object]] = []
    cluster_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write",
        lambda **kwargs: storage_probe_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd",
        lambda **kwargs: cluster_calls.append(kwargs),
    )

    with pytest.raises(PreflightBlockedError, match="compute.instance.count"):
        provisioning.provision_if_absent(project="proj")

    assert storage_probe_calls == []
    assert cluster_calls == []


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
    assert "k8s:validated stable GPU health and CUDA vectorAdd" in result.actions


def test_reused_cluster_runs_requested_skypilot_smoke(
    tmp_path: Path, monkeypatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._run_skypilot_smoke",
        lambda *args, **kwargs: seen.append(("smoke", *args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        lambda *args, **kwargs: seen.append(("readiness", *args, kwargs)) or {},
    )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        sky_smoke=True,
        accelerator="RTXPRO6000:1",
        sky_bin="/opt/npa/sky",
    )

    assert result.status == "ok"
    assert "sky-smoke:passed" in result.actions
    assert [item[0] for item in seen] == ["readiness", "smoke"]
    readiness = seen[0]
    assert readiness[1] == ["RTXPRO6000:1"]
    assert readiness[-1]["kubeconfig"] == kubeconfig
    assert readiness[-1]["sky_bin"] == "/opt/npa/sky"
    assert readiness[-1]["label_known_gpus"] is True
    assert seen[1] == (
        "smoke",
        kubeconfig,
        "npa-cluster",
        "npa-cluster",
        "RTXPRO6000:1",
        {"sky_bin": "/opt/npa/sky"},
    )


def test_fresh_cluster_uses_the_same_readiness_then_smoke_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "fresh-kubeconfig"
    seen: list[tuple[str, object]] = []

    def up(**kwargs):  # noqa: ANN003, ANN202
        seen.append(("up", kwargs))

    monkeypatch.setattr("npa.cli.cluster.terraform_lifecycle.up_cmd", up)
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        lambda *_args, **kwargs: seen.append(("readiness", kwargs)) or {},
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._run_skypilot_smoke",
        lambda *_args, **kwargs: seen.append(("smoke", kwargs)),
    )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        sky_smoke=True,
        accelerator="RTXPRO6000:1",
        sky_bin="/opt/npa/sky",
    )

    assert result.status == "ok"
    assert [item[0] for item in seen] == ["up", "readiness", "smoke"]
    up_kwargs = seen[0][1]
    assert up_kwargs["sky_smoke"] is False
    assert up_kwargs["sky_bin"] == "/opt/npa/sky"
    assert seen[1][1]["label_known_gpus"] is True
    assert seen[1][1]["sky_bin"] == "/opt/npa/sky"
    assert seen[2][1]["sky_bin"] == "/opt/npa/sky"


def test_cached_smoke_without_accelerator_keeps_auto_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        lambda accelerators, **_kwargs: seen.append(("readiness", accelerators)) or {},
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._run_skypilot_smoke",
        lambda *_args, **_kwargs: seen.append(("smoke", _args[3])),
    )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        sky_smoke=True,
        sky_bin="/opt/npa/sky",
    )

    assert result.status == "ok"
    assert seen == [("readiness", []), ("smoke", "")]


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
        seen["kubeconfig"] = str(kwargs["kubeconfig"])
        seen["label_known_gpus"] = kwargs["label_known_gpus"]
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
        "label_known_gpus": True,
    }
    assert any("SkyPilot discovery=ready" in action for action in result.actions)


def test_green_preflight_rolls_back_only_new_cluster_on_readiness_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    from npa.orchestration.skypilot import k8s_gpu_catalog
    from npa.provisioning_journal import current_operation

    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "new-kubeconfig"

    def create_cluster(**_kwargs) -> None:
        operation = current_operation()
        assert operation is not None
        operation.transition("resource-created")
        operation.record_resource(
            resource_type="managed_kubernetes_cluster",
            requested_name="npa-cluster",
            provider_id="cluster-created-by-test-operation",
            ownership="created_by_this_operation",
            ownership_source="terraform-output-and-state",
            project_id="project-1",
        )

    monkeypatch.setattr("npa.cli.cluster.terraform_lifecycle.up_cmd", create_cluster)
    monkeypatch.setattr(
        k8s_gpu_catalog,
        "wait_for_kubernetes_accelerators",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            k8s_gpu_catalog.KubernetesGpuCatalogError("GPU not discoverable")
        ),
    )
    down = mocker.patch("npa.cli.cluster.terraform_lifecycle.down_cmd")

    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=kubeconfig,
        accelerator="RTXPRO6000:1",
    )

    assert result.status == "partial"
    assert result.gpu_readiness == "failed"
    down.assert_called_once()
    assert down.call_args.kwargs["cluster_id"] == "cluster-created-by-test-operation"
    assert any("rollback:removed only cluster resources" in a for a in result.actions)


def test_readiness_failure_preserves_preexisting_cluster_and_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    from npa.orchestration.skypilot import k8s_gpu_catalog

    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "existing-kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.setattr(
        k8s_gpu_catalog,
        "wait_for_kubernetes_accelerators",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            k8s_gpu_catalog.KubernetesGpuCatalogError("GPU not discoverable")
        ),
    )
    down = mocker.patch("npa.cli.cluster.terraform_lifecycle.down_cmd")

    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=kubeconfig,
        accelerator="RTXPRO6000:1",
    )

    assert result.status == "partial"
    assert kubeconfig.is_file()
    assert result.storage_bucket == "s3://bucket/checkpoints/"
    down.assert_not_called()


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
    assert (
        "export KUBECONFIG=/home/op/.npa/clusters/npa-cluster/kubeconfig"
        in result.output
    )


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

    assert result.status == "ready"
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
