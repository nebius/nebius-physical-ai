from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa import provisioning
from npa.clients import config, credentials

runner = CliRunner()


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/graphics:operator",
        "registry.example/graphics@sha256:" + "1" * 64,
    ],
)
def test_provision_cli_preserves_explicit_graphics_image(monkeypatch, image):
    from npa.cli.main import app

    observed = []

    def capture_provision(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(status="ready", to_dict=lambda: {"status": "ready"})

    monkeypatch.setattr("npa.cli.provision.provision_if_absent", capture_provision)
    result = runner.invoke(
        app,
        [
            "provision-if-absent",
            "--gpu-graphics-smoke-image",
            image,
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed[0]["gpu_graphics_smoke_image"] == image


def test_graphics_default_quarantine_precedes_storage_and_cluster_mutations(
    tmp_path, monkeypatch
):
    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    def unexpected_mutation(*_args, **_kwargs):
        pytest.fail(
            "quarantined graphics image must fail before storage or cluster mutation"
        )

    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write", unexpected_mutation
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd", unexpected_mutation
    )
    with pytest.raises(ValueError, match="quarantined public release"):
        provisioning.provision_if_absent(
            project="proj", gpu_workload_profile="rtx-rendering"
        )


def test_provision_cli_renders_graphics_quarantine_as_a_cli_error(
    tmp_path, monkeypatch
):
    from npa.cli.main import app

    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "provision-if-absent",
            "--project",
            "proj",
            "--gpu-workload-profile",
            "rtx-rendering",
        ],
    )

    assert result.exit_code != 0
    assert "quarantined public release" in result.output
    assert "--gpu-graphics-smoke-image" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("skip_k8s,dry_run", [(True, False), (False, True)])
def test_graphics_default_not_resolved_when_no_health_work_runs(
    tmp_path, monkeypatch, skip_k8s, dry_run
):
    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        provisioning,
        "resolve_graphics_smoke_image",
        lambda *_args: pytest.fail("unused image must not resolve"),
    )
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *_args: False)

    result = provisioning.provision_if_absent(
        project="proj",
        gpu_workload_profile="rtx-rendering",
        skip_s3=True,
        skip_k8s=skip_k8s,
        dry_run=dry_run,
    )

    assert result.status == ("ok" if skip_k8s else "ready")


# The external tools provisioning is allowed to resolve. Anything else reaching
# _require_bin is a new dependency that should be noticed, not silently satisfied.
PROVISIONING_BINARIES = frozenset({"terraform", "nebius", "kubectl"})


@pytest.fixture(autouse=True)
def _successful_storage_probe(monkeypatch):
    resolved_binaries: list[str] = []
    from npa.clients import storage_validation
    from npa.clients.storage_validation import StorageProbeResult
    from npa.orchestration.skypilot import local_api

    monkeypatch.setattr(local_api, "_require_linux_host", lambda: None)

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
    # SkyPilot context enablement has its own CLI tests. Provisioning tests
    # assert orchestration and must never invoke a real SkyPilot process.
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._check_skypilot_kubernetes",
        lambda *_args, **_kwargs: ("sky", {}, "config"),
    )

    # provision_if_absent resolves kubectl before calling the validator stubbed
    # above, so without this the cached-cluster tests pass only on a machine that
    # happens to have kubectl installed. Nothing here ever executes the binary.
    #
    # Stand in for the filesystem lookup only, and record what was asked for: a
    # blanket passthrough would also fake terraform and nebius resolution and let a
    # regression that stopped resolving anything at all go unnoticed.
    def _resolve_without_touching_the_filesystem(binary: str) -> str:
        assert binary in PROVISIONING_BINARIES, f"unexpected binary resolved: {binary}"
        resolved_binaries.append(binary)
        return f"/usr/bin/{binary}"

    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._require_bin",
        _resolve_without_touching_the_filesystem,
    )
    return resolved_binaries


def _write_runtime(tmp_path: Path, monkeypatch, *, region: str = "eu-north1") -> None:
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
                        "region": region,
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


def _capture_mk8s_backend_plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from npa import cluster_backends

    backend = cluster_backends.get_backend("mk8s")
    seen: dict[str, object] = {}

    class CapturingBackend:
        def plan(self, desired):  # noqa: ANN001, ANN202 - backend test double
            seen["desired"] = desired
            seen["plan"] = backend.plan(desired)
            return seen["plan"]

        def preflight(self, desired, request):  # noqa: ANN001, ANN202
            return backend.preflight(desired, request)

    monkeypatch.setattr(
        cluster_backends, "get_backend", lambda _name: CapturingBackend()
    )
    return seen


def _failed_empty_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, region: str = "eu-north1"
):
    from npa.provisioning_journal import ProvisioningOperation, operation_context

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa provision-if-absent",
        project_alias="proj",
        project_id="project-1",
        tenant_id="tenant-1",
        region=region,
        resource_type="cluster",
        requested_name="failed-cluster",
        resume_command="npa provision-if-absent",
    )
    with operation_context(operation):
        operation.record_preflight_plan(
            {
                "project_alias": "proj",
                "project_id": "project-1",
                "tenant_id": "tenant-1",
                "region": region,
                "topology": {
                    "cluster_name": "failed-cluster",
                    "cpu_nodes": 1,
                    "gpu_nodes": 1,
                    "gpu_platform": "gpu-h200-sxm",
                    "gpu_preset": "1gpu-16vcpu-200gb",
                    "accelerator": "H200:1",
                    "gpu_preemptible": False,
                    "capacity_block_group": "",
                },
            }
        )
        operation.transition("mutating")
        operation.record_failure(RuntimeError("Command failed (1): terraform apply"))
        operation.transition("recovery-required")
        operation.record_rollback(
            attempted=False, completed=False, removed=[], preserved=[]
        )
    return operation


def _retained_standard_recovery_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    terraform_dir: Path | None = None,
    region: str = "eu-north1",
) -> Path:
    """Create retained material for the one supported failed no-S3 producer."""

    from npa.cluster import state as cluster_state

    cluster_root = tmp_path / ".npa" / "clusters"
    monkeypatch.setattr(cluster_state, "CLUSTERS_DIR", cluster_root)
    backend = (
        cluster_state.cluster_dir("failed-cluster")
        / "backend-state"
        / "project-1"
        / "failed-cluster"
    )
    recipe = backend / "k8s-training"
    source = Path.cwd() / "deploy" / "cluster" / "vendor" / "nebius-solutions-library"
    shutil.copytree(source / "k8s-training", recipe)
    shutil.copytree(source / "modules", backend / "modules")
    _materialize_standard_recipe(recipe, region=region)
    (backend / ".npa-fleet-env.json").write_text(
        json.dumps(
            {
                "backend": "mk8s",
                "cluster_name": "failed-cluster",
                "context": "retained-backend-context",
                "project_id": "project-1",
                "tenant_id": "tenant-1",
                "region": region,
                "gpu_driver_mode": "managed-image",
                "managed_driver_preset": "cuda13.0",
                "status": "provisioning",
                "subnet_id": "vpcsubnet-retained",
                "provider_rpc_deadlines": {
                    "source_sha256": _source_recipe_hashes(source, region=region),
                    "materialized_provider_sha256": _sha256_file(
                        recipe / "provider.tf"
                    ),
                    "apply_timeout_minutes": 120,
                    "inserted_defaults": {
                        "auth_timeout": "120m",
                        "per_retry_timeout": "120m",
                        "timeout": "120m",
                    },
                    "preserved_operator_fields": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (recipe / "terraform.tfvars").write_text(_standard_tfvars(), encoding="utf-8")
    (recipe / "terraform.tfstate").write_text(
        json.dumps(_empty_terraform_state()), encoding="utf-8"
    )
    (recipe / ".terraform.lock.hcl").write_text("# retained lock\n", encoding="utf-8")
    (recipe / "README.md").write_text("retained standard recipe\n", encoding="utf-8")
    selected_dir = terraform_dir or (Path.cwd() / "deploy" / "cluster").resolve()
    evidence = tmp_path / "retained-events.json"
    service_dir = tmp_path / "service"
    service_dir.mkdir()
    (service_dir / "failed.stderr").write_text(
        "rpc FailedPrecondition: allocation rejected before resource creation\n",
        encoding="utf-8",
    )
    (service_dir / "failed.stderr").chmod(0o600)
    command = _recorded_provision_command(selected_dir, service_dir)
    evidence.write_text(
        json.dumps({"scope": "retained operator events", "records": [command]}),
        encoding="utf-8",
    )
    evidence.chmod(0o600)
    return evidence


def _materialize_standard_recipe(recipe: Path, *, region: str) -> None:
    """Apply the same deterministic source transformations as the producer."""

    from npa.cluster_backends.mk8s_render import (
        patch_explicit_region_defaults,
        patch_provider_domain,
    )
    from npa.cluster_backends.provider_rpc import _hcl_providers, _materialized_text

    locals_file = recipe / "locals.tf"
    locals_file.write_text(
        patch_explicit_region_defaults(locals_file.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    provider_file = recipe / "provider.tf"
    provider = patch_provider_domain(provider_file.read_text(encoding="utf-8"), region)
    provider_file.write_text(
        _materialized_text(
            _hcl_providers(provider_file, provider)[0],
            provider,
            {"auth_timeout": "120m", "per_retry_timeout": "120m", "timeout": "120m"},
            {},
        ),
        encoding="utf-8",
    )


def _source_recipe_hashes(source: Path, *, region: str) -> dict[str, str]:
    """Return source hashes before the provider-default materialization."""

    from npa.cluster_backends.mk8s_render import (
        patch_explicit_region_defaults,
        patch_provider_domain,
    )

    hashes = {}
    for path in sorted((source / "k8s-training").glob("*.tf")):
        payload = path.read_bytes()
        if path.name == "locals.tf":
            payload = patch_explicit_region_defaults(payload.decode("utf-8")).encode()
        if path.name == "provider.tf":
            payload = patch_provider_domain(payload.decode("utf-8"), region).encode()
        hashes[path.name] = _sha256_bytes(payload)
    return hashes


def _sha256_file(path: Path) -> str:
    """Return a fixture file hash without embedding source literals in a test."""

    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(payload: bytes) -> str:
    """Return a deterministic fixture digest."""

    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _standard_tfvars() -> str:
    """Return the supported standard H200 backend shape as parsed Terraform."""

    values = {
        "cluster_name": '"failed-cluster"',
        "ssh_public_key": '"ssh-ed25519 synthetic"',
        "mk8s_cluster_public_endpoint": "true",
        "enable_k8s_node_group_sa": "false",
        "enable_egress_gateway": "false",
        "cpu_nodes_public_ips": "false",
        "gpu_nodes_public_ips": "false",
        "cpu_nodes_preemptible": "false",
        "gpu_nodes_preemptible": "false",
        "mig_strategy": '"none"',
        "gpu_nodes_driverfull_image": "true",
        "gpu_nodes_driver_preset": '"cuda13.0"',
        "custom_driver": "false",
        "cpu_nodes_fixed_count": "1",
        "cpu_nodes_platform": '"cpu-d3"',
        "cpu_nodes_preset": '"8vcpu-32gb"',
        "cpu_disk_size": '"128"',
        "gpu_nodes_fixed_count_per_group": "1",
        "gpu_node_groups": "1",
        "gpu_nodes_platform": '"gpu-h200-sxm"',
        "gpu_nodes_preset": '"1gpu-16vcpu-200gb"',
        "gpu_disk_size": '"1023"',
        "enable_gpu_cluster": "false",
        "enable_filestore": "false",
        "existing_filestore": '""',
        "filestore_disk_size_gibibytes": "1024",
        "filestore_mount_path": '"/mnt/data"',
        "filestore_mount_tag": '"data"',
        "filesystem_csi": '{ chart_version = "0.1.6" }',
        "enable_nebius_o11y_agent": "false",
        "enable_grafana": "false",
        "enable_prometheus": "false",
        "collectK8sClusterMetrics": "false",
        "enable_kuberay_cluster": "false",
        "enable_kuberay_service": "false",
        "enable_opa_gatekeeper": "false",
        "loki": "{ enabled = false, replication_factor = 1 }",
    }
    return "\n".join(f"{key} = {value}" for key, value in values.items()) + "\n"


def _empty_terraform_state() -> dict[str, object]:
    """Return retained state with no provider resource instances."""

    return {
        "version": 4,
        "terraform_version": "1.9.0",
        "serial": 1,
        "lineage": "synthetic",
        "outputs": {},
        "check_results": None,
        "resources": [
            {"mode": "managed", "type": "random_string", "instances": [{}]},
            {"mode": "managed", "type": "time_static", "instances": [{}]},
            {
                "mode": "managed",
                "type": "nebius_compute_v1_gpu_cluster",
                "instances": [],
            },
        ],
    }


def _recorded_provision_command(
    terraform_dir: Path, service_dir: Path
) -> dict[str, object]:
    """Build a collected failed invocation without executing its shell script."""

    script = "\n".join(
        [
            f"service_dir={shlex.quote(str(service_dir))}",
            "hy-npa-task provision-if-absent --project proj --cluster-name failed-cluster "
            '--context failed-cluster --kubeconfig "$service_dir/failed.kubeconfig" '
            f"--terraform-dir {shlex.quote(str(terraform_dir))} --skip-s3 --gpu-nodes 1 "
            "--gpu-platform gpu-h200-sxm --gpu-preset 1gpu-16vcpu-200gb --on-demand "
            '--accelerator H200:1 --output-format json > "$service_dir/out" '
            '2> "$service_dir/failed.stderr"',
            "exit $provision_rc",
        ]
    )
    return {
        "command": "/bin/bash -lc " + shlex.quote(script),
        "exit_code": 2,
        "status": "failed",
        "item_id": "synthetic-item",
        "turn_id": "synthetic-turn",
        "output": "retained failure",
    }


def _reconciliation_provider_stubs(
    monkeypatch: pytest.MonkeyPatch, *, region: str = "eu-north1"
) -> list[str]:
    from npa.clients import nebius
    from npa.cluster.api import CompleteClusterInventory, MK8sClient

    calls: list[str] = []

    def project_identity(*_args, **_kwargs) -> SimpleNamespace:
        calls.append("project")
        return SimpleNamespace(region=region)

    def cluster_inventory(*_args, **_kwargs) -> CompleteClusterInventory:
        calls.append("cluster")
        return CompleteClusterInventory((), "a" * 64)

    monkeypatch.setattr(
        nebius,
        "get_project_identity",
        project_identity,
    )
    monkeypatch.setattr(
        MK8sClient,
        "list_clusters_complete",
        cluster_inventory,
    )
    return calls


def _reserved_capacity_plan():
    from npa.provisioning_preflight import WholePathPreflightPlan, resolve_topology

    return WholePathPreflightPlan(
        project_alias="proj",
        project_id="project-1",
        tenant_id="tenant-1",
        region="eu-north1",
        topology=resolve_topology(
            cluster_name="reserved-cluster",
            gpu_nodes=1,
            gpu_platform="gpu-rtx6000",
            gpu_preset="1gpu-24vcpu-218gb",
            capacity_block_group="capacityblockgroup-example",
            preemptible=False,
        ),
        decision="ready",
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


def test_reconcile_empty_failed_provision_requires_fresh_named_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    calls = _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="failed-cluster",
        context_name="failed-cluster",
        reconcile_empty_failure=True,
        recovery_evidence=evidence,
    )

    payload = operation.read()
    checkpoint = payload["checkpoints"]["empty-failure-provider-absence"]
    assert result.status == "reconciled"
    assert payload["phase"] == "rolled-back"
    assert payload["last_error"] == "Command failed (1): terraform apply"
    assert checkpoint["provider_inventory"]["cluster"]["count"] == 0
    assert len(checkpoint["original_journal_sha256"]) == 64
    assert checkpoint["retained_producer"]["supported_profile"] == (
        "standard-no-s3-mk8s-single-gpu"
    )
    assert checkpoint["provider_inventory"]["node_group"]["absence_basis"]
    assert calls == ["project", "cluster"]


def test_reconcile_empty_failed_provision_binds_non_eu_recipe_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retained producer graph includes the selected provider-domain patch."""

    region = "us-central1"
    _write_runtime(tmp_path, monkeypatch, region=region)
    operation = _failed_empty_operation(tmp_path, monkeypatch, region=region)
    evidence = _retained_standard_recovery_evidence(
        tmp_path, monkeypatch, region=region
    )
    _reconciliation_provider_stubs(monkeypatch, region=region)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="failed-cluster",
        context_name="failed-cluster",
        reconcile_empty_failure=True,
        recovery_evidence=evidence,
    )

    assert result.status == "reconciled"
    assert operation.read()["phase"] == "rolled-back"


def test_reconcile_empty_failed_provision_refuses_recorded_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    operation.record_config_mutation(store="config", fields=["selected_project"])
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)
    provider_calls = _reconciliation_provider_stubs(monkeypatch)

    with pytest.raises(provisioning.OperationJournalError, match="owned or mutated"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"
    assert provider_calls == []


def test_reconcile_empty_failed_provision_refuses_live_named_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cluster.api import CompleteClusterInventory
    from npa.cluster.api import MK8sClient

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    monkeypatch.setattr(
        MK8sClient,
        "list_clusters_complete",
        lambda *_args, **_kwargs: CompleteClusterInventory(
            (SimpleNamespace(name="failed-cluster"),), "a" * 64
        ),
    )

    with pytest.raises(
        provisioning.OperationJournalError, match="cluster is still present"
    ):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"


def test_reconcile_empty_failed_provision_refuses_journal_change_during_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.cluster.api import CompleteClusterInventory, MK8sClient

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    def change_journal(*_args, **_kwargs) -> CompleteClusterInventory:
        operation.path.write_text(operation.path.read_text() + "\n", encoding="utf-8")
        return CompleteClusterInventory((), "a" * 64)

    monkeypatch.setattr(MK8sClient, "list_clusters_complete", change_journal)

    with pytest.raises(provisioning.OperationJournalError, match="changed during"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"


def test_reconcile_empty_failed_provision_refuses_retained_network_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained VPC ownership record is outside the reused-subnet recovery shape."""

    from npa.cluster import state as cluster_state

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    backend = (
        cluster_state.cluster_dir("failed-cluster")
        / "backend-state"
        / "project-1"
        / "failed-cluster"
    )
    (backend.parent / ".npa-fleet-network.json").write_text("{}", encoding="utf-8")
    calls = _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    with pytest.raises(provisioning.OperationJournalError, match="network or subnet"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"
    assert calls == []


def test_reconcile_empty_failed_provision_refuses_custom_terraform_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained custom recipe cannot reuse the standard empty-recovery path."""

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(
        tmp_path, monkeypatch, terraform_dir=tmp_path / "custom-recipe"
    )
    calls = _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    with pytest.raises(provisioning.OperationJournalError, match="does not select"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"
    assert calls == []


def test_reconcile_empty_failed_provision_refuses_recipe_graph_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained module byte drift blocks recovery before provider inventory."""

    from npa.cluster import state as cluster_state

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    module = (
        cluster_state.cluster_dir("failed-cluster")
        / "backend-state"
        / "project-1"
        / "failed-cluster"
        / "modules"
        / "o11y"
        / "loki.tf"
    )
    module.write_text(
        module.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8"
    )
    calls = _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    with pytest.raises(provisioning.OperationJournalError, match="modules differ"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"
    assert calls == []


def test_reconcile_empty_failed_provision_refuses_timeout_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout never proves that a failed create left no provider resources."""

    _write_runtime(tmp_path, monkeypatch)
    operation = _failed_empty_operation(tmp_path, monkeypatch)
    evidence = _retained_standard_recovery_evidence(tmp_path, monkeypatch)
    operation.record_failure(RuntimeError("terraform apply timeout"))
    calls = _reconciliation_provider_stubs(monkeypatch)
    monkeypatch.setattr(provisioning, "_owner_is_alive", lambda _pid: False)

    with pytest.raises(
        provisioning.OperationJournalError, match="completed apply rejection"
    ):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="failed-cluster",
            reconcile_empty_failure=True,
            recovery_evidence=evidence,
        )

    assert operation.read()["phase"] == "recovery-required"
    assert calls == []


def test_provision_dry_run_keeps_absent_project_quota_unknown_after_tenant_rbac_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty project response must not authorize a quota-backed mutation."""
    _write_runtime(tmp_path, monkeypatch)
    from npa import provisioning_preflight
    from npa.clients import nebius

    calls: list[str] = []

    def tenant_denied(parent_id: str, _region: str, _names):
        calls.append(parent_id)
        raise nebius.NebiusError("PermissionDenied")

    monkeypatch.setattr(provisioning_preflight, "read_provider_quotas", tenant_denied)
    monkeypatch.setattr(
        nebius,
        "list_quota_allowances",
        lambda parent_id: calls.append(parent_id) or {"items": []},
    )

    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=tmp_path / "missing-kubeconfig",
        dry_run=True,
        skip_s3=True,
    )

    assert result.status == "unknown"
    assert calls == ["tenant-1", "project-1"]
    scope = next(
        item
        for item in result.preflight["checks"]
        if item["name"] == "quota_evidence_scope"
    )
    assert scope["status"] == "unknown"


def test_provision_blocks_mutation_when_project_quota_is_absent_after_rbac_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The project fallback is fail-closed before Terraform can create nodes."""
    _write_runtime(tmp_path, monkeypatch)
    from npa import provisioning_preflight
    from npa.clients import nebius
    from npa.provisioning_preflight import PreflightBlockedError

    monkeypatch.setattr(
        provisioning_preflight,
        "read_provider_quotas",
        lambda *_args: (_ for _ in ()).throw(nebius.NebiusError("PermissionDenied")),
    )
    monkeypatch.setattr(nebius, "list_quota_allowances", lambda _parent: {"items": []})
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd",
        lambda **_kwargs: pytest.fail("quota-blocked plan must not create a cluster"),
    )

    with pytest.raises(PreflightBlockedError, match="quota"):
        provisioning.provision_if_absent(
            project="proj",
            kubeconfig=tmp_path / "missing-kubeconfig",
            skip_s3=True,
        )


def test_provision_if_absent_dry_run_preserves_strict_reserved_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    seen = _capture_mk8s_backend_plan(monkeypatch)

    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=tmp_path / "missing-kubeconfig",
        dry_run=True,
        skip_s3=True,
        gpu_nodes=1,
        gpu_platform="gpu-b200-sxm",
        gpu_preset="8gpu-160vcpu-1792gb",
        gpu_driver_mode="managed-image",
        capacity_block_group="capacityblockgroup-example",
        infiniband_fabric="us-central1-b",
    )

    assert result.status == "ready"
    assert result.preflight["topology"]["capacity_block_group"] == (
        "capacityblockgroup-example"
    )
    assert result.preflight["topology"]["gpu_preemptible"] is False
    assert not any(
        quota["name"].startswith("compute.instance.gpu.")
        for quota in result.preflight["quotas"]
    )
    assert seen["desired"].gpu_nodes.capacity_block_group == (
        "capacityblockgroup-example"
    )
    assert seen["desired"].gpu_nodes.preemptible is False
    assert seen["plan"]["gpu_reservation"] == "strict"
    assert any("provider_mutation=false" in action for action in result.actions)


def test_reserved_capacity_recovery_uses_the_effective_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    from npa.provisioning_journal import ProvisioningOperation

    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setattr(provisioning, "_has_cached_kubeconfig", lambda *_a, **_k: False)
    applied: dict[str, object] = {}
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle.up_cmd",
        lambda **kwargs: applied.update(kwargs),
    )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="reserved-cluster",
        kubeconfig=tmp_path / "reserved-kubeconfig",
        skip_s3=True,
        _resolved_plan=_reserved_capacity_plan(),
    )

    journal = ProvisioningOperation(result.operation_id).read()
    resume_argv = journal["recovery_commands"]["resume_argv"]
    group_index = resume_argv.index("--capacity-block-group")
    assert resume_argv[group_index + 1] == "capacityblockgroup-example"
    assert "--on-demand" in resume_argv
    assert "--preemptible" not in resume_argv
    assert journal["preflight_plan"]["topology"]["capacity_block_group"] == (
        "capacityblockgroup-example"
    )
    assert applied["capacity_block_group"] == "capacityblockgroup-example"
    assert applied["preemptible"] is False


def test_empty_failed_operation_can_start_a_new_topology_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU preflight failure must not block a later CPU-only retry forever."""
    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    calls: list[dict] = []

    def up_cmd(**kwargs):
        calls.append(kwargs)
        if kwargs["gpu_nodes"] == 1:
            raise RuntimeError("synthetic pre-resource failure")

    monkeypatch.setattr("npa.cli.cluster.terraform_lifecycle.up_cmd", up_cmd)
    kubeconfig = tmp_path / "missing-kubeconfig"
    with pytest.raises(RuntimeError, match="synthetic pre-resource failure"):
        provisioning.provision_if_absent(
            project="proj",
            cluster_name="retry-shape",
            kubeconfig=kubeconfig,
            skip_s3=True,
            cpu_nodes=0,
            gpu_nodes=1,
        )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="retry-shape",
        kubeconfig=kubeconfig,
        skip_s3=True,
        cpu_nodes=1,
        gpu_nodes=0,
    )

    assert result.operation_id.endswith("-r1")
    assert [call["gpu_nodes"] for call in calls] == [1, 0]
    from npa.provisioning_journal import ProvisioningOperation

    first = ProvisioningOperation(result.operation_id.removesuffix("-r1")).read()
    second = ProvisioningOperation(result.operation_id).read()
    assert first["phase"] == "rolled-back"
    assert first["rollback"]["completed"] is True
    assert second["preflight_plan"]["topology"]["gpu_nodes"] == 0
    assert second["preflight_plan"]["topology"]["cpu_nodes"] == 1


def test_provision_if_absent_preflight_uses_terraform_disk_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("TF_VAR_cpu_disk_size", "96")
    monkeypatch.setenv("TF_VAR_gpu_disk_size", "600")

    result = provisioning.provision_if_absent(
        project="proj",
        kubeconfig=tmp_path / "missing-kubeconfig",
        dry_run=True,
        skip_s3=True,
    )

    topology = result.preflight["topology"]
    assert topology["cpu_disk_gib"] == 96
    assert topology["gpu_disk_gib"] == 600
    assert topology["required_network_ssd_gib"] == "696"


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
    _successful_storage_probe,
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
    # The fixture fakes the lookup, so this is the assertion that keeps the
    # resolution itself covered: reusing a kubeconfig must still demand kubectl.
    assert "kubectl" in _successful_storage_probe


def test_reused_cluster_runs_requested_skypilot_smoke(
    tmp_path: Path, monkeypatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._check_skypilot_kubernetes",
        lambda *args, **kwargs: seen.append(("check", *args, kwargs)),
    )
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
    assert [item[0] for item in seen] == ["check", "readiness", "smoke"]
    assert seen[0] == (
        "check",
        kubeconfig,
        "npa-cluster",
        {"sky_bin": "/opt/npa/sky"},
    )
    readiness = seen[1]
    assert readiness[1] == ["RTXPRO6000:1"]
    assert readiness[-1]["kubeconfig"] == kubeconfig
    assert readiness[-1]["sky_bin"] == "/opt/npa/sky"
    assert readiness[-1]["label_known_gpus"] is True
    assert seen[2] == (
        "smoke",
        kubeconfig,
        "npa-cluster",
        "npa-cluster",
        "RTXPRO6000:1",
        {"sky_bin": "/opt/npa/sky", "credentials_checked": True},
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
        "npa.cli.cluster.terraform_lifecycle._check_skypilot_kubernetes",
        lambda *_args, **kwargs: seen.append(("check", kwargs)),
    )
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
    assert [item[0] for item in seen] == ["up", "check", "readiness", "smoke"]
    up_kwargs = seen[0][1]
    assert up_kwargs["sky_smoke"] is False
    assert up_kwargs["sky_bin"] == "/opt/npa/sky"
    assert seen[1][1]["sky_bin"] == "/opt/npa/sky"
    assert seen[2][1]["label_known_gpus"] is True
    assert seen[2][1]["sky_bin"] == "/opt/npa/sky"
    assert seen[3][1]["sky_bin"] == "/opt/npa/sky"
    assert seen[3][1]["credentials_checked"] is True


def test_skypilot_smoke_uses_the_accelerator_name_resolved_by_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_runtime(tmp_path, monkeypatch)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    requested = "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    resolved = "NVIDIA-RTX-PRO-6000-BLACKWELL-SERVER-EDITION:1"
    seen: list[str] = []
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        lambda *_args, **_kwargs: {
            requested: SimpleNamespace(resolved=resolved),
        },
    )
    monkeypatch.setattr(
        "npa.cli.cluster.terraform_lifecycle._run_skypilot_smoke",
        lambda *_args, **_kwargs: seen.append(_args[3]),
    )

    result = provisioning.provision_if_absent(
        project="proj",
        cluster_name="npa-cluster",
        kubeconfig=kubeconfig,
        sky_smoke=True,
        accelerator=requested,
    )

    assert result.status == "ok"
    assert seen == [resolved]


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


def test_provision_cli_runs_a_successful_empty_failure_reconciliation(mocker) -> None:
    """The explicit recovery result is terminal, not a partial provision."""
    from npa.cli.main import app

    reconcile = mocker.patch(
        "npa.cli.provision.provision_if_absent",
        return_value=provisioning.ProvisionIfAbsentResult(
            status="reconciled",
            project="proj",
            cluster_name="failed-cluster",
            actions=["recovery:terminalized failed empty provision"],
        ),
    )

    result = runner.invoke(
        app,
        [
            "provision-if-absent",
            "--project",
            "proj",
            "--cluster-name",
            "failed-cluster",
            "--reconcile-empty-failure",
            "--recovery-evidence",
            "/private/recovery-evidence.json",
        ],
    )

    assert result.exit_code == 0
    assert reconcile.call_args.kwargs["reconcile_empty_failure"] is True
    assert reconcile.call_args.kwargs["recovery_evidence"] == Path(
        "/private/recovery-evidence.json"
    )
    assert "status: reconciled" in result.output


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
