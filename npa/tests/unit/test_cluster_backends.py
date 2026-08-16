from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from npa.cluster_backends import BackendOwnershipError, get_backend, persisted_backend
from npa.cluster_backends.mk8s import (
    MK8sApplyRequest,
    MK8sDestroyRequest,
    MK8sExecutionScope,
    MK8sProjectIdentity,
    MK8sStatusRequest,
)
from npa.cluster_backends.mk8s_model import MK8sDesired, MK8sNodePool
from npa.cluster_backends.soperator import SoperatorApplyRequest
from npa.fleet.lifecycle import fleet_status, plan_fleet
from npa.fleet.spec import (
    ClusterSpec,
    FleetSpec,
    FleetSpecError,
    NodePoolSpec,
    ProjectSpec,
    spec_from_mapping,
)
from npa.soperator.spec import spec_from_mapping as soperator_spec_from_mapping


def _legacy_mk8s_mapping() -> dict:
    return {
        "apiVersion": "npa.fleet/v0.0.1",
        "name": "compatible",
        "region": "us-central1",
        "defaults": {
            "cpu_nodes": {
                "count": 1,
                "platform": "cpu-d3",
                "preset": "8vcpu-32gb",
            }
        },
        "projects": [{"project_id": "project-test"}],
    }


def _soperator_envelope() -> dict:
    return {
        "name": "slurm",
        "backend": "soperator",
        "soperator": {
            "workers": [{"name": "cpu", "platform": "cpu-d3", "preset": "8vcpu-32gb"}]
        },
    }


def test_native_mk8s_desired_validation_is_fail_closed() -> None:
    desired = MK8sDesired(
        name="native",
        gpu_nodes=MK8sNodePool(
            count=1,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            capacity_block_group="capacity-test",
            preemptible=True,
        ),
    )
    with pytest.raises(ValueError, match="strict reserved.*preemptible"):
        get_backend("mk8s").validate(desired)


def test_legacy_fleet_defaults_to_mk8s_without_plan_drift() -> None:
    legacy = spec_from_mapping(_legacy_mk8s_mapping())
    explicit_mapping = _legacy_mk8s_mapping()
    explicit_mapping["projects"][0]["clusters"] = [
        {
            "backend": "mk8s",
            "mk8s": {
                "cpu_nodes": {
                    "count": 1,
                    "platform": "cpu-d3",
                    "preset": "8vcpu-32gb",
                }
            },
        }
    ]
    explicit = spec_from_mapping(explicit_mapping)

    old_cluster = legacy.projects[0].clusters[0]
    new_cluster = explicit.projects[0].clusters[0]
    assert old_cluster.backend_name() == "mk8s"
    assert old_cluster == new_cluster
    assert "backend" not in plan_fleet(legacy)["projects"][0]["clusters"][0]
    assert plan_fleet(explicit)["projects"][0]["clusters"][0]["backend"] == "mk8s"


def test_mixed_fleet_plans_through_native_backend_adapters() -> None:
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {"name": "kube", "backend": "mk8s", "mk8s": {}},
        _soperator_envelope(),
    ]
    spec = spec_from_mapping(mapping)
    plan = plan_fleet(spec)

    clusters = {item["name"]: item for item in plan["projects"][0]["clusters"]}
    assert clusters["kube"]["backend"] == "mk8s"
    expected_soperator = get_backend("soperator").plan(
        spec.projects[0].clusters[1].soperator
    )
    expected_soperator["region"] = "us-central1"
    assert clusters["slurm"] == {
        "backend": "soperator",
        **expected_soperator,
    }


def test_standalone_and_one_target_fleet_share_complete_mig_contract(
    tmp_path, monkeypatch
) -> None:
    from npa.cluster_backends import mk8s_execution

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {
            "name": "mig",
            "backend": "mk8s",
            "mk8s": {
                "cpu_nodes": {"count": 0},
                "gpu_nodes": {
                    "count": 2,
                    "platform": "gpu-rtx6000",
                    "preset": "1gpu-24vcpu-218gb",
                    "capacity_block_group": "capacity-test",
                },
                "gpu_health_timeout_minutes": 37,
                "gpu_cuda_smoke_image": "registry.example/cuda-smoke:pinned",
                "mig": {"enabled": True},
            },
        }
    ]
    fleet = spec_from_mapping(mapping)
    fleet_desired = fleet.projects[0].clusters[0]
    standalone_desired = ClusterSpec(
        **{
            name: getattr(fleet_desired, name)
            for name in fleet_desired.__dataclass_fields__
            if name not in {"backend_explicit"}
        }
    )
    backend = get_backend("mk8s")

    standalone_plan = backend.plan(standalone_desired)
    fleet_plan = backend.plan(fleet_desired)
    assert standalone_plan == fleet_plan
    assert standalone_plan["mig"] == {"strategy": "mixed", "config": "all-balanced"}
    assert standalone_plan["gpu_health_timeout_minutes"] == 37
    assert standalone_plan["gpu_cuda_smoke"] is True
    standalone_render = backend.materialize(
        standalone_desired, MK8sApplyRequest()
    ).deployment_inputs
    fleet_render = backend.materialize(
        fleet_desired, MK8sApplyRequest()
    ).deployment_inputs
    assert standalone_render == fleet_render

    monkeypatch.setattr(
        mk8s_execution, "is_verified_unchanged_target", lambda **_kwargs: True
    )
    request = MK8sApplyRequest(
        **_mk8s_execution_identity(fleet, fleet.projects[0]),
        fleet_root=tmp_path,
        tenant_id="tenant-test",
        region="us-central1",
        nebius_bin="nebius",
        provider_env={},
        provider_preflight=True,
    )
    assert backend.preflight(standalone_desired, request) == backend.preflight(
        fleet_desired, request
    )

    calls: list[dict] = []

    class Report:
        nodes = [object(), object()]

        def as_dict(self):
            return {"nodes": 2}

    status_request = MK8sStatusRequest(
        kubeconfig=tmp_path / "kubeconfig",
        mig_verifier=lambda **kwargs: calls.append(kwargs) or Report(),
    )
    assert backend.verify(standalone_desired, status_request) == backend.verify(
        fleet_desired, status_request
    )
    assert calls[0]["reconcile"] is calls[1]["reconcile"] is True
    assert calls[0]["timeout_seconds"] == calls[1]["timeout_seconds"] == 37 * 60
    assert calls[0]["cuda_smoke_image"] == calls[1]["cuda_smoke_image"]


def test_backend_materializers_preserve_native_deployment_inputs() -> None:
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {"name": "kube", "backend": "mk8s", "mk8s": {}},
        _soperator_envelope(),
    ]
    spec = spec_from_mapping(mapping)
    mk8s = spec.projects[0].clusters[0]
    soperator = spec.projects[0].clusters[1].soperator
    assert soperator is not None

    mk8s_materialized = get_backend("mk8s").materialize(mk8s, MK8sApplyRequest())
    sop_materialized = get_backend("soperator").materialize(
        soperator, SoperatorApplyRequest()
    )
    assert (
        "cpu_nodes_fixed_count = 1"
        in mk8s_materialized.deployment_inputs["terraform_tfvars"]
    )
    assert (
        "slurm_nodeset_workers"
        in sop_materialized.deployment_inputs["terraform_tfvars"]
    )


def test_standalone_and_one_target_fleet_share_soperator_contract() -> None:
    native_mapping = {
        "apiVersion": "npa.soperator/v0.0.1",
        "name": "slurm",
        "workers": [
            {
                "name": "gpu",
                "platform": "gpu-b200-sxm",
                "preset": "8gpu-160vcpu-1792gb",
                "size": 1,
                "fabric": "us-central1-b",
                "capacity_block_group": "capacity-test",
                "docker_cache": True,
            }
        ],
    }
    standalone = soperator_spec_from_mapping(native_mapping)
    fleet_mapping = _legacy_mk8s_mapping()
    fleet_mapping["projects"][0]["clusters"] = [
        {
            "name": "slurm",
            "backend": "soperator",
            "soperator": {
                key: value
                for key, value in native_mapping.items()
                if key not in {"apiVersion", "name"}
            },
        }
    ]
    fleet = spec_from_mapping(fleet_mapping)
    from_fleet = fleet.projects[0].clusters[0].soperator
    assert from_fleet is not None
    backend = get_backend("soperator")

    assert standalone == from_fleet
    assert backend.plan(standalone) == backend.plan(from_fleet)
    assert (
        backend.materialize(standalone, SoperatorApplyRequest()).deployment_inputs
        == backend.materialize(from_fleet, SoperatorApplyRequest()).deployment_inputs
    )


def test_mk8s_materializer_preserves_preemptible_capacity_mode() -> None:
    desired = ClusterSpec(
        name="spot",
        gpu_nodes=NodePoolSpec(
            count=1,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            preemptible=True,
        ),
    )
    rendered = (
        get_backend("mk8s")
        .materialize(desired, MK8sApplyRequest())
        .deployment_inputs["terraform_tfvars"]
    )
    assert "gpu_nodes_preemptible        = true" in rendered


def test_cross_backend_fields_fail_precisely() -> None:
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {**_soperator_envelope(), "gpu_nodes": {"count": 2}}
    ]
    with pytest.raises(FleetSpecError, match="unsupported mk8s/flat field.*gpu_nodes"):
        spec_from_mapping(mapping)


def test_explicit_mk8s_envelope_rejects_unknown_nested_fields() -> None:
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {
            "backend": "mk8s",
            "mk8s": {
                "cpu_nodes": {
                    "count": 1,
                    "platform": "cpu-d3",
                    "preset": "8vcpu-32gb",
                    "typo": True,
                }
            },
        }
    ]
    with pytest.raises(FleetSpecError, match=r"cluster\.mk8s\.cpu_nodes.*typo"):
        spec_from_mapping(mapping)


def test_duplicate_soperator_physical_names_across_projects_fail() -> None:
    mapping = _legacy_mk8s_mapping()
    mapping["projects"] = [
        {"project_id": "project-a", "clusters": [_soperator_envelope()]},
        {"project_id": "project-b", "clusters": [_soperator_envelope()]},
    ]
    with pytest.raises(
        FleetSpecError, match="duplicate soperator cluster name 'slurm'"
    ):
        spec_from_mapping(mapping).validate()


def test_old_state_is_mk8s_and_backend_mismatch_fails_closed(tmp_path) -> None:
    assert persisted_backend({"status": "deployed"}) == "mk8s"
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [_soperator_envelope()]
    spec = spec_from_mapping(mapping)
    root = tmp_path / spec.name
    root.mkdir(parents=True)
    (root / "fleet-state.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "clusters": [
                    {
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                        "status": "deployed",
                    }
                ],
            }
        )
    )

    with pytest.raises(BackendOwnershipError, match="belongs to backend 'mk8s'"):
        fleet_status(spec, work_root=tmp_path)


def test_corrupt_fleet_inventory_fails_closed(tmp_path) -> None:
    spec = spec_from_mapping(_legacy_mk8s_mapping())
    root = tmp_path / spec.name
    root.mkdir(parents=True)
    (root / "fleet-state.json").write_text("{broken")
    with pytest.raises(RuntimeError, match="fleet inventory.*unreadable"):
        fleet_status(spec, work_root=tmp_path)


def test_fleet_inventory_write_failure_is_not_warning_only(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_write_json_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        lifecycle._write_fleet_state(tmp_path, {"clusters": []})


def test_mixed_deploy_dispatches_soperator_adapter_without_cli_shellout(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {"name": "kube", "backend": "mk8s", "mk8s": {}},
        _soperator_envelope(),
    ]
    spec = spec_from_mapping(mapping)
    seen: list[str] = []

    class Adapter:
        def apply(self, desired, request):
            seen.append(desired.name)
            assert (
                request.work_root
                == tmp_path / "compatible/project-test/slurm/soperator"
            )
            return {"name": desired.name, "status": "deployed"}

    monkeypatch.setattr(
        lifecycle,
        "_deploy_mk8s_fleet",
        lambda *_args, **_kwargs: {
            "name": "compatible",
            "clusters": [
                {
                    "project_key": "project-test",
                    "cluster_name": "kube",
                    "status": "deployed",
                }
            ],
        },
    )
    monkeypatch.setattr(lifecycle, "get_backend", lambda name: Adapter())
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_args: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda region: region)
    monkeypatch.setattr(
        lifecycle,
        "resolve_project_id",
        lambda *_args, **_kwargs: ("project-test", False),
    )
    monkeypatch.setattr(
        lifecycle, "ensure_subnet", lambda *_args, **_kwargs: ("subnet-test", "")
    )
    monkeypatch.setattr(lifecycle, "_upsert_fleet_state", lambda *_args: None)

    result = lifecycle.deploy_fleet(spec, work_root=tmp_path)

    assert seen == ["slurm"]
    assert [item.get("backend", "mk8s") for item in result["clusters"]] == [
        "mk8s",
        "soperator",
    ]


def test_soperator_backend_status_delegates_real_native_status(monkeypatch) -> None:
    from npa.soperator import lifecycle
    from npa.cluster_backends.soperator import SoperatorStatusRequest
    from npa.soperator.spec import SoperatorSpec

    seen: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "cluster_status",
        lambda name, **_kwargs: (
            seen.append(name)
            or {"name": name, "status": "running", "sinfo": "idle\n", "workers": []}
        ),
    )

    status = get_backend("soperator").status(
        SoperatorSpec(name="slurm"), SoperatorStatusRequest()
    )

    assert seen == ["slurm"]
    assert status["status"] == "running"
    assert status["sinfo"] == "idle\n"


def test_soperator_sidecar_keeps_auth_and_observability_profiles_separate(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import _load_env_sidecar, _write_env_sidecar

    _write_env_sidecar(
        tmp_path,
        region="region-test",
        tenant_id="tenant-test",
        project_id="project-test",
        subnet_id="subnet-test",
        o11y_profile="telemetry-profile",
        auth_profile="operator-profile",
    )

    saved = _load_env_sidecar(tmp_path)
    assert saved is not None
    assert saved["auth_profile"] == "operator-profile"
    assert saved["o11y_profile"] == "telemetry-profile"


def test_destroy_rejects_noncanonical_soperator_state_root_before_adapter(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [_soperator_envelope()]
    spec = spec_from_mapping(mapping)
    root = tmp_path / spec.name
    root.mkdir(parents=True)
    (root / "fleet-state.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "backend": "soperator",
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                        "backend_state_root": str(tmp_path / "outside"),
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        lifecycle,
        "get_backend",
        lambda _name: pytest.fail("adapter must not receive an untrusted state root"),
    )

    with pytest.raises(ValueError, match="canonical fleet-owned root"):
        lifecycle.destroy_fleet(spec, work_root=tmp_path)


def test_standalone_mk8s_apply_adopts_native_one_target_result(
    tmp_path, monkeypatch
) -> None:
    from npa.cluster import state
    from npa.cluster_backends import mk8s_execution

    monkeypatch.setattr(state, "CLUSTERS_DIR", tmp_path / "clusters")
    source = tmp_path / "source-kubeconfig"
    source.write_text(
        "apiVersion: v1\nkind: Config\ncurrent-context: fleet-generated\n"
        "contexts:\n- name: fleet-generated\n  context: {cluster: c, user: u}\n"
    )
    desired = ClusterSpec(
        name="provider-name",
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="8vcpu-32gb"),
    )
    project = ProjectSpec(
        name="standalone-target", project_id="project-test", clusters=[desired]
    )
    spec = FleetSpec(
        name="standalone",
        tenant_id="tenant-test",
        region="region-test",
        projects=[project],
    )

    def native_apply(**_kwargs):
        return {
            "status": "deployed",
            "cluster_id": "cluster-test",
            "kube_context": "fleet-generated",
            "kubeconfig": str(source),
        }

    target = tmp_path / "requested-kubeconfig"
    backend_root = tmp_path / "clusters" / "requested" / "backend-state"
    monkeypatch.setattr(mk8s_execution, "deploy_cluster", native_apply)

    result = get_backend("mk8s").apply(
        desired,
        MK8sApplyRequest(
            **_mk8s_execution_identity(spec, project),
            project_id="project-test",
            region="region-test",
            tenant_id="tenant-test",
            fleet_root=backend_root,
            recipe_root=tmp_path / "recipe",
            terraform_bin="terraform",
            nebius_bin="nebius",
            standalone_context="requested",
            standalone_kubeconfig=target,
        ),
    )

    assert result["ownership"] == "standalone-shared-mk8s-backend"
    assert "current-context: requested" in target.read_text()
    metadata = json.loads((tmp_path / "clusters/requested/metadata.json").read_text())
    assert metadata["backend"] == "mk8s"
    assert metadata["backend_cluster_id"] == "cluster-test"
    assert metadata["backend_state_root"] == str(backend_root.resolve())


def test_mk8s_preflight_reuses_provider_verified_zero_increment(
    tmp_path, monkeypatch
) -> None:
    from npa.cluster_backends import mk8s_execution
    from npa.fleet import quotas

    desired = ClusterSpec(
        name="mig",
        gpu_nodes=NodePoolSpec(
            count=2,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            capacity_block_group="reservation-test",
        ),
    )
    project = ProjectSpec(name="target", project_id="project-test", clusters=[desired])
    spec = FleetSpec(
        name="standalone",
        tenant_id="tenant-test",
        region="region-test",
        projects=[project],
    )
    monkeypatch.setattr(
        mk8s_execution, "is_verified_unchanged_target", lambda **_kwargs: True
    )
    monkeypatch.setattr(
        quotas,
        "preflight_region",
        lambda **_kwargs: pytest.fail("zero-increment target must not consume quota"),
    )

    result = get_backend("mk8s").preflight(
        desired,
        MK8sApplyRequest(
            **_mk8s_execution_identity(spec, project),
            tenant_id="tenant-test",
            region="region-test",
            nebius_bin="nebius",
            provider_env={},
            provider_preflight=True,
            fleet_root=tmp_path,
        ),
    )

    assert result["incremental_demand"] == 0
    assert result["capacity_quota"] == "provider-verified-zero-increment"


def test_id_backed_standalone_target_proves_zero_increment_without_name_match(
    tmp_path, monkeypatch
) -> None:
    import subprocess

    from npa.cluster_backends import mk8s_execution
    from npa.fleet import quotas
    from npa.fleet.tfvars import render_tfvars

    desired = ClusterSpec(name="existing", allow_control_plane_only=True)
    project = ProjectSpec(project_id="project-test", clusters=[desired])
    spec = FleetSpec(
        name="standalone",
        tenant_id="tenant-test",
        region="region-test",
        projects=[project],
    )
    install_dir = tmp_path / project.key() / desired.name
    (install_dir / "k8s-training").mkdir(parents=True)
    (install_dir / ".npa-fleet-env.json").write_text(
        json.dumps(
            {
                "status": "deployed",
                "project_id": "project-test",
                "cluster_id": "cluster-test",
                "tenant_id": "tenant-test",
                "region": "region-test",
                "cluster_name": "existing",
            }
        )
    )
    (install_dir / "k8s-training/terraform.tfvars").write_text(render_tfvars(desired))
    monkeypatch.setattr(
        mk8s_execution,
        "_get_project",
        lambda *_args: {
            "metadata": {
                "id": "project-test",
                "name": "real-cloud-project-name",
                "parentId": "tenant-test",
            },
            "spec": {"region": "region-test"},
        },
    )

    def provider_capture(args, **_kwargs):
        payload = (
            {
                "metadata": {
                    "id": "cluster-test",
                    "name": "existing",
                    "parentId": "project-test",
                },
                "status": {"state": "RUNNING"},
            }
            if "cluster" in args and "get" in args
            else {"items": []}
        )
        return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider_capture)
    monkeypatch.setattr(
        quotas,
        "preflight_region",
        lambda **_kwargs: pytest.fail("verified target must have zero demand"),
    )

    result = get_backend("mk8s").preflight(
        desired,
        MK8sApplyRequest(
            **_mk8s_execution_identity(spec, project),
            tenant_id="tenant-test",
            region="region-test",
            nebius_bin="nebius",
            provider_env={},
            provider_preflight=True,
            fleet_root=tmp_path,
        ),
    )

    assert result["capacity_quota"] == "provider-verified-zero-increment"


def test_mk8s_backend_destroy_calls_backend_owned_exact_destroy(
    tmp_path, monkeypatch
) -> None:
    from npa.cluster_backends import mk8s_execution

    desired = ClusterSpec(name="cluster", allow_control_plane_only=True)
    project = ProjectSpec(project_id="project-test", clusters=[desired])
    spec = FleetSpec(name="standalone", projects=[project])
    destroyed: list[str] = []
    monkeypatch.setattr(
        mk8s_execution,
        "destroy_cluster",
        lambda **_kwargs: (
            destroyed.append(_kwargs["cluster"].name)
            or {"cluster_name": "cluster", "status": "destroyed"}
        ),
    )

    result = get_backend("mk8s").destroy(
        desired,
        MK8sDestroyRequest(
            **_mk8s_execution_identity(spec, project),
            fleet_root=tmp_path,
            terraform_bin="terraform",
            nebius_bin="nebius",
        ),
    )

    assert result and result["status"] == "destroyed"
    assert destroyed == ["cluster"]


def test_mk8s_backend_owns_standalone_terraform_apply(monkeypatch, tmp_path) -> None:
    from npa.cluster_backends import process

    spec = spec_from_mapping(_legacy_mk8s_mapping())
    cluster = spec.projects[0].clusters[0]
    seen: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(
        process,
        "run_stream",
        lambda args, **kwargs: seen.append((args, kwargs["cwd"])),
    )

    result = get_backend("mk8s").apply(
        cluster,
        MK8sApplyRequest(
            terraform_command=("terraform", "apply", "-auto-approve"),
            terraform_cwd=tmp_path,
            terraform_env={},
            terraform_timeout_seconds=60,
            command_runner=process.run_stream,
        ),
    )

    assert seen == [(["terraform", "apply", "-auto-approve"], tmp_path)]
    assert result["status"] == "applied"


def test_mig_verification_contract_is_backend_owned(tmp_path) -> None:
    from npa.cluster_backends.mk8s import MK8sStatusRequest

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {
            "name": "mig",
            "backend": "mk8s",
            "mk8s": {
                "gpu_nodes": {
                    "count": 2,
                    "platform": "gpu-rtx6000",
                    "preset": "1gpu-24vcpu-218gb",
                    "capacity_block_group": "capacity-test",
                },
                "cpu_nodes": {"count": 0},
                "gpu_health_timeout_minutes": 37,
                "gpu_cuda_smoke_image": "registry.example/cuda-smoke:pinned",
                "mig": {"enabled": True},
            },
        }
    ]
    cluster = spec_from_mapping(mapping).projects[0].clusters[0]
    seen: dict = {}

    class Report:
        nodes = [object(), object()]

        def as_dict(self):
            return {"nodes": 2}

    def verify(**kwargs):
        seen.update(kwargs)
        return Report()

    result = get_backend("mk8s").verify(
        cluster,
        MK8sStatusRequest(
            kubeconfig=tmp_path / "kubeconfig",
            kubectl_bin="kubectl-test",
            mig_verifier=verify,
        ),
    )

    assert seen["expected_nodes"] == 2
    assert seen["reconcile"] is True
    assert seen["timeout_seconds"] == 37 * 60
    assert seen["cuda_smoke_image"] == "registry.example/cuda-smoke:pinned"
    assert result["mig"] == {"nodes": 2}


def test_project_network_cleanup_waits_for_every_backend_inventory(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [_soperator_envelope()]
    spec = spec_from_mapping(mapping)
    fleet_root = tmp_path / spec.name
    project_root = fleet_root / "project-test"
    project_root.mkdir(parents=True)
    network_state = project_root / ".npa-fleet-network.json"
    network_state.write_text(
        json.dumps(
            {
                "project_id": "project-test",
                "created_network_id": "network-test",
                "subnet_id": "subnet-test",
            }
        )
    )
    (fleet_root / "fleet-state.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "backend": "soperator",
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        lifecycle,
        "_reclaim_created_network",
        lambda *_args, **_kwargs: pytest.fail("surviving target owns the network"),
    )

    result = lifecycle._reclaim_unused_project_networks(
        spec,
        fleet_root=fleet_root,
        nebius_bin="nebius",
        prefix="",
        only_projects=None,
        profile=None,
        on_status=None,
    )

    assert result == []
    assert network_state.exists()


def test_soperator_fleet_apply_honors_concurrency(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"] = [
        {"project_id": "project-a", "clusters": [_soperator_envelope()]},
        {
            "project_id": "project-b",
            "clusters": [{**_soperator_envelope(), "name": "slurm-b"}],
        },
    ]
    spec = spec_from_mapping(mapping)
    barrier = threading.Barrier(2)
    threads: set[int] = set()

    class Adapter:
        def apply(self, desired, request):
            threads.add(threading.get_ident())
            barrier.wait(timeout=5)
            return {"name": desired.name, "status": "deployed"}

    monkeypatch.setattr(lifecycle, "get_backend", lambda _name: Adapter())
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_args: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda region: region)
    monkeypatch.setattr(
        lifecycle,
        "resolve_project_id",
        lambda _bin, _tenant, project, **_kwargs: (project.project_id, False),
    )
    monkeypatch.setattr(
        lifecycle, "ensure_subnet", lambda *_args, **_kwargs: ("subnet-test", "")
    )

    result = lifecycle.deploy_fleet(spec, work_root=tmp_path, concurrency=2)

    assert len(threads) == 2
    assert result["deployed"] == 2


def test_reserved_soperator_preflight_precedes_subnet_mutation(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    envelope = _soperator_envelope()
    envelope["soperator"]["workers"] = [
        {
            "name": "gpu",
            "platform": "gpu-b200-sxm",
            "preset": "8gpu-160vcpu-1792gb",
            "fabric": "us-central1-b",
            "capacity_block_group": "capacity-test",
        }
    ]
    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [envelope]
    spec = spec_from_mapping(mapping)

    class Adapter:
        def preflight(self, desired, request):
            assert request.provider_preflight is True
            raise ValueError("reserved capacity incompatible")

    monkeypatch.setattr(lifecycle, "get_backend", lambda _name: Adapter())
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_args: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda region: region)
    monkeypatch.setattr(
        lifecycle,
        "resolve_project_id",
        lambda *_args, **_kwargs: ("project-test", False),
    )
    monkeypatch.setattr(
        lifecycle,
        "ensure_subnet",
        lambda *_args, **_kwargs: pytest.fail("subnet mutation preceded preflight"),
    )

    result = lifecycle.deploy_fleet(spec, work_root=tmp_path)
    assert result["clusters"][0]["status"] == "error"
    assert "reserved capacity incompatible" in result["clusters"][0]["error"]


def test_strict_reservation_proves_nonpreemptible_when_provider_omits_false() -> None:
    from npa.cluster_backends import mk8s_execution
    from npa.fleet.spec import NodePoolSpec

    pool = NodePoolSpec(
        count=2,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacity-test",
    )
    payload = {
        "spec": {
            "fixedNodeCount": "2",
            "template": {
                "resources": {
                    "platform": "gpu-rtx6000",
                    "preset": "1gpu-24vcpu-218gb",
                },
                "reservationPolicy": {
                    "policy": "STRICT",
                    "reservationIds": ["capacity-test"],
                },
            },
        },
        "status": {"state": "RUNNING"},
    }

    assert mk8s_execution._provider_node_group_matches_pool(payload, pool) is True


def test_successful_soperator_destroy_removes_backend_root_and_reclaims_network(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [_soperator_envelope()]
    spec = spec_from_mapping(mapping)
    fleet_root = tmp_path / spec.name
    backend_root = fleet_root / "project-test/slurm/soperator"
    backend_root.mkdir(parents=True)
    (backend_root / "retained-source").mkdir()
    project_root = fleet_root / "project-test"
    (project_root / ".npa-fleet-network.json").write_text(
        json.dumps(
            {
                "project_id": "project-test",
                "created_network_id": "network-test",
                "subnet_id": "subnet-test",
            }
        )
    )
    (fleet_root / "fleet-state.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "backend": "soperator",
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                        "backend_state_root": str(backend_root),
                    }
                ]
            }
        )
    )
    reclaimed: list[str] = []

    class Adapter:
        def destroy(self, desired, request):
            assert request.work_root == backend_root
            return {"status": "destroyed"}

    monkeypatch.setattr(lifecycle, "get_backend", lambda _name: Adapter())
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle,
        "_reclaim_created_network",
        lambda _bin, _project, network, *_args, **_kwargs: (
            reclaimed.append(network) or []
        ),
    )

    result = lifecycle.destroy_fleet(spec, work_root=tmp_path)

    assert result["failed"] == 0
    assert not backend_root.exists()
    assert reclaimed == ["network-test"]
    assert not (project_root / ".npa-fleet-network.json").exists()


def test_mixed_destroy_dispatches_both_adapters_and_keeps_state_collision_free(
    tmp_path, monkeypatch
) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {"name": "kube", "backend": "mk8s", "mk8s": {}},
        _soperator_envelope(),
    ]
    spec = spec_from_mapping(mapping)
    fleet_root = tmp_path / spec.name
    sop_root = fleet_root / "project-test/slurm/soperator"
    sop_root.mkdir(parents=True)
    (fleet_root / "fleet-state.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "backend": "mk8s",
                        "project_key": "project-test",
                        "cluster_name": "kube",
                    },
                    {
                        "backend": "soperator",
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                        "backend_state_root": str(sop_root),
                    },
                ]
            }
        )
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        lifecycle,
        "_destroy_mk8s_fleet",
        lambda subset, **_kwargs: (
            calls.append(("mk8s", subset.projects[0].clusters[0].name))
            or {
                "clusters": [
                    {
                        "backend": "mk8s",
                        "project_key": "project-test",
                        "cluster_name": "kube",
                        "status": "destroyed",
                    }
                ]
            }
        ),
    )

    class SoperatorAdapter:
        def destroy(self, desired, request):
            calls.append(("soperator", desired.name))
            assert request.work_root == sop_root
            return {"backend": "soperator", "status": "destroyed"}

    monkeypatch.setattr(lifecycle, "get_backend", lambda name: SoperatorAdapter())
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_reclaim_unused_project_networks", lambda *_args, **_kwargs: []
    )

    result = lifecycle.destroy_fleet(spec, work_root=tmp_path)

    assert calls == [("mk8s", "kube"), ("soperator", "slurm")]
    assert {(item["backend"], item["cluster_name"]) for item in result["clusters"]} == {
        ("mk8s", "kube"),
        ("soperator", "slurm"),
    }
    inventory = json.loads((fleet_root / "fleet-state.json").read_text())
    assert inventory["clusters"] == [
        {
            "backend": "mk8s",
            "project_key": "project-test",
            "cluster_name": "kube",
        }
    ]


def test_mixed_status_aggregates_native_backend_failure(tmp_path, monkeypatch) -> None:
    from npa.fleet import lifecycle

    mapping = _legacy_mk8s_mapping()
    mapping["projects"][0]["clusters"] = [
        {"name": "kube", "backend": "mk8s", "mk8s": {}},
        _soperator_envelope(),
    ]
    spec = spec_from_mapping(mapping)
    fleet_root = tmp_path / spec.name
    fleet_root.mkdir(parents=True)
    (fleet_root / "fleet-state.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "backend": "mk8s",
                        "project_key": "project-test",
                        "cluster_name": "kube",
                        "status": "deployed",
                    },
                    {
                        "backend": "soperator",
                        "project_key": "project-test",
                        "cluster_name": "slurm",
                        "status": "provisioning",
                    },
                ]
            }
        )
    )

    class Adapter:
        def status(self, desired, request):
            if desired.name == "kube":
                return {"backend": "mk8s", "status": "deployed"}
            raise RuntimeError("controller is not ready")

    monkeypatch.setattr(lifecycle, "get_backend", lambda _name: Adapter())

    result = lifecycle.fleet_status(spec, work_root=tmp_path)
    statuses = {item["cluster_name"]: item for item in result["clusters"]}
    assert statuses["kube"]["status"] == "deployed"
    assert statuses["slurm"]["status"] == "status-error"
    assert "controller is not ready" in statuses["slurm"]["error"]


def _mk8s_execution_identity(spec, project) -> dict:
    return {
        "scope": MK8sExecutionScope(
            fleet_name=spec.name,
            tenant_id=spec.tenant_id,
            region=spec.region,
            project_prefix=spec.project_prefix,
        ),
        "project": MK8sProjectIdentity(
            project_key=project.key(),
            project_id=project.project_id,
            project_name=project.name,
            expected_provider_name=project.display_name(spec.project_prefix),
        ),
    }
