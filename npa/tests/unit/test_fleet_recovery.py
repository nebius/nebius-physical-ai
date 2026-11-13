"""Exercise fail-closed adoption of one interrupted Fleet apply."""

from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import replace
from pathlib import Path
import shutil
from subprocess import CompletedProcess

import pytest

from npa.cluster_backends import mk8s_execution
from npa.cluster_backends.mk8s_model import as_mk8s_desired
from npa.cluster_backends.mk8s_render import render_tfvars
from npa.fleet import lifecycle, recovery
from npa.fleet.release import default_mk8s_release
from npa.fleet.spec import (
    ClusterSpec,
    FleetSpec,
    NodePoolSpec,
    ObjectStorageSpec,
    ProjectSpec,
)


def _spec() -> FleetSpec:
    return FleetSpec(
        name="recovery-test",
        tenant_id="tenant-test",
        region="test-region",
        profile="operator-test",
        ssh_public_key="ssh-test",
        projects=[
            ProjectSpec(
                name="team",
                project_id="project-test",
                clusters=[
                    ClusterSpec(
                        name="recover",
                        cpu_nodes=NodePoolSpec(
                            count=1,
                            platform="cpu-d3",
                            preset="48vcpu-192gb",
                        ),
                    )
                ],
            )
        ],
    )


def _install(tmp_path: Path) -> tuple[FleetSpec, Path]:
    spec = _spec()
    cluster = spec.projects[0].clusters[0]
    install_dir = tmp_path / spec.name / "team" / cluster.name
    install_dir.mkdir(parents=True, mode=0o700)
    workdir = install_dir / "k8s-training"
    source = _repo_root() / "deploy/cluster/vendor/nebius-solutions-library"
    shutil.copytree(source / "k8s-training", workdir)
    shutil.copytree(source / "modules", install_dir / "modules")
    desired = as_mk8s_desired(cluster)
    (workdir / "terraform.tfvars").write_text(
        render_tfvars(desired, ssh_public_key="ssh-test", recipe_dir=workdir),
        encoding="utf-8",
    )
    _write_json(install_dir / ".npa-backend-owner.json", _owner(spec, cluster))
    _write_json(install_dir / ".npa-fleet-env.json", _sidecar(spec, cluster))
    _write_json(workdir / "terraform.tfstate", _state(spec, cluster))
    return spec, install_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owner(spec: FleetSpec, cluster: ClusterSpec) -> dict[str, str]:
    return {
        "fleet_name": spec.name,
        "project_key": "team",
        "cluster_name": cluster.name,
        "backend": "mk8s",
    }


def _sidecar(spec: FleetSpec, cluster: ClusterSpec) -> dict[str, object]:
    return {
        "backend": "mk8s",
        "tenant_id": spec.tenant_id,
        "project_id": "project-test",
        "region": spec.region,
        "subnet_id": "subnet-test",
        "cluster_name": cluster.name,
        "context": "fleet-recovery-test-team-recover",
        "profile": spec.profile,
        "status": "provisioning",
        "mk8s_recipe_release": default_mk8s_release().persistence(),
    }


def _state(spec: FleetSpec, cluster: ClusterSpec) -> dict[str, object]:
    return {
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_cluster",
                "instances": [
                    {
                        "attributes": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": cluster.name,
                        }
                    }
                ],
            },
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_node_group",
                "instances": [
                    {
                        "attributes": {
                            "template": {
                                "network_interfaces": [{"subnet_id": "subnet-test"}]
                            }
                        }
                    }
                ],
            },
        ]
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_provider(monkeypatch, calls: dict[str, object]) -> None:
    monkeypatch.setattr(lifecycle, "_resolve_ssh_public_key", lambda value: value)
    monkeypatch.setattr(lifecycle, "_require_bin", lambda value: value)
    monkeypatch.setattr(lifecycle, "_nebius_cli_env", lambda: {"PATH": ""})

    def resolve(**kwargs):
        calls["verify"] = kwargs
        return "cluster-test"

    def kubeconfig(_bin, _cluster_id, path, *_args):
        calls["kubeconfig"] = path
        path.write_text("apiVersion: v1\n", encoding="utf-8")

    def persist(**kwargs):
        calls["persist"] = kwargs

    monkeypatch.setattr(lifecycle, "_resolve_verified_unchanged_target", resolve)
    monkeypatch.setattr(lifecycle, "_write_kubeconfig", kubeconfig)
    monkeypatch.setattr(lifecycle, "_persist_npa_cluster_identity", persist)
    monkeypatch.setattr(
        lifecycle,
        "_get_project",
        lambda *_args: {
            "metadata": {"id": "project-test", "parent_id": "tenant-test"},
            "spec": {"region": "test-region"},
        },
    )


def _mixed_unmaterialized_resolver(monkeypatch, calls, malformed_part):
    valid = {
        "metadata": {
            "id": "cluster-test",
            "parent_id": "project-test",
            "name": "recover",
        },
        "status": {"state": "RUNNING"},
    }
    if malformed_part == "item":
        malformed = []
    else:
        malformed = {"metadata": {}, "status": {}}
        malformed[malformed_part] = []
    result = CompletedProcess([], 0, json.dumps({"items": [valid, malformed]}), "")
    monkeypatch.setattr(
        mk8s_execution, "_run_capture", lambda *_args, **_kwargs: result
    )

    def resolve(**_kwargs):
        calls["provider_resolution"] = True
        return mk8s_execution._unmaterialized_state_provider_candidate(
            project_id="project-test",
            cluster_name="recover",
            nebius_bin="nebius",
            profile="operator-test",
            env={},
        )

    return resolve


def _error_summary(status: str = "error", **overrides: object) -> dict[str, object]:
    return {
        "backend": "mk8s",
        "project_key": "team",
        "cluster_name": "recover",
        "region": "test-region",
        "status": status,
        "error": "historical pre-registration failure",
        **overrides,
    }


def _ordinary_error_summary(
    install_dir: Path, status: str = "error", **overrides: object
) -> dict[str, object]:
    """Build the bounded diagnostics emitted by the ordinary Fleet failure path."""

    diagnostics = {
        "project_id": "project-test",
        "install_dir": str(install_dir),
        "object_storage": {"enabled": False},
        "terraform_log": str(install_dir / "deploy.log"),
    }
    diagnostics.update(overrides)
    return _error_summary(status, **diagnostics)


def _write_summary(path: Path, entries: list[object]) -> bytes:
    path.write_text(json.dumps({"clusters": entries}), encoding="utf-8")
    return path.read_bytes()


def _provider_pool(
    name: str,
    *,
    platform: str,
    preset: str,
    count: int,
    reservation_id: str = "",
    state: str = "RUNNING",
) -> dict[str, object]:
    """Build one complete provider node-group response for retained recovery.

    Args:
        name: Exact node-group name.
        platform: Provider platform selector.
        preset: Provider compute preset.
        count: Required fixed node count.
        reservation_id: Strict capacity-block policy ID when required.
        state: Provider node-group lifecycle state.

    Returns:
        One complete provider node-group response.

    Raises:
        None.
    """

    template: dict[str, object] = {
        "resources": {"platform": platform, "preset": preset},
        "preemptible": False,
    }
    if reservation_id:
        template["reservation_policy"] = {
            "policy": "STRICT",
            "reservation_ids": [reservation_id],
        }
    return {
        "metadata": {
            "id": "nodegroup-" + name,
            "parent_id": "cluster-test",
            "name": name,
        },
        "spec": {"fixed_node_count": count, "template": template},
        "status": {"state": state},
    }


def test_recovers_one_verified_target_without_touching_terraform(tmp_path, monkeypatch):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_before = (workdir / "terraform.tfstate").read_bytes()
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("stale lock is retained", encoding="utf-8")
    lock_before = lock_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert result["status"] == "recovered-pending-validation"
    assert result["cluster_id"] == "cluster-test"
    assert result["mk8s_recipe_release"] == default_mk8s_release().persistence()
    assert (workdir / "terraform.tfstate").read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before
    assert calls["verify"]["project"].project_id == "project-test"
    assert calls["verify"]["cluster"].name == "recover"
    assert calls["persist"]["cluster_id"] == "cluster-test"
    sidecar = json.loads((install_dir / ".npa-fleet-env.json").read_text())
    assert sidecar["status"] == "recovered-pending-validation"
    state = json.loads((tmp_path / spec.name / "fleet-state.json").read_text())
    assert state["clusters"] == [result]
    assert state["unresolved"] == 1


def test_recovery_rejects_mismatched_owner_before_provider_or_registration(
    tmp_path, monkeypatch
):
    spec, install_dir = _install(tmp_path)
    marker = install_dir / ".npa-backend-owner.json"
    _write_json(marker, {**json.loads(marker.read_text()), "backend": "soperator"})
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(recovery.FleetRecoveryError, match="owner marker"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()


def test_recovery_rejects_terminal_provider_mismatch_before_registration(
    tmp_path, monkeypatch
):
    spec, install_dir = _install(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_resolve_verified_unchanged_target", lambda **_: "")

    with pytest.raises(recovery.FleetRecoveryError, match="could not be verified"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert "kubeconfig" not in calls
    assert not (install_dir / "kubeconfig").exists()


def test_recovery_resolves_one_unmaterialized_state_identity(tmp_path, monkeypatch):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = [{}]
    _write_json(state_path, state)
    state_before = state_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert result["cluster_id"] == "cluster-test"
    assert state_path.read_bytes() == state_before
    assert calls["verify"]["cluster"].name == "recover"
    sidecar = json.loads((install_dir / ".npa-fleet-env.json").read_text())
    assert sidecar["cluster_id"] == "cluster-test"


def test_recovery_resolves_exact_empty_instances_only_after_verified_provider(
    tmp_path, monkeypatch
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = []
    _write_json(state_path, state)
    state_before = state_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert result["cluster_id"] == "cluster-test"
    assert state_path.read_bytes() == state_before
    assert calls["verify"]["cluster"].name == "recover"
    assert calls["persist"]["cluster_id"] == "cluster-test"


@pytest.mark.parametrize("status", ["error", "incomplete"])
def test_recovery_supersedes_identity_free_error_summary_after_verification(
    tmp_path, monkeypatch, status
):
    spec, install_dir = _install(tmp_path)
    summary_path = tmp_path / spec.name / "fleet-state.json"
    _write_summary(summary_path, [_error_summary(status)])
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert result["status"] == "recovered-pending-validation"
    assert "verify" in calls
    assert calls["persist"]["cluster_id"] == "cluster-test"
    assert json.loads(summary_path.read_text())["clusters"] == [result]
    assert (install_dir / "kubeconfig").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [_error_summary(cluster_id="cluster-test")],
        [_error_summary(kube_context="fleet-recovery-test-team-recover")],
        [_error_summary(kubeconfig="/owner-private/kubeconfig")],
        [_error_summary(mk8s_recipe_release={"id": "release-test"})],
        [_error_summary(project_id="other-project")],
        [_error_summary(backend="soperator")],
        [_error_summary(region="other-region")],
        [
            {
                "project_key": "team",
                "cluster_name": "recover",
                "region": "test-region",
                "status": "error",
                "error": "historical pre-registration failure",
            }
        ],
        [_error_summary(region="")],
        [_error_summary(status="provisioning")],
        [_error_summary(status=[])],
        [_error_summary(error=[])],
        [_error_summary(), _error_summary()],
    ],
    ids=[
        "cluster-id",
        "context",
        "kubeconfig",
        "release",
        "wrong-project-id",
        "wrong-backend",
        "wrong-region",
        "missing-backend",
        "missing-region",
        "non-error-status",
        "malformed-status",
        "malformed-error",
        "ambiguous-error-records",
    ],
)
def test_recovery_rejects_identity_bearing_or_ambiguous_summary_without_mutation(
    tmp_path, monkeypatch, entries
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state_before = state_path.read_bytes()
    backup_path = workdir / "terraform.tfstate.backup"
    backup_path.write_bytes(state_before)
    backup_before = backup_path.read_bytes()
    sidecar_path = install_dir / ".npa-fleet-env.json"
    sidecar_before = sidecar_path.read_bytes()
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("retained lock", encoding="utf-8")
    lock_before = lock_path.read_bytes()
    summary_path = tmp_path / spec.name / "fleet-state.json"
    summary_before = _write_summary(summary_path, entries)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(recovery.FleetRecoveryError, match="already registered"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}
    assert state_path.read_bytes() == state_before
    assert backup_path.read_bytes() == backup_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert lock_path.read_bytes() == lock_before
    assert not (install_dir / "kubeconfig").exists()
    assert summary_path.read_bytes() == summary_before


@pytest.mark.parametrize("status", ["error", "incomplete"])
def test_normal_lifecycle_reconciles_exact_ordinary_error_summary_without_persistence(
    tmp_path, monkeypatch, status
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    backup_path.write_bytes(state_path.read_bytes())
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("retained lock", encoding="utf-8")
    summary_path = tmp_path / spec.name / "fleet-state.json"
    _write_summary(summary_path, [_ordinary_error_summary(install_dir, status)])
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "backup": backup_path,
        "lock": lock_path,
        "summary": summary_path,
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.reconcile_retained_fleet_target(spec, work_root=tmp_path)

    assert result["status"] == "reconciled-pending-validation"
    assert set(calls) == {"verify"}
    assert not (install_dir / "kubeconfig").exists()
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


def test_normal_lifecycle_reconciles_matching_enabled_object_storage_summary(
    tmp_path, monkeypatch
):
    spec, install_dir = _install(tmp_path)
    spec.projects[0].object_storage = ObjectStorageSpec(
        enabled=True, storage_class="enhanced", size_gibibytes=2048
    )
    summary_path = tmp_path / spec.name / "fleet-state.json"
    _write_summary(
        summary_path,
        [
            _ordinary_error_summary(
                install_dir,
                object_storage={
                    "enabled": True,
                    "status": "ready",
                    "storage_class": "enhanced",
                    "size_gibibytes": 2048,
                    "writable": True,
                    "retained_on_destroy": True,
                },
            )
        ],
    )
    summary_before = summary_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    result = recovery.reconcile_retained_fleet_target(spec, work_root=tmp_path)

    assert result["status"] == "reconciled-pending-validation"
    assert set(calls) == {"verify"}
    assert summary_path.read_bytes() == summary_before
    assert not (install_dir / "kubeconfig").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "other-project"),
        ("project_id", []),
        ("install_dir", "/outside-selected-installation"),
        ("install_dir", []),
        ("install_dir", "\0"),
        ("terraform_log", "/outside-selected-installation/deploy.log"),
        ("terraform_log", []),
        ("terraform_log", "\0"),
        ("object_storage", {"enabled": True}),
        ("object_storage", []),
        ("unexpected_diagnostic", "unexpected"),
        ("cluster_id", "identity-bearing"),
    ],
    ids=[
        "wrong-project-id",
        "malformed-project-id",
        "wrong-install-dir",
        "malformed-install-dir",
        "embedded-nul-install-dir",
        "log-outside-install-dir",
        "malformed-terraform-log",
        "embedded-nul-terraform-log",
        "wrong-object-storage",
        "malformed-object-storage",
        "unknown-extra-field",
        "durable-cluster-identity",
    ],
)
def test_normal_lifecycle_rejects_unsafe_ordinary_error_diagnostics_without_mutation(
    tmp_path, monkeypatch, field, value
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    backup_path.write_bytes(state_path.read_bytes())
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("retained lock", encoding="utf-8")
    summary_path = tmp_path / spec.name / "fleet-state.json"
    entry = _ordinary_error_summary(install_dir)
    entry[field] = value
    _write_summary(summary_path, [entry])
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "backup": backup_path,
        "lock": lock_path,
        "summary": summary_path,
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(recovery.FleetRecoveryError, match="already registered"):
        recovery.reconcile_retained_fleet_target(spec, work_root=tmp_path)

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"project_key": "other", "cluster_name": "other", "status": "error"}],
    ],
    ids=["empty-summary", "unrelated-summary"],
)
def test_recovery_preserves_nonmatching_summary_when_provider_refuses(
    tmp_path, monkeypatch, entries
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state_before = state_path.read_bytes()
    sidecar_path = install_dir / ".npa-fleet-env.json"
    sidecar_before = sidecar_path.read_bytes()
    summary_path = tmp_path / spec.name / "fleet-state.json"
    summary_before = _write_summary(summary_path, entries)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    def reject_provider(**kwargs):
        calls["verify"] = kwargs
        return ""

    monkeypatch.setattr(
        lifecycle, "_resolve_verified_unchanged_target", reject_provider
    )

    with pytest.raises(recovery.FleetRecoveryError, match="could not be verified"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert set(calls) == {"verify"}
    assert state_path.read_bytes() == state_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert not (install_dir / "kubeconfig").exists()
    assert summary_path.read_bytes() == summary_before


@pytest.mark.parametrize("status", ["error", "incomplete"])
def test_recovery_preserves_admitted_error_summary_when_provider_refuses(
    tmp_path, monkeypatch, status
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    owner_path = install_dir / ".npa-backend-owner.json"
    owner_before = owner_path.read_bytes()
    state_path = workdir / "terraform.tfstate"
    state_before = state_path.read_bytes()
    backup_path = workdir / "terraform.tfstate.backup"
    backup_path.write_bytes(state_before)
    backup_before = backup_path.read_bytes()
    sidecar_path = install_dir / ".npa-fleet-env.json"
    sidecar_before = sidecar_path.read_bytes()
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("retained lock", encoding="utf-8")
    lock_before = lock_path.read_bytes()
    summary_path = tmp_path / spec.name / "fleet-state.json"
    summary_before = _write_summary(
        summary_path, [_ordinary_error_summary(install_dir, status)]
    )
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    def reject_provider(**kwargs):
        calls["verify"] = kwargs
        return ""

    monkeypatch.setattr(
        lifecycle, "_resolve_verified_unchanged_target", reject_provider
    )

    with pytest.raises(recovery.FleetRecoveryError, match="could not be verified"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert set(calls) == {"verify"}
    assert owner_path.read_bytes() == owner_before
    assert state_path.read_bytes() == state_before
    assert backup_path.read_bytes() == backup_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert lock_path.read_bytes() == lock_before
    assert not (install_dir / "kubeconfig").exists()
    assert summary_path.read_bytes() == summary_before


@pytest.mark.parametrize("malformed_part", ["item", "metadata", "status"])
@pytest.mark.parametrize(
    "fleet_summary",
    [
        b'{\n  "clusters": [],\n  "unresolved": 0\n}\n',
        b'{"clusters":[{"project_key":"other","cluster_name":"other"}],"unresolved":0}\n',
    ],
    ids=["empty-summary", "unrelated-summary"],
)
def test_recovery_rejects_empty_state_mixed_malformed_inventory_before_persistence(
    tmp_path, monkeypatch, malformed_part, fleet_summary
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = []
    _write_json(state_path, state)
    state_before = state_path.read_bytes()
    sidecar_path = install_dir / ".npa-fleet-env.json"
    sidecar_before = sidecar_path.read_bytes()
    lock_path = workdir / ".terraform.tfstate.lock.info"
    lock_path.write_text("retained lock", encoding="utf-8")
    lock_before = lock_path.read_bytes()
    summary_path = tmp_path / spec.name / "fleet-state.json"
    summary_path.write_bytes(fleet_summary)
    summary_before = summary_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(
        lifecycle,
        "_resolve_verified_unchanged_target",
        _mixed_unmaterialized_resolver(monkeypatch, calls, malformed_part),
    )

    with pytest.raises(recovery.FleetRecoveryError, match="could not be verified"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {"provider_resolution": True}
    assert state_path.read_bytes() == state_before
    assert sidecar_path.read_bytes() == sidecar_before
    assert lock_path.read_bytes() == lock_before
    assert not (install_dir / "kubeconfig").exists()
    assert summary_path.read_bytes() == summary_before


@pytest.mark.parametrize(
    "instances", [[{}], []], ids=["missing-attributes", "empty-list"]
)
def test_recovery_rejects_unmaterialized_state_without_verified_identity(
    tmp_path, monkeypatch, instances
):
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = instances
    _write_json(state_path, state)
    state_before = state_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_resolve_verified_unchanged_target", lambda **_: "")

    with pytest.raises(recovery.FleetRecoveryError, match="could not be verified"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert state_path.read_bytes() == state_before
    assert "kubeconfig" not in calls
    assert "persist" not in calls


@pytest.mark.parametrize(
    "instances", [[{}], []], ids=["missing-attributes", "empty-list"]
)
def test_recovery_rejects_live_owner_before_unmaterialized_provider_resolution(
    tmp_path, monkeypatch, instances
):
    spec, install_dir = _install(tmp_path)
    state_path = install_dir / "k8s-training" / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = instances
    _write_json(state_path, state)
    state_before = state_path.read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    def live_owner(_workdir: Path) -> None:
        raise recovery.FleetRecoveryError(
            "a live Terraform apply still owns this Fleet target"
        )

    monkeypatch.setattr(
        recovery,
        "_assert_no_live_apply_owner",
        live_owner,
    )

    with pytest.raises(recovery.FleetRecoveryError, match="live Terraform apply"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}
    assert state_path.read_bytes() == state_before
    assert not (install_dir / "kubeconfig").exists()


def test_recovery_rechecks_apply_owner_before_registration(tmp_path, monkeypatch):
    spec, install_dir = _install(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    checks: list[Path] = []

    def no_live_apply_owner(workdir: Path) -> None:
        checks.append(workdir)
        if len(checks) == 2:
            raise recovery.FleetRecoveryError("live Terraform apply owns target")

    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", no_live_apply_owner)

    with pytest.raises(recovery.FleetRecoveryError, match="live Terraform apply"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert len(checks) == 2
    assert "verify" in calls
    assert "kubeconfig" not in calls
    assert not (install_dir / "kubeconfig").exists()


def test_recovery_rejects_tampered_release_and_state_subnet(tmp_path, monkeypatch):
    spec, install_dir = _install(tmp_path)
    sidecar_path = install_dir / ".npa-fleet-env.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["mk8s_recipe_release"]["bundle_sha256"] = "0" * 64
    _write_json(sidecar_path, sidecar)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(recovery.FleetRecoveryError, match="immutable release"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    sidecar["mk8s_recipe_release"] = default_mk8s_release().persistence()
    _write_json(sidecar_path, sidecar)
    state_path = install_dir / "k8s-training" / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][1]["instances"][0]["attributes"]["template"][
        "network_interfaces"
    ][0]["subnet_id"] = "wrong-subnet"
    _write_json(state_path, state)

    with pytest.raises(recovery.FleetRecoveryError, match="subnet differs"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()


def test_recovery_rejects_non_private_target_before_provider(tmp_path, monkeypatch):
    spec, install_dir = _install(tmp_path)
    install_dir.chmod(0o755)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(recovery.FleetRecoveryError, match="not owner-private"):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}


def test_recovery_resolves_undeclared_tenant_and_region(tmp_path, monkeypatch):
    spec, _install_dir = _install(tmp_path)
    spec.tenant_id = ""
    spec.region = ""
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(
        lifecycle,
        "_resolve_tenant_id",
        lambda _binary, _explicit, _profile: "tenant-test",
    )
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _explicit: "test-region")

    result = recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert result["status"] == "recovered-pending-validation"
    assert calls["verify"]["tenant_id"] == "tenant-test"
    assert calls["verify"]["region"] == "test-region"


def test_live_terraform_apply_owner_is_rejected_from_hermetic_proc_view(tmp_path):
    workdir = tmp_path / "k8s-training"
    workdir.mkdir()
    process = tmp_path / "proc" / "123"
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(workdir)
    (process / "cmdline").write_bytes(b"/usr/bin/terraform\0apply\0")

    with pytest.raises(recovery.FleetRecoveryError, match="live Terraform apply"):
        recovery._assert_no_live_apply_owner(workdir, proc_root=tmp_path / "proc")


def test_live_terraform_provider_child_is_rejected_from_hermetic_proc_view(tmp_path):
    workdir = tmp_path / "k8s-training"
    workdir.mkdir()
    process = tmp_path / "proc" / "124"
    process.mkdir(parents=True)
    (process / "cwd").symlink_to(tmp_path)
    provider = workdir / ".terraform" / "providers" / "provider-nebius"
    (process / "cmdline").write_bytes(str(provider).encode() + b"\0")

    with pytest.raises(recovery.FleetRecoveryError, match="live Terraform apply"):
        recovery._assert_no_live_apply_owner(workdir, proc_root=tmp_path / "proc")


def test_live_terraform_state_lock_holder_is_rejected_from_hermetic_proc_view(
    tmp_path,
):
    workdir = tmp_path / "k8s-training"
    workdir.mkdir()
    lock = workdir / ".terraform.tfstate.lock.info"
    lock.write_text("retained", encoding="utf-8")
    process = tmp_path / "proc" / "125"
    (process / "fd").mkdir(parents=True)
    (process / "cwd").symlink_to(tmp_path)
    (process / "cmdline").write_bytes(b"/usr/bin/helper\0")
    (process / "fd" / "4").symlink_to(lock)

    with pytest.raises(recovery.FleetRecoveryError, match="Terraform state lock"):
        recovery._assert_no_live_apply_owner(workdir, proc_root=tmp_path / "proc")


def _unmaterialized_retained_target(tmp_path: Path) -> tuple[FleetSpec, Path, dict]:
    spec, install_dir = _install(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    state = json.loads(state_path.read_text())
    state["resources"][0]["instances"] = []
    _write_json(state_path, state)
    (workdir / ".terraform.tfstate.lock.info").write_text(
        "retained lock", encoding="utf-8"
    )
    summary_path = tmp_path / spec.name / "fleet-state.json"
    _write_summary(summary_path, [_error_summary()])
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": summary_path,
    }
    return spec, install_dir, {name: path.read_bytes() for name, path in paths.items()}


def _backup_root_retained_target(tmp_path: Path) -> tuple[FleetSpec, Path, dict]:
    """Build the exact retained current-state/backup-root compatibility shape."""

    spec, install_dir, _before = _unmaterialized_retained_target(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    state = json.loads(state_path.read_text())
    state["lineage"] = "retained-backup-root-lineage"
    state["serial"] = 2
    backup = deepcopy(state)
    backup["serial"] = 1
    root = backup["resources"][0]
    state["resources"] = state["resources"][1:]
    assert root["type"] == "nebius_mk8s_v1_cluster"
    assert root["instances"] == []
    _write_json(state_path, state)
    _write_json(backup_path, backup)
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "backup": backup_path,
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    return spec, install_dir, {name: path.read_bytes() for name, path in paths.items()}


def _zero_node_subnet_retained_target(tmp_path: Path) -> tuple[FleetSpec, Path, dict]:
    """Build the exact backup-root shape with no Terraform node-group resource."""

    spec, install_dir, _before = _backup_root_retained_target(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    state = json.loads(state_path.read_text())
    backup = json.loads(backup_path.read_text())
    for payload in (state, backup):
        payload["resources"] = [
            resource
            for resource in payload["resources"]
            if resource["type"] != "nebius_mk8s_v1_node_group"
        ]
    _write_json(state_path, state)
    _write_json(backup_path, backup)
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "backup": backup_path,
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    return spec, install_dir, {name: path.read_bytes() for name, path in paths.items()}


def test_generic_recovery_rejects_backup_root_compatibility_shape(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _backup_root_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)

    with pytest.raises(
        recovery.FleetRecoveryError,
        match="Terraform state does not contain one exact mk8s cluster",
    ):
        recovery.recover_interrupted_fleet_apply(spec, work_root=tmp_path)

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": install_dir / "k8s-training" / "terraform.tfstate",
        "backup": install_dir / "k8s-training" / "terraform.tfstate.backup",
        "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


@pytest.mark.parametrize(
    "invalid_shape",
    [
        "materialized-root",
        "deposed-root",
        "duplicate-root",
        "malformed-root",
        "module-scoped-root",
        "common-module-cluster",
        "current-malformed-root",
        "lineage-divergent",
        "serial-regressing",
        "common-resource-divergent",
        "current-only-resource",
        "extra-backup-resource",
    ],
)
def test_normal_lifecycle_rejects_invalid_backup_root_before_provider_or_persistence(
    tmp_path, monkeypatch, invalid_shape
):
    spec, install_dir, _before = _backup_root_retained_target(tmp_path)
    workdir = install_dir / "k8s-training"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    state = json.loads(state_path.read_text())
    backup = json.loads(backup_path.read_text())
    root = backup["resources"][0]
    if invalid_shape == "materialized-root":
        root["instances"] = [{"attributes": {"id": "cluster-test"}}]
    elif invalid_shape == "deposed-root":
        root["instances"] = [{"deposed": "retained"}]
    elif invalid_shape == "duplicate-root":
        backup["resources"].append(deepcopy(root))
    elif invalid_shape == "malformed-root":
        root["instances"] = {"not": "a-list"}
    elif invalid_shape == "module-scoped-root":
        root["module"] = "module.retained"
    elif invalid_shape == "common-module-cluster":
        common_module_cluster = {
            "mode": "managed",
            "type": "nebius_mk8s_v1_cluster",
            "name": "retained-module-cluster",
            "module": "module.retained",
            "instances": [],
        }
        state["resources"].append(common_module_cluster)
        backup["resources"].append(deepcopy(common_module_cluster))
    elif invalid_shape == "current-malformed-root":
        state["resources"].append(
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_cluster",
                "instances": {"not": "a-list"},
            }
        )
    elif invalid_shape == "lineage-divergent":
        backup["lineage"] = "different-lineage"
    elif invalid_shape == "serial-regressing":
        backup["serial"] = state["serial"] + 1
    elif invalid_shape == "common-resource-divergent":
        backup["resources"][1]["name"] = "changed-common-resource"
    elif invalid_shape == "current-only-resource":
        state["resources"].append(
            {
                "mode": "managed",
                "type": "nebius_compute_v1_filesystem",
                "instances": [],
            }
        )
    elif invalid_shape == "extra-backup-resource":
        backup["resources"].append(
            {
                "mode": "managed",
                "type": "nebius_compute_v1_filesystem",
                "instances": [],
            }
        )
    else:
        raise AssertionError(f"unhandled invalid shape: {invalid_shape}")
    _write_json(state_path, state)
    _write_json(backup_path, backup)
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": state_path,
        "backup": backup_path,
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    monkeypatch.setattr(
        mk8s_execution,
        "_run_capture",
        lambda *_args, **_kwargs: pytest.fail("invalid backup shape called provider"),
    )

    with pytest.raises(
        RuntimeError, match="failed exact normal-lifecycle reconciliation"
    ):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


def test_normal_lifecycle_resumes_only_missing_retained_cpu_pool(tmp_path, monkeypatch):
    spec, install_dir, before = _unmaterialized_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    commands: list[list[str]] = []
    sleeps: list[float] = []
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    pool_lists = 0

    def provider(command, **_kwargs):
        nonlocal pool_lists
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            pool_lists += 1
            payload = (
                {"items": []}
                if pool_lists < 3
                else {
                    "items": [
                        _provider_pool(
                            "recover-ng-cpu",
                            platform="cpu-d3",
                            preset="48vcpu-192gb",
                            count=1,
                            state=("PROVISIONING" if pool_lists == 3 else "RUNNING"),
                        )
                    ]
                }
            )
        elif "node-group" in command and "create" in command:
            payload = {
                "metadata": {
                    "id": "nodegroup-test",
                    "parent_id": "cluster-test",
                    "name": command[command.index("--name") + 1],
                }
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    result = lifecycle.deploy_fleet(
        spec,
        work_root=tmp_path,
        k8s_training_dir=install_dir,
        preflight=False,
    )

    assert result["clusters"][0]["status"] == "recovered-pending-validation"
    assert result["clusters"][0]["cluster_id"] == "cluster-test"
    assert result["unresolved"] == 1
    assert "kubeconfig" in calls
    assert "persist" in calls
    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert len(creates) == 1
    assert creates[0][creates[0].index("--name") + 1] == "recover-ng-cpu"
    assert pool_lists >= 4
    assert sleeps
    assert all(seconds == 30.0 for seconds in sleeps)
    assert not any("terraform" in command for command in commands)
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": install_dir / "k8s-training" / "terraform.tfstate",
            "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        if name in {"sidecar", "summary"}:
            assert path.read_bytes() != expected
        else:
            assert path.read_bytes() == expected


def test_normal_lifecycle_refuses_registration_when_terminal_pool_never_converges(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _unmaterialized_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monotonic_values = iter((0.0, 0.0, 0.0, 60.0))
    sleeps: list[float] = []
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    commands: list[list[str]] = []
    pool_lists = 0

    def provider(command, **_kwargs):
        nonlocal pool_lists
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            pool_lists += 1
            payload = (
                {"items": []}
                if pool_lists < 3
                else {
                    "items": [
                        _provider_pool(
                            "recover-ng-cpu",
                            platform="cpu-d3",
                            preset="48vcpu-192gb",
                            count=1,
                            state="PROVISIONING",
                        )
                    ]
                }
            )
        elif "node-group" in command and "create" in command:
            payload = {
                "metadata": {
                    "id": "nodegroup-test",
                    "parent_id": "cluster-test",
                    "name": command[command.index("--name") + 1],
                }
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(RuntimeError, match="did not reach terminal state"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
            timeout_minutes=1,
        )

    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert [command[command.index("--name") + 1] for command in creates] == [
        "recover-ng-cpu"
    ]
    assert pool_lists >= 4
    assert sleeps
    assert all(seconds == 30.0 for seconds in sleeps)
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    assert not any("terraform" in command for command in commands)
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": install_dir / "k8s-training" / "terraform.tfstate",
            "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        assert path.read_bytes() == expected


def test_normal_lifecycle_refuses_registration_when_postcreate_inventory_is_not_terminal(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _zero_node_subnet_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monotonic_values = iter((0.0, 0.0, 60.0))
    sleeps: list[float] = []
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    commands: list[list[str]] = []
    postcreate_inventory_reads = 0
    created = False

    def provider(command, **_kwargs):
        nonlocal created, postcreate_inventory_reads
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            if created:
                postcreate_inventory_reads += 1
            payload = {"items": []}
        elif "node-group" in command and "create" in command:
            created = True
            payload = {
                "metadata": {
                    "id": "nodegroup-test",
                    "parent_id": "cluster-test",
                    "name": command[command.index("--name") + 1],
                }
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(RuntimeError, match="did not reach terminal state"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
            timeout_minutes=1,
        )

    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert [command[command.index("--name") + 1] for command in creates] == [
        "recover-ng-cpu"
    ]
    assert postcreate_inventory_reads >= 2
    assert sleeps == [30.0]
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    assert not any("terraform" in command for command in commands)
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": install_dir / "k8s-training" / "terraform.tfstate",
            "backup": install_dir / "k8s-training" / "terraform.tfstate.backup",
            "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        assert path.read_bytes() == expected


def test_retained_pool_quota_demand_counts_only_verified_missing_pools() -> None:
    from npa.cluster_backends.quotas import required_quotas, required_reservations

    cluster = ClusterSpec(
        name="recover",
        cpu_nodes=NodePoolSpec(
            count=2,
            platform="cpu-d3",
            preset="48vcpu-192gb",
        ),
        gpu_nodes=NodePoolSpec(
            count=2,
            platform="gpu-rtx6000",
            preset="1gpu-24vcpu-218gb",
            capacity_block_group="capacityblockgroup-test",
        ),
    )

    demand = lifecycle._RetainedPoolQuotaDemand(cluster, ["recover-ng-gpu-1"])

    assert demand.cpu_count() == 0
    assert demand.gpu_count() == 1
    assert required_quotas([demand])["compute.instance.count"] == 1
    requirements = required_reservations([demand], "test-region")
    assert requirements["capacityblockgroup-test"].required_gpus == 1

    with pytest.raises(ValueError, match="undeclared missing node group"):
        lifecycle._RetainedPoolQuotaDemand(cluster, ["unexpected-node-group"])


def test_normal_lifecycle_resumes_zero_subnet_cpu_and_strict_gpu_target(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _zero_node_subnet_retained_target(tmp_path)
    cluster = spec.projects[0].clusters[0]
    cluster.enable_filestore = True
    cluster.filesystem_csi_chart_repository = "oci://charts.example.invalid/nebius"
    cluster.gpu_nodes = NodePoolSpec(
        count=1,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-test",
    )
    bound_cluster = replace(
        cluster,
        gpu_nodes=replace(
            cluster.gpu_nodes,
            compute_reservation_id="capacityblockgroup-test",
        ),
    )
    workdir = install_dir / "k8s-training"
    state = json.loads((workdir / "terraform.tfstate").read_text())
    backup_path = workdir / "terraform.tfstate.backup"
    backup = json.loads(backup_path.read_text())
    filesystem = {
        "mode": "managed",
        "type": "nebius_compute_v1_filesystem",
        "name": "shared-filesystem",
        "instances": [{"attributes": {"id": "filesystem-test"}}],
    }
    state["resources"].append(filesystem)
    backup["resources"].append(deepcopy(filesystem))
    _write_json(workdir / "terraform.tfstate", state)
    _write_json(backup_path, backup)
    tfvars = render_tfvars(
        as_mk8s_desired(bound_cluster), ssh_public_key="ssh-test", recipe_dir=workdir
    )
    (workdir / "terraform.tfvars").write_text(
        tfvars,
        encoding="utf-8",
    )
    before["state"] = (workdir / "terraform.tfstate").read_bytes()
    before["backup"] = backup_path.read_bytes()
    before["lock"] = (workdir / ".terraform.tfstate.lock.info").read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_bind_strict_compute_reservations",
        lambda clusters, **_kwargs: {
            key: (bound_cluster if key == ("team", "recover") else value)
            for key, value in clusters.items()
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    monotonic_values = iter((0.0, 0.0))
    sleeps: list[float] = []
    monkeypatch.setattr(recovery.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    commands: list[list[str]] = []
    pool_lists = 0
    postcreate_inventory_reads = 0
    created = False

    def provider(command, **_kwargs):
        nonlocal created, pool_lists, postcreate_inventory_reads
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            pool_lists += 1
            if created:
                postcreate_inventory_reads += 1
            payload = (
                {}
                if not created or postcreate_inventory_reads == 1
                else {
                    "items": [
                        _provider_pool(
                            "recover-ng-cpu",
                            platform="cpu-d3",
                            preset="48vcpu-192gb",
                            count=1,
                        ),
                        _provider_pool(
                            "recover-ng-gpu-0",
                            platform="gpu-rtx6000",
                            preset="1gpu-24vcpu-218gb",
                            count=1,
                            reservation_id="capacityblockgroup-test",
                        ),
                    ]
                }
            )
        elif "node-group" in command and "create" in command:
            created = True
            payload = {
                "metadata": {
                    "id": "nodegroup-" + command[command.index("--name") + 1],
                    "parent_id": "cluster-test",
                    "name": command[command.index("--name") + 1],
                }
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    result = lifecycle.deploy_fleet(
        spec,
        work_root=tmp_path,
        k8s_training_dir=install_dir,
        preflight=False,
    )

    assert result["clusters"][0]["status"] == "recovered-pending-validation"
    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert [command[command.index("--name") + 1] for command in creates] == [
        "recover-ng-cpu",
        "recover-ng-gpu-0",
    ]
    assert pool_lists == 4
    assert postcreate_inventory_reads == 2
    assert sleeps == [30.0]
    gpu_command = creates[1]
    assert (
        gpu_command[gpu_command.index("--template-reservation-policy-policy") + 1]
        == "strict"
    )
    assert (
        gpu_command[
            gpu_command.index("--template-reservation-policy-reservation-ids") + 1
        ]
        == "capacityblockgroup-test"
    )
    filesystems = json.loads(
        gpu_command[gpu_command.index("--template-filesystems") + 1]
    )
    assert filesystems == [
        {
            "attach_mode": "read_write",
            "mount_tag": "data",
            "existing_filesystem": {"id": "filesystem-test"},
        }
    ]
    assert any("capacityblockgroup-test" in value for value in gpu_command)
    assert not any("cluster" in command and "create" in command for command in commands)
    assert not any("terraform" in command for command in commands)
    assert (workdir / "terraform.tfstate").read_bytes() == before["state"]
    assert backup_path.read_bytes() == before["backup"]
    assert (workdir / ".terraform.tfstate.lock.info").read_bytes() == before["lock"]


def test_normal_lifecycle_empty_node_group_mapping_refusal_preserves_retained_bytes(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _zero_node_subnet_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    commands: list[list[str]] = []

    def provider(command, **_kwargs):
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
            return CompletedProcess(command, 0, json.dumps(payload), "")
        if "node-group" in command and "list" in command:
            return CompletedProcess(command, 0, "{}", "")
        if "node-group" in command and "create" in command:
            return CompletedProcess(command, 1, "", "provider refused")
        pytest.fail(f"unexpected provider command: {command}")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(RuntimeError, match="could not create retained node group"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert [command[command.index("--name") + 1] for command in creates] == [
        "recover-ng-cpu"
    ]
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    assert not any("terraform" in command for command in commands)
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": install_dir / "k8s-training" / "terraform.tfstate",
        "backup": install_dir / "k8s-training" / "terraform.tfstate.backup",
        "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"metadata": {}},
        {"items": [], "next_page_token": "next"},
        {"items": [{"metadata": []}]},
        {
            "items": [
                _provider_pool(
                    "unexpected-pool",
                    platform="cpu-d3",
                    preset="48vcpu-192gb",
                    count=1,
                )
            ]
        },
    ],
    ids=["absent", "missing-items", "paged", "malformed-item", "unexpected-pool"],
)
def test_normal_lifecycle_rejects_noncanonical_empty_node_group_inventory(
    tmp_path, monkeypatch, payload
):
    spec, install_dir, before = _zero_node_subnet_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    commands: list[list[str]] = []

    def provider(command, **_kwargs):
        commands.append(command)
        if "cluster" in command and "list" in command:
            response = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
            return CompletedProcess(command, 0, json.dumps(response), "")
        if "node-group" in command and "list" in command:
            return CompletedProcess(command, 0, json.dumps(payload), "")
        pytest.fail(f"unexpected provider command: {command}")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(
        RuntimeError, match="failed exact normal-lifecycle reconciliation"
    ):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert not any(
        "node-group" in command and "create" in command for command in commands
    )
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    assert not any("terraform" in command for command in commands)
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": install_dir / "k8s-training" / "terraform.tfstate",
        "backup": install_dir / "k8s-training" / "terraform.tfstate.backup",
        "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


def test_normal_lifecycle_refuses_zero_subnet_when_provider_has_a_pool(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _zero_node_subnet_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    commands: list[list[str]] = []

    def provider(command, **_kwargs):
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            payload = {
                "items": [
                    _provider_pool(
                        "recover-ng-cpu",
                        platform="cpu-d3",
                        preset="48vcpu-192gb",
                        count=1,
                    )
                ]
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(
        RuntimeError, match="requires zero retained provider node groups"
    ):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert not any(
        "node-group" in command and "create" in command for command in commands
    )
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": install_dir / "k8s-training" / "terraform.tfstate",
            "backup": install_dir / "k8s-training" / "terraform.tfstate.backup",
            "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        assert path.read_bytes() == expected


@pytest.mark.parametrize("subnet_shape", ["missing-sidecar", "mismatching", "multiple"])
def test_normal_lifecycle_refuses_invalid_state_subnet_before_provider_or_persistence(
    tmp_path, monkeypatch, subnet_shape
):
    spec, install_dir, _before = _unmaterialized_retained_target(tmp_path)
    workdir = install_dir / "k8s-training"
    sidecar_path = install_dir / ".npa-fleet-env.json"
    state_path = workdir / "terraform.tfstate"
    backup_path = workdir / "terraform.tfstate.backup"
    sidecar = json.loads(sidecar_path.read_text())
    state = json.loads(state_path.read_text())
    _write_json(backup_path, deepcopy(state))
    if subnet_shape == "missing-sidecar":
        sidecar["subnet_id"] = ""
        _write_json(sidecar_path, sidecar)
    elif subnet_shape == "mismatching":
        state["resources"][1]["instances"][0]["attributes"]["template"][
            "network_interfaces"
        ][0]["subnet_id"] = "other-subnet"
        _write_json(state_path, state)
    elif subnet_shape == "multiple":
        duplicate = deepcopy(state["resources"][1]["instances"][0])
        duplicate["attributes"]["template"]["network_interfaces"][0]["subnet_id"] = (
            "other-subnet"
        )
        state["resources"][1]["instances"].append(duplicate)
        _write_json(state_path, state)
    else:
        raise AssertionError(f"unhandled subnet shape: {subnet_shape}")
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": sidecar_path,
        "state": state_path,
        "backup": backup_path,
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("invalid subnet must not enter Terraform apply"),
    )
    monkeypatch.setattr(
        mk8s_execution,
        "_run_capture",
        lambda *_args, **_kwargs: pytest.fail("invalid subnet called provider"),
    )

    with pytest.raises(RuntimeError, match="failed exact normal-lifecycle"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert calls == {}
    assert not (install_dir / "kubeconfig").exists()
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


def test_retained_resume_rejects_nonmatching_saved_selector_before_provider(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _unmaterialized_retained_target(tmp_path)
    cluster = spec.projects[0].clusters[0]
    cluster.gpu_nodes = NodePoolSpec(
        count=1,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-test",
    )
    bound_cluster = replace(
        cluster,
        gpu_nodes=replace(
            cluster.gpu_nodes,
            compute_reservation_id="capacityblockgroup-test",
        ),
    )
    workdir = install_dir / "k8s-training"
    (workdir / "terraform.tfvars").write_text(
        render_tfvars(
            as_mk8s_desired(bound_cluster),
            ssh_public_key="ssh-test",
            recipe_dir=workdir,
        ).replace("capacityblockgroup-test", "other-capacity-block"),
        encoding="utf-8",
    )
    paths = {
        "owner": install_dir / ".npa-backend-owner.json",
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": workdir / "terraform.tfstate",
        "lock": workdir / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    bytes_before = {name: path.read_bytes() for name, path in paths.items()}
    _prepare_provider(monkeypatch, {})
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_bind_strict_compute_reservations",
        lambda clusters, **_kwargs: {
            key: (bound_cluster if key == ("team", "recover") else value)
            for key, value in clusters.items()
        },
    )
    monkeypatch.setattr(
        mk8s_execution,
        "_run_capture",
        lambda *_args, **_kwargs: pytest.fail("material-input refusal called provider"),
    )

    with pytest.raises(RuntimeError, match="materialized inputs differ"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    for name, path in paths.items():
        assert path.read_bytes() == bytes_before[name]


def test_normal_lifecycle_refuses_retained_target_when_provider_verification_fails(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _unmaterialized_retained_target(tmp_path)
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(
        recovery,
        "_verify_retained_pool_resume",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            recovery.FleetRecoveryError("provider topology differs")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("provider refusal must not enter Terraform apply"),
    )
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)

    with pytest.raises(RuntimeError, match="failed exact normal-lifecycle"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert "kubeconfig" not in calls
    assert "persist" not in calls
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": install_dir / "k8s-training" / "terraform.tfstate",
            "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        assert path.read_bytes() == expected


def test_normal_lifecycle_provider_refusal_preserves_absent_owner_marker(
    tmp_path, monkeypatch
):
    spec, install_dir, _before = _unmaterialized_retained_target(tmp_path)
    marker = install_dir / ".npa-backend-owner.json"
    marker.unlink()
    paths = {
        "sidecar": install_dir / ".npa-fleet-env.json",
        "state": install_dir / "k8s-training" / "terraform.tfstate",
        "lock": install_dir / "k8s-training" / ".terraform.tfstate.lock.info",
        "summary": tmp_path / spec.name / "fleet-state.json",
    }
    before = {name: path.read_bytes() for name, path in paths.items()}
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")

    def provider_refusal(*_args, **_kwargs):
        calls["provider_refusal"] = True
        raise RuntimeError("provider refused retained target")

    monkeypatch.setattr(lifecycle, "_verify_retained_pool_target", provider_refusal)
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("provider refusal must not enter Terraform apply"),
    )

    with pytest.raises(RuntimeError, match="provider refused retained target"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    assert calls == {"provider_refusal": True}
    assert not marker.exists()
    assert not (install_dir / "kubeconfig").exists()
    for name, path in paths.items():
        assert path.read_bytes() == before[name]


def test_normal_lifecycle_refuses_registration_before_terminal_strict_pool_proof(
    tmp_path, monkeypatch
):
    spec, install_dir, before = _unmaterialized_retained_target(tmp_path)
    cluster = spec.projects[0].clusters[0]
    cluster.gpu_nodes = NodePoolSpec(
        count=1,
        platform="gpu-rtx6000",
        preset="1gpu-24vcpu-218gb",
        capacity_block_group="capacityblockgroup-test",
    )
    bound_cluster = replace(
        cluster,
        gpu_nodes=replace(
            cluster.gpu_nodes,
            compute_reservation_id="capacityblockgroup-test",
        ),
    )
    workdir = install_dir / "k8s-training"
    tfvars = render_tfvars(
        as_mk8s_desired(bound_cluster), ssh_public_key="ssh-test", recipe_dir=workdir
    )
    (workdir / "terraform.tfvars").write_text(
        tfvars,
        encoding="utf-8",
    )
    before["state"] = (workdir / "terraform.tfstate").read_bytes()
    before["lock"] = (workdir / ".terraform.tfstate.lock.info").read_bytes()
    calls: dict[str, object] = {}
    _prepare_provider(monkeypatch, calls)
    monkeypatch.setattr(lifecycle, "_assert_terraform_version", lambda _: None)
    monkeypatch.setattr(lifecycle, "_resolve_tenant_id", lambda *_: "tenant-test")
    monkeypatch.setattr(lifecycle, "_resolve_region", lambda _: "test-region")
    monkeypatch.setattr(recovery, "_assert_no_live_apply_owner", lambda _workdir: None)
    monkeypatch.setattr(
        lifecycle,
        "_bind_strict_compute_reservations",
        lambda clusters, **_kwargs: {
            key: (bound_cluster if key == ("team", "recover") else value)
            for key, value in clusters.items()
        },
    )
    monkeypatch.setattr(
        lifecycle,
        "_deploy_one_cluster",
        lambda **_: pytest.fail("retained target must not enter Terraform apply"),
    )
    commands: list[list[str]] = []
    pool_lists = 0

    def provider(command, **_kwargs):
        nonlocal pool_lists
        commands.append(command)
        if "cluster" in command and "list" in command:
            payload = {
                "items": [
                    {
                        "metadata": {
                            "id": "cluster-test",
                            "parent_id": "project-test",
                            "name": "recover",
                        },
                        "status": {"state": "RUNNING"},
                    }
                ]
            }
        elif "node-group" in command and "list" in command:
            pool_lists += 1
            payload = (
                {"items": []}
                if pool_lists < 3
                else {
                    "items": [
                        _provider_pool(
                            "recover-ng-cpu",
                            platform="cpu-d3",
                            preset="48vcpu-192gb",
                            count=1,
                        ),
                        _provider_pool(
                            "recover-ng-gpu-0",
                            platform="gpu-rtx6000",
                            preset="1gpu-24vcpu-218gb",
                            count=1,
                            reservation_id="other-compute-reservation",
                        ),
                    ]
                }
            )
        elif "node-group" in command and "create" in command:
            payload = {
                "metadata": {
                    "id": "nodegroup-" + command[command.index("--name") + 1],
                    "parent_id": "cluster-test",
                    "name": command[command.index("--name") + 1],
                }
            }
        else:
            pytest.fail(f"unexpected provider command: {command}")
        return CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(mk8s_execution, "_run_capture", provider)

    with pytest.raises(RuntimeError, match="node-group topology differs"):
        lifecycle.deploy_fleet(
            spec,
            work_root=tmp_path,
            k8s_training_dir=install_dir,
            preflight=False,
        )

    creates = [
        command
        for command in commands
        if "node-group" in command and "create" in command
    ]
    assert [command[command.index("--name") + 1] for command in creates] == [
        "recover-ng-cpu",
        "recover-ng-gpu-0",
    ]
    assert pool_lists == 3
    assert "kubeconfig" not in calls
    assert "persist" not in calls
    assert not any("terraform" in command for command in commands)
    for name, expected in before.items():
        path = {
            "owner": install_dir / ".npa-backend-owner.json",
            "sidecar": install_dir / ".npa-fleet-env.json",
            "state": workdir / "terraform.tfstate",
            "lock": workdir / ".terraform.tfstate.lock.info",
            "summary": tmp_path / spec.name / "fleet-state.json",
        }[name]
        assert path.read_bytes() == expected
