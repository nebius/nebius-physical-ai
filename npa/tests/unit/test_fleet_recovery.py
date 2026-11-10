"""Exercise fail-closed adoption of one interrupted Fleet apply."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from npa.cluster_backends.mk8s_model import as_mk8s_desired
from npa.cluster_backends.mk8s_render import render_tfvars
from npa.fleet import lifecycle, recovery
from npa.fleet.release import default_mk8s_release
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec


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
