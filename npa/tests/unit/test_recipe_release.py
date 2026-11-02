"""Exercise immutable MK8s and Soperator source-selection recovery rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from npa.cluster_backends import mk8s_execution
from npa.cluster_backends.mk8s_model import as_mk8s_desired
from npa.fleet import release as releases
from npa.fleet.lifecycle import _cluster_for_mk8s_release, _find_vendored_recipe_root
from npa.fleet.release import (
    LegacyMK8sReleaseRecoveryError,
    MK8sReleaseError,
    default_mk8s_release,
    recipe_bundle_sha256,
    select_installation_release,
    filesystem_csi_makes_default_storage_class,
    validate_release_source,
)
from npa.fleet.spec import ClusterSpec, NodePoolSpec
from npa.soperator import lifecycle
from npa.soperator.spec import DEFAULT_SOLUTIONS_LIBRARY_REF


def _write_mk8s_sidecar(install_dir: Path, release: dict[str, str]) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    (install_dir / ".npa-fleet-env.json").write_text(
        json.dumps({"backend": "mk8s", "mk8s_recipe_release": release}),
        encoding="utf-8",
    )


def _recipe_root(path: Path) -> Path:
    (path / "k8s-training").mkdir(parents=True)
    (path / "modules").mkdir()
    (path / "k8s-training" / "variables.tf").write_text('variable "name" {}\n')
    return path


def test_new_install_selects_current_date_release_once(tmp_path: Path) -> None:
    selected = select_installation_release(tmp_path / "new")

    assert selected.release_id == "mk8s-v2026-10-22"
    assert selected.upstream_tag == "main-v2026-10-12"
    assert len(selected.upstream_commit) == 40


def test_existing_install_keeps_exact_saved_release(tmp_path: Path) -> None:
    release = default_mk8s_release()
    install_dir = tmp_path / "install"
    _write_mk8s_sidecar(install_dir, release.persistence())

    assert select_installation_release(install_dir) == release


def test_prior_qualified_contract_remains_reconcilable(tmp_path: Path) -> None:
    """A new default must not invalidate an already persisted release."""

    prior = releases.resolve_mk8s_release("mk8s-v2026-10-18")
    root = _recipe_root(tmp_path / "recipe")
    lock = root.parent / ".terraform.lock.hcl"
    lock.write_text("provider lock\n")
    expected = replace(
        prior,
        bundle_sha256=recipe_bundle_sha256(root),
        provider_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
    )

    validate_release_source(expected, root)


def test_saved_prior_release_is_reselected_without_revendoring(
    tmp_path: Path,
) -> None:
    prior = releases.resolve_mk8s_release("mk8s-v2026-10-18")
    install_dir = tmp_path / "install"
    _write_mk8s_sidecar(install_dir, prior.persistence())

    assert select_installation_release(install_dir, prior.release_id) == prior
    assert select_installation_release(install_dir) == prior


def test_historical_release_keeps_filesystem_default_but_new_release_does_not():
    assert (
        filesystem_csi_makes_default_storage_class(
            releases.resolve_mk8s_release("mk8s-v2026-10-19")
        )
        is True
    )
    assert filesystem_csi_makes_default_storage_class(default_mk8s_release()) is False


def test_saved_release_preserves_effective_filesystem_default_without_rejecting_cpu_only_install():
    old_release = releases.resolve_mk8s_release("mk8s-v2026-10-19")
    cpu_only = ClusterSpec(name="cpu-only", cpu_nodes=NodePoolSpec(count=1))
    filesystem = replace(
        cpu_only,
        name="filesystem",
        enable_filestore=True,
        filesystem_csi_chart_repository="oci://example.invalid/filesystem-csi",
    )

    assert (
        _cluster_for_mk8s_release(
            cpu_only, old_release
        ).filesystem_csi_make_default_storage_class
        is False
    )
    assert (
        _cluster_for_mk8s_release(
            filesystem, old_release
        ).filesystem_csi_make_default_storage_class
        is True
    )
    assert (
        _cluster_for_mk8s_release(
            filesystem, default_mk8s_release()
        ).filesystem_csi_make_default_storage_class
        is False
    )


def test_materialized_saved_release_keeps_legacy_storage_default(
    tmp_path: Path,
) -> None:
    """A saved installation must render from its retained recipe, not current bytes."""

    source = _find_vendored_recipe_root()
    assert source is not None
    old_release = releases.resolve_mk8s_release("mk8s-v2026-10-19")
    legacy_install = tmp_path / "legacy-install"
    shutil.copytree(source, legacy_install)
    legacy_variables = legacy_install / "k8s-training" / "variables.tf"
    legacy_source = legacy_variables.read_text(encoding="utf-8")
    assert (
        "make_default_storage_class          = optional(bool, false)" in legacy_source
    )
    legacy_variables.write_text(
        legacy_source.replace(
            "make_default_storage_class          = optional(bool, false)",
            "make_default_storage_class          = optional(bool, true)",
            1,
        ),
        encoding="utf-8",
    )
    _write_mk8s_sidecar(legacy_install, old_release.persistence())
    legacy_cluster = ClusterSpec(
        name="legacy",
        cpu_nodes=NodePoolSpec(count=1),
        enable_filestore=True,
        filesystem_csi_chart_repository="oci://example.invalid/filesystem-csi",
        filesystem_csi_make_default_storage_class=(
            filesystem_csi_makes_default_storage_class(old_release)
        ),
    )

    legacy_workdir = mk8s_execution._prepare_install_dir(
        legacy_install,
        recipe_root=legacy_install,
        region="us-central1",
        cluster=as_mk8s_desired(legacy_cluster),
        ssh_public_key="ssh-test",
    )

    assert legacy_workdir == legacy_install / "k8s-training"
    assert "make_default_storage_class = true" in (
        legacy_workdir / "terraform.tfvars"
    ).read_text(encoding="utf-8")
    assert "optional(bool, true)" in legacy_variables.read_text(encoding="utf-8")

    new_install = tmp_path / "new-install"
    new_release = default_mk8s_release()
    new_cluster = replace(
        legacy_cluster,
        name="new",
        filesystem_csi_make_default_storage_class=(
            filesystem_csi_makes_default_storage_class(new_release)
        ),
    )
    new_workdir = mk8s_execution._prepare_install_dir(
        new_install,
        recipe_root=source,
        region="us-central1",
        cluster=as_mk8s_desired(new_cluster),
        ssh_public_key="ssh-test",
    )

    assert "make_default_storage_class = false" in (
        new_workdir / "terraform.tfvars"
    ).read_text(encoding="utf-8")


def test_saved_release_rejects_an_in_place_default_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = default_mk8s_release()
    install_dir = tmp_path / "install"
    _write_mk8s_sidecar(install_dir, release.persistence())
    later = replace(release, release_id="mk8s-v2099-01-01")
    monkeypatch.setitem(releases._MK8S_RELEASES, later.release_id, later)

    with pytest.raises(MK8sReleaseError, match="reviewed migration or replacement"):
        select_installation_release(install_dir, requested_release_id=later.release_id)


def test_modified_recipe_bundle_is_rejected(tmp_path: Path) -> None:
    root = _recipe_root(tmp_path / "recipe")
    lock = root.parent / ".terraform.lock.hcl"
    lock.write_text("provider lock\n")
    expected = replace(
        default_mk8s_release(),
        bundle_sha256=recipe_bundle_sha256(root),
        provider_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        configuration_contract_sha256=releases.configuration_contract_sha256(),
    )
    validate_release_source(expected, root)
    (root / "modules" / "changed.tf").write_text("changed\n")

    with pytest.raises(MK8sReleaseError, match="does not match"):
        validate_release_source(expected, root)


def test_modified_provider_lock_or_contract_is_rejected(tmp_path: Path) -> None:
    root = _recipe_root(tmp_path / "recipe")
    lock = root.parent / ".terraform.lock.hcl"
    lock.write_text("provider lock\n")
    expected = replace(
        default_mk8s_release(),
        bundle_sha256=recipe_bundle_sha256(root),
        provider_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        configuration_contract_sha256=releases.configuration_contract_sha256(),
    )
    lock.write_text("changed provider lock\n")

    with pytest.raises(MK8sReleaseError, match="provider lock does not match"):
        validate_release_source(expected, root)

    lock.write_text("provider lock\n")
    with pytest.raises(MK8sReleaseError, match="component/configuration contract"):
        validate_release_source(
            replace(expected, configuration_contract_sha256="0" * 64), root
        )


def test_legacy_mk8s_state_requires_actionable_recovery(tmp_path: Path) -> None:
    install_dir = tmp_path / "legacy"
    install_dir.mkdir()
    (install_dir / ".npa-fleet-env.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LegacyMK8sReleaseRecoveryError, match="reviewed recovery"):
        select_installation_release(install_dir)


def test_soperator_existing_install_retains_saved_source_commit(tmp_path: Path) -> None:
    root = tmp_path / "work"
    install_dir = (
        root / "nebius-solutions-library" / "soperator" / "installations" / "slurm"
    )
    install_dir.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install_dir,
        region="region",
        tenant_id="tenant",
        project_id="project",
        subnet_id="subnet",
        o11y_profile="",
        solutions_library_ref=DEFAULT_SOLUTIONS_LIBRARY_REF,
    )

    selected = lifecycle._select_solutions_library_ref(
        "slurm", terraform_dir=None, work_root=root, requested_ref=None
    )

    assert selected == DEFAULT_SOLUTIONS_LIBRARY_REF


def test_soperator_legacy_state_requires_explicit_recovery_ref(tmp_path: Path) -> None:
    root = tmp_path / "work"
    install_dir = (
        root / "nebius-solutions-library" / "soperator" / "installations" / "slurm"
    )
    install_dir.mkdir(parents=True)
    (install_dir / "terraform.tfstate").write_text("{}", encoding="utf-8")

    with pytest.raises(
        lifecycle.UpstreamContractError, match="original immutable --ref"
    ):
        lifecycle._select_solutions_library_ref(
            "slurm", terraform_dir=None, work_root=root, requested_ref=None
        )
