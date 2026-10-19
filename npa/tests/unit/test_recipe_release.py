"""Exercise immutable MK8s and Soperator source-selection recovery rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from npa.fleet import release as releases
from npa.fleet.release import (
    LegacyMK8sReleaseRecoveryError,
    MK8sReleaseError,
    default_mk8s_release,
    recipe_bundle_sha256,
    select_installation_release,
    validate_release_source,
)
from npa.soperator import lifecycle
from npa.soperator.spec import DEFAULT_SOLUTIONS_LIBRARY_REF


def _write_mk8s_sidecar(install_dir: Path, release: dict[str, str]) -> None:
    install_dir.mkdir(parents=True)
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

    assert selected.release_id == "mk8s-v2026-10-18"
    assert selected.upstream_tag == "main-v2026-10-12"
    assert len(selected.upstream_commit) == 40


def test_existing_install_keeps_exact_saved_release(tmp_path: Path) -> None:
    release = default_mk8s_release()
    install_dir = tmp_path / "install"
    _write_mk8s_sidecar(install_dir, release.persistence())

    assert select_installation_release(install_dir) == release


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
