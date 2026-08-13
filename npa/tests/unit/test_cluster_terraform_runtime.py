from __future__ import annotations

from pathlib import Path

import pytest

from npa.cli.cluster import terraform_runtime


def test_isolated_data_cleanup_failure_stays_owned_and_reportable(
    monkeypatch, tmp_path: Path
) -> None:
    terraform_dir = tmp_path / "deploy" / "cluster"
    terraform_dir.mkdir(parents=True)
    original_rmtree = terraform_runtime.shutil.rmtree
    scratch: Path | None = None

    def fail_rmtree(path, *args, **kwargs):
        raise OSError("filesystem busy")

    monkeypatch.setattr(terraform_runtime.shutil, "rmtree", fail_rmtree)
    with pytest.raises(
        terraform_runtime.TerraformDataCleanupError,
        match="npa cleanup --full --yes",
    ):
        with terraform_runtime.isolated_terraform_data_dir(
            terraform_dir, "cluster-a"
        ) as data_dir:
            scratch = data_dir
            (data_dir / "providers").mkdir()

    assert scratch is not None and scratch.exists()
    residue = terraform_runtime.collect_terraform_residue(start=tmp_path)
    assert any(item.path == scratch and item.removable for item in residue)

    monkeypatch.setattr(terraform_runtime.shutil, "rmtree", original_rmtree)
    item = next(item for item in residue if item.path == scratch)
    assert terraform_runtime.remove_terraform_residue(item) == ""
    assert not scratch.exists()


def test_isolated_data_creation_failure_never_leaves_an_unowned_run_dir(
    monkeypatch, tmp_path: Path
) -> None:
    terraform_dir = tmp_path / "deploy" / "cluster"
    terraform_dir.mkdir(parents=True)
    monkeypatch.setattr(
        terraform_runtime.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    with pytest.raises(RuntimeError, match="Could not create isolated Terraform data"):
        with terraform_runtime.isolated_terraform_data_dir(
            terraform_dir, "cluster-a"
        ):
            raise AssertionError("unreachable")

    root = terraform_runtime.terraform_scratch_root()
    assert not root.exists() or list(root.iterdir()) == []


def test_destroy_evidence_ignores_a_legacy_provider_cache(
    tmp_path: Path,
) -> None:
    terraform_dir = tmp_path / "deploy" / "cluster"
    cache = terraform_dir / ".terraform" / "providers"
    cache.mkdir(parents=True)

    assert not terraform_runtime.has_destroy_evidence(
        terraform_dir, "never-created"
    )


def test_apply_inventory_is_durable_evidence_before_kubeconfig(
    tmp_path: Path,
) -> None:
    terraform_dir = tmp_path / "deploy" / "cluster"
    terraform_dir.mkdir(parents=True)

    inventory = terraform_runtime.record_terraform_inventory(
        "interrupted", terraform_dir
    )

    assert inventory.stat().st_mode & 0o077 == 0
    assert terraform_runtime.has_destroy_evidence(terraform_dir, "interrupted")


def test_source_cache_symlink_is_reported_but_never_removed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    terraform_dir = repo / "deploy" / "cluster"
    terraform_dir.mkdir(parents=True)
    for name in ("main.tf", "versions.tf", ".terraform.lock.hcl"):
        (terraform_dir / name).write_text("# test\n")
    outside = tmp_path / "outside"
    (outside / "providers").mkdir(parents=True)
    (outside / "providers" / "keep").write_text("user data")
    (terraform_dir / ".terraform").symlink_to(outside, target_is_directory=True)

    item = next(
        item
        for item in terraform_runtime.collect_terraform_residue(start=repo)
        if item.path == terraform_dir / ".terraform"
    )

    assert not item.removable
    assert "validation" in terraform_runtime.remove_terraform_residue(item)
    assert (outside / "providers" / "keep").read_text() == "user data"


def test_source_cache_with_symlinked_parent_is_never_removed(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    outside = tmp_path / "outside-deploy" / "cluster"
    cache = outside / ".terraform" / "providers"
    cache.mkdir(parents=True)
    (cache / "keep").write_text("user data")
    for name in ("main.tf", "versions.tf", ".terraform.lock.hcl"):
        (outside / name).write_text("# test\n")
    (repo / "deploy").symlink_to(outside.parent, target_is_directory=True)

    item = next(
        item
        for item in terraform_runtime.collect_terraform_residue(start=repo)
        if item.path == repo / "deploy" / "cluster" / ".terraform"
    )

    assert not item.removable
    assert "validation" in terraform_runtime.remove_terraform_residue(item)
    assert (cache / "keep").read_text() == "user data"
