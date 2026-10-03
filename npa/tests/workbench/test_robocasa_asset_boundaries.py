"""Adversarial namespace controls for RoboCasa asset publication."""

import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from npa.workbench.robocasa import capabilities


def _candidate(tmp_path: Path):
    archive = next(
        item
        for item in capabilities._asset_archives()
        if item.filename == "objaverse.zip"
    )
    assets = tmp_path / "assets"
    receipt = capabilities._asset_receipt_path(assets / ".npa_asset_fetch", archive)
    source = tmp_path / "source.zip"
    with ZipFile(source, "w") as zipped:
        zipped.writestr("objaverse/new.xml", "<mujoco/>")
    outside = tmp_path / "outside"
    (outside / "objaverse").mkdir(parents=True)
    marker = outside / "objaverse" / "keep.txt"
    marker.write_text("outside-owned", encoding="utf-8")
    return archive, assets, receipt, source, outside, marker


def test_publication_rejects_ancestor_symlink_and_preserves_outside(tmp_path):
    archive, assets, receipt, source, outside, marker = _candidate(tmp_path)
    (assets / "objects").symlink_to(outside, target_is_directory=True)
    with pytest.raises(capabilities.RoboCasaError, match="unsafe.*publication"):
        capabilities._stage_publish_and_receipt(archive, source, assets, receipt)
    assert marker.read_text(encoding="utf-8") == "outside-owned"
    assert not (outside / "objaverse" / "new.xml").exists()
    assert not receipt.exists()
    assert not capabilities._asset_receipt_is_valid(receipt, archive, assets)


def test_publication_rejects_parent_replacement_during_staging(tmp_path, monkeypatch):
    archive, assets, receipt, source, outside, marker = _candidate(tmp_path)
    objects = assets / "objects"
    (objects / "objaverse").mkdir(parents=True)
    original_marker = objects / "objaverse" / "original.txt"
    original_marker.write_text("original-owned", encoding="utf-8")
    extract = capabilities._extract_validated_zip_file

    def replace_parent_after_extract(*args):
        extract(*args)
        objects.rename(assets / "detached-objects")
        objects.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        capabilities, "_extract_validated_zip_file", replace_parent_after_extract
    )
    with pytest.raises(capabilities.RoboCasaError, match="unsafe.*publication"):
        capabilities._stage_publish_and_receipt(archive, source, assets, receipt)
    assert marker.read_text(encoding="utf-8") == "outside-owned"
    assert (
        assets / "detached-objects/objaverse/original.txt"
    ).read_text() == "original-owned"
    assert not (outside / "objaverse/new.xml").exists()
    assert not receipt.exists()


@pytest.mark.parametrize(
    "relative", [".", ".npa_asset_fetch", ".npa_asset_fetch/receipts"]
)
def test_receipt_creation_rejects_symlinked_ancestors(tmp_path, relative):
    assets = tmp_path / "assets"
    assets.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("outside-owned", encoding="utf-8")
    selected = assets if relative == "." else assets / relative
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected == assets:
        assets.rmdir()
    selected.symlink_to(outside, target_is_directory=True)
    archive = capabilities._asset_archives()[0]
    with pytest.raises(capabilities.RoboCasaError, match="unsafe.*receipt"):
        capabilities._asset_receipt_path(assets / ".npa_asset_fetch", archive)
    assert list(outside.iterdir()) == [marker]


def test_asset_lock_rejects_existing_symlink_without_modifying_target(tmp_path):
    assets = tmp_path / "assets"
    state = assets / ".npa_asset_fetch"
    state.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("outside-owned", encoding="utf-8")
    (state / "fetch.lock").symlink_to(outside)
    with pytest.raises(capabilities.RoboCasaError, match="unsafe.*state"):
        with capabilities._asset_fetch_lock(assets):
            pytest.fail("symlinked lock accepted")
    assert outside.read_text(encoding="utf-8") == "outside-owned"


def test_receipt_fifo_is_rejected_without_waiting_for_a_writer(tmp_path, monkeypatch):
    archive, assets, receipt, source, _outside, _marker = _candidate(tmp_path)
    capabilities._stage_publish_and_receipt(archive, source, assets, receipt)
    assert capabilities._asset_receipt_is_valid(receipt, archive, assets)
    validate_held_receipt = capabilities._held_asset_receipt_is_valid
    held_receipt_checks = []

    def record_held_receipt_check(*args):
        held_receipt_checks.append(args)
        return validate_held_receipt(*args)

    monkeypatch.setattr(
        capabilities, "_held_asset_receipt_is_valid", record_held_receipt_check
    )
    receipt.unlink()
    os.mkfifo(receipt)
    assert not capabilities._asset_receipt_is_valid(receipt, archive, assets)
    assert len(held_receipt_checks) == 1


def test_receipt_parent_swap_cannot_publish_outside(tmp_path, monkeypatch):
    archive, assets, receipt, source, outside, marker = _candidate(tmp_path)
    extract = capabilities._extract_validated_zip_file
    displaced = assets / ".npa_asset_fetch/detached-receipts"

    def replace_receipt_parent(*args):
        extract(*args)
        receipt.parent.rename(displaced)
        receipt.parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        capabilities, "_extract_validated_zip_file", replace_receipt_parent
    )
    with pytest.raises(capabilities.RoboCasaError, match="unsafe.*publication"):
        capabilities._stage_publish_and_receipt(archive, source, assets, receipt)
    assert marker.read_text(encoding="utf-8") == "outside-owned"
    assert not (outside / receipt.name).exists()
    assert not (displaced / receipt.name).exists()
