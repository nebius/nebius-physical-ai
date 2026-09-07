"""Reject pre-created shared or redirected golden output directories."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.cosmos.nano_video import NanoVideoError, write_json
from npa.workbench.cosmos.nano_video_golden import _private_root
from npa.workbench.cosmos import nano_video_augment_client as recovery


def test_new_private_root_and_existing_owned_root_preserve_real_outputs(tmp_path):
    root = tmp_path / "results"
    assert _private_root(root) == root
    assert root.stat().st_mode & 0o777 == 0o700
    write_json(root / "result.json", {"status": "prepared"})
    before = (root / "result.json").read_bytes()
    assert _private_root(root) == root
    assert (root / "result.json").read_bytes() == before


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_shared_directory_is_refused_before_artifact_write(tmp_path, mode):
    root = tmp_path / "results"
    root.mkdir()
    root.chmod(mode)
    with pytest.raises(NanoVideoError, match="owned directory"):
        _private_root(root)
    assert not list(root.iterdir())


def test_symlink_is_refused_even_when_target_is_private(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "results"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(NanoVideoError, match="owned directory"):
        _private_root(link)
    assert not list(target.iterdir())


def test_another_owner_is_refused_even_with_private_mode(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir(mode=0o700)
    info = root.lstat()
    monkeypatch.setattr(Path, "lstat", lambda self: SimpleNamespace(
        st_mode=info.st_mode, st_uid=os.getuid() + 1,
    ))
    with pytest.raises(NanoVideoError, match="owned directory"):
        _private_root(root)


def test_recovery_preserves_private_artifacts_and_refuses_duplicate_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path))
    root = recovery._private_root("s3://example-bucket/result", fresh=True)
    write_json(root / "result.json", {"status": "prepared"})
    before = (root / "result.json").read_bytes()
    assert recovery._private_root("s3://example-bucket/result", fresh=False) == root
    assert (root / "result.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        recovery._private_root("s3://example-bucket/result", fresh=True)


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_recovery_refuses_shared_directory(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path))
    root = recovery._private_root("s3://example-bucket/result", fresh=True)
    root.chmod(mode)
    with pytest.raises(recovery.AugmentationClientError, match="owned and private"):
        recovery._private_root("s3://example-bucket/result", fresh=False)
    assert not list(root.iterdir())


def test_recovery_refuses_redirected_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path))
    root = recovery._private_root("s3://example-bucket/result", fresh=True)
    root.rmdir()
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(recovery.AugmentationClientError, match="symlink"):
        recovery._private_root("s3://example-bucket/result", fresh=False)
    assert not list(target.iterdir())


def test_recovery_refuses_another_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path))
    root = recovery._private_root("s3://example-bucket/result", fresh=True)
    info = root.lstat()
    monkeypatch.setattr(Path, "lstat", lambda self: SimpleNamespace(
        st_mode=info.st_mode, st_uid=os.getuid() + 1,
    ))
    with pytest.raises(recovery.AugmentationClientError, match="owned and private"):
        recovery._private_root("s3://example-bucket/result", fresh=False)
