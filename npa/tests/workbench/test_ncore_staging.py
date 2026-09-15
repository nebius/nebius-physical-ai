"""Real filesystem attacks and overlapping allocations for private staging."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from npa.workbench import ncore_staging as staging


@pytest.mark.parametrize("ancestor", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_rejects_symlinks_without_touching_their_targets(tmp_path, ancestor, dangling):
    outside = tmp_path / "outside"
    if not dangling:
        outside.mkdir(mode=0o700)
        (outside / "keep").write_bytes(b"unchanged")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    root = link / "child" if ancestor else link
    with pytest.raises(staging.PrivateStagingError, match="real directories"):
        with staging.private_staging_directory(root, prefix="test-"):
            pytest.fail("followed a symlink")
    if dangling:
        assert not outside.exists()
    else:
        assert list(outside.iterdir()) == [outside / "keep"]
        assert (outside / "keep").read_bytes() == b"unchanged"


def test_symlink_inserted_between_mkdir_and_open_is_rejected(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    root = tmp_path / "staging"
    original_open = os.open
    parent_inode = tmp_path.stat().st_ino
    attacked = []

    def replace_then_open(path, flags, *args, **kwargs):
        if (
            path == root.name
            and kwargs.get("dir_fd") is not None
            and os.fstat(kwargs["dir_fd"]).st_ino == parent_inode
        ):
            root.rename(tmp_path / "displaced")
            root.symlink_to(outside, target_is_directory=True)
            attacked.append(True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(staging.os, "open", replace_then_open)
    with pytest.raises(staging.PrivateStagingError, match="real directories"):
        staging.private_directory(root)
    assert attacked == [True]
    assert not list(outside.iterdir())


@pytest.mark.parametrize("ancestor", [False, True])
def test_rejects_foreign_ownership_even_without_write_permission(
    monkeypatch, tmp_path, ancestor
):
    root = tmp_path / "foreign"
    root.mkdir(mode=0o700)
    foreign_inode = root.stat().st_ino
    original_fstat = os.fstat

    def foreign_owner(fd):
        info = original_fstat(fd)
        if info.st_ino == foreign_inode:
            return SimpleNamespace(st_uid=os.geteuid() + 1, st_mode=info.st_mode)
        return info

    monkeypatch.setattr(staging.os, "fstat", foreign_owner)
    with pytest.raises(staging.PrivateStagingError, match="ownership|owned"):
        staging.private_directory(root / "child" if ancestor else root)
    assert not list(root.iterdir())


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777, 0o1700])
def test_existing_parent_must_be_private_without_chmod(tmp_path, mode):
    root = tmp_path / "staging"
    root.mkdir()
    root.chmod(mode)
    with pytest.raises(staging.PrivateStagingError, match="mode 0700"):
        staging.private_directory(root)
    assert stat.S_IMODE(root.stat().st_mode) == mode
    assert not list(root.iterdir())


@pytest.mark.parametrize("mode", [0o770, 0o777])
def test_writable_ancestor_is_rejected_before_creating_children(tmp_path, mode):
    root = tmp_path / "shared"
    root.mkdir()
    root.chmod(mode)
    with pytest.raises(staging.PrivateStagingError, match="ancestor"):
        staging.private_directory(root / "private")
    assert not list(root.iterdir())


def test_root_owned_sticky_ancestor_allows_owned_private_child(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    inode = shared.stat().st_ino
    original_fstat = os.fstat

    def root_owner(fd):
        info = original_fstat(fd)
        if info.st_ino == inode:
            return SimpleNamespace(st_uid=0, st_mode=info.st_mode)
        return info

    monkeypatch.setattr(staging.os, "fstat", root_owner)
    with staging.private_staging_directory(shared / "private", prefix="colmap-") as path:
        assert path.parent == shared / "private"
        assert path.stat().st_uid == os.geteuid()
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert not path.exists()


def test_regular_file_and_parent_traversal_are_rejected(tmp_path):
    file = tmp_path / "file"
    file.write_bytes(b"unchanged")
    with pytest.raises(staging.PrivateStagingError, match="real directories"):
        staging.private_directory(file)
    with pytest.raises(staging.PrivateStagingError, match="parent traversal"):
        staging.private_directory(tmp_path / "created" / ".." / "staging")
    assert file.read_bytes() == b"unchanged"
    assert not (tmp_path / "created").exists()


def test_default_is_expanded_only_on_execution_and_is_private(monkeypatch, tmp_path):
    # expanduser uses the current process home, not Path.home at import time.
    original_expanduser = Path.expanduser

    def expanduser(path):
        if path.parts[0] == "~":
            return tmp_path.joinpath("home", *path.parts[1:])
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", expanduser)
    assert staging.DEFAULT_COLMAP_CACHE_DIR.parts[0] == "~"
    with staging.private_staging_directory(
        staging.DEFAULT_COLMAP_CACHE_DIR, prefix="colmap-"
    ) as allocated:
        assert allocated.is_relative_to(tmp_path / "home")
        for path in [allocated, *allocated.parents]:
            if path == tmp_path:
                break
            assert path.stat().st_uid == os.geteuid()
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
        (allocated / "source").write_bytes(b"private")
    assert not allocated.exists()
    assert allocated.parent.is_dir()


def test_concurrent_first_use_allocates_distinct_generations_and_cleans_only_own(
    tmp_path,
):
    root = tmp_path / "new" / "shared"
    barrier = Barrier(2)

    def allocate():
        barrier.wait()
        context = staging.private_staging_directory(root, prefix="colmap-")
        return context, context.__enter__()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(allocate) for _ in range(2)]
        allocations = [future.result() for future in futures]
    first, second = (path for _, path in allocations)
    try:
        assert first != second
        assert first.parent == second.parent == root
        (first / "source").write_bytes(b"first")
        (second / "source").write_bytes(b"second")
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        allocations[0][0].__exit__(None, None, None)
        assert not first.exists()
        assert (second / "source").read_bytes() == b"second"
    finally:
        for context, _ in allocations:
            context.__exit__(None, None, None)
    assert not list(root.iterdir())


def test_failure_cleanup_preserves_existing_parent_contents(tmp_path):
    root = staging.private_directory(tmp_path / "staging")
    (root / "keep").write_bytes(b"preexisting")
    with pytest.raises(RuntimeError, match="conversion failed"):
        with staging.private_staging_directory(root, prefix="colmap-") as allocated:
            (allocated / "source").write_bytes(b"partial")
            raise RuntimeError("conversion failed")
    assert not allocated.exists()
    assert list(root.iterdir()) == [root / "keep"]
    assert (root / "keep").read_bytes() == b"preexisting"
