"""Reject absent/mismatched OptiX weights and protect native mounts during staging."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from npa.workbench.isaac_arena import optix_payload
from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.runtime import _require_optix_runtime
from npa.workbench.isaac_arena.viewport_graphics import _optix_library_sha256


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "image/usr/share/nvidia/nvoptix.bin"
    target.parent.mkdir(parents=True, mode=0o755)
    source = tmp_path / "extracted/usr/share/nvidia/nvoptix.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"exact-signed-driver-denoiser-weights")
    mounts = tmp_path / "mountinfo"
    mounts.write_text("100 99 0:1 / / rw - overlay overlay rw\n")
    monkeypatch.setattr(optix_payload, "_WEIGHTS_PATH", target)
    monkeypatch.setattr(optix_payload, "_MOUNTINFO", mounts)
    return source.parents[3], source, target, mounts


def test_weights_are_atomic_readback_verified_and_not_source_symlink(weights):
    extracted, source, target, _ = weights
    assert optix_payload._native_optix_weights() is None
    evidence = optix_payload._prepare_optix_weights(extracted)
    assert target.read_bytes() == source.read_bytes()
    assert not target.is_symlink()
    assert target.stat().st_ino != source.stat().st_ino
    assert target.stat().st_mode & 0o777 == 0o444
    assert evidence["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert evidence["bytes"] == source.stat().st_size
    assert evidence["placement"] == "container_overlay"
    assert evidence["installed_in_container"] is True
    assert evidence["installed_on_node"] is False
    assert evidence["baked"] is evidence["published"] is False
    assert list(target.parent.iterdir()) == [target]


def test_identical_existing_native_weights_are_read_without_writing(weights):
    extracted, source, target, mounts = weights
    target.write_bytes(source.read_bytes())
    before = target.stat()
    mounts.write_text("100 99 8:1 / / rw - ext4 /dev/root rw\n")
    assert optix_payload._prepare_optix_weights(extracted)["placement"] == "native_runtime"
    assert target.stat().st_ino == before.st_ino
    assert target.stat().st_mtime_ns == before.st_mtime_ns


def test_existing_mismatched_weights_are_never_overwritten(weights):
    extracted, _, target, _ = weights
    target.write_bytes(b"different-driver")
    before = target.read_bytes()
    with pytest.raises(IsaacArenaError, match="differs from the matching driver"):
        optix_payload._prepare_optix_weights(extracted)
    assert target.read_bytes() == before


@pytest.mark.parametrize("kind", ["empty", "symlink", "directory", "fifo"])
def test_native_weights_must_be_a_nonempty_regular_file(weights, kind):
    _, source, target, _ = weights
    if kind == "empty":
        target.touch()
    elif kind == "symlink":
        target.symlink_to(source)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    with pytest.raises(IsaacArenaError, match="regular file"):
        optix_payload._native_optix_weights()


def test_missing_matching_package_payload_cannot_pass(weights):
    extracted, source, target, _ = weights
    source.unlink()
    with pytest.raises(IsaacArenaError, match="package is missing OptiX weights"):
        optix_payload._prepare_optix_weights(extracted)
    assert not target.exists()


@pytest.mark.parametrize("mountpoint", ["parent", "target", "ancestor"])
def test_container_root_overlay_does_not_authorize_a_bind_mount(weights, mountpoint):
    extracted, _, target, mounts = weights
    paths = {"parent": target.parent, "target": target, "ancestor": target.parents[2]}
    # The same major/minor and fstype are deliberately insufficient: mount ID
    # and mount root distinguish this otherwise identical-looking bind mount.
    mounts.write_text(mounts.read_text()
                      + f"101 100 0:1 /host {paths[mountpoint]} rw - overlay overlay rw\n")
    with pytest.raises(IsaacArenaError, match="mounted destination"):
        optix_payload._prepare_optix_weights(extracted)
    assert not target.exists()


@pytest.mark.parametrize("mountinfo", ["", "malformed", "100 99 8:1 / / rw - ext4 /dev/root rw\n",
                                     "100 99 0:1 /host/subtree / rw - overlay overlay rw\n"])
def test_unknown_or_host_root_mount_is_rejected(weights, mountinfo):
    extracted, _, target, mounts = weights
    mounts.write_text(mountinfo)
    with pytest.raises(IsaacArenaError, match="mount|overlay"):
        optix_payload._prepare_optix_weights(extracted)
    assert not target.exists()


@pytest.mark.parametrize("mode", [0o777, 0o775])
def test_shared_writable_directory_is_rejected(weights, mode):
    extracted, _, target, _ = weights
    target.parent.chmod(mode)
    with pytest.raises(IsaacArenaError, match="private writable directory"):
        optix_payload._prepare_optix_weights(extracted)
    assert not target.exists()


def test_destination_owner_must_be_task_user(weights, monkeypatch):
    extracted, _, target, _ = weights
    monkeypatch.setattr(optix_payload.os, "geteuid", lambda: target.parent.stat().st_uid + 1)
    with pytest.raises(IsaacArenaError, match="private writable directory"):
        optix_payload._prepare_optix_weights(extracted)


def test_parent_symlink_is_rejected(weights):
    extracted, _, target, _ = weights
    moved = target.parent.with_name("other")
    target.parent.rename(moved)
    target.parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(IsaacArenaError, match="image-owned directory"):
        optix_payload._prepare_optix_weights(extracted)
    assert not (moved / target.name).exists()


@pytest.mark.parametrize("identical", [False, True])
def test_racing_destination_creation_cannot_overwrite_bytes(weights, monkeypatch, identical):
    extracted, source, target, _ = weights
    link = os.link
    contents = source.read_bytes() if identical else b"racing-native-payload"

    def race(*args, **kwargs):
        target.write_bytes(contents)
        return link(*args, **kwargs)

    monkeypatch.setattr(optix_payload.os, "link", race)
    if identical:
        optix_payload._prepare_optix_weights(extracted)
    else:
        with pytest.raises(IsaacArenaError, match="differs from the matching driver"):
            optix_payload._prepare_optix_weights(extracted)
    assert target.read_bytes() == contents
    assert list(target.parent.iterdir()) == [target]


def test_copy_corruption_is_rejected_before_destination_publication(weights, monkeypatch):
    extracted, _, target, _ = weights
    monkeypatch.setattr(optix_payload.shutil, "copyfileobj",
                        lambda source, copy: copy.write(b"corrupted"))
    with pytest.raises(IsaacArenaError, match="hash verification"):
        optix_payload._prepare_optix_weights(extracted)
    assert not target.exists()
    assert not list(target.parent.iterdir())


def test_mount_change_after_directory_open_is_rejected_before_first_write(weights, monkeypatch):
    extracted, _, target, mounts = weights
    open_file = os.open

    def remount(path, flags, *args, **kwargs):
        descriptor = open_file(path, flags, *args, **kwargs)
        if path == target.parent:
            mounts.write_text(mounts.read_text()
                              + f"101 100 0:1 /host {target.parent} rw - overlay overlay rw\n")
        return descriptor

    monkeypatch.setattr(optix_payload.os, "open", remount)
    with pytest.raises(IsaacArenaError, match="mounted destination"):
        optix_payload._prepare_optix_weights(extracted)
    assert not list(target.parent.iterdir())


def test_directory_replacement_after_open_is_rejected_before_first_write(weights, monkeypatch):
    extracted, _, target, _ = weights
    open_file = os.open
    original = target.parent.with_name("original")

    def replace(path, flags, *args, **kwargs):
        descriptor = open_file(path, flags, *args, **kwargs)
        if path == target.parent:
            path.rename(original)
            path.mkdir()
        return descriptor

    monkeypatch.setattr(optix_payload.os, "open", replace)
    with pytest.raises(IsaacArenaError, match="destination changed"):
        optix_payload._prepare_optix_weights(extracted)
    assert not list(target.parent.iterdir())
    assert not list(original.iterdir())


@pytest.mark.parametrize("message", [
    "Unable to load denoiser weights: Could not open optix denoiser weights file",
    "optixDenoiserCreate(ctx, kind, options, &denoiser) failed.",
    "Optix Error: OPTIX_ERROR_INTERNAL_ERROR.",
    "[Error] [rtx.optixdenoising.plugin] New native failure description",
])
def test_renderer_failure_cannot_pass_from_settings_readback(message):
    with pytest.raises(IsaacArenaError, match="settings alone do not prove denoising"):
        _require_optix_runtime("OptiX enabled=true\n" + message)


def test_optix_settings_without_runtime_errors_are_not_rejected():
    _require_optix_runtime("OptiX enabled=true\nMetrics: {'success_rate': 1.0}")


@pytest.mark.parametrize("kind", ["missing", "empty", "wrong-version", "symlink"])
def test_fallback_requires_exact_package_optix_library(tmp_path, kind):
    library = tmp_path / "libnvoptix.so.580.173.02"
    alternative = tmp_path / "libnvoptix.so.580.999.01"
    alternative.write_bytes(b"wrong-library")
    if kind == "empty":
        library.touch()
    elif kind == "symlink":
        library.symlink_to(alternative)
    elif kind == "wrong-version":
        library.write_bytes(b"correct-version")
    (tmp_path / "libnvoptix.so.1").symlink_to(alternative)
    with pytest.raises(IsaacArenaError, match="exact-driver OptiX library"):
        _optix_library_sha256(tmp_path, "580.173.02")
