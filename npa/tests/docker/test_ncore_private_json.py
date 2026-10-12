"""Private acceptance files must not depend on the launching shell's umask."""

import json
import os
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from ncore_publication import process  # noqa: E402


@pytest.mark.parametrize("mask", [0o000, 0o022, 0o077])
def test_private_json_is_owner_only_before_writing(tmp_path, monkeypatch, mask):
    target = tmp_path / "receipt.json"
    original_fdopen = os.fdopen
    observed_modes = []

    def checked_fdopen(descriptor, *args, **kwargs):
        observed_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(process.os, "fdopen", checked_fdopen)
    previous = os.umask(mask)
    try:
        process.write_json(target, {"scope": "synthetic"})
        assert os.umask(mask) == mask
    finally:
        os.umask(previous)

    assert observed_modes == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text()) == {"scope": "synthetic"}


def test_private_json_refuses_existing_file_without_changing_it(tmp_path):
    target = tmp_path / "receipt.json"
    target.write_bytes(b"retained original evidence\n")
    target.chmod(0o640)
    before = target.stat()

    with pytest.raises(FileExistsError):
        process.write_json(target, {"replacement": "forbidden"})

    assert target.read_bytes() == b"retained original evidence\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert target.stat().st_ino == before.st_ino


@pytest.mark.parametrize("existing_target", [False, True])
def test_private_json_refuses_symlinks_without_touching_target(
    tmp_path, existing_target
):
    target = tmp_path / "other.json"
    if existing_target:
        target.write_bytes(b"unrelated evidence\n")
    link = tmp_path / "receipt.json"
    link.symlink_to(target)

    with pytest.raises(FileExistsError):
        process.write_json(link, {"replacement": "forbidden"})

    assert link.is_symlink()
    assert target.exists() is existing_target
    if existing_target:
        assert target.read_bytes() == b"unrelated evidence\n"
