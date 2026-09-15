"""Final receipt identity and complete readback, using synthetic outputs only."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import adjudicate as A  # noqa: E402


@pytest.fixture
def output(tmp_path):
    tmp_path.chmod(0o700)
    with A.W.authorized_roots(tmp_path, ROOT):
        directory, held = A.W.create_output(tmp_path / "output")
        try:
            yield directory, held
        finally:
            os.close(held)


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "hardlink", "mode"])
def test_final_name_or_file_metadata_change_is_refused(output, monkeypatch, kind):
    directory, held = output
    original = A.os.fsync
    changed = []

    def mutate(fd):
        original(fd)
        path = directory / "adjudication.json"
        if fd != held or not path.exists() or changed:
            return
        changed.append(True)
        if kind == "mode":
            path.chmod(0o644)
        elif kind == "hardlink":
            os.link(path, directory / "extra-link")
        else:
            old = directory / "retained-original"
            path.rename(old)
            if kind == "regular":
                path.write_bytes(old.read_bytes())
                path.chmod(0o600)
            elif kind == "symlink":
                path.symlink_to(old)
            else:
                os.mkfifo(path, 0o600)

    monkeypatch.setattr(A.os, "fsync", mutate)
    with pytest.raises(A.W.ScanError):
        A.write_result(directory, held, {"status": "synthetic"})
    assert changed
    assert not (directory / "adjudication.json").exists()
    assert not (directory / "adjudication.json.pending").exists()


@pytest.mark.parametrize("failure", ["initial_fstat", "read", "interrupt"])
def test_failure_closes_original_file_descriptor(output, monkeypatch, failure):
    directory, held = output
    original_open, original_fstat = A.os.open, A.os.fstat
    opened = []

    def opening(name, *args, **kwargs):
        fd = original_open(name, *args, **kwargs)
        if name == "adjudication.json.pending":
            opened.append(fd)
        return fd

    def stat_failed(fd):
        if fd in opened:
            raise OSError("synthetic-stat")
        return original_fstat(fd)

    def read_failed(fd, size):
        assert fd in opened
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise OSError("synthetic-read")

    monkeypatch.setattr(A.os, "open", opening)
    if failure == "initial_fstat":
        monkeypatch.setattr(A.os, "fstat", stat_failed)
    else:
        monkeypatch.setattr(A.os, "read", read_failed)
    with pytest.raises((OSError, KeyboardInterrupt)):
        A.write_result(directory, held, {"status": "synthetic"})
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            original_fstat(fd)
    assert list(directory.iterdir()) == []


def test_same_size_restored_mtime_mutation_is_refused(output, monkeypatch):
    directory, held = output
    original = A.os.fsync
    changed = []

    def mutate(fd):
        original(fd)
        path = directory / "adjudication.json"
        if fd == held and path.exists() and not changed:
            before = path.stat()
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"accepted", b"rejected"))
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed.append(True)

    monkeypatch.setattr(A.os, "fsync", mutate)
    with pytest.raises(A.W.ScanError, match="output_bytes_changed"):
        A.write_result(directory, held, {"status": "accepted"})
    assert changed and list(directory.iterdir()) == []


def test_readback_conserves_complete_large_output(output):
    import json

    directory, held = output
    value = {"status": "synthetic", "evidence": "x" * (2 * 1024 * 1024 + 7)}
    A.write_result(directory, held, value)
    path = directory / "adjudication.json"
    assert path.read_bytes() == (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_nlink == 1
