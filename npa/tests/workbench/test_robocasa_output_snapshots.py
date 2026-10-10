"""Fail-closed local snapshot boundaries before conditional S3 publication."""

import asyncio
import hashlib
import io
import os
import stat

import pytest

from npa.workbench.robocasa import capabilities


def _snapshot(root, path, content=b"verified snapshot"):
    info = root.stat()
    return capabilities._verified_output_spool(
        root,
        path,
        (info.st_dev, info.st_ino),
        digest=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
    )


@pytest.mark.parametrize(
    "content",
    [b"", b"verified snapshot", b"x" * 1048580],
    ids=["empty", "small", "over-read-chunk"],
)
def test_snapshot_is_private_seekable_and_closed(tmp_path, content):
    source = tmp_path / "output.bin"
    source.write_bytes(content)
    with _snapshot(tmp_path, source, content) as spool:
        assert stat.S_IMODE(os.fstat(spool.fileno()).st_mode) == 0o600
        assert spool.seekable()
        assert spool.read() == content
        spool.seek(0)
        assert spool.read() == content
    assert spool.closed


@pytest.mark.parametrize(
    "failure", [OSError("network failed"), asyncio.CancelledError()]
)
def test_snapshot_closes_on_publication_failure_or_cancellation(tmp_path, failure):
    source = tmp_path / "output.bin"
    source.write_bytes(b"verified snapshot")
    with pytest.raises(type(failure)):
        with _snapshot(tmp_path, source) as spool:
            raise failure
    assert spool.closed


@pytest.mark.parametrize("replacement", [b"short", b"x" * 17, b"x" * 18])
def test_snapshot_rejects_changed_source_before_publication(tmp_path, replacement):
    source = tmp_path / "output.bin"
    source.write_bytes(replacement)
    with pytest.raises(capabilities.RoboCasaError, match="size.*changed"):
        with _snapshot(tmp_path, source):
            pytest.fail("changed bytes must never reach publication")


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_snapshot_never_reads_unsafe_source(tmp_path, kind):
    source = tmp_path / "output.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"verified snapshot")
    if kind == "symlink":
        source.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(source)
    else:
        os.link(outside, source)
    with pytest.raises((OSError, capabilities.RoboCasaError)):
        with _snapshot(tmp_path, source):
            pytest.fail("unsafe source must never reach publication")
    assert outside.read_bytes() == b"verified snapshot"


@pytest.mark.parametrize("mutation", ["file", "parent", "root"])
def test_snapshot_detects_namespace_change_during_read(tmp_path, monkeypatch, mutation):
    root = tmp_path / "outputs"
    parent = root / "nested"
    parent.mkdir(parents=True)
    source = parent / "output.bin"
    source.write_bytes(b"verified snapshot")
    fill = capabilities._fill_verified_output_spool

    def mutate_after_read(*args, **kwargs):
        fill(*args, **kwargs)
        target = {"file": source, "parent": parent, "root": root}[mutation]
        target.rename(target.with_name("original-" + target.name))
        if mutation == "file":
            target.write_bytes(b"verified snapshot")
        else:
            target.mkdir()

    monkeypatch.setattr(capabilities, "_fill_verified_output_spool", mutate_after_read)
    with pytest.raises((OSError, capabilities.RoboCasaError)):
        with _snapshot(root, source):
            pytest.fail("changed namespace must never reach publication")


def test_snapshot_detects_in_place_write_during_read(tmp_path, monkeypatch):
    source = tmp_path / "output.bin"
    source.write_bytes(b"verified snapshot")
    fill = capabilities._fill_verified_output_spool

    def mutate_after_read(*args, **kwargs):
        fill(*args, **kwargs)
        source.write_bytes(b"different snapshot")

    monkeypatch.setattr(capabilities, "_fill_verified_output_spool", mutate_after_read)
    with pytest.raises(capabilities.RoboCasaError, match="identity changed"):
        with _snapshot(tmp_path, source):
            pytest.fail("source mutation must never reach publication")


def test_snapshot_rejects_short_spool_write():
    class ShortWrite(io.BytesIO):
        def write(self, data):
            return super().write(data[:-1])

    with pytest.raises(capabilities.RoboCasaError, match="write was incomplete"):
        capabilities._fill_verified_output_spool(
            io.BytesIO(b"abc"),
            ShortWrite(),
            digest=hashlib.sha256(b"abc").hexdigest(),
            byte_count=3,
        )


def test_snapshot_read_failure_closes_private_spool(tmp_path, monkeypatch):
    source = tmp_path / "output.bin"
    source.write_bytes(b"verified snapshot")
    temporary_file = capabilities.tempfile.TemporaryFile
    opened = []

    def record_spool(*args, **kwargs):
        spool = temporary_file(*args, **kwargs)
        opened.append(spool)
        return spool

    def fail_read(*_args, **_kwargs):
        raise OSError("injected read failure")

    monkeypatch.setattr(capabilities.tempfile, "TemporaryFile", record_spool)
    monkeypatch.setattr(capabilities, "_fill_verified_output_spool", fail_read)
    with pytest.raises(OSError, match="injected read failure"):
        with _snapshot(tmp_path, source):
            pytest.fail("failed read must never reach publication")
    assert len(opened) == 1 and opened[0].closed
