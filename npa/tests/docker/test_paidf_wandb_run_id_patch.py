"""Synthetic source and filesystem contracts for the pinned DIG W&B patch."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import stat
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/paidf-anomalygen-sky/patch_wandb_run_id.py"
)
SOURCE = b"""import wandb.util

def fresh():
    return wandb.util.generate_id()

def retry():
    return wandb.util.generate_id()
"""


def expected_patch(source: bytes) -> bytes:
    return source.replace(
        b"import wandb.util\n",
        b"import wandb.util\nfrom wandb.sdk.lib.runid import generate_id as _npa_generate_wandb_id\n",
    ).replace(b"wandb.util.generate_id()", b"_npa_generate_wandb_id()")


@pytest.fixture
def patcher():
    spec = importlib.util.spec_from_file_location(
        "paidf_wandb_run_id_patch_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic(patcher, monkeypatch):
    def bind(source=SOURCE, expected=None):
        if expected is None:
            expected = expected_patch(source)
        monkeypatch.setattr(
            patcher, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest()
        )
        monkeypatch.setattr(
            patcher, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest()
        )
        return source, expected

    bind()
    return bind


def test_transform_changes_both_calls_and_imports_published_generator(
    patcher, synthetic
):
    source, expected = synthetic()
    original = bytes(source)
    actual = patcher.transform(source)
    assert actual == expected
    assert source == original
    tree = ast.parse(actual)
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls == ["_npa_generate_wandb_id", "_npa_generate_wandb_id"]
    imported = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert len(imported) == 1
    assert imported[0].module == "wandb.sdk.lib.runid"
    assert [(alias.name, alias.asname) for alias in imported[0].names] == [
        ("generate_id", "_npa_generate_wandb_id")
    ]
    assert b"wandb.util.generate_id()" not in actual


@pytest.mark.parametrize(
    "source",
    [SOURCE + b"\n", SOURCE.replace(b"fresh", b"other"), b"", expected_patch(SOURCE)],
)
def test_transform_rejects_unreviewed_source_bytes(patcher, synthetic, source):
    with pytest.raises(ValueError):
        patcher.transform(source)


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace(b"import wandb.util\n", b"import wandb\n"),
        b"import wandb.util\n" + SOURCE,
        SOURCE.replace(b"wandb.util.generate_id()", b"None", 1),
        SOURCE + b"\ndef extra():\n    return wandb.util.generate_id()\n",
    ],
)
def test_transform_checks_anchors_even_with_synthetic_hashes(
    patcher, synthetic, source
):
    synthetic(source)
    with pytest.raises(ValueError):
        patcher.transform(source)


def test_transform_verifies_expected_patched_bytes(patcher, synthetic, monkeypatch):
    monkeypatch.setattr(patcher, "PATCHED_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        patcher.transform(SOURCE)


def test_transform_compiles_hash_bound_output(patcher, synthetic):
    source = SOURCE + b"\ndef broken(:\n"
    synthetic(source)
    with pytest.raises((ValueError, SyntaxError)):
        patcher.transform(source)


def test_read_regular_returns_actual_bytes_and_identity(patcher, tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    raw, info = patcher.read_regular(path)
    assert raw == SOURCE
    assert (info.st_dev, info.st_ino, info.st_uid, info.st_gid, info.st_size) == (
        path.stat().st_dev,
        path.stat().st_ino,
        os.getuid(),
        os.getgid(),
        len(SOURCE),
    )
    assert stat.S_ISREG(info.st_mode)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
def test_read_regular_rejects_unsafe_entries_before_open(
    patcher, tmp_path, monkeypatch, kind
):
    path = tmp_path / "source.py"
    target = tmp_path / "target.py"
    target.write_bytes(SOURCE)
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    original_open = os.open

    def reject_open(file, *args, **kwargs):
        if Path(file) == path:
            pytest.fail("unsafe source was opened before its file-type/link check")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(patcher.os, "open", reject_open)
    with pytest.raises(ValueError):
        patcher.read_regular(path)
    assert target.read_bytes() == SOURCE


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_replace_regular_preserves_permissions_and_ownership(
    patcher, synthetic, tmp_path, mode
):
    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    path.chmod(mode)
    raw, info = patcher.read_regular(path)
    expected = patcher.transform(raw)
    patcher.replace_regular(path, info, expected)
    after = path.stat()
    assert path.read_bytes() == expected
    assert stat.S_IMODE(after.st_mode) == mode
    assert (after.st_uid, after.st_gid) == (info.st_uid, info.st_gid)
    assert after.st_ino != info.st_ino
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize(
    "mutation", ["replacement", "same_size_write", "hardlink", "symlink", "mode"]
)
def test_replace_regular_refuses_stale_original_identity(
    patcher, synthetic, tmp_path, mutation, monkeypatch
):
    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    _, info = patcher.read_regular(path)
    replacement = tmp_path / "concurrent.py"
    if mutation == "replacement":
        replacement.write_bytes(SOURCE)
        os.replace(replacement, path)
    elif mutation == "same_size_write":
        freeze_same_inode_timestamps(patcher, path, info, monkeypatch)
        path.write_bytes(SOURCE.replace(b"fresh", b"other"))
    elif mutation == "hardlink":
        os.link(path, replacement)
    elif mutation == "symlink":
        path.rename(replacement)
        path.symlink_to(replacement)
    else:
        path.chmod(0o400)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        patcher.replace_regular(path, info, expected_patch(SOURCE))
    assert path.read_bytes() == before


def test_atomic_replace_failure_preserves_original_and_cleans_temporary(
    patcher, synthetic, tmp_path, monkeypatch
):
    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    _, info = patcher.read_regular(path)
    observed = []

    def reject_replace(source, destination, *args, **kwargs):
        temporary = Path(source)
        assert temporary.parent == path.parent and temporary != path
        assert Path(destination) == path
        assert temporary.read_bytes() == expected_patch(SOURCE)
        assert path.read_bytes() == SOURCE and path.stat().st_ino == info.st_ino
        observed.append(temporary)
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(patcher.os, "replace", reject_replace)
    with pytest.raises(OSError, match="synthetic rename failure"):
        patcher.replace_regular(path, info, expected_patch(SOURCE))
    assert len(observed) == 1
    assert path.read_bytes() == SOURCE and path.stat().st_ino == info.st_ino
    assert set(tmp_path.iterdir()) == {path}


def test_temporary_write_failure_preserves_original(
    patcher, synthetic, tmp_path, monkeypatch
):
    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    _, info = patcher.read_regular(path)

    def fail_fsync(fd):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(patcher.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="synthetic fsync failure"):
        patcher.replace_regular(path, info, expected_patch(SOURCE))
    assert path.read_bytes() == SOURCE and path.stat().st_ino == info.st_ino
    assert set(tmp_path.iterdir()) == {path}


def freeze_same_inode_timestamps(patcher, path, info, monkeypatch):
    """Model a same-size write inside one filesystem timestamp interval."""
    original_lstat = Path.lstat
    original_fstat = patcher.os.fstat

    def stable_fields(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
        )

    def collided(value):
        return info if stable_fields(value) == stable_fields(info) else value

    def lstat(self, *args, **kwargs):
        value = original_lstat(self, *args, **kwargs)
        return collided(value) if self == path else value

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(patcher.os, "fstat", lambda fd: collided(original_fstat(fd)))


def test_read_regular_rejects_same_inode_mutation_during_read(
    patcher, tmp_path, monkeypatch
):
    import builtins

    path = tmp_path / "source.py"
    path.write_bytes(SOURCE)
    freeze_same_inode_timestamps(patcher, path, path.stat(), monkeypatch)
    original_open = builtins.open

    class MutatingRead:
        def __init__(self, stream):
            self.stream = stream
            self.changed = False

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, *args, **kwargs):
            raw = self.stream.read(*args, **kwargs)
            if not self.changed:
                self.changed = True
                with original_open(path, "wb") as output:
                    output.write(SOURCE.replace(b"fresh", b"other"))
            return raw

    def intercept(file, *args, **kwargs):
        stream = original_open(file, *args, **kwargs)
        return MutatingRead(stream) if Path(file) == path else stream

    monkeypatch.setattr(patcher, "open", intercept, raising=False)
    with pytest.raises(ValueError, match="changed"):
        patcher.read_regular(path)


def test_read_regular_refuses_replaced_named_entry(patcher, tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    replacement = tmp_path / "replacement.py"
    path.write_bytes(SOURCE)
    replacement.write_bytes(SOURCE)
    original_open = patcher.os.open

    def replace_after_open(name, flags, *args, **kwargs):
        fd = original_open(name, flags, *args, **kwargs)
        if Path(name) == path:
            os.replace(replacement, path)
        return fd

    monkeypatch.setattr(patcher.os, "open", replace_after_open)
    with pytest.raises(ValueError, match="changed"):
        patcher.read_regular(path)


@pytest.mark.parametrize("failure", [None, "direct", "transitive", "url"])
def test_default_dependency_gate_checks_direct_and_transitive_requirements(
    patcher, monkeypatch, failure
):
    requirements = {
        "wandb": ["pydantic>=2.10", "not-installed; extra == 'optional'"],
        "opentelemetry-api": ["typing-extensions>=4.5"],
    }
    versions = {"pydantic": "2.13.0", "typing-extensions": "4.15.0"}
    if failure == "direct":
        versions["pydantic"] = "2.0.0"
    elif failure == "transitive":
        versions["typing-extensions"] = "4.0.0"
    elif failure == "url":
        requirements["wandb"] = ["pydantic @ https://example.invalid/unreviewed.whl"]
    monkeypatch.setattr(patcher.metadata, "requires", requirements.__getitem__)
    monkeypatch.setattr(patcher.metadata, "version", versions.__getitem__)
    if failure:
        with pytest.raises(ValueError, match="runtime dependency"):
            patcher.verify_dependencies()
    else:
        patcher.verify_dependencies()
