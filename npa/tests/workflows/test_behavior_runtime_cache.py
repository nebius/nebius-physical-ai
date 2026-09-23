from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest

from npa.workflows.behavior_challenge import runtime_cache
from npa.workflows.behavior_challenge.runtime_cache import prepare_runtime

MANIFEST_URI = "s3://private-runtime/manifest.json"


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.calls: list[str] = []

    def download_file(self, uri: str, path: str) -> None:
        self.calls.append(uri)
        Path(path).write_bytes(self.objects[uri])


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cache_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, str | None]]:
    """Capture cache metadata and regular-file bytes without following links."""
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        digest = _sha(path.read_bytes()) if path.is_file() else None
        result[relative] = (
            metadata.st_mode,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_size,
            digest,
        )
    return result


def _tar(entries: list[tuple[str, str, bytes | str]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        for kind, name, value in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if name.endswith("python") else 0o644
            if kind == "file":
                assert isinstance(value, bytes)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                member.size = 0
                archive.addfile(member)
            elif kind == "hardlink":
                member.type = tarfile.LNKTYPE
                member.linkname = str(value)
                member.size = 0
                archive.addfile(member)
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = str(value)
                member.size = 0
                archive.addfile(member)
            elif kind == "fifo":
                member.type = tarfile.FIFOTYPE
                member.size = 0
                archive.addfile(member)
            else:
                raise AssertionError(kind)
    return stream.getvalue()


def _manifest_bytes(archives: list[tuple[str, bytes]]) -> bytes:
    value = {
        "schema": "npa.behavior.private-runtime.v1",
        "allowed_directories": ["work", ".local/python"],
        "archives": [
            {"uri": uri, "sha256": _sha(payload), "bytes": len(payload)}
            for uri, payload in archives
        ],
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _runtime_objects(
    *, extra_first: list[tuple[str, str, bytes | str]] | None = None
) -> tuple[dict[str, bytes], bytes]:
    first = _tar(
        [
            ("file", ".local/python/bin/python", b"immutable-python"),
            ("file", "work/runtime.txt", b"runtime"),
            *(extra_first or []),
        ]
    )
    second = _tar(
        [
            ("file", "work/data/original", b"shared"),
            ("hardlink", "work/data/copy", "work/data/original"),
        ]
    )
    archives = [
        ("s3://private-runtime/base.tar.gz", first),
        ("s3://private-runtime/data.tar.gz", second),
    ]
    manifest = _manifest_bytes(archives)
    return {MANIFEST_URI: manifest, **dict(archives)}, manifest


def _run(
    tmp_path: Path,
    objects: dict[str, bytes],
    manifest: bytes,
) -> tuple[dict[str, Any], FakeStorage, FakeStorage, Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    workspace = home / "work"
    manifest_storage = FakeStorage({MANIFEST_URI: objects[MANIFEST_URI]})
    archive_storage = FakeStorage(
        {key: value for key, value in objects.items() if key != MANIFEST_URI}
    )
    receipt = prepare_runtime(
        manifest_storage,
        archive_storage,
        MANIFEST_URI,
        _sha(manifest),
        home,
        workspace,
    )
    return receipt, manifest_storage, archive_storage, home, workspace


def test_prepare_runtime_restores_verified_archives_and_reuses_clean_base(
    tmp_path: Path,
) -> None:
    objects, manifest = _runtime_objects()
    receipt, manifest_storage, archive_storage, home, workspace = _run(
        tmp_path, objects, manifest
    )

    assert receipt["schema"] == "npa.behavior.runtime-ready.v2"
    assert receipt["manifest_sha256"] == _sha(manifest)
    assert receipt["python_directory"] == ".local/python"
    assert (home / ".local/python/bin/python").read_bytes() == b"immutable-python"
    assert (workspace / "runtime.txt").read_bytes() == b"runtime"
    assert (workspace / "data/copy").read_bytes() == b"shared"
    assert (
        os.stat(workspace / "data/original").st_ino
        == os.stat(workspace / "data/copy").st_ino
    )
    assert manifest_storage.calls == [MANIFEST_URI]
    assert archive_storage.calls == [
        "s3://private-runtime/base.tar.gz",
        "s3://private-runtime/data.tar.gz",
    ]

    shutil.rmtree(home / ".local/python")
    reused = prepare_runtime(
        manifest_storage,
        archive_storage,
        MANIFEST_URI,
        _sha(manifest),
        home,
        workspace,
    )
    assert reused == receipt
    assert (home / ".local/python/bin/python").read_bytes() == b"immutable-python"
    assert manifest_storage.calls == [MANIFEST_URI, MANIFEST_URI]
    assert len(archive_storage.calls) == 2


def test_prepare_runtime_reuses_identical_python_without_replacing_it(
    tmp_path: Path,
) -> None:
    objects, manifest = _runtime_objects()
    receipt, manifest_storage, archive_storage, home, workspace = _run(
        tmp_path, objects, manifest
    )
    python = home / ".local/python/bin/python"
    original = python.stat()

    repeated = prepare_runtime(
        manifest_storage, archive_storage, MANIFEST_URI, _sha(manifest), home, workspace
    )

    assert repeated == receipt
    assert python.stat().st_ino == original.st_ino
    assert python.stat().st_mtime_ns == original.st_mtime_ns
    assert python.read_bytes() == b"immutable-python"
    assert len(archive_storage.calls) == 2


def test_ready_cache_is_read_only_while_restoring_external_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects, manifest = _runtime_objects()
    receipt, manifests, archives, home, workspace = _run(tmp_path, objects, manifest)
    cache = workspace / ".npa-runtime-cache"
    cache_mode = cache.stat().st_mode & 0o777
    archive_calls = list(archives.calls)
    shutil.rmtree(home / ".local/python")
    before = _cache_snapshot(cache)
    writes: list[Path] = []

    def reject_cache_write(path: Path, _value: bytes) -> None:
        writes.append(path)
        raise AssertionError("ready cache must not be rewritten")

    monkeypatch.setattr(runtime_cache, "_atomic_bytes", reject_cache_write)
    cache.chmod(0o555)
    try:
        reused = prepare_runtime(
            manifests, archives, MANIFEST_URI, _sha(manifest), home, workspace
        )
    finally:
        cache.chmod(cache_mode)

    assert reused == receipt
    assert writes == []
    assert archives.calls == archive_calls
    assert _cache_snapshot(cache) == before
    assert (home / ".local/python/bin/python").read_bytes() == b"immutable-python"


@pytest.mark.parametrize("change", ["bytes", "mode", "symlink"])
def test_prepare_runtime_rejects_changed_existing_python(
    tmp_path: Path, change: str
) -> None:
    objects, manifest = _runtime_objects()
    _, manifest_storage, archive_storage, home, workspace = _run(
        tmp_path, objects, manifest
    )
    python = home / ".local/python/bin/python"
    if change == "bytes":
        python.write_bytes(b"different-python")
    elif change == "mode":
        python.chmod(python.stat().st_mode ^ 0o100)
    else:
        python.unlink()
        python.symlink_to(workspace / "runtime.txt")

    with pytest.raises(ValueError, match="identity differs|cannot contain symlinks"):
        prepare_runtime(
            manifest_storage,
            archive_storage,
            MANIFEST_URI,
            _sha(manifest),
            home,
            workspace,
        )


@pytest.mark.parametrize(
    "entry, message",
    [
        (("file", "/absolute", b"bad"), "safe POSIX path"),
        (("file", "work/../escape", b"bad"), "safe POSIX path"),
        (("file", "other/file", b"bad"), "outside manifest-allowed"),
        (
            ("file", "work/.npa-runtime-cache/runtime-ready.json", b"bad"),
            "reserved runtime cache state",
        ),
        (("symlink", "work/link", "work/runtime.txt"), "unsupported member"),
        (("fifo", "work/pipe", b""), "unsupported member"),
        (("hardlink", "work/link", "work/missing"), "extracted regular file"),
    ],
)
def test_prepare_runtime_rejects_unsafe_members(
    tmp_path: Path,
    entry: tuple[str, str, bytes | str],
    message: str,
) -> None:
    objects, manifest = _runtime_objects(extra_first=[entry])

    with pytest.raises(ValueError, match=message):
        _run(tmp_path, objects, manifest)


def test_prepare_runtime_rejects_duplicate_member(tmp_path: Path) -> None:
    objects, manifest = _runtime_objects(
        extra_first=[("file", "work/runtime.txt", b"duplicate")]
    )

    with pytest.raises(ValueError, match="duplicate member"):
        _run(tmp_path, objects, manifest)


def test_prepare_runtime_rejects_changed_archive_identity(tmp_path: Path) -> None:
    objects, manifest = _runtime_objects()
    objects["s3://private-runtime/base.tar.gz"] += b"changed"

    with pytest.raises(ValueError, match="archive identity differs"):
        _run(tmp_path, objects, manifest)


def test_prepare_runtime_validates_manifest_before_creating_workspace(
    tmp_path: Path,
) -> None:
    objects, manifest = _runtime_objects()
    home = tmp_path / "home"
    home.mkdir()
    workspace = home / "work"
    manifest_storage = FakeStorage({MANIFEST_URI: objects[MANIFEST_URI]})
    archive_storage = FakeStorage({})

    with pytest.raises(ValueError, match="manifest SHA-256 differs"):
        prepare_runtime(
            manifest_storage,
            archive_storage,
            MANIFEST_URI,
            "0" * 64,
            home,
            workspace,
        )
    assert not workspace.exists()
    assert archive_storage.calls == []


def test_prepare_runtime_retries_interrupted_first_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects, manifest = _runtime_objects()
    original = runtime_cache._extract_archive
    attempts = 0

    def interrupt(archive: Path, home: Path, allowed: list[Any]) -> None:
        nonlocal attempts
        attempts += 1
        original(archive, home, allowed)
        if attempts == 1:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(runtime_cache, "_extract_archive", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(tmp_path, objects, manifest)

    monkeypatch.setattr(runtime_cache, "_extract_archive", original)
    home = tmp_path / "home"
    workspace = home / "work"
    cached_manifest = workspace / ".npa-runtime-cache/runtime-manifest.json"
    cached_manifest_stat = cached_manifest.stat()
    manifest_storage = FakeStorage({MANIFEST_URI: objects[MANIFEST_URI]})
    archive_storage = FakeStorage(
        {key: value for key, value in objects.items() if key != MANIFEST_URI}
    )
    receipt = prepare_runtime(
        manifest_storage,
        archive_storage,
        MANIFEST_URI,
        _sha(manifest),
        home,
        workspace,
    )
    assert receipt["schema"] == "npa.behavior.runtime-ready.v2"
    assert (home / ".local/python/bin/python").read_bytes() == b"immutable-python"
    assert cached_manifest.stat().st_ino == cached_manifest_stat.st_ino
    assert cached_manifest.stat().st_mtime_ns == cached_manifest_stat.st_mtime_ns


def test_prepare_runtime_rejects_changed_python_cache(tmp_path: Path) -> None:
    objects, manifest = _runtime_objects()
    _, manifest_storage, archive_storage, home, workspace = _run(
        tmp_path, objects, manifest
    )
    shutil.rmtree(home / ".local/python")
    cache_file = workspace / ".npa-runtime-cache/python-base/bin/python"
    cache_file.write_bytes(b"changed")

    with pytest.raises(ValueError, match="cache is incomplete or changed"):
        prepare_runtime(
            manifest_storage,
            archive_storage,
            MANIFEST_URI,
            _sha(manifest),
            home,
            workspace,
        )


def test_prepare_runtime_rejects_changed_marker(tmp_path: Path) -> None:
    objects, manifest = _runtime_objects()
    _, manifest_storage, archive_storage, home, workspace = _run(
        tmp_path, objects, manifest
    )
    shutil.rmtree(home / ".local/python")
    marker = next((workspace / ".npa-runtime-cache").glob("archive-0-*.json"))
    value = json.loads(marker.read_text())
    value["archive_bytes"] += 1
    marker.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="marker identity differs"):
        prepare_runtime(
            manifest_storage,
            archive_storage,
            MANIFEST_URI,
            _sha(manifest),
            home,
            workspace,
        )


def test_prepare_runtime_uses_separate_storage_clients(tmp_path: Path) -> None:
    objects, manifest = _runtime_objects()
    _, manifest_storage, archive_storage, _, _ = _run(tmp_path, objects, manifest)

    assert set(manifest_storage.calls) == {MANIFEST_URI}
    assert MANIFEST_URI not in archive_storage.calls
    assert all(uri.endswith(".tar.gz") for uri in archive_storage.calls)


@pytest.mark.parametrize("legacy_manifest_replaced", [False, True])
def test_changed_manifest_preserves_ready_cache_before_restore(
    tmp_path: Path,
    legacy_manifest_replaced: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, manifest = _runtime_objects()
    _, manifests, archives, home, workspace = _run(tmp_path, objects, manifest)
    changed = json.loads(manifest)
    changed["archives"] = changed["archives"][:1]
    requested = (json.dumps(changed, sort_keys=True) + "\n").encode()
    manifests.objects[MANIFEST_URI] = requested
    cache = workspace / ".npa-runtime-cache"
    if legacy_manifest_replaced:
        (cache / "runtime-manifest.json").write_bytes(requested)
    before = {p.name: p.read_bytes() for p in cache.glob("*.json")}
    archive_calls = list(archives.calls)
    shutil.rmtree(home / ".local/python")

    def reject_cache_write(_path: Path, _value: bytes) -> None:
        raise AssertionError("different manifest must fail before cache mutation")

    monkeypatch.setattr(runtime_cache, "_atomic_bytes", reject_cache_write)

    with pytest.raises(ValueError, match="different manifest.*isolated workspace"):
        prepare_runtime(
            manifests, archives, MANIFEST_URI, _sha(requested), home, workspace
        )

    assert {p.name: p.read_bytes() for p in cache.glob("*.json")} == before
    assert archives.calls == archive_calls
    assert not (home / ".local/python").exists()
    assert (cache / "python-base/bin/python").read_bytes() == b"immutable-python"
    assert (workspace / "data/original").read_bytes() == b"shared"


def test_changed_manifest_does_not_clean_interrupted_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects, manifest = _runtime_objects()
    original = runtime_cache._extract_archive

    def interrupt(archive: Path, home: Path, allowed: list[Any]) -> None:
        original(archive, home, allowed)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(runtime_cache, "_extract_archive", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(tmp_path, objects, manifest)
    home, workspace = tmp_path / "home", tmp_path / "home/work"
    changed = json.loads(manifest)
    changed["archives"] = changed["archives"][:1]
    requested = json.dumps(changed).encode()
    manifests = FakeStorage({MANIFEST_URI: requested})
    archives = FakeStorage(objects)
    python = home / ".local/python/bin/python"
    inode = python.stat().st_ino

    with pytest.raises(ValueError, match="different manifest"):
        prepare_runtime(
            manifests, archives, MANIFEST_URI, _sha(requested), home, workspace
        )

    assert archives.calls == []
    assert python.stat().st_ino == inode
    assert python.read_bytes() == b"immutable-python"
    assert (
        workspace / ".npa-runtime-cache/runtime-manifest.json"
    ).read_bytes() == manifest


@pytest.mark.parametrize("name", ["runtime-manifest.json", "runtime-ready.json"])
def test_runtime_cache_rejects_identity_symlink(tmp_path: Path, name: str) -> None:
    objects, manifest = _runtime_objects()
    _, manifests, archives, home, workspace = _run(tmp_path, objects, manifest)
    path = workspace / ".npa-runtime-cache" / name
    original = path.read_bytes()
    target = tmp_path / "other.json"
    target.write_bytes(original)
    path.unlink()
    path.symlink_to(target)
    calls = list(archives.calls)

    with pytest.raises(ValueError, match="safe regular file"):
        prepare_runtime(
            manifests, archives, MANIFEST_URI, _sha(manifest), home, workspace
        )

    assert target.read_bytes() == original
    assert path.is_symlink()
    assert archives.calls == calls
