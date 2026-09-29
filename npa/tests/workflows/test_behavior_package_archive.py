"""Exercise manifest package extraction against malformed and valid archives."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge.package_archive import (
    build_manifest_archive,
    extract_manifest_archive,
    verify_manifest_archive,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(files: dict[str, tuple[bytes, int]]) -> bytes:
    value = {
        "payload_count": len(files),
        "payloads": [
            {
                "path": path,
                "bytes": len(payload),
                "sha256": _sha(payload),
                "mode": oct(mode),
            }
            for path, (payload, mode) in files.items()
        ],
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _archive(
    path: Path,
    files: dict[str, tuple[bytes, int]],
    *,
    root: str = ".",
    extra: dict[str, bytes] | None = None,
    directories: tuple[str, ...] = (),
) -> tuple[Path, str]:
    manifest = _manifest(files)
    prefix = "" if root == "." else root + "/"
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, mode in [
            ("MANIFEST.json", manifest, 0o644),
            *((name, data, mode) for name, (data, mode) in files.items()),
            *((name, data, 0o644) for name, data in (extra or {}).items()),
        ]:
            member = tarfile.TarInfo(prefix + name)
            member.mode = mode
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        for name in directories:
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            member.type = tarfile.DIRTYPE
            bundle.addfile(member)
    return path, _sha(manifest)


def _raw_archive(path: Path, rows: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload in rows:
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return path


def test_flat_archive_verifies_and_extracts_exact_modes(tmp_path: Path) -> None:
    files = {
        "worker.py": (b"print('ok')\n", 0o755),
        "data/config.json": (b"{}\n", 0o644),
    }
    archive, manifest_sha = _archive(tmp_path / "worker.tar.gz", files)
    value = verify_manifest_archive(
        archive, expected_root=".", expected_manifest_sha256=manifest_sha
    )
    extracted = extract_manifest_archive(
        archive,
        tmp_path / "out",
        expected_root=".",
        expected_manifest_sha256=manifest_sha,
    )
    assert value["payload_count"] == 2
    assert (extracted / "worker.py").read_bytes() == files["worker.py"][0]
    assert (extracted / "worker.py").stat().st_mode & 0o777 == 0o755


def test_nested_explicit_root_allows_structural_parent_directories(
    tmp_path: Path,
) -> None:
    files = {"worker.py": (b"pass\n", 0o644)}
    archive, manifest_sha = _archive(
        tmp_path / "worker.tar.gz",
        files,
        root="outer/package",
        directories=("outer", "outer/package"),
    )
    extracted = extract_manifest_archive(
        archive,
        tmp_path / "out",
        expected_root="outer/package",
        expected_manifest_sha256=manifest_sha,
    )
    assert (extracted / "worker.py").read_bytes() == b"pass\n"


def test_explicit_root_rejects_enclosing_directory_mismatch(tmp_path: Path) -> None:
    archive, manifest_sha = _archive(
        tmp_path / "worker.tar.gz", {"worker.py": (b"pass\n", 0o644)}, root="package"
    )
    with pytest.raises(ValueError, match="explicit root"):
        verify_manifest_archive(
            archive, expected_root=".", expected_manifest_sha256=manifest_sha
        )


def test_explicit_root_rejects_undeclared_sibling_directory(tmp_path: Path) -> None:
    files = {"worker.py": (b"pass\n", 0o644)}
    archive, manifest_sha = _archive(
        tmp_path / "worker.tar.gz",
        files,
        root="package",
        directories=("outside",),
    )
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="outside the explicit root"):
        extract_manifest_archive(
            archive,
            destination,
            expected_root="package",
            expected_manifest_sha256=manifest_sha,
        )
    assert not destination.exists()


def test_archive_rejects_file_ancestor_before_extraction(tmp_path: Path) -> None:
    files = {"a": (b"parent", 0o644), "a/b": (b"child", 0o644)}
    archive, manifest_sha = _archive(tmp_path / "worker.tar.gz", files)
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="file topology differs"):
        extract_manifest_archive(
            archive,
            destination,
            expected_root=".",
            expected_manifest_sha256=manifest_sha,
        )
    assert not destination.exists()


def test_archive_rejects_file_ancestor_of_directory_before_extraction(
    tmp_path: Path,
) -> None:
    files = {"a": (b"parent", 0o644)}
    archive, manifest_sha = _archive(
        tmp_path / "worker.tar.gz", files, directories=("a/b",)
    )
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="file topology differs"):
        extract_manifest_archive(
            archive,
            destination,
            expected_root=".",
            expected_manifest_sha256=manifest_sha,
        )
    assert not destination.exists()


def test_archive_rejects_undeclared_bytecode_and_mode_change(tmp_path: Path) -> None:
    files = {"worker.py": (b"pass\n", 0o644)}
    extra = {"__pycache__/worker.cpython-311.pyc": b"undeclared"}
    archive, manifest_sha = _archive(tmp_path / "extra.tar.gz", files, extra=extra)
    with pytest.raises(ValueError, match="file sets differ"):
        verify_manifest_archive(
            archive, expected_root=".", expected_manifest_sha256=manifest_sha
        )
    changed, changed_sha = _archive(tmp_path / "mode.tar.gz", files)
    with tarfile.open(changed, "r:gz") as source:
        rows = [
            (member, source.extractfile(member).read())
            for member in source
            if member.isfile()
        ]
    with tarfile.open(changed, "w:gz") as output:
        for member, payload in rows:
            if member.name == "worker.py":
                member.mode = 0o755
            output.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="mode differs"):
        verify_manifest_archive(
            changed, expected_root=".", expected_manifest_sha256=changed_sha
        )


@pytest.mark.parametrize("alias", ["./worker.py", "dir//worker.py", "dir/./worker.py"])
def test_archive_rejects_noncanonical_member_aliases(
    tmp_path: Path, alias: str
) -> None:
    files = {"worker.py": (b"pass\n", 0o644)}
    manifest = _manifest(files)
    archive = _raw_archive(
        tmp_path / "alias.tar.gz",
        [("MANIFEST.json", manifest), (alias, files["worker.py"][0])],
    )
    with pytest.raises(ValueError, match="archive member path differs"):
        verify_manifest_archive(
            archive, expected_root=".", expected_manifest_sha256=_sha(manifest)
        )


def test_files_mapping_cannot_override_its_path_key(tmp_path: Path) -> None:
    payload = b"pass\n"
    value = {
        "files": {
            "worker.py": {
                "path": "other.py",
                "bytes": len(payload),
                "sha256": _sha(payload),
                "mode": "0o644",
            }
        }
    }
    manifest = (json.dumps(value) + "\n").encode()
    archive = _raw_archive(
        tmp_path / "override.tar.gz",
        [("MANIFEST.json", manifest), ("worker.py", payload)],
    )
    with pytest.raises(TypeError, match="file row differs"):
        verify_manifest_archive(
            archive, expected_root=".", expected_manifest_sha256=_sha(manifest)
        )


def _source(tmp_path: Path, files: dict[str, tuple[bytes, int]]) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(files)
    (source / "MANIFEST.json").write_bytes(manifest)
    for name, (payload, mode) in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)
    return source, _sha(manifest)


@pytest.mark.parametrize("root", [".", "outer/package"])
def test_builder_roundtrips_reproducibly_without_unlisted_files(
    tmp_path: Path, root: str
) -> None:
    files = {"worker.py": (b"pass\n", 0o755), "data/caf\u00e9.json": (b"{}", 0o644)}
    source, digest = _source(tmp_path, files)
    (source / "._worker.py").write_bytes(b"macOS resource fork")
    (source / "private-token").write_bytes(b"must not enter the archive")
    outputs = [tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"]
    for number, output in enumerate(outputs):
        os.utime(source / "worker.py", (number + 10, number + 10))
        build_manifest_archive(
            source, output, expected_root=root, expected_manifest_sha256=digest
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    extracted = extract_manifest_archive(
        outputs[0],
        tmp_path / "readback",
        expected_root=root,
        expected_manifest_sha256=digest,
    )
    assert {
        str(p.relative_to(extracted)) for p in extracted.rglob("*") if p.is_file()
    } == {"MANIFEST.json", *files}
    for name, (payload, mode) in files.items():
        assert (extracted / name).read_bytes() == payload
        assert (extracted / name).stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("change", ["payload", "mode", "manifest", "missing"])
def test_builder_mismatch_publishes_nothing_and_cleans_owned_scratch(
    tmp_path: Path, change: str
) -> None:
    source, digest = _source(tmp_path, {"worker.py": (b"pass\n", 0o755)})
    if change == "payload":
        (source / "worker.py").write_bytes(b"changed")
    elif change == "mode":
        (source / "worker.py").chmod(0o644)
    elif change == "manifest":
        (source / "MANIFEST.json").write_bytes(b"{}")
    else:
        (source / "worker.py").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        build_manifest_archive(
            source,
            tmp_path / "result.tar.gz",
            expected_root=".",
            expected_manifest_sha256=digest,
        )
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("link", ["root", "ancestor", "payload", "manifest"])
def test_builder_rejects_symlinked_source_paths(tmp_path: Path, link: str) -> None:
    source, digest = _source(tmp_path, {"data/worker.py": (b"pass\n", 0o755)})
    target = (
        source
        if link == "root"
        else source
        / {
            "ancestor": "data",
            "payload": "data/worker.py",
            "manifest": "MANIFEST.json",
        }[link]
    )
    moved = tmp_path / "outside"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(OSError):
        build_manifest_archive(
            source,
            tmp_path / "result.tar.gz",
            expected_root=".",
            expected_manifest_sha256=digest,
        )
    assert not (tmp_path / "result.tar.gz").exists()


def test_builder_rejects_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    source, digest = _source(tmp_path, {"worker.py": (b"pass\n", 0o755)})
    (source / "worker.py").unlink()
    os.mkfifo(source / "worker.py")
    with pytest.raises(ValueError, match="regular file"):
        build_manifest_archive(
            source,
            tmp_path / "result.tar.gz",
            expected_root=".",
            expected_manifest_sha256=digest,
        )
    assert not (tmp_path / "result.tar.gz").exists()


def test_builder_cannot_overwrite_a_concurrently_created_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workflows.behavior_challenge import package_archive

    source, digest = _source(tmp_path, {"worker.py": (b"pass\n", 0o755)})
    output = tmp_path / "result.tar.gz"
    verify = package_archive.verify_manifest_archive

    def concurrent_writer(*args: object, **kwargs: object) -> dict:
        result = verify(*args, **kwargs)
        output.write_bytes(b"another publisher")
        return result

    monkeypatch.setattr(package_archive, "verify_manifest_archive", concurrent_writer)
    with pytest.raises(FileExistsError):
        build_manifest_archive(
            source, output, expected_root=".", expected_manifest_sha256=digest
        )
    assert output.read_bytes() == b"another publisher"
    assert set(tmp_path.iterdir()) == {source, output}
