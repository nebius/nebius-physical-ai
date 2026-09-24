"""Verify and extract manifest-backed workflow packages without hidden payloads."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} path differs")
    if value.startswith("/") or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError(f"{label} path differs")
    path = PurePosixPath(value)
    if path.as_posix() != value:
        raise ValueError(f"{label} path differs")
    return path


def _root(value: str) -> PurePosixPath | None:
    if value == ".":
        return None
    return _relative(value, "package root")


def _identity(stream: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _rows(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError("package manifest must be a JSON object")
    raw = value.get("payloads")
    if raw is None:
        files = value.get("files")
        if not isinstance(files, dict):
            raise TypeError("package manifest files differ")
        raw = []
        for path, row in files.items():
            if not isinstance(row, dict) or "path" in row:
                raise TypeError("package manifest file row differs")
            raw.append({"path": path, **row})
    if not isinstance(raw, list) or not raw:
        raise ValueError("package manifest payloads differ")
    rows: dict[str, dict[str, Any]] = {}
    for item in raw:
        path, row = _row(item)
        if path in rows:
            raise ValueError("package manifest has duplicate payload paths")
        rows[path] = row
    if "payload_count" in value and value["payload_count"] != len(rows):
        raise ValueError("package manifest payload count differs")
    return rows


def _row(value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise TypeError("package manifest payload row differs")
    path = _relative(value.get("path"), "payload").as_posix()
    size, digest, mode = value.get("bytes"), value.get("sha256"), value.get("mode")
    if type(size) is not int or size < 0 or _SHA256.fullmatch(str(digest)) is None:
        raise ValueError(f"package payload identity differs: {path}")
    if not isinstance(mode, str) or not re.fullmatch(r"0o[0-7]{3}", mode):
        raise ValueError(f"package payload mode differs: {path}")
    return path, {"bytes": size, "sha256": digest, "mode": mode}


def _archive_files(bundle: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    files: dict[str, tarfile.TarInfo] = {}
    seen: set[str] = set()
    for member in bundle.getmembers():
        path = _relative(member.name.rstrip("/"), "archive member").as_posix()
        if path in seen or not (member.isfile() or member.isdir()):
            raise ValueError("package archive member differs")
        seen.add(path)
        if member.isfile():
            files[path] = member
    return files


def _inside_root(path: str, root: PurePosixPath | None) -> str | None:
    member = PurePosixPath(path)
    if root is None:
        return member.as_posix()
    try:
        return member.relative_to(root).as_posix()
    except ValueError:
        return None


def _manifest(bundle: tarfile.TarFile, member: tarfile.TarInfo, expected: str) -> dict:
    stream = bundle.extractfile(member)
    if stream is None:
        raise ValueError("package manifest payload is absent")
    payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("package manifest bytes differ")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError("package manifest must be a JSON object")
    return value


def _verify_payloads(
    bundle: tarfile.TarFile,
    files: dict[str, tarfile.TarInfo],
    root: PurePosixPath | None,
    rows: dict[str, dict[str, Any]],
) -> None:
    observed = {
        relative
        for path in files
        if (relative := _inside_root(path, root)) != "MANIFEST.json"
    }
    if None in observed or observed != set(rows):
        raise ValueError("package archive and manifest file sets differ")
    prefix = "" if root is None else root.as_posix() + "/"
    for path, expected in rows.items():
        member = files[prefix + path]
        stream = bundle.extractfile(member)
        if stream is None or _identity(stream) != {
            key: expected[key] for key in ("bytes", "sha256")
        }:
            raise ValueError(f"package payload differs: {path}")
        if oct(member.mode & 0o777) != expected["mode"]:
            raise ValueError(f"package payload mode differs: {path}")


def _verify_bundle(
    bundle: tarfile.TarFile, root: PurePosixPath | None, expected_sha256: str
) -> dict[str, Any]:
    files = _archive_files(bundle)
    manifest_name = "MANIFEST.json" if root is None else f"{root}/MANIFEST.json"
    if manifest_name not in files:
        raise ValueError("package manifest is absent at the explicit root")
    manifest = _manifest(bundle, files[manifest_name], expected_sha256)
    _verify_payloads(bundle, files, root, _rows(manifest))
    return manifest


def verify_manifest_archive(
    archive: Path, *, expected_root: str, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Verify an archive's explicit root and complete manifest file bijection.

    Args:
        archive: Local gzip-compressed tar archive to inspect.
        expected_root: Exact archive prefix, or ``.`` for a flat archive.
        expected_manifest_sha256: SHA-256 of the manifest at that root.
    Returns:
        The verified manifest JSON object.
    Raises:
        OSError: The archive cannot be read.
        ValueError: Paths, identities, modes, root, or inventory differ.
        TypeError: The manifest shape differs.
        tarfile.TarError: The input is not a supported tar archive.
    """
    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected package manifest SHA-256 differs")
    root = _root(expected_root)
    with tarfile.open(archive, "r:gz") as bundle:
        return _verify_bundle(bundle, root, expected_manifest_sha256)


def extract_manifest_archive(
    archive: Path,
    destination: Path,
    *,
    expected_root: str,
    expected_manifest_sha256: str,
) -> Path:
    """Verify a package archive completely, then extract it into a new directory.

    Args:
        archive: Local gzip-compressed tar archive to inspect and extract.
        destination: New directory that receives the archive bytes.
        expected_root: Exact archive prefix, or ``.`` for a flat archive.
        expected_manifest_sha256: SHA-256 of the manifest at that root.
    Returns:
        Extracted package root, equal to destination for a flat archive.
    Raises:
        FileExistsError: The destination already exists or is a symlink.
        OSError: Archive bytes cannot be read or written.
        ValueError: The verified archive contract differs.
        TypeError: The manifest shape differs.
        tarfile.TarError: The input is not a supported tar archive.
    """
    root = _root(expected_root)
    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected package manifest SHA-256 differs")
    with tarfile.open(archive, "r:gz") as bundle:
        _verify_bundle(bundle, root, expected_manifest_sha256)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("package destination must be new")
        destination.mkdir()
        _extract_verified(bundle, destination)
    return destination if root is None else destination / Path(*root.parts)


def _extract_verified(bundle: tarfile.TarFile, destination: Path) -> None:
    for member in bundle.getmembers():
        relative = _relative(member.name.rstrip("/"), "member")
        target = destination / Path(*relative.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            _write_member(bundle, member, target)


def _write_member(
    bundle: tarfile.TarFile, member: tarfile.TarInfo, target: Path
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    stream = bundle.extractfile(member)
    if stream is None:
        raise ValueError("package member payload is absent")
    with target.open("xb") as output:
        while chunk := stream.read(1024 * 1024):
            output.write(chunk)
    target.chmod(member.mode & 0o777)
