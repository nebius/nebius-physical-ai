"""Optional immutable archive/restore companion for quiescent native Ray CLIP results."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlsplit

from botocore.exceptions import BotoCoreError, ClientError

from archive_inventory import canonical, digest, read_json, require_digest, safe_name, validate_result

from npa.clients.storage import StorageClient, StorageError, StoragePreconditionFailed

SCHEMA = "npa.ray-clip-archive.v1"


def _directory(path):
    """Open an absolute directory through non-link components, retaining its descriptor."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Use an absolute directory without traversal")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _write_chunk(output, chunk):
    with memoryview(chunk) as view:
        while view:
            view = view[os.write(output, view):]


def _scan_file(descriptor, info, destination):
    checksum, size = hashlib.sha256(), 0
    output = None
    try:
        if destination is not None:
            output = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        while chunk := os.read(descriptor, 1024 * 1024):
            checksum.update(chunk)
            size += len(chunk)
            if output is not None:
                _write_chunk(output, chunk)
        if output is not None:
            os.fsync(output)
    finally:
        if output is not None:
            os.close(output)
    if _signature(info) != _signature(os.fstat(descriptor)) or size != info.st_size:
        raise ValueError("Source changed while reading a file")
    return {"sha256": checksum.hexdigest(), "size": size}


def _scan_entry(directory, entry, name, destination, files, signatures):
    info = entry.stat(follow_symlinks=False)
    is_directory = stat.S_ISDIR(info.st_mode)
    if not is_directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError("Result tree contains a link or nonregular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if is_directory:
        flags |= os.O_DIRECTORY
    child = os.open(entry.name, flags, dir_fd=directory)
    try:
        if _signature(info) != _signature(os.fstat(child)):
            raise ValueError("Source changed during inventory")
        if is_directory:
            if destination is not None:
                (destination / name).mkdir(mode=0o700)
            _scan_directory(child, name, destination, files, signatures)
        else:
            signatures[name] = _signature(info)
            target = destination / name if destination is not None else None
            files[name] = _scan_file(child, info, target)
    finally:
        os.close(child)


def _scan_directory(directory, relative, destination, files, signatures):
    before = _signature(os.fstat(directory))
    signatures[relative] = before
    with os.scandir(directory) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        name = safe_name(f"{relative}/{entry.name}" if relative else entry.name)
        _scan_entry(directory, entry, name, destination, files, signatures)
    if _signature(os.fstat(directory)) != before:
        raise ValueError("Source directory changed during inventory")


def _scan(descriptor, destination=None):
    """Read regular files through anchored descriptors; optionally freeze private copies."""
    files, signatures = {}, {}
    _scan_directory(descriptor, "", destination, files, signatures)
    if not files:
        raise ValueError("Empty result tree")
    return files, signatures


def _prefix(uri):
    parsed = urlsplit(uri)
    if (parsed.scheme != "s3" or not parsed.netloc or parsed.username or parsed.password
            or parsed.port or parsed.query or parsed.fragment or "%" in uri
            or not parsed.path.startswith("/")):
        raise ValueError("Use an unsigned S3 URI with an explicit archive prefix")
    key = parsed.path[1:].rstrip("/")
    safe_name(key)
    return f"s3://{parsed.netloc}/{key}"


def _read(storage, uri):
    value = storage.read_bytes_with_etag(uri)
    if value is None:
        raise ValueError("An archive object is missing")
    return value[0]


def _create_verified(storage, uri, payload):
    """Conditionally create or verify an identical concurrent retry; never overwrite."""
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        # A competing writer is acceptable only if its actual bytes match.
        existing = _read(storage, uri)
        if existing != payload:
            raise ValueError("Archive prefix contains conflicting bytes") from None
    if _read(storage, uri) != payload:
        raise ValueError("Archive object readback differs")


def _archive_payloads(storage, prefix, frozen, files):
    for name, entry in files.items():
        content = (frozen / name).read_bytes()
        if len(content) != entry["size"] or digest(content) != entry["sha256"]:
            raise ValueError("Frozen source changed")
        _create_verified(storage, f"{prefix}/files/{name}", content)
    for name, entry in files.items():
        content = _read(storage, f"{prefix}/files/{name}")
        if len(content) != entry["size"] or digest(content) != entry["sha256"]:
            raise ValueError("Archive payload changed before completion")


def _recheck_source(source, descriptor, files, signatures):
    # The completed marker must not describe a tree changed during storage I/O.
    if _scan(descriptor) != (files, signatures):
        raise ValueError("Source changed before archive completion")
    current = _directory(source)
    try:
        if _signature(os.fstat(current)) != signatures[""]:
            raise ValueError("Source root changed before archive completion")
    finally:
        os.close(current)


def archive(source: Path, output_uri: str, storage: StorageClient) -> dict:
    """Publish one complete, quiescent CLIP result tree without overwriting objects.

    Args:
        source: Absolute, completed result directory with no active writers.
        output_uri: Explicit private S3 prefix reserved for this result.
        storage: Existing configured NPA storage client.
    Returns:
        Non-identifying completion-manifest hash, result format and file counts.
    Raises:
        ValueError: Source validation, immutability or byte verification fails.
        OSError: Local files cannot be safely accessed.
        StorageError, BotoCoreError, ClientError: Storage fails; partial objects may remain.
    """
    prefix = _prefix(output_uri)
    descriptor = _directory(source)
    try:
        with tempfile.TemporaryDirectory(prefix="clip-archive-") as temporary:
            frozen = Path(temporary)
            files, signatures = _scan(descriptor, frozen)
            kind = validate_result(frozen, files)
            manifest = {"schema": SCHEMA, "format": kind, "files": files}
            payload = canonical(manifest)
            _archive_payloads(storage, prefix, frozen, files)
            _recheck_source(source, descriptor, files, signatures)
            _create_verified(storage, f"{prefix}/complete.json", payload)
            return _receipt(manifest, payload)
    finally:
        os.close(descriptor)


def _manifest(payload, expected_digest):
    if digest(payload) != require_digest(expected_digest):
        raise ValueError("Completion manifest hash differs")
    manifest = read_json(payload)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "format", "files"}
            or manifest["schema"] != SCHEMA or manifest["format"] not in {"basic", "advanced"}
            or not isinstance(manifest["files"], dict) or not manifest["files"]):
        raise ValueError("Invalid archive completion manifest")
    for name, entry in manifest["files"].items():
        safe_name(name)
        if (not isinstance(entry, dict) or set(entry) != {"sha256", "size"}
                or type(entry["size"]) is not int or entry["size"] < 0):
            raise ValueError("Invalid archive file entry")
        require_digest(entry["sha256"])
        if any(str(parent) in manifest["files"] for parent in Path(name).parents if str(parent) != "."):
            raise ValueError("Archive file conflicts with a directory")
    if payload != canonical(manifest):
        raise ValueError("Noncanonical completion manifest")
    return manifest


def _expose(parent, staging, destination):
    """Atomically publish on Linux without replacing even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError("Restore requires Linux renameat2 with RENAME_NOREPLACE")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # Complete fallible preparation before the single publication boundary.
    os.fsync(parent)
    if rename(parent, os.fsencode(staging), parent, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("Restore destination already exists")
        raise OSError(code, "Atomic restore publication failed")


def _restore_payloads(storage, prefix, staging, files):
    for name, entry in files.items():
        content = _read(storage, f"{prefix}/files/{name}")
        if len(content) != entry["size"] or digest(content) != entry["sha256"]:
            raise ValueError("Archived payload is corrupt")
        target = staging / name
        directory = staging
        for component in Path(name).parts[:-1]:
            directory /= component
            directory.mkdir(mode=0o700, exist_ok=True)
        with target.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())


def _validate_staging(staging, manifest):
    kind = validate_result(staging, manifest["files"])
    descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        observed, _ = _scan(descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if kind != manifest["format"] or observed != manifest["files"]:
        raise ValueError("Restored inventory differs")


def _recheck_restore_parent(destination, info):
    current = _directory(destination.parent)
    try:
        current_info = os.fstat(current)
        if ((current_info.st_dev, current_info.st_ino) != (info.st_dev, info.st_ino)
                or current_info.st_uid != os.getuid() or current_info.st_mode & 0o077):
            raise ValueError("Restore parent changed before publication")
    finally:
        os.close(current)


def _restore_parent(parent, destination):
    info = os.fstat(parent)
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Restore parent must be owned by this user with mode 0700")
    if os.path.lexists(f"/proc/self/fd/{parent}/{destination.name}"):
        raise FileExistsError("Restore destination already exists")
    return info


def restore(input_uri: str, destination: Path, manifest_sha256: str, storage: StorageClient) -> dict:
    """Restore listed, hash-bound objects through private staging into a new directory.

    Args:
        input_uri: Explicit archive prefix previously passed to archive.
        destination: New absolute directory in an existing owner-only parent.
        manifest_sha256: Trusted completion hash returned by archive.
        storage: Existing configured NPA storage client.
    Returns:
        Verified completion identity and counts after atomic publication.
    Raises:
        ValueError: Manifest, payloads or reconstructed result formats differ.
        OSError: Destination is unsafe, already exists or cannot be published.
        StorageError, BotoCoreError, ClientError: Remote data cannot be read.
    """
    prefix = _prefix(input_uri)
    destination = Path(destination)
    safe_name(destination.name)
    parent = _directory(destination.parent)
    staging = None
    try:
        info = _restore_parent(parent, destination)
        payload = _read(storage, f"{prefix}/complete.json")
        manifest = _manifest(payload, manifest_sha256)
        # Anchor staging to the verified open parent, even if its pathname is
        # replaced. Recheck that pathname before publishing through the same fd.
        staging = Path(tempfile.mkdtemp(prefix=".clip-restore-", dir=f"/proc/self/fd/{parent}"))
        _restore_payloads(storage, prefix, staging, manifest["files"])
        _validate_staging(staging, manifest)
        _recheck_restore_parent(destination, info)
        _expose(parent, staging.name, destination.name)
        staging = None
        return _receipt(manifest, payload)
    finally:
        try:
            if staging is not None:
                shutil.rmtree(staging)
        finally:
            os.close(parent)


def _receipt(manifest, payload):
    return {"manifest_sha256": digest(payload), "format": manifest["format"],
            "files": len(manifest["files"]),
            "bytes": sum(entry["size"] for entry in manifest["files"].values())}


def main(argv=None) -> int:
    """Parse the optional companion CLI; keep paths and provider failures private.

    Args:
        argv: Argument strings, or None to use the process arguments.
    Returns:
        Zero on success, one for an expected failure, or two for an unexpected error.
    Raises:
        SystemExit: Argument parsing requests help or rejects invalid arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("archive", help="Archive a completed quiescent local tree")
    upload.add_argument("--input-path", required=True, type=Path)
    upload.add_argument("--output-path", required=True)
    download = commands.add_parser("restore", help="Verify and restore into a new private directory")
    download.add_argument("--input-path", required=True)
    download.add_argument("--output-path", required=True, type=Path)
    download.add_argument("--manifest-sha256", required=True)
    options = parser.parse_args(argv)
    try:
        storage = StorageClient.from_environment()
        if options.command == "archive":
            result = archive(options.input_path, options.output_path, storage)
        else:
            result = restore(options.input_path, options.output_path, options.manifest_sha256, storage)
    except (ValueError, OSError, StorageError, BotoCoreError, ClientError):
        print("CLIP archive operation failed; verify source, destination, integrity and storage access.", file=sys.stderr)
        return 1
    except Exception:
        # Native format readers can include private paths in unexpected errors.
        # Keep their details private while distinguishing a bug from bad input.
        print("CLIP archive encountered an unexpected error; report it with private diagnostic evidence.", file=sys.stderr)
        return 2
    print(canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
