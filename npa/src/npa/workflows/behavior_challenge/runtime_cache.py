"""Restore immutable BEHAVIOR runtime archives into operator-selected paths."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_SCHEMA = "npa.behavior.private-runtime.v1"
_READY_SCHEMA = "npa.behavior.runtime-ready.v2"
_MARKER_SCHEMA = "npa.behavior.runtime-archive-marker.v1"
_STATE_DIRECTORY = ".npa-runtime-cache"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _s3_uri(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact S3 object URI")
    parsed = urlparse(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError(f"{label} must be an exact S3 object URI")
    return value


def _safe_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe POSIX path relative to home")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe POSIX path relative to home")
    return path


def _archive(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"uri", "sha256", "bytes"}:
        raise ValueError("Each runtime archive requires uri, sha256, and bytes")
    if not isinstance(value["sha256"], str) or not _SHA256.fullmatch(value["sha256"]):
        raise ValueError("Runtime archive SHA-256 is invalid")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ValueError("Runtime archive byte count must be positive")
    return {
        "uri": _s3_uri(value["uri"], "archive uri"),
        "sha256": value["sha256"],
        "bytes": value["bytes"],
    }


def _validate_manifest(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "allowed_directories", "archives"}
        or value.get("schema") != _MANIFEST_SCHEMA
    ):
        raise ValueError("Unsupported or malformed private runtime manifest")
    directories = value["allowed_directories"]
    archives = value["archives"]
    if not isinstance(directories, list) or len(directories) != 2:
        raise ValueError("Runtime manifest requires workspace and Python directories")
    allowed = [_safe_relative(item, "allowed directory") for item in directories]
    if len(set(allowed)) != len(allowed) or _paths_overlap(allowed[0], allowed[1]):
        raise ValueError("Runtime allowed directories must be distinct and disjoint")
    if not isinstance(archives, list) or not archives:
        raise ValueError("Runtime manifest requires at least one archive")
    frozen = [_archive(item) for item in archives]
    if len({item["sha256"] for item in frozen}) != len(frozen):
        raise ValueError("Runtime archive identities must be unique")
    return {
        "schema": _MANIFEST_SCHEMA,
        "allowed_directories": [str(path) for path in allowed],
        "archives": frozen,
    }


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _basic_operator_paths(home: Path, workspace: Path) -> tuple[Path, Path]:
    if home.is_symlink() or not home.is_absolute() or not home.is_dir():
        raise ValueError(
            "target home must be an existing absolute non-symlink directory"
        )
    resolved_home = home.resolve()
    if workspace.is_symlink() or not workspace.is_absolute():
        raise ValueError("workspace must be an absolute non-symlink path")
    try:
        workspace.relative_to(home)
    except ValueError as exc:
        raise ValueError("workspace must be contained by target home") from exc
    _reject_symlink_ancestors(workspace, home)
    resolved_workspace = workspace.resolve()
    if not resolved_workspace.is_relative_to(resolved_home):
        raise ValueError("workspace must be contained by target home")
    return resolved_home, resolved_workspace


def _operator_paths(
    home: Path, workspace: Path, manifest: dict[str, Any]
) -> tuple[Path, Path, list[PurePosixPath]]:
    resolved_home, resolved_workspace = _basic_operator_paths(home, workspace)
    relative_workspace = PurePosixPath(
        resolved_workspace.relative_to(resolved_home).as_posix()
    )
    allowed = [PurePosixPath(item) for item in manifest["allowed_directories"]]
    if relative_workspace not in allowed:
        raise ValueError(
            "Manifest allowed directories do not bind the operator workspace"
        )
    return resolved_home, resolved_workspace, allowed


def _checked_target(
    name: str, home: Path, allowed: list[PurePosixPath]
) -> tuple[PurePosixPath, Path]:
    relative = _safe_relative(name, "archive member")
    if _STATE_DIRECTORY in relative.parts:
        raise ValueError("Archive member targets reserved runtime cache state")
    if not any(relative.is_relative_to(directory) for directory in allowed):
        raise ValueError("Archive member is outside manifest-allowed directories")
    target = home.joinpath(*relative.parts)
    resolved = target.resolve(strict=False)
    roots = [
        home.joinpath(*directory.parts).resolve(strict=False) for directory in allowed
    ]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Archive member escapes its allowed destination")
    _reject_symlink_ancestors(target, home)
    return relative, target


def _reject_symlink_ancestors(target: Path, home: Path) -> None:
    current = home
    for part in target.relative_to(home).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Archive destination contains a filesystem symlink")


def _restore_hardlink(
    member: tarfile.TarInfo, home: Path, allowed: list[PurePosixPath]
) -> None:
    _, destination = _checked_target(member.name, home, allowed)
    _, original = _checked_target(member.linkname, home, allowed)
    if destination == original or not original.is_file() or original.is_symlink():
        raise ValueError("Archive hardlink target must be an extracted regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise ValueError("Archive hardlink destination has a conflicting type")
        destination.unlink()
    try:
        os.link(original, destination, follow_symlinks=False)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(original, destination, follow_symlinks=False)
        shutil.copystat(original, destination, follow_symlinks=False)


def _extract_archive(archive: Path, home: Path, allowed: list[PurePosixPath]) -> None:
    names: set[PurePosixPath] = set()
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            relative, target = _checked_target(member.name, home, allowed)
            if relative in names:
                raise ValueError("Runtime archive contains a duplicate member")
            names.add(relative)
            if not (member.isfile() or member.isdir() or member.islnk()):
                raise ValueError("Runtime archive contains an unsupported member type")
            if member.islnk():
                _checked_target(member.linkname, home, allowed)
                filtered = tarfile.data_filter(member, str(home))
                if filtered is None:
                    raise ValueError("Runtime archive hardlink was rejected")
                _restore_hardlink(filtered, home, allowed)
                continue
            if target.exists() and target.is_symlink():
                raise ValueError("Runtime archive destination is a symlink")
            source.extract(member, home, filter="data")


def _tree_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Restored Python base must be a non-symlink directory")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Restored Python base cannot contain symlinks")
        relative = path.relative_to(root).as_posix()
        mode = path.stat().st_mode & 0o777
        if path.is_dir():
            rows.append({"path": relative, "type": "directory", "mode": mode})
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": _digest(path),
                }
            )
        else:
            raise ValueError("Restored Python base contains a special file")
    return _canonical_digest(rows)


def _snapshot_python(python_target: Path, cache: Path) -> str:
    if cache.exists() or cache.is_symlink():
        raise ValueError("Python base cache already exists before snapshot")
    temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copytree(python_target, temporary)
        digest = _tree_digest(temporary)
        os.replace(temporary, cache)
        return digest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _restore_python(python_target: Path, cache: Path, expected_sha256: str) -> None:
    if python_target.exists() or python_target.is_symlink():
        raise ValueError("Fresh worker Python destination is unexpectedly occupied")
    if _tree_digest(cache) != expected_sha256:
        raise ValueError("Immutable Python base cache identity differs")
    python_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache, python_target)
    if _tree_digest(python_target) != expected_sha256:
        raise ValueError("Restored Python base identity differs")


def _download_verified(storage: Any, item: dict[str, Any], target: Path) -> None:
    if target.exists() and (
        target.stat().st_size != item["bytes"] or _digest(target) != item["sha256"]
    ):
        target.unlink()
    if not target.exists():
        storage.download_file(item["uri"], str(target))
    if target.stat().st_size != item["bytes"] or _digest(target) != item["sha256"]:
        raise ValueError("Downloaded runtime archive identity differs")


def _marker_value(
    index: int, item: dict[str, Any], state: str, python_sha256: str | None = None
) -> dict[str, Any]:
    value = {
        "schema": _MARKER_SCHEMA,
        "archive_index": index,
        "archive_sha256": item["sha256"],
        "archive_bytes": item["bytes"],
        "state": state,
    }
    if python_sha256 is not None:
        value["python_base_sha256"] = python_sha256
    return value


def _marker_state(path: Path, index: int, item: dict[str, Any]) -> dict | None:
    if not path.exists():
        return None
    value = _load_json(path, "runtime archive marker")
    restoring = _marker_value(index, item, "restoring")
    if value == restoring:
        return value
    if index == 0 and value.get("python_base_sha256"):
        complete = _marker_value(index, item, "complete", value["python_base_sha256"])
    else:
        complete = _marker_value(index, item, "complete")
    if value != complete:
        raise ValueError("Runtime archive marker identity differs")
    return value


def _retry_cleanup(index: int, python_target: Path, python_cache: Path) -> None:
    if index != 0:
        return
    for path in (python_target, python_cache):
        if path.is_symlink():
            raise ValueError("Interrupted Python restore left a symlink")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise ValueError("Interrupted Python restore left a conflicting file")


def _restore_archive(
    storage: Any,
    item: dict[str, Any],
    index: int,
    state_root: Path,
    home: Path,
    allowed: list[PurePosixPath],
    python_target: Path,
    python_cache: Path,
) -> dict[str, Any]:
    marker = state_root / f"archive-{index}-{item['sha256']}.json"
    state = _marker_state(marker, index, item)
    if state and state["state"] == "complete":
        return state
    if state:
        _retry_cleanup(index, python_target, python_cache)
    elif index == 0 and (python_target.exists() or python_target.is_symlink()):
        raise ValueError("Fresh worker Python destination is unexpectedly occupied")
    _atomic_json(marker, _marker_value(index, item, "restoring"))
    archive = state_root / f"archive-{index}-{item['sha256']}.tar.gz"
    _download_verified(storage, item, archive)
    _extract_archive(archive, home, allowed)
    python_sha = _snapshot_python(python_target, python_cache) if index == 0 else None
    complete = _marker_value(index, item, "complete", python_sha)
    _atomic_json(marker, complete)
    archive.unlink()
    return complete


def _ready_receipt(
    manifest_sha256: str,
    manifest: dict[str, Any],
    python_directory: PurePosixPath,
    python_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": _READY_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "allowed_directories": manifest["allowed_directories"],
        "python_directory": str(python_directory),
        "python_base_sha256": python_sha256,
        "archives": [
            {"sha256": item["sha256"], "bytes": item["bytes"]}
            for item in manifest["archives"]
        ],
    }


def _reuse_ready(
    ready_path: Path,
    expected: dict[str, Any],
    python_target: Path,
    python_cache: Path,
) -> dict[str, Any]:
    actual = _load_json(ready_path, "runtime ready receipt")
    if actual != expected:
        raise ValueError("Runtime ready receipt identity differs")
    _restore_python(python_target, python_cache, expected["python_base_sha256"])
    return actual


def _download_manifest(
    storage: Any, uri: str, expected_sha256: str, home: Path
) -> bytes:
    with tempfile.TemporaryDirectory(
        prefix=".npa-runtime-manifest-", dir=home
    ) as temporary:
        path = Path(temporary) / "manifest.json"
        storage.download_file(uri, str(path))
        if _digest(path) != expected_sha256:
            raise ValueError("Runtime manifest SHA-256 differs")
        return path.read_bytes()


def _prepare_state(workspace: Path, manifest_bytes: bytes) -> Path:
    if workspace.exists() and not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    workspace.mkdir(parents=True, exist_ok=True)
    state_root = workspace / _STATE_DIRECTORY
    if state_root.is_symlink() or (state_root.exists() and not state_root.is_dir()):
        raise ValueError("Runtime cache state path is not a safe directory")
    state_root.mkdir(exist_ok=True)
    _atomic_bytes(state_root / "runtime-manifest.json", manifest_bytes)
    return state_root


def _restore_archives(
    storage: Any,
    manifest: dict[str, Any],
    state_root: Path,
    home: Path,
    allowed: list[PurePosixPath],
    python_target: Path,
    python_cache: Path,
) -> list[dict[str, Any]]:
    return [
        _restore_archive(
            storage,
            item,
            index,
            state_root,
            home,
            allowed,
            python_target,
            python_cache,
        )
        for index, item in enumerate(manifest["archives"])
    ]


def _ensure_python_target(target: Path, cache: Path, expected_sha256: str) -> None:
    if not target.exists():
        _restore_python(target, cache, expected_sha256)
    elif target.is_symlink() or _tree_digest(target) != expected_sha256:
        raise ValueError("Restored Python base identity differs")


def _runtime_context(
    storage: Any,
    manifest_uri: str,
    manifest_sha256: str,
    home: Path,
    workspace: Path,
) -> tuple[dict[str, Any], Path, Path, list[PurePosixPath], Path]:
    _s3_uri(manifest_uri, "manifest_uri")
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256")
    resolved_home, _ = _basic_operator_paths(home, workspace)
    manifest_bytes = _download_manifest(
        storage, manifest_uri, manifest_sha256, resolved_home
    )
    try:
        manifest_value = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("runtime manifest is not valid JSON") from exc
    manifest = _validate_manifest(manifest_value)
    resolved_home, resolved_workspace, allowed = _operator_paths(
        home, workspace, manifest
    )
    state_root = _prepare_state(resolved_workspace, manifest_bytes)
    return manifest, resolved_home, resolved_workspace, allowed, state_root


def _finish_runtime(
    manifest_sha256: str,
    manifest: dict[str, Any],
    python_relative: PurePosixPath,
    python_target: Path,
    python_cache: Path,
    markers: list[dict[str, Any]],
    state_root: Path,
) -> dict[str, Any]:
    python_sha = markers[0].get("python_base_sha256")
    if not _valid_python_sha(python_sha) or _tree_digest(python_cache) != python_sha:
        raise ValueError("Runtime Python base cache is incomplete or changed")
    expected = _ready_receipt(manifest_sha256, manifest, python_relative, python_sha)
    ready_path = state_root / "runtime-ready.json"
    if ready_path.exists():
        return _reuse_ready(ready_path, expected, python_target, python_cache)
    _ensure_python_target(python_target, python_cache, python_sha)
    _atomic_json(ready_path, expected)
    return expected


def _prepare_runtime(
    storage_for_manifest: Any,
    storage_for_archives: Any,
    manifest_uri: str,
    manifest_sha256: str,
    home: Path,
    workspace: Path,
) -> dict[str, Any]:
    manifest, resolved_home, resolved_workspace, allowed, state_root = _runtime_context(
        storage_for_manifest, manifest_uri, manifest_sha256, home, workspace
    )
    workspace_relative = PurePosixPath(
        resolved_workspace.relative_to(resolved_home).as_posix()
    )
    python_relative = next(path for path in allowed if path != workspace_relative)
    python_target = resolved_home.joinpath(*python_relative.parts)
    python_cache = state_root / "python-base"
    markers = _restore_archives(
        storage_for_archives,
        manifest,
        state_root,
        resolved_home,
        allowed,
        python_target,
        python_cache,
    )
    return _finish_runtime(
        manifest_sha256,
        manifest,
        python_relative,
        python_target,
        python_cache,
        markers,
        state_root,
    )


def prepare_runtime(
    storage_for_manifest: Any,
    storage_for_archives: Any,
    manifest_uri: str,
    manifest_sha256: str,
    home: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Restore verified runtime archives and an immutable Python base.

    Args:
        storage_for_manifest: Client authorized for the exact manifest object.
        storage_for_archives: Client authorized for the archive objects.
        manifest_uri: Exact S3 manifest object URI.
        manifest_sha256: SHA-256 of the complete manifest bytes.
        home: Existing absolute target home directory.
        workspace: Absolute runtime workspace contained by ``home``.

    Returns:
        Receipt binding the manifest, archives, allowed paths, and Python tree.

    Raises:
        ValueError: An identity, path, archive member, or cache state differs.
        OSError: A storage or filesystem operation fails.
        tarfile.TarError: A runtime archive cannot be decoded.
    """
    return _prepare_runtime(
        storage_for_manifest,
        storage_for_archives,
        manifest_uri,
        manifest_sha256,
        home,
        workspace,
    )


def _valid_python_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
