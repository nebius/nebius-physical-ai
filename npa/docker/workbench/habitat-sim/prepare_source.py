#!/usr/bin/env python3
"""Fetch and materialize the exact public Habitat-Sim build source closure."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import BinaryIO, Callable
import urllib.parse
import urllib.request


class SourceError(ValueError):
    """Report an immutable source or projection boundary failure.

    Args:
        *args: Error details forwarded to ValueError.

    Returns:
        None.

    Raises:
        None.
    """


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise SourceError("source archive redirects are not accepted")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SourceError(f"{label} path is unsafe")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise SourceError(f"{label} path is unsafe")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise SourceError(f"{label} path is unsafe")
    return PurePosixPath(value)


def _component_name(value: object) -> str:
    relative = _relative_path(value, "source component")
    if len(relative.parts) != 1:
        raise SourceError("source component must be one path component")
    return relative.as_posix()


def _dependency_relative(value: object) -> PurePosixPath:
    relative = _relative_path(value, "dependency target")
    if relative.parts[:2] != ("src", "deps") or len(relative.parts) < 3:
        raise SourceError("dependency target must be a src/deps descendant")
    return relative


def _validate_path_list(values: object, label: str) -> None:
    if not isinstance(values, list):
        raise SourceError(f"{label} paths must be a list")
    for value in values:
        _relative_path(value, label)


def _validate_archive_paths(item: dict[str, object]) -> None:
    _component_name(item.get("name", "habitat-sim"))
    _relative_path(item["license_path"], "source license")
    _validate_path_list(item.get("excluded_archive_links", []), "excluded link")


def _validate_dependency_paths(manifest: dict[str, object]) -> None:
    names = {"habitat-sim"}
    targets: list[PurePosixPath] = []
    for item in manifest["dependencies"]:
        _validate_archive_paths(item)
        name = _component_name(item["name"])
        target = _dependency_relative(item["target"])
        if name in names or any(
            target.is_relative_to(old) or old.is_relative_to(target) for old in targets
        ):
            raise SourceError("source dependency names and targets must not overlap")
        names.add(name)
        targets.append(target)
        if name not in manifest["dependency_projection"]:
            raise SourceError("source dependency projection is missing")
    for name, selected in manifest["dependency_projection"].items():
        _component_name(name)
        _validate_path_list(selected, "dependency projection")
        for value in selected:
            _component_name(value)


def _validate_manifest_paths(manifest: dict[str, object]) -> None:
    """Reject untrusted path spellings before opening output or fetching bytes."""
    try:
        if manifest["schema_version"] != "npa.habitat-sim.source-manifest.v1":
            raise SourceError("unsupported source manifest")
        source = manifest["source"]
        _validate_archive_paths(source)
        for item in source["metadata_patches"]:
            _relative_path(item["path"], "source metadata patch")
        for item in source["required_projection_files"]:
            _relative_path(item["path"], "required source projection")
        _validate_path_list(source["final_projection"], "final projection")
        _validate_dependency_paths(manifest)
        for item in manifest["internal_vendored"]:
            _dependency_relative(item["root"])
            for member in item["files"]:
                _relative_path(member["path"], "vendored source member")
        _validate_path_list(manifest["forbidden_paths"], "forbidden source")
    except (KeyError, TypeError, AttributeError) as error:
        raise SourceError("source manifest path records are malformed") from error


def _owned_staging_entry(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise SourceError("source staging path must be owned and not shared or linked")


def _contained_source_path(root: Path, value: object, label: str) -> Path:
    relative = _relative_path(value, label)
    _owned_staging_entry(root)
    if not root.is_dir():
        raise SourceError("source staging root must be a directory")
    target = root
    for part in relative.parts:
        target = target / part
        try:
            _owned_staging_entry(target)
        except FileNotFoundError:
            continue
    if not target.resolve().is_relative_to(root.resolve()):
        raise SourceError("source staging path escapes the owned root")
    return target


def _archive_request(item: dict[str, object]) -> tuple[urllib.request.Request, int]:
    url = str(item["archive_url"])
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "codeload.github.com":
        raise SourceError("source archive must use official GitHub codeload HTTPS")
    expected_bytes = item.get("archive_bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise SourceError("source archive byte count must be a positive integer")
    request = urllib.request.Request(
        url, headers={"User-Agent": "npa-source-preparer/1"}
    )
    return request, expected_bytes


def _verify_response(response: BinaryIO, url: str, expected_bytes: int) -> None:
    if response.geturl() != url:
        raise SourceError("source archive redirected away from pinned locator")
    headers = getattr(response, "headers", None)
    declared_length = headers.get("Content-Length") if headers is not None else None
    if declared_length is None:
        return
    try:
        declared_bytes = int(declared_length)
    except (TypeError, ValueError) as error:
        raise SourceError("source archive Content-Length is invalid") from error
    if declared_bytes < 0 or declared_bytes != expected_bytes:
        raise SourceError("source archive Content-Length does not match pinned size")


def _stream_archive(
    response: BinaryIO, stream: BinaryIO, expected: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        remaining = expected - size
        chunk = response.read(min(1024 * 1024, remaining + 1))
        if not chunk:
            return size, digest.hexdigest()
        if len(chunk) > remaining:
            raise SourceError("source archive exceeds pinned byte count")
        digest.update(chunk)
        stream.write(chunk)
        size += len(chunk)


def _download(
    item: dict[str, object],
    directory: Path,
    opener: Callable[..., BinaryIO] | None = None,
) -> Path:
    request, expected_bytes = _archive_request(item)
    name = _component_name(item.get("name", "habitat-sim"))
    target = _contained_source_path(directory, f"{name}.tar.gz", "source archive")
    open_request = opener or urllib.request.build_opener(_RefuseRedirect()).open
    created_target = False
    try:
        with open_request(request, timeout=120) as response:
            _verify_response(response, request.full_url, expected_bytes)
            with target.open("xb") as stream:
                created_target = True
                size, digest = _stream_archive(response, stream, expected_bytes)
        if size != expected_bytes or digest != item["archive_sha256"]:
            raise SourceError("source archive size or SHA-256 mismatch")
    except Exception:
        if created_target:
            target.unlink(missing_ok=True)
        raise
    return target


def _archive_root(members: list[tarfile.TarInfo]) -> str:
    if not members:
        raise SourceError("source archive is empty")
    roots: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SourceError("unsafe source archive path")
        roots.add(path.parts[0])
        if member.isdev() or member.isfifo():
            raise SourceError("source archive contains unsupported link or device")
        if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise SourceError("source archive contains unsupported member type")
    if len(roots) != 1:
        raise SourceError("source archive must have one top-level directory")
    return roots.pop()


def _safe_members(
    archive: tarfile.TarFile, excluded_links: list[str]
) -> tuple[str, list[tarfile.TarInfo]]:
    members = archive.getmembers()
    root = _archive_root(members)
    excluded = set(excluded_links)
    observed: set[str] = set()
    selected: list[tarfile.TarInfo] = []
    for member in members:
        relative = PurePosixPath(member.name).relative_to(root).as_posix()
        if member.issym() or member.islnk():
            if relative not in excluded:
                raise SourceError(f"undeclared source archive link: {relative}")
            observed.add(relative)
            continue
        selected.append(member)
    if observed != excluded:
        raise SourceError("declared source archive link was not present")
    return root, selected


def _extract(
    archive_path: Path, target: Path, excluded_links: list[str] | None = None
) -> None:
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        root, members = _safe_members(archive, excluded_links or [])
        for member in members:
            relative = PurePosixPath(member.name).relative_to(root)
            if not relative.parts:
                continue
            destination = target.joinpath(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SourceError("regular source member cannot be read")
            with destination.open("xb") as stream:
                shutil.copyfileobj(source, stream)


def _verify_license(root: Path, item: dict[str, object]) -> None:
    path = _contained_source_path(root, item["license_path"], "source license")
    if not path.is_file() or _sha(path) != item["license_sha256"]:
        raise SourceError(
            f"license bytes changed for {item.get('name', 'habitat-sim')}"
        )
    expected_size = item.get("license_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise SourceError("source license size changed")


def _verify_required_source_files(
    root: Path, required: list[dict[str, object]]
) -> None:
    for item in required:
        relative = _relative_path(item["path"], "required source projection")
        path = _contained_source_path(root, item["path"], "required source projection")
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _sha(path) != item["sha256"]
        ):
            raise SourceError(f"required source projection file changed: {relative}")


def _patched_metadata(original: bytes, patch: dict[str, object]) -> bytes:
    relative = patch["path"]
    if (
        len(original) != patch["upstream_file_bytes"]
        or hashlib.sha256(original).hexdigest() != patch["upstream_file_sha256"]
    ):
        raise SourceError(f"source metadata patch input changed: {relative}")
    preimage = str(patch["preimage_utf8"]).encode("utf-8")
    postimage = str(patch["postimage_utf8"]).encode("utf-8")
    if (
        not preimage
        or hashlib.sha256(preimage).hexdigest() != patch["preimage_sha256"]
        or hashlib.sha256(postimage).hexdigest() != patch["postimage_sha256"]
    ):
        raise SourceError(f"source metadata patch bytes changed: {relative}")
    if patch["expected_match_count"] != 1 or original.count(preimage) != 1:
        raise SourceError(f"source metadata patch preimage changed: {relative}")
    patched = original.replace(preimage, postimage)
    if (
        len(patched) != patch["patched_file_bytes"]
        or hashlib.sha256(patched).hexdigest() != patch["patched_file_sha256"]
    ):
        raise SourceError(f"source metadata patch result changed: {relative}")
    return patched


def _apply_metadata_patches(root: Path, patches: list[dict[str, object]]) -> None:
    for patch in patches:
        path = _contained_source_path(root, patch["path"], "source metadata patch")
        if not path.is_file():
            raise SourceError(
                f"source metadata patch target is invalid: {patch['path']}"
            )
        path.write_bytes(_patched_metadata(path.read_bytes(), patch))


def _required_pbr_paths(required: list[dict[str, object]]) -> set[PurePosixPath]:
    selected: set[PurePosixPath] = set()
    for item in required:
        relative = _relative_path(item["path"], "required PBR projection")
        try:
            pbr_relative = relative.relative_to("data/pbr")
        except ValueError as error:
            raise SourceError(
                "required PBR projection path is outside data/pbr"
            ) from error
        if (
            not pbr_relative.parts
            or ".." in pbr_relative.parts
            or pbr_relative in selected
        ):
            raise SourceError("required PBR projection path is empty or duplicated")
        selected.add(pbr_relative)
    if not selected:
        raise SourceError("required PBR projection must not be empty")
    return selected


def _prune_pbr(root: Path, required: list[dict[str, object]]) -> None:
    pbr = root / "data/pbr"
    selected = _required_pbr_paths(required)
    for path in pbr.rglob("*"):
        relative = PurePosixPath(path.relative_to(pbr).as_posix())
        if path.is_file() and relative not in selected:
            path.unlink()
    directories = sorted(
        (path for path in pbr.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        if not any(path.iterdir()):
            path.rmdir()
    observed = {
        PurePosixPath(path.relative_to(pbr).as_posix())
        for path in pbr.rglob("*")
        if path.is_file()
    }
    if observed != selected:
        raise SourceError("required PBR projection changed during pruning")


def _prune_parent(root: Path, required: list[dict[str, object]]) -> None:
    allowed_top = {
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "src",
        "src_python",
        "data",
    }
    for path in list(root.iterdir()):
        if path.name not in allowed_top:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    data = root / "data"
    for path in list(data.iterdir()):
        if path.name not in {"default.physics_config.json", "pbr"}:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    _prune_pbr(root, required)
    for relative in ("src/deps/rlr-audio-propagation", "src/deps/glfw"):
        shutil.rmtree(root / relative, ignore_errors=True)
    basis = root / "src/deps/basis-universal"
    for path in list(basis.iterdir()):
        if path.name != "transcoder":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for relative in ("src/tests", "src/utils/viewer"):
        shutil.rmtree(root / relative, ignore_errors=True)


def _prune_dependency(root: Path, allowed: list[str]) -> None:
    selected = set(allowed)
    if not selected:
        raise SourceError("dependency projection must not be empty")
    for path in list(root.iterdir()):
        if path.name not in selected:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    missing = selected - {path.name for path in root.iterdir()}
    if missing:
        raise SourceError("dependency projection path is missing")


def _assert_forbidden(root: Path, forbidden: list[str]) -> None:
    for relative in forbidden:
        if _contained_source_path(root, relative, "forbidden source").exists():
            raise SourceError(f"forbidden source path survived: {relative}")
    names = "\n".join(
        path.relative_to(root).as_posix().lower() for path in root.rglob("*")
    )
    for token in ("rlr-audio",):
        if token in names:
            raise SourceError(f"forbidden source token survived: {token}")


def _verify_internal_vendored(root: Path, components: list[dict[str, object]]) -> None:
    for component in components:
        relative = _dependency_relative(component["root"])
        selected_root = _contained_source_path(root, str(relative), "vendored root")
        expected = {
            str(_relative_path(row["path"], "vendored source member")): str(
                row["sha256"]
            )
            for row in component["files"]
        }
        observed = {
            path.relative_to(selected_root).as_posix(): _sha(path)
            for path in selected_root.rglob("*")
            if path.is_file()
        }
        if observed != expected:
            raise SourceError(
                f"vendored source closure changed for {component['name']}"
            )


def _inventory(root: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SourceError("source projection must not contain symbolic links")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    return {
        "schema_version": "npa.habitat-sim.source-projection.v1",
        "file_count": len(files),
        "files": files,
    }


def _materialize_parent(manifest: dict[str, object], output: Path, temp: Path) -> None:
    source = dict(manifest["source"])
    source["name"] = "habitat-sim"
    parent_archive = _download(source, temp)
    _extract(parent_archive, output, source.get("excluded_archive_links", []))
    _verify_license(output, source)
    _apply_metadata_patches(output, source["metadata_patches"])
    _verify_required_source_files(output, source["required_projection_files"])
    _prune_parent(output, source["required_projection_files"])
    _verify_required_source_files(output, source["required_projection_files"])
    _verify_internal_vendored(output, manifest["internal_vendored"])


def _materialize_dependency(
    dependency: dict[str, object], output: Path, temp: Path, selected: list[str]
) -> None:
    relative = _dependency_relative(dependency["target"])
    target = _contained_source_path(output, str(relative), "dependency target")
    archive = _download(dependency, temp)
    if target.exists():
        shutil.rmtree(target)
    _extract(archive, target, dependency.get("excluded_archive_links", []))
    _verify_license(target, dependency)
    _prune_dependency(target, selected)


def _materialize(manifest: dict[str, object], output: Path, temp: Path) -> bytes:
    """Verify the complete projection in private staging before publication."""
    _validate_manifest_paths(manifest)
    _materialize_parent(manifest, output, temp)
    for dependency in manifest["dependencies"]:
        selected = manifest["dependency_projection"][dependency["name"]]
        _materialize_dependency(dependency, output, temp, selected)
    _assert_forbidden(output, manifest["forbidden_paths"])
    inventory = _inventory(output)
    inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    expected = manifest["expected_projection"]
    if (
        inventory["file_count"] != expected["file_count"]
        or hashlib.sha256(inventory_bytes).hexdigest() != expected["inventory_sha256"]
    ):
        raise SourceError("materialized source projection changed")
    return inventory_bytes


def _owned_destination(path: Path, stack: ExitStack) -> Path:
    if path.name in {"", ".", ".."}:
        raise SourceError("invalid source output name")
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    stack.callback(os.close, descriptor)
    info = os.fstat(descriptor)
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise SourceError("source output parent must be owned and not shared writable")
    destination = Path(f"/proc/self/fd/{descriptor}") / path.name
    if os.path.lexists(destination):
        raise SourceError("source output already exists")
    return destination


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish on Linux atomically, refusing even a concurrent empty directory."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise SourceError("atomic no-clobber directory publication is unavailable")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        raise OSError(ctypes.get_errno(), "source publication refused")


def _publish_projection(staged: Path, output: Path, inventory: Path | None) -> None:
    inventory_source = staged.parent / "inventory.json"
    linked = False
    try:
        if inventory is not None:
            os.link(inventory_source, inventory, follow_symlinks=False)
            linked = True
        _rename_noreplace(staged, output)
    except BaseException:
        if linked:
            try:
                observed, owned = inventory.lstat(), inventory_source.lstat()
                if (observed.st_dev, observed.st_ino) == (owned.st_dev, owned.st_ino):
                    inventory.unlink()
            except OSError:
                print("warning: source inventory cleanup failed", file=sys.stderr)
        raise


def _stage_owned_projection(
    manifest: dict[str, object], output: Path, inventory: Path | None
) -> None:
    temp = Path(tempfile.mkdtemp(prefix=".npa-habitat-source-", dir=output.parent))
    try:
        staged = temp / "projection"
        archives = temp / "archives"
        archives.mkdir(mode=0o700)
        inventory_bytes = _materialize(manifest, staged, archives)
        (temp / "inventory.json").write_bytes(inventory_bytes)
        _publish_projection(staged, output, inventory)
    finally:
        try:
            shutil.rmtree(temp)
        except OSError:
            print("warning: temporary source staging cleanup failed", file=sys.stderr)


def stage(
    manifest: dict[str, object], output: Path, inventory_output: Path | None = None
) -> None:
    """Expose only a fully verified projection without replacing existing output.

    Args:
        manifest: Immutable source records with canonical relative member paths.
        output: New projection directory beneath an owned, non-shared parent.
        inventory_output: Optional new file for the verified projection inventory.

    Returns:
        None.

    Raises:
        SourceError: A manifest, source, staging or projection check fails.
        OSError: Source access or atomic no-clobber publication fails.
    """
    _validate_manifest_paths(manifest)
    with ExitStack() as stack:
        output = _owned_destination(output, stack)
        inventory = (
            _owned_destination(inventory_output, stack)
            if inventory_output is not None
            else None
        )
        _stage_owned_projection(manifest, output, inventory)


def main(argv: list[str] | None = None) -> int:
    """Validate the source manifest and publish its verified source projection.

    Args:
        argv: CLI arguments, or None to read the process arguments.

    Returns:
        Zero after successful verified publication.

    Raises:
        SourceError: The manifest or source projection fails validation.
        OSError: Manifest access, staging or publication fails.
        ValueError: The manifest is not valid JSON.
        SystemExit: Argument parsing fails or help is requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "npa.habitat-sim.source-manifest.v1":
        raise SourceError("unsupported source manifest")
    stage(manifest, args.output, args.inventory_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
