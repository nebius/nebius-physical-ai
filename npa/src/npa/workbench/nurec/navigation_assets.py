"""Contain and inspect navigation USD dependencies before composing any stage."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from npa.clients.storage import StorageClient, StorageError, safe_s3_download_target

_USD = {".usd", ".usda", ".usdc"}


def contained_file(root: Path, name: str) -> Path:
    """Resolve an existing plain relative file inside a bundle.

    Args:
        root: Bundle containment boundary.
        name: Relative POSIX asset name.
    Returns:
        Contained regular file.
    Raises:
        ValueError: Unsafe, missing, or symbolic file.
    """
    root = root.resolve()
    if not isinstance(name, str) or not name or any(c in name for c in ":\\[]<>"):
        raise ValueError("asset must be a plain relative bundle path")
    try:
        target = safe_s3_download_target(root, name, "")
    except StorageError as exc:
        raise ValueError("asset path escapes its bundle") from exc
    raw = root / name
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("symbolic assets are unsupported")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("bundle asset is missing or empty")
    return target


def sha256(path: Path) -> str:
    """Hash artifact bytes without loading the complete asset into memory.

    Args:
        path: Regular file to hash.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: File cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: str, destination: Path) -> Path:
    """Snapshot a local or contained S3 input directory into private staging.

    Args:
        source: Local directory or S3 prefix.
        destination: Fresh staging directory.
    Returns:
        Staged input directory.
    Raises:
        ValueError: Input is absent, symbolic, or uses an unsupported URI.
        StorageError: S3 inputs cannot be downloaded or contained.
        OSError: Copy fails.
    """
    if not source:
        raise ValueError(
            "input-path requires a contained scene or calibrated scan bundle"
        )
    if source.startswith("s3://"):
        StorageClient.from_environment().download_directory(source, str(destination))
        return destination
    root = Path(source)
    if "://" in source or not root.is_dir() or root.is_symlink():
        raise ValueError("input-path must be a local directory or s3:// prefix")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("input bundle contains a symbolic or special file")
    shutil.copytree(root, destination)
    return destination


def publish(directory: Path, destination: str) -> None:
    """Publish verified artifacts to a new local directory or S3 prefix.

    Args:
        directory: Verified private output directory.
        destination: Local path or S3 prefix.
    Returns:
        None.
    Raises:
        ValueError: Empty, unsupported, or occupied destination.
        StorageError: Conditional S3 publication fails.
        ClientError: The S3 provider rejects an artifact write.
        OSError: Local output exists or copying fails.
    """
    from npa.workbench.nurec.navigation_publication import publish_immutable

    publish_immutable(directory, destination)


def _extract_package(path: Path, root: Path) -> Path:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if not members or len(set(names)) != len(names):
            raise ValueError("USDZ must contain unique members")
        for member in members:
            try:
                target = safe_s3_download_target(root, member.filename, "")
            except StorageError as exc:
                raise ValueError("USDZ member escapes its bundle") from exc
            mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(mode) or member.compress_type != 0:
                raise ValueError("USDZ requires stored regular files only")
            if mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("USDZ special member is unsupported")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    first = contained_file(root, names[0])
    if first.suffix.lower() not in _USD:
        raise ValueError("USDZ first member must be its USD root layer")
    return first


def _layer_dependencies(path: Path) -> list[str]:
    from pxr import Sdf, UsdUtils

    layer = Sdf.Layer.FindOrOpen(str(path))
    if not layer:
        raise ValueError("cannot read USD layer")
    if layer.ListAllTimeSamples():
        raise ValueError("navigation scene inputs must be static, without time samples")

    layer.Traverse(
        Sdf.Path.absoluteRootPath,
        lambda spec_path: _check_layer_spec(layer.GetObjectAtPath(spec_path)),
    )
    copy = Sdf.Layer.CreateAnonymous()
    copy.TransferContent(layer)
    dependencies = []
    UsdUtils.ModifyAssetPaths(copy, lambda asset: dependencies.append(asset) or asset)
    composition = set(layer.GetCompositionAssetDependencies())
    if any(Path(name).suffix.lower() not in _USD | {".usdz"} for name in composition):
        raise ValueError("composition dependencies must be USD or USDZ assets")
    return sorted(set(dependencies) | composition)


def _check_layer_spec(spec) -> None:
    from pxr import Sdf

    if spec is None:
        return  # Connection/relationship target paths do not have object specs.
    if spec.HasInfo("clips"):
        raise ValueError("value clips are unsupported in static scene inputs")
    if isinstance(spec, Sdf.PrimSpec):
        identifiers = [spec.typeName]
        if spec.HasInfo("apiSchemas"):
            operations = spec.GetInfo("apiSchemas")
            for field in (
                "explicitItems",
                "addedItems",
                "prependedItems",
                "appendedItems",
                "deletedItems",
                "orderedItems",
            ):
                identifiers.extend(getattr(operations, field))
        if any(name.startswith(("OmniScripting", "OmniGraph")) for name in identifiers):
            raise ValueError(
                "executable OmniScripting or OmniGraph content is unsupported"
            )
    if isinstance(spec, Sdf.PropertySpec) and (
        spec.name.startswith(("omni:scripting:", "omni:graph:"))
        or spec.name in {"node:type", "node:typeVersion"}
    ):
        raise ValueError("executable scripting or OmniGraph properties are unsupported")


def audit_asset(path: Path, boundary: Path) -> set[Path]:
    """Reject external dependencies before USD stage composition or packaging.

    Args:
        path: USD or USDZ root asset.
        boundary: Allowed input tree root.
    Returns:
        Closed set of regular source files consumed by the asset.
    Raises:
        ValueError: Unsupported, external, missing, animated, or executable dependency.
    """
    return _audit_tree(path, boundary, package=False)


def _audit_tree(path: Path, boundary: Path, *, package: bool) -> set[Path]:
    pending, visited = [path], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current.suffix.lower() == ".usdz":
            _audit_package(current)
        elif current.suffix.lower() in _USD:
            for name in _layer_dependencies(current):
                target = _dependency(current.parent, boundary, name, package)
                if not target.is_relative_to(boundary.resolve()):
                    raise ValueError("USD dependency escapes bundle")
                pending.append(target)
    return visited


def _dependency(parent: Path, boundary: Path, name: str, package: bool) -> Path:
    try:
        return contained_file(parent, name)
    except ValueError:
        if not package:
            raise
        # USD package resolution falls back to the archive root for asset paths.
        return contained_file(boundary, name)


def _audit_package(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="npa-usdz-audit-") as temporary:
        root = Path(temporary).resolve()
        first = _extract_package(path, root)
        _audit_tree(first, root, package=True)
        for member in root.rglob("*"):
            if member.suffix.lower() in _USD | {".usdz"}:
                _audit_tree(member, root, package=True)
