#!/usr/bin/env python3
"""Fetch and materialize the exact public Habitat-Sim build source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import BinaryIO, Callable
import urllib.parse
import urllib.request


class SourceError(ValueError):
    """An immutable source or projection boundary failed."""


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise SourceError("source archive redirects are not accepted")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    item: dict[str, object],
    directory: Path,
    opener: Callable[..., BinaryIO] | None = None,
) -> Path:
    url = str(item["archive_url"])
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "codeload.github.com":
        raise SourceError("source archive must use official GitHub codeload HTTPS")
    target = directory / f"{item.get('name', 'habitat-sim')}.tar.gz"
    request = urllib.request.Request(
        url, headers={"User-Agent": "npa-source-preparer/1"}
    )
    open_request = opener or urllib.request.build_opener(_RefuseRedirect()).open
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            open_request(request, timeout=120) as response,
            target.open("xb") as stream,
        ):
            if response.geturl() != url:
                raise SourceError("source archive redirected away from pinned locator")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
                size += len(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if size != item["archive_bytes"] or digest.hexdigest() != item["archive_sha256"]:
        target.unlink(missing_ok=True)
        raise SourceError("source archive size or SHA-256 mismatch")
    return target


def _safe_members(
    archive: tarfile.TarFile, excluded_links: list[str]
) -> tuple[str, list[tarfile.TarInfo]]:
    members = archive.getmembers()
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
    root = roots.pop()
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
    path = root / str(item["license_path"])
    if not path.is_file() or _sha(path) != item["license_sha256"]:
        raise SourceError(
            f"license bytes changed for {item.get('name', 'habitat-sim')}"
        )
    expected_size = item.get("license_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise SourceError("source license size changed")


def _prune_parent(root: Path) -> None:
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
        if path.name != "default.physics_config.json":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
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
        if (root / relative).exists():
            raise SourceError(f"forbidden source path survived: {relative}")
    names = "\n".join(
        path.relative_to(root).as_posix().lower() for path in root.rglob("*")
    )
    for token in ("rlr-audio",):
        if token in names:
            raise SourceError(f"forbidden source token survived: {token}")


def _verify_internal_vendored(root: Path, components: list[dict[str, object]]) -> None:
    for component in components:
        selected_root = root / str(component["root"])
        expected = {str(row["path"]): str(row["sha256"]) for row in component["files"]}
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


def stage(
    manifest: dict[str, object], output: Path, inventory_output: Path | None = None
) -> None:
    if output.exists():
        raise SourceError("source output already exists")
    with tempfile.TemporaryDirectory(prefix="npa-habitat-source-") as temp_name:
        temp = Path(temp_name)
        source = dict(manifest["source"])
        source["name"] = "habitat-sim"
        parent_archive = _download(source, temp)
        _extract(parent_archive, output, source.get("excluded_archive_links", []))
        _verify_license(output, source)
        _prune_parent(output)
        _verify_internal_vendored(output, manifest["internal_vendored"])
        for dependency in manifest["dependencies"]:
            archive = _download(dependency, temp)
            target = output / dependency["target"]
            if target.exists():
                shutil.rmtree(target)
            _extract(archive, target, dependency.get("excluded_archive_links", []))
            _verify_license(target, dependency)
            projections = manifest["dependency_projection"]
            _prune_dependency(target, projections[dependency["name"]])
        _assert_forbidden(output, manifest["forbidden_paths"])
        inventory = _inventory(output)
        inventory_bytes = (
            json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        ).encode()
        expected = manifest["expected_projection"]
        if (
            inventory["file_count"] != expected["file_count"]
            or hashlib.sha256(inventory_bytes).hexdigest()
            != expected["inventory_sha256"]
        ):
            raise SourceError("materialized source projection changed")
        if inventory_output is not None:
            inventory_output.write_bytes(inventory_bytes)


def main(argv: list[str] | None = None) -> int:
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
