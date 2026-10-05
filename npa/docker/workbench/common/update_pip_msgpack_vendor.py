"""Replace pip's vulnerable MessagePack vendor copy with the installed fixed wheel."""

from __future__ import annotations

import importlib
from importlib.metadata import distribution, version
from pathlib import Path
import shutil

import msgpack
import pip


def main() -> None:
    """Keep actual vendor bytes, import behavior, and vendor metadata consistent."""
    fixed_version = "1.2.1"
    if version("msgpack") != fixed_version:
        raise RuntimeError("the exact fixed MessagePack wheel must be installed first")
    vendor = Path(pip.__file__).resolve().parent / "_vendor"
    destination = vendor / "msgpack"
    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError(
            "pip's MessagePack vendor directory must be an ordinary directory"
        )
    shutil.rmtree(destination)
    shutil.copytree(Path(msgpack.__file__).resolve().parent, destination)
    package = distribution("msgpack")
    licenses = [
        item for item in package.files or () if ".dist-info/licenses/" in str(item)
    ]
    if not licenses:
        raise RuntimeError("the fixed MessagePack wheel must retain its license files")
    for item in licenses:
        license_path = Path(package.locate_file(item))
        shutil.copy2(license_path, destination / license_path.name)
    for metadata in vendor.glob("msgpack-*.dist-info"):
        if metadata.is_symlink() or not metadata.is_dir():
            raise RuntimeError(
                "MessagePack vendor metadata must be an ordinary directory"
            )
        shutil.rmtree(metadata)
    vendor_manifest = vendor / "vendor.txt"
    lines = vendor_manifest.read_text().splitlines()
    matches = [
        index for index, line in enumerate(lines) if line.startswith("msgpack==")
    ]
    if len(matches) != 1:
        raise RuntimeError("pip must declare exactly one MessagePack vendor version")
    lines[matches[0]] = f"msgpack=={fixed_version}"
    vendor_manifest.write_text("\n".join(lines) + "\n")
    importlib.invalidate_caches()
    vendored = importlib.import_module("pip._vendor.msgpack")
    if vendored.__version__ != fixed_version:
        raise RuntimeError("pip did not import the fixed MessagePack vendor bytes")
    payload = {"bootstrap": [1, "fixed"]}
    if vendored.unpackb(vendored.packb(payload), raw=False) != payload:
        raise RuntimeError("fixed MessagePack vendor round trip failed")


if __name__ == "__main__":
    main()
