"""Bake fixed pip vendor bytes and remove its unused legacy metadata backend."""

from __future__ import annotations

import importlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil
import sys

import pip
from pip._internal.metadata import select_backend

FIXED_VERSIONS = {"msgpack": "1.2.1", "urllib3": "2.8.0"}


def replace_vendor_package(vendor: Path, name: str, fixed_version: str) -> None:
    """Copy the installed fixed wheel's module and license into pip's vendor tree."""
    if version(name) != fixed_version:
        raise RuntimeError(f"the exact fixed {name} wheel must be installed first")
    module = importlib.import_module(name)
    destination = vendor / name
    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError(f"pip's {name} vendor must be an ordinary directory")
    shutil.rmtree(destination)
    shutil.copytree(Path(module.__file__).resolve().parent, destination)
    package = distribution(name)
    licenses = [
        item for item in package.files or () if ".dist-info/licenses/" in str(item)
    ]
    if not licenses:
        raise RuntimeError(f"the fixed {name} wheel must retain its license files")
    for item in licenses:
        license_path = Path(package.locate_file(item))
        shutil.copy2(license_path, destination / license_path.name)
    importlib.invalidate_caches()
    vendored = importlib.import_module(f"pip._vendor.{name}")
    if vendored.__version__ != fixed_version:
        raise RuntimeError(f"pip did not import the fixed {name} vendor bytes")


def update_inventory(vendor: Path) -> None:
    """Describe only the vendor bytes retained in this Python 3.11 image."""
    manifest = vendor / "vendor.txt"
    lines = manifest.read_text().splitlines()
    for name, fixed_version in FIXED_VERSIONS.items():
        matches = [
            i for i, line in enumerate(lines) if line.strip().startswith(f"{name}==")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"pip must declare exactly one {name} vendor version")
        i = matches[0]
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = f"{indent}{name}=={fixed_version}"
    if sum(line.startswith("setuptools==") for line in lines) != 1:
        raise RuntimeError("pip must declare exactly one legacy setuptools subset")
    manifest.write_text(
        "\n".join(line for line in lines if not line.startswith("setuptools==")) + "\n"
    )
    bom_path = vendor / "bom.cdx.json"
    bom = json.loads(bom_path.read_text())
    replacements: dict[str, str | None] = {}
    for name in (*FIXED_VERSIONS, "setuptools"):
        components = [item for item in bom["components"] if item["name"] == name]
        if len(components) != 1:
            raise RuntimeError(f"pip must inventory exactly one {name} component")
        component = components[0]
        previous_ref = component["bom-ref"]
        if name == "setuptools":
            replacements[previous_ref] = None
            bom["components"].remove(component)
        else:
            fixed_version = FIXED_VERSIONS[name]
            fixed_ref = f"pkg:pypi/{name}@{fixed_version}"
            replacements[previous_ref] = fixed_ref
            component.update(version=fixed_version, purl=fixed_ref)
            component["bom-ref"] = fixed_ref
    dependencies = []
    for item in bom["dependencies"]:
        ref = replacements.get(item["ref"], item["ref"])
        if ref is None:
            continue
        item["ref"] = ref
        if "dependsOn" in item:
            item["dependsOn"] = [
                replacements.get(dep, dep)
                for dep in item["dependsOn"]
                if replacements.get(dep, dep) is not None
            ]
        dependencies.append(item)
    bom["dependencies"] = dependencies
    bom["version"] += 1
    bom_path.write_text(json.dumps(bom, indent=2) + "\n")


def main() -> None:
    """Verify the supported metadata backend before changing the vendor closure."""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "this vendor closure supports only the pinned Python 3.11 image"
        )
    if select_backend().__name__ != "pip._internal.metadata.importlib":
        raise RuntimeError("pip must use its default importlib metadata backend")
    vendor = Path(pip.__file__).resolve().parent / "_vendor"
    legacy = vendor / "pkg_resources"
    if legacy.is_symlink() or not legacy.is_dir():
        raise RuntimeError(
            "pip's legacy metadata backend must be an ordinary directory"
        )
    for name, fixed_version in FIXED_VERSIONS.items():
        replace_vendor_package(vendor, name, fixed_version)
    shutil.rmtree(legacy)
    update_inventory(vendor)
    msgpack = importlib.import_module("pip._vendor.msgpack")
    payload = {"bootstrap": [1, "fixed"]}
    if msgpack.unpackb(msgpack.packb(payload), raw=False) != payload:
        raise RuntimeError("fixed MessagePack vendor round trip failed")
    if select_backend().__name__ != "pip._internal.metadata.importlib":
        raise RuntimeError("pip metadata backend changed during vendor replacement")


if __name__ == "__main__":
    main()
