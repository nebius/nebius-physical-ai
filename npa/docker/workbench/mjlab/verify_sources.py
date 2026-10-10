"""Verify that every image-layer package retains its corresponding source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent
DOC = "usr/share/doc/npa-mjlab/"


def _inventory(archive: Path) -> dict:
    helper = ROOT.parent / "ncore/base_sources.py"
    spec = importlib.util.spec_from_file_location("mjlab_layer_inventory", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inventory(archive)


def _control_files(archive: Path, inventory: dict) -> dict[str, bytes]:
    names = {
        DOC + "bootstrap-sources.json",
        DOC + "python-sources/python-sources.lock.json",
        DOC + "python-sources/notices.json",
    }
    result = {}
    with tarfile.open(archive) as outer:
        manifest = json.load(outer.extractfile("manifest.json"))[0]
        for ordinal, name in enumerate(manifest["Layers"]):
            with tarfile.open(fileobj=outer.extractfile(name), mode="r|*") as layer:
                for member in layer:
                    path = member.name.removeprefix("./")
                    if path not in names:
                        continue
                    if inventory["final_files"][path]["layer"] != ordinal:
                        continue
                    if not member.isfile() or member.size > 1024 * 1024:
                        raise ValueError("invalid source control member")
                    result[path] = layer.extractfile(member).read()
    if result.keys() != names:
        raise ValueError("source control files are absent from the final image")
    return result


def _require_file(files: dict, path: str, size: int, digest: str) -> None:
    item = files.get(path, {})
    if item.get("type") != "0" or (item.get("size"), item.get("sha256")) != (
        size,
        digest,
    ):
        raise ValueError("corresponding source is absent, deleted or changed")


def _verify_debian(inventory: dict, records: list[dict]) -> int:
    required = {
        (item["source"], item["source_version"])
        for layer in inventory["layers"]
        for item in layer["debian_packages"]
    }
    if {(item["source"], item["version"]) for item in records} != required:
        raise ValueError("source population differs from all-layer package versions")
    count = 0
    for item in records:
        prefix = f"{DOC}ubuntu-sources/{item['source']}/{item['version']}/"
        for artifact in item["artifacts"]:
            _require_file(
                inventory["final_files"],
                prefix + artifact["name"],
                artifact["bytes"],
                artifact["sha256"],
            )
            count += 1
    return count


def _verify_python(inventory: dict, controls: dict, lock_path: Path) -> int:
    content = lock_path.read_bytes()
    if controls[DOC + "python-sources/python-sources.lock.json"] != content:
        raise ValueError("delivered Python source lock differs from reviewed source")
    lock = json.loads(content)
    files = inventory["final_files"]
    for item in lock["artifacts"]:
        _require_file(
            files,
            DOC + "python-sources/" + item["filename"],
            item["size"],
            item["sha256"],
        )
    notices = json.loads(controls[DOC + "python-sources/notices.json"])
    if notices.keys() != {item["name"] for item in lock["artifacts"]}:
        raise ValueError("Python source notice population is incomplete")
    for records in notices.values():
        for item in records:
            path = DOC + "python-sources/notices/" + item["sha256"] + ".txt"
            if files.get(path, {}).get("sha256") != item["sha256"]:
                raise ValueError("delivered native license text differs from source")
    return len(lock["artifacts"])


def _report(
    inventory: dict,
    records: list[dict],
    debian_count: int,
    python_count: int,
    lock_path: Path,
) -> dict:
    return {
        "schema": "npa.mjlab.source-delivery-verification.v1",
        "archive_sha256": inventory["archive_sha256"],
        "config_sha256": inventory["config_sha256"],
        "layers": len(inventory["layers"]),
        "debian_source_components": len(records),
        "debian_source_artifacts": debian_count,
        "python_source_artifacts": python_count,
        "python_source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "valid": True,
    }


def main() -> None:
    """Verify exact source delivery against the complete saved-image layer graph.

    Args:
        None. Image and lock paths are supplied on the command line.
    Returns:
        None.
    Raises:
        ValueError, OSError: Image ancestry, source coverage or delivery fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument(
        "--python-lock", type=Path, default=ROOT / "python-sources.lock.json"
    )
    args = parser.parse_args()
    inventory = _inventory(args.image_archive)
    controls = _control_files(args.image_archive, inventory)
    records = json.loads(controls[DOC + "bootstrap-sources.json"])
    debian_count = _verify_debian(inventory, records)
    python_count = _verify_python(inventory, controls, args.python_lock)
    report = _report(inventory, records, debian_count, python_count, args.python_lock)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
