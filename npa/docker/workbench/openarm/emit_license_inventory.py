"""Emit the installed Python distribution license inventory for the image."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _licenses(metadata: importlib.metadata.PackageMetadata) -> list[str]:
    values = []
    expression = str(metadata.get("License-Expression", "")).strip()
    legacy = str(metadata.get("License", "")).strip()
    if expression:
        values.append(expression)
    if legacy and legacy.upper() != "UNKNOWN":
        values.append(legacy)
    values.extend(
        value.removeprefix("License :: ")
        for value in metadata.get_all("Classifier", [])
        if value.startswith("License :: ")
    )
    return sorted(set(values))


def main() -> int:
    destination = Path("/usr/share/doc/npa-openarm/python-license-inventory.json")
    locks = [
        Path("/opt/npa/docker/workbench/common/isaac-oss-deps.txt"),
        Path("/opt/npa/openarm/mujoco-requirements.txt"),
    ]
    packages = []
    for distribution in importlib.metadata.distributions():
        metadata = distribution.metadata
        name = str(metadata.get("Name", distribution.name)).strip()
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "licenses": _licenses(metadata),
                "home_page": str(metadata.get("Home-page", "")).strip(),
            }
        )
    payload = {
        "schema": "npa.python-license-inventory.v1",
        "packages": sorted(packages, key=lambda row: row["name"].lower()),
        "locks": [{"path": str(path), "sha256": _sha256(path)} for path in locks],
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
