"""Emit the installed Python distribution license inventory for the image."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


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


def _license_files(
    distribution: importlib.metadata.Distribution,
) -> list[dict[str, str]]:
    records = []
    for entry in distribution.files or []:
        name = str(entry).lower()
        if not any(marker in name for marker in ("license", "copying", "notice")):
            continue
        path = Path(distribution.locate_file(entry))
        if path.is_file():
            records.append({"path": str(entry), "sha256": _sha256(path)})
    return sorted(records, key=lambda row: row["path"].lower())


def _category(name: str, licenses: list[str]) -> str:
    if not name.lower().startswith("nvidia-"):
        return "declared-license"
    signals = " ".join(licenses).lower()
    if "proprietary" in signals and any(
        marker in signals for marker in ("apache", "bsd")
    ):
        return "nvidia-mixed-license-signals"
    if "proprietary" in signals:
        return "nvidia-proprietary-redistributable-component"
    return "nvidia-declared-license"


def _package_record(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    metadata = distribution.metadata
    name = str(metadata.get("Name", distribution.name)).strip()
    licenses = _licenses(metadata)
    if not licenses:
        raise RuntimeError(f"installed distribution has no license signal: {name}")
    return {
        "name": name,
        "version": distribution.version,
        "licenses": licenses,
        "license_category": _category(name, licenses),
        "license_files": _license_files(distribution),
        "home_page": str(metadata.get("Home-page", "")).strip(),
    }


def main() -> int:
    """Write a fail-closed inventory for every installed Python distribution."""
    destination = Path("/usr/share/doc/npa-openarm/python-license-inventory.json")
    locks = [
        Path("/opt/npa/docker/workbench/common/isaac-oss-deps.txt"),
        Path("/opt/npa/openarm/mujoco-requirements.txt"),
        Path("/opt/npa/openarm/security-requirements.txt"),
    ]
    packages = [_package_record(row) for row in importlib.metadata.distributions()]
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
