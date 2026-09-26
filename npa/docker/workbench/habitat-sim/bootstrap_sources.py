#!/usr/bin/env python3
"""Deliver and verify official source for the neutral Ubuntu bootstrap packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def fields(text: str) -> dict[str, str]:
    """Read one Debian control paragraph, including folded checksum fields."""
    result: dict[str, str] = {}
    key = ""
    for line in text.splitlines():
        if line.startswith(" ") and key:
            result[key] += "\n" + line.strip()
        elif ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
        elif line.endswith(":"):
            key = line[:-1]
            result[key] = ""
    return result


def packages(root: Path) -> list[tuple[str, str]]:
    """Include both original base packages and final installed package versions."""
    result = set()
    for name in ("base-status", "runtime-status"):
        for paragraph in (root / name).read_text().split("\n\n"):
            row = fields(paragraph)
            if row.get("Status") != "install ok installed":
                continue
            source = row.get("Source", row["Package"])
            match = re.fullmatch(r"([a-z0-9][a-z0-9+.-]*)(?: \(([^)]+)\))?", source)
            if match is None:
                raise ValueError("invalid source package identity")
            result.add((match[1], match[2] or row["Version"]))
    return sorted(result)


def digest(path: Path) -> str:
    """Hash all bytes without retaining source archives in memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_files(directory: Path, source: str, version: str) -> list[dict]:
    """Require one exact .dsc and its complete SHA-256 source population."""
    descriptors = list(directory.glob("*.dsc"))
    if len(descriptors) != 1:
        raise ValueError("expected one source descriptor")
    descriptor = descriptors[0]
    row = fields(descriptor.read_text())
    if (row.get("Source"), row.get("Version")) != (source, version):
        raise ValueError("source descriptor identity mismatch")
    expected = {}
    for line in row["Checksums-Sha256"].splitlines():
        if not line.strip():
            continue
        sha256, size, name = line.split()
        if Path(name).name != name or not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ValueError("invalid source artifact record")
        expected[name] = (int(size), sha256)
    observed = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unexpected source artifact type")
        identity = (path.stat().st_size, digest(path))
        detached_signature = path.name.endswith(".asc") and path.name[:-4] in expected
        if (
            path != descriptor
            and not detached_signature
            and expected.get(path.name) != identity
        ):
            raise ValueError("source descriptor checksum mismatch")
        observed.append(
            {"name": path.name, "bytes": identity[0], "sha256": identity[1]}
        )
    names = {entry["name"] for entry in observed}
    extras = names - set(expected) - {descriptor.name}
    if not set(expected) <= names or any(
        not name.endswith(".asc") or name[:-4] not in expected for name in extras
    ):
        raise ValueError("incomplete corresponding source")
    return observed


def main(mode: str, root: Path) -> None:
    """Fetch through signed snapshot APT metadata or verify the delivered bundle."""
    records = []
    source_root = root / "ubuntu-sources"
    for source, version in packages(root):
        directory = source_root / source / version
        if mode == "fetch":
            directory.mkdir(parents=True, exist_ok=False)
            subprocess.run(
                [
                    "apt-get",
                    "source",
                    "--download-only",
                    "--only-source",
                    f"{source}={version}",
                ],
                cwd=directory,
                check=True,
            )
        records.append(
            {
                "source": source,
                "version": version,
                "artifacts": source_files(directory, source, version),
            }
        )
    manifest = root / "bootstrap-sources.json"
    if mode == "fetch":
        manifest.write_text(json.dumps(records, sort_keys=True, indent=2) + "\n")
    elif mode != "verify" or json.loads(manifest.read_text()) != records:
        raise ValueError("bootstrap source inventory mismatch")
    print(f"Verified accompanying source for {len(records)} Ubuntu components")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
