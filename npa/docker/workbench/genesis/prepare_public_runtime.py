"""Bind the reduced public Genesis dependency metadata to reviewed wheel sources."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re


SOURCES = {
    "genesis_world-0.4.6.dist-info": {
        "metadata_sha256": "757b986468652e0eba17e701edf50e02c5522dfe2de708881db4d87a0431c40a",
        "excluded_dependencies": {"tetgen"},
    },
    "lerobot-0.4.4.dist-info": {
        "metadata_sha256": "ad580fb6d2ecf379a98ad700e9dde5541184fbf7779d4c3e5cdd9519d06f9e4f",
        "excluded_dependencies": {"diffusers", "wandb", "torchcodec"},
    },
}


def _metadata_replacement(directory: Path, source: dict) -> tuple[bytes, bytes]:
    original = (directory / "METADATA").read_bytes()
    if hashlib.sha256(original).hexdigest() != source["metadata_sha256"]:
        raise RuntimeError(f"unreviewed dependency metadata: {directory.name}")
    removed = set()
    kept = []
    for line in original.splitlines(keepends=True):
        match = re.match(rb"Requires-Dist: ([a-zA-Z0-9_-]+)", line)
        dependency = match[1].decode().lower() if match else ""
        if dependency in source["excluded_dependencies"]:
            if dependency in removed:
                raise RuntimeError(f"duplicate dependency: {dependency}")
            removed.add(dependency)
        else:
            kept.append(line)
    if removed != source["excluded_dependencies"]:
        raise RuntimeError(f"missing reviewed dependencies: {directory.name}")
    return original, b"".join(kept)


def _record_replacement(directory: Path, replacement: bytes) -> str:
    record_path = directory / "RECORD"
    rows = list(csv.reader(io.StringIO(record_path.read_text())))
    metadata_name = f"{directory.name}/METADATA"
    selected = [row for row in rows if row[0] == metadata_name]
    if len(selected) != 1:
        raise RuntimeError(f"invalid METADATA record: {directory.name}")
    digest = base64.urlsafe_b64encode(hashlib.sha256(replacement).digest())
    selected[0][1:] = ["sha256=" + digest.decode().rstrip("="), str(len(replacement))]
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def _prepare(site_packages: Path, *, check_only: bool) -> list[dict]:
    changes = []
    for name, source in SOURCES.items():
        directory = site_packages / name
        original, replacement = _metadata_replacement(directory, source)
        record = _record_replacement(directory, replacement)
        changes.append((directory, original, replacement, record))
    if not check_only:
        for dependency in ("tetgen", "diffusers", "wandb", "torchcodec"):
            if list(site_packages.glob(f"{dependency}*.dist-info")):
                raise RuntimeError(f"excluded distribution is installed: {dependency}")
        for directory, _, replacement, record in changes:
            (directory / "METADATA").write_bytes(replacement)
            (directory / "RECORD").write_text(record)
    return [
        {
            "distribution": directory.name,
            "original_metadata_sha256": hashlib.sha256(original).hexdigest(),
            "reduced_metadata_sha256": hashlib.sha256(replacement).hexdigest(),
            "excluded_dependencies": sorted(
                SOURCES[directory.name]["excluded_dependencies"]
            ),
            "metadata_record_updated": not check_only,
        }
        for directory, original, replacement, _ in changes
    ]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    receipt = _prepare(arguments.site_packages, check_only=arguments.check_only)
    print(json.dumps({"check_only": arguments.check_only, "distributions": receipt}))


if __name__ == "__main__":
    _main()
