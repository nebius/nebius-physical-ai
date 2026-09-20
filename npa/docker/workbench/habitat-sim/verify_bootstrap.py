#!/usr/bin/env python3
"""Verify Habitat absence and accompanying bootstrap source in every saved layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import bootstrap_sources as sources

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import scan_image_omniverse_payload as payload_scan  # noqa: E402

DOC = "usr/share/doc/npa-habitat-sim/"
FORBIDDEN = re.compile(
    r"(?:^|/)(?:habitat_sim(?:/|[-.])|libLLVM|libnvidia|libcuda|libcudart)"
    r"|(?:^|/)site-packages/(?:numpy|scipy|numba|llvmlite|matplotlib)(?:/|[-.])"
    r"|(?:^|/)(?:habitat-test-scenes\.zip|skokloster-castle\.(?:glb|navmesh))$"
)


def inspect(archive_path: Path) -> dict:
    """Reuse the repository graph reader, then hash each delivered source member."""
    findings = []
    entries = 0
    for path in payload_scan._iter_tarball(archive_path):
        entries += 1
        if FORBIDDEN.search(path):
            findings.append({"code": "baked_runtime_payload", "path": path})
    files, text = {}, {}
    observed_packages = set()
    with tarfile.open(archive_path) as outer:
        manifest = json.load(outer.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise ValueError("one runtime image required")
        for layer in manifest[0]["Layers"]:
            with tarfile.open(fileobj=outer.extractfile(layer), mode="r|*") as archive:
                for member in archive:
                    path = member.name.removeprefix("./").lstrip("/")
                    parent, _, name = path.rpartition("/")
                    if name.startswith(".wh."):
                        removed = parent if name == ".wh..wh..opq" else "/".join(
                            part for part in (parent, name.removeprefix(".wh.")) if part
                        )
                        if (not removed or DOC.startswith(removed.rstrip("/") + "/")
                                or removed.startswith(DOC)):
                            raise ValueError("whiteout affects delivered bootstrap source")
                    if not (path.startswith(DOC) or path == "var/lib/dpkg/status"):
                        continue
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("bootstrap source member must be regular")
                    stream = archive.extractfile(member)
                    sha256, count = hashlib.sha256(), 0
                    retain = path.endswith((".dsc", "bootstrap-sources.json", "status"))
                    chunks = []
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        count += len(chunk)
                        sha256.update(chunk)
                        if retain:
                            if count > 1024 * 1024:
                                raise ValueError("bootstrap control member exceeds limit")
                            chunks.append(chunk)
                    if count != member.size:
                        raise ValueError("truncated source member")
                    identity = (count, sha256.hexdigest())
                    if "/ubuntu-sources/" in path and path in files and files[path] != identity:
                        raise ValueError("changed source bytes in ancestor layer")
                    files[path] = identity
                    if retain:
                        text[path] = b"".join(chunks).decode()
                    if path == "var/lib/dpkg/status":
                        observed_packages.update(package_identities(text[path]))
    records = json.loads(text[DOC + "bootstrap-sources.json"])
    delivered = {(row["source"], row["version"]) for row in records}
    if delivered != observed_packages or len(delivered) != len(records):
        raise ValueError("source coverage differs from actual layer package population")
    expected_paths = set()
    for row in records:
        prefix = f"{DOC}ubuntu-sources/{row['source']}/{row['version']}/"
        descriptors = [entry for entry in row["artifacts"] if entry["name"].endswith(".dsc")]
        if len(descriptors) != 1:
            raise ValueError("one source descriptor required")
        descriptor = sources.fields(text[prefix + descriptors[0]["name"]])
        if (descriptor.get("Source"), descriptor.get("Version")) != (row["source"], row["version"]):
            raise ValueError("source descriptor identity mismatch")
        for artifact in row["artifacts"]:
            if Path(artifact["name"]).name != artifact["name"]:
                raise ValueError("invalid source artifact path")
            path = prefix + artifact["name"]
            if path in expected_paths or files.get(path) != (artifact["bytes"], artifact["sha256"]):
                raise ValueError("source artifact identity mismatch")
            expected_paths.add(path)
        declared = {entry["name"] for entry in row["artifacts"]}
        for line in descriptor["Checksums-Sha256"].splitlines():
            if not line.strip():
                continue
            digest, size, name = line.split()
            if name not in declared or files.get(prefix + name) != (int(size), digest):
                raise ValueError("descriptor source checksum mismatch")
    actual_paths = {path for path in files if path.startswith(DOC + "ubuntu-sources/")}
    if actual_paths != expected_paths:
        raise ValueError("unexpected corresponding source population")
    return {"valid": not findings, "entries_read": entries,
            "source_components": len(delivered), "source_artifacts": len(expected_paths),
            "source_bytes": sum(files[path][0] for path in expected_paths),
            "findings": findings}


def package_identities(status: str) -> set[tuple[str, str]]:
    """Resolve exact source versions from actual inherited and installed status bytes."""
    result = set()
    for paragraph in status.split("\n\n"):
        row = sources.fields(paragraph)
        if row.get("Status") != "install ok installed":
            continue
        match = re.fullmatch(r"([a-z0-9][a-z0-9+.-]*)(?: \(([^)]+)\))?", row.get("Source", row["Package"]))
        if match is None:
            raise ValueError("invalid package source identity")
        result.add((match[1], match[2] or row["Version"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.docker_save)
    args.json.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)
