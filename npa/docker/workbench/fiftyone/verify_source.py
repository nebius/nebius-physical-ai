"""Verify the MongoDB executable and source annex delivered in the FiftyOne image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


_REQUIRED_SOURCE_FILES = (
    "LICENSE-Community.txt",
    "README.md",
    "SConstruct",
    "buildscripts/scons.py",
    "docs/building.md",
    "etc/pip/compile-requirements.txt",
    "src/mongo/db/mongod_main.cpp",
    "src/third_party/wiredtiger/SConscript",
)
_NOTICES = ("LICENSE-Community.txt", "MPL-2", "THIRD-PARTY-NOTICES", "SOURCE.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(path: Path, source: dict) -> int:
    if path.stat().st_size != source["archive_bytes"]:
        raise ValueError(
            "MongoDB source archive size differs from the reviewed archive"
        )
    if _sha256(path) != source["archive_sha256"]:
        raise ValueError(
            "MongoDB source archive checksum differs from the reviewed archive"
        )
    prefix = f"mongo-{source['git_revision']}/"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name: member for member in members}
        if len(names) != len(members):
            raise ValueError("MongoDB source archive contains duplicate paths")
        for relative in _REQUIRED_SOURCE_FILES:
            member = names.get(prefix + relative)
            if member is None or not member.isfile():
                raise ValueError(
                    f"MongoDB source archive lacks required file: {relative}"
                )
    return len(members)


def _verify_binary(mongod: Path, manifest: dict) -> dict:
    binary = manifest["binary"]
    if (
        mongod.stat().st_size != binary["file_bytes"]
        or _sha256(mongod) != binary["file_sha256"]
    ):
        raise ValueError("MongoDB executable checksum differs from the reviewed binary")
    output = subprocess.check_output([str(mongod), "--version"], text=True)
    marker = "Build Info: "
    if marker not in output:
        raise ValueError("MongoDB did not report its build identity")
    build = json.loads(output.split(marker, 1)[1])
    if build["version"] != manifest["version"]:
        raise ValueError("MongoDB executable version differs from its source manifest")
    if build["gitVersion"] != binary["git_revision"]:
        raise ValueError("MongoDB executable revision differs from its source manifest")
    if manifest["source"]["git_origin_revision"] != build["gitVersion"]:
        raise ValueError(
            "MongoDB public source origin differs from the executable revision"
        )
    environment = build["environment"]
    if (
        environment["distmod"] != binary["platform"]
        or environment["distarch"] != binary["architecture"]
    ):
        raise ValueError("MongoDB executable platform differs from its source manifest")
    if build["modules"]:
        raise ValueError("MongoDB executable contains undeclared modules")
    return build


def _verify_delivery(notices: Path, mongod: Path, manifest: dict) -> None:
    for name in _NOTICES:
        if not (notices / name).read_bytes():
            raise ValueError(f"MongoDB source delivery is missing its notice: {name}")
    if (mongod.parent / "MONGODB_SOURCE.md").read_bytes() != (
        notices / "SOURCE.md"
    ).read_bytes():
        raise ValueError("MongoDB source directions are missing beside the executable")
    version = json.loads((notices / "version.json").read_text())
    if version != {
        "version": manifest["version"],
        "githash": manifest["source"]["git_revision"],
    }:
        raise ValueError("MongoDB source-build version metadata is inconsistent")


def main() -> int:
    """Check executable/source identity, integrity, and recipient-readable delivery.

    Args:
        None. Paths are read from command-line arguments.

    Returns:
        Zero after every source-delivery check succeeds.

    Raises:
        ValueError: The delivered artifacts have inconsistent identity or content.
        OSError: A required artifact cannot be read or the executable cannot run.
        subprocess.CalledProcessError: MongoDB's version command fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongod", type=Path, required=True)
    parser.add_argument("--notices-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.notices_dir / "source.json").read_text())
    build = _verify_binary(args.mongod, manifest)
    members = _verify_archive(
        args.notices_dir / "mongodb-source.tar.gz", manifest["source"]
    )
    _verify_delivery(args.notices_dir, args.mongod, manifest)
    print(
        json.dumps(
            {
                "version": build["version"],
                "source_members": members,
                "source_delivery": "verified",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
