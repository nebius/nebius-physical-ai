"""Fetch exact Ubuntu sources and verify their public accompanying source annex."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import httpx

ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "robotwin-corresponding-sources.tar"


def _fetch_artifact(row, destination):
    path = destination / row["path"]
    if path.exists():
        raw = path.read_bytes()
    else:
        response = httpx.get(row["url"], follow_redirects=True)
        response.raise_for_status()
        raw = response.content
    if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Ubuntu source archive differs from its exact lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _tar_member(archive, name, raw):
    member = tarfile.TarInfo(name)
    member.size = len(raw)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(raw))


def build(destination):
    """Generate a deterministic source annex from the committed artifact lock.

    Args:
        destination: Private external directory for source files and the archive.
    Returns:
        The archive path, SHA-256, byte count and source-lock SHA-256.
    Raises:
        ValueError: An artifact does not match the lock.
        OSError: Download or filesystem access failed.
    """
    lock_bytes = (ROOT / "corresponding-sources.json").read_bytes()
    lock = json.loads(lock_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda row: _fetch_artifact(row, destination), lock["artifacts"]))
    output = destination / ARCHIVE_NAME
    with tarfile.open(output, "w") as archive:
        for row in sorted(lock["artifacts"], key=lambda item: item["path"]):
            _tar_member(archive, row["path"], (destination / row["path"]).read_bytes())
        _tar_member(archive, "source-lock.json", lock_bytes)
    return {
        "archive": ARCHIVE_NAME,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "source_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def verify_public(source_sha):
    """Prove recipients can anonymously download the exact source annex.

    Args:
        source_sha: Full commit whose development image is being published.
    Returns:
        None.
    Raises:
        ValueError: Public content differs from the committed source identity.
        OSError: Anonymous download failed.
    """
    if re.fullmatch("[0-9a-f]{40}", source_sha) is None:
        raise ValueError("A full source commit is required")
    expected = json.loads((ROOT / "source-bundle.json").read_text())
    base = "https://github.com/nebius/nebius-physical-ai/releases/download/"
    base += f"robotwin-sources-dev-{source_sha}/"
    response = httpx.get(base + "source-manifest.json", follow_redirects=True)
    response.raise_for_status()
    manifest = response.json()
    if manifest != {**expected, "source_revision": source_sha}:
        raise ValueError("Public source manifest differs from the source commit")
    digest = hashlib.sha256()
    size = 0
    with httpx.stream("GET", base + ARCHIVE_NAME, follow_redirects=True) as response:
        response.raise_for_status()
        for data in response.iter_bytes(1024 * 1024):
            size += len(data)
            digest.update(data)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Public source annex differs from its byte identity")


def main():
    """Build or anonymously verify the RoboTwin source annex.

    Args:
        None; arguments are provided on the command line.
    Returns:
        Zero after successful completion.
    Raises:
        ValueError: A source identity fails verification.
        OSError: A required source cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--verify-public", metavar="SOURCE_SHA")
    args = parser.parse_args()
    if args.output_dir:
        print(json.dumps(build(args.output_dir), sort_keys=True))
    else:
        verify_public(args.verify_public)
        print("RoboTwin corresponding source annex verified anonymously")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
