"""Bind a RoboTwin Docker archive to its independently inspected image identity."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tarfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_byte_scan import (
    core as W,
    prepare as P,
    oci_graph as G,
    ncore_verification as N,
)

SCHEMA = "npa.robotwin.image-verification.v1"


def inspect(fd, length, expected_id):
    """Inspect the attested OCI graph and its exact Docker compatibility view.

    Args:
        fd: Open archive descriptor.
        length: Exact archive size.
        expected_id: Independently inspected image index digest.
    Returns:
        The bound graph with ordered runtime layers.
    Raises:
        W.ScanError: The graph or compatibility view is inconsistent.
    """
    return G.inspect(
        fd, length, expected_id, platform=N.PLATFORM, allow_docker_manifest=True
    )


bind = N.bind


def _layer_population(fd, row):
    def decoded():
        raw = W.Slice(fd, row["offset"], row["size"])
        compressed = os.pread(fd, 2, row["offset"]) == b"\x1f\x8b"
        return W.GzipReader(raw) if compressed else raw

    digest = hashlib.sha256()
    reader = decoded()
    while data := reader.read(W.CHUNK):
        digest.update(data)
    W.require(
        "sha256:" + digest.hexdigest() == row["diff_id"], "robotwin_layer_diff_id"
    )
    files = size = 0
    with tarfile.open(fileobj=decoded(), mode="r|") as layer:
        for member in layer:
            if not member.isfile():
                continue
            count = 0
            with layer.extractfile(member) as body:
                while data := body.read(W.CHUNK):
                    count += len(data)
            W.require(count == member.size, "robotwin_regular_file_size")
            files += 1
            size += count
    return files, size


def verify(archive_binding, expected_id):
    """Verify graph and layer counts before the separate mandatory byte scan.

    Args:
        archive_binding: Owner-only archive path and SHA-256 binding.
        expected_id: Image identity obtained independently from Docker inspect.
    Returns:
        A report binding every layer and its complete regular-file population.
    Raises:
        W.ScanError: The archive or inspected identity is inconsistent.
    """
    with W.bound_open(archive_binding) as (_path, fd, info):
        result = inspect(fd, info.st_size, expected_id)
        layers = result["layers"]
        counts = [_layer_population(fd, row) for row in layers]
        W.require(
            W.descriptor_digest(fd) == archive_binding["sha256"],
            "robotwin_archive_changed",
        )
        return {
            "schema_version": SCHEMA,
            "valid": True,
            "expected_image_id": expected_id,
            "image_index_digest": expected_id,
            "archive_sha256": archive_binding["sha256"],
            "image_manifest_digest": result["image_manifest_digest"],
            "image_config_digest": result["image_config_digest"],
            "verified_layer_diff_ids": [row["diff_id"] for row in layers],
            "layer_count": len(layers),
            "regular_files_read": sum(row[0] for row in counts),
            "content_bytes_read": sum(row[1] for row in counts),
        }


def main():
    """Write a private graph receipt for the RoboTwin complete-byte gate.

    Args:
        None; paths and image identity are command-line arguments.
    Returns:
        Zero after successful verification.
    Raises:
        W.ScanError: The selected archive fails verification.
    """
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis-root", "trusted-root", "archive", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    args = parser.parse_args()
    with W.authorized_roots(args.analysis_root, args.trusted_root):
        report = verify(P.binding(args.archive), args.expected_image_id)
        directory, descriptor = W.create_output(args.output_dir)
        try:
            W.write_private_json(directory, "verification.json", report)
        finally:
            os.close(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
