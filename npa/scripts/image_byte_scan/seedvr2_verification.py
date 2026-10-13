"""Bind a SeedVR2 direct-manifest archive without claiming OCI attestations."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tarfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W, oci_graph as G, prepare as P
else:
    from . import core as W, oci_graph as G, prepare as P

SCHEMA = "npa.seedvr2.direct-manifest-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}


def inspect(fd, length, expected_id):
    """Inspect one unattested Docker/OCI manifest and its exact Docker view.

    Args:
        fd: Open archive descriptor.
        length: Exact archive byte length.
        expected_id: Independently inspected manifest digest, never an index ID.
    Returns:
        Closed graph identities and ordered ancestor-layer ranges.
    Raises:
        W.ScanError: The archive or graph violates the direct-manifest contract.
    """
    return G.inspect(
        fd,
        length,
        expected_id,
        platform=PLATFORM,
        allow_docker_manifest=True,
        direct_manifest=True,
    )


def bind(result, verification, expected_id):
    """Rebind a direct-manifest receipt to freshly inspected archive bytes.

    Args:
        result: Fresh graph inspection.
        verification: Original independent layer-population receipt.
        expected_id: Required manifest digest.
    Returns:
        None.
    Raises:
        W.ScanError: Identity, graph mode or population metadata disagrees.
    """
    _bind_identity(result, verification, expected_id)
    W.require(
        type(verification["layer_count"]) is int
        and len(result["layers"]) == verification["layer_count"]
        and [row["diff_id"] for row in result["layers"]]
        == verification["verified_layer_diff_ids"],
        "seedvr_direct_layer_binding",
    )
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(
            type(verification[key]) is int and verification[key] >= 0,
            "seedvr_direct_population",
        )


def _bind_identity(result, verification, expected_id):
    W.require(
        verification["schema_version"] == SCHEMA
        and verification["valid"] is True
        and verification["expected_image_id"]
        == expected_id
        == verification["image_manifest_digest"]
        and "image_index_digest" not in verification
        and type(verification["attestation_manifest_count"]) is int
        and verification["attestation_manifest_count"] == 0,
        "seedvr_direct_identity",
    )
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(result[key] == verification[key], "seedvr_direct_graph_binding")


def _decoded(fd, row):
    compressed = row["descriptor"]["mediaType"].endswith(("+gzip", ".gzip"))
    magic = os.pread(fd, min(2, row["size"]), row["offset"])
    W.require((magic == b"\x1f\x8b") == compressed, "seedvr_direct_layer_codec")
    if compressed:
        W.gzip_header(W.Slice(fd, row["offset"], row["size"]))
    raw = W.Slice(fd, row["offset"], row["size"])
    return W.GzipReader(raw) if compressed else raw


def _layer_population(fd, row):
    reader, digest = _decoded(fd, row), hashlib.sha256()
    while data := reader.read(W.CHUNK):
        digest.update(data)
    W.require(
        "sha256:" + digest.hexdigest() == row["diff_id"], "seedvr_direct_layer_diff_id"
    )
    files = size = 0
    with tarfile.open(fileobj=_decoded(fd, row), mode="r|") as layer:
        for member in layer:
            if not member.isfile():
                continue
            count = 0
            with layer.extractfile(member) as body:
                while data := body.read(W.CHUNK):
                    count += len(data)
            W.require(count == member.size, "seedvr_direct_regular_size")
            files += 1
            size += count
    return files, size


def verify(archive_binding, expected_id):
    """Verify exact graph bytes, decoded digests and ancestor regular counts.

    Args:
        archive_binding: Private archive path and SHA-256 binding.
        expected_id: Independently inspected manifest digest.
    Returns:
        A structural receipt; not secret, payload, quality or release approval.
    Raises:
        W.ScanError: Archive identity or decoded-layer integrity fails.
    """
    with W.bound_open(archive_binding) as (_path, fd, info):
        result = inspect(fd, info.st_size, expected_id)
        counts = [_layer_population(fd, row) for row in result["layers"]]
        W.require(
            W.descriptor_digest(fd) == archive_binding["sha256"],
            "seedvr_direct_archive_changed",
        )
        return {
            "schema_version": SCHEMA,
            "valid": True,
            "expected_image_id": expected_id,
            "archive_sha256": archive_binding["sha256"],
            "image_manifest_digest": result["image_manifest_digest"],
            "image_config_digest": result["image_config_digest"],
            "attestation_manifest_count": 0,
            "verified_layer_diff_ids": [row["diff_id"] for row in result["layers"]],
            "layer_count": len(result["layers"]),
            "regular_files_read": sum(row[0] for row in counts),
            "content_bytes_read": sum(row[1] for row in counts),
        }


def main(argv=None):
    """Write a private direct-manifest receipt with sanitized terminal status.

    Args:
        argv: Optional explicit command arguments; defaults to the process argv.
    Returns:
        Zero for verified structure; one for a refused input.
    Raises:
        None; supported input failures are reported without private paths.
    """
    os.umask(0o077)
    held = None
    try:
        with W.cancellation_scope():
            parser = W.SanitizedArgumentParser(description=__doc__)
            for name in ("analysis-root", "trusted-root", "archive", "output-dir"):
                parser.add_argument("--" + name, type=Path, required=True)
            parser.add_argument("--expected-image-id", required=True)
            args = parser.parse_args(argv)
            with W.authorized_roots(args.analysis_root, args.trusted_root):
                result = verify(P.binding(args.archive), args.expected_image_id)
                directory, held = W.create_output(args.output_dir)
                W.write_private_json(directory, "verification.json", result)
        print("SeedVR2 direct-manifest verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("SeedVR2 direct-manifest verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
