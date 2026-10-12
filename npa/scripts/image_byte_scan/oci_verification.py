#!/usr/bin/env python3
"""Bind original OCI archives and ancestor populations before complete-byte scans.

This is structural verification, not product acceptance, finding adjudication,
attestation trust, or permission to publish an image.
"""

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

SCHEMA = "npa.image.oci-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}


def inspect(fd, length, expected_id):
    """Inspect the complete original OCI graph for the supported runtime platform.

    Args:
        fd: Open archive descriptor.
        length: Exact archive size in bytes.
        expected_id: Independently obtained image index digest.
    Returns:
        The verified graph and ordered layer ranges.
    Raises:
        W.ScanError: Identity, graph closure, or platform verification fails.
    """
    return G.inspect(fd, length, expected_id, platform=PLATFORM)


def bind(result, verification, expected_id):
    """Bind a retained verifier population to an independently inspected graph.

    Args:
        result: Newly inspected graph.
        verification: Retained graph and population report.
        expected_id: Independently obtained image index digest.
    Returns:
        None.
    Raises:
        W.ScanError: Identity, ordered layers, or population fields disagree.
    """
    W.require(
        verification["expected_image_id"]
        == expected_id
        == verification["image_index_digest"],
        "oci_expected_identity",
    )
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(result[key] == verification[key], "oci_verifier_image_binding")
    layers = result["layers"]
    W.require(
        type(verification["layer_count"]) is int
        and len(layers) == verification["layer_count"]
        and [row["diff_id"] for row in layers]
        == verification["verified_layer_diff_ids"],
        "oci_verifier_layer_binding",
    )
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(
            type(verification[key]) is int and verification[key] >= 0,
            "oci_verifier_population",
        )


def _decoded_layer(fd, row):
    raw = W.Slice(fd, row["offset"], row["size"])
    compressed = row["descriptor"]["mediaType"].endswith("+gzip")
    magic = os.pread(fd, min(2, row["size"]), row["offset"])
    W.require((magic == b"\x1f\x8b") == compressed, "oci_layer_codec_binding")
    if compressed:
        W.gzip_header(W.Slice(fd, row["offset"], row["size"]))
    return W.GzipReader(raw) if compressed else raw


def _layer_population(fd, row):
    reader, value = _decoded_layer(fd, row), hashlib.sha256()
    while data := reader.read(W.CHUNK):
        value.update(data)
    W.require("sha256:" + value.hexdigest() == row["diff_id"], "oci_layer_diff_id")
    files = size = 0
    # The byte scanner independently checks physical headers, padding and counts.
    with tarfile.open(fileobj=_decoded_layer(fd, row), mode="r|") as layer:
        for member in layer:
            if not member.isfile():
                continue
            count = 0
            with layer.extractfile(member) as body:
                while data := body.read(W.CHUNK):
                    count += len(data)
            W.require(count == member.size, "oci_regular_file_size")
            files += 1
            size += count
    return files, size


def _graph_report(archive_binding, expected_id, result, counts):
    return {
        "valid": True,
        "expected_image_id": expected_id,
        "image_index_digest": expected_id,
        "archive_sha256": archive_binding["sha256"],
        "image_manifest_digest": result["image_manifest_digest"],
        "image_config_digest": result["image_config_digest"],
        "verified_layer_diff_ids": [row["diff_id"] for row in result["layers"]],
        "layer_count": len(result["layers"]),
        "regular_files_read": sum(row[0] for row in counts),
        "content_bytes_read": sum(row[1] for row in counts),
    }


def verify_graph(archive_binding, expected_id):
    """Verify original archive identity, graph, decoded layers and ancestor counts.

    Args:
        archive_binding: Owner-only archive path and SHA-256 binding.
        expected_id: Independently obtained image index digest.
    Returns:
        Structural graph evidence without a product-specific schema.
    Raises:
        W.ScanError: The graph, layer bytes, population, or input binding fails.
    """
    with W.bound_open(archive_binding) as (_path, fd, info):
        result = inspect(fd, info.st_size, expected_id)
        counts = [_layer_population(fd, row) for row in result["layers"]]
        W.require(
            W.descriptor_digest(fd) == archive_binding["sha256"], "oci_archive_changed"
        )
        return _graph_report(archive_binding, expected_id, result, counts)


def verify(archive_binding, expected_id):
    """Produce generic structural evidence for the separate complete-byte scan.

    Args:
        archive_binding: Owner-only archive path and SHA-256 binding.
        expected_id: Independently obtained image index digest.
    Returns:
        A generic OCI verification report, without product acceptance.
    Raises:
        W.ScanError: Archive, graph, layer, or population verification fails.
    """
    return {"schema_version": SCHEMA, **verify_graph(archive_binding, expected_id)}


def _arguments(argv):
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    """Write an owner-only generic OCI graph report with sanitized diagnostics.

    Args:
        argv: Command arguments, or None for the process arguments.
    Returns:
        Zero on successful structural verification, otherwise one.
    Raises:
        None.
    """
    os.umask(0o077)
    held = None
    try:
        with W.cancellation_scope():
            args = _arguments(argv)
            with W.authorized_roots(args.analysis_root, args.trusted_root):
                result = verify(P.binding(args.archive), args.expected_image_id)
                directory, held = W.create_output(args.output_dir)
                W.write_private_json(directory, "verification.json", result)
        print("OCI graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("OCI graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
