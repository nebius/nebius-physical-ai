#!/usr/bin/env python3
"""Prepare NCore OCI graph/count input for the complete-byte scanner.

This receipt verifies archive identity, graph closure, decoded layer digests and
the all-ancestor regular-file population. It does not authorize publication or
replace the mandatory byte scan, confidentiality policy, or other image gates.
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

SCHEMA = "npa.ncore.oci-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}


def inspect(fd, length, expected_id):
    return G.inspect(fd, length, expected_id, platform=PLATFORM)


def bind(result, verification, expected_id):
    W.require(verification["expected_image_id"] == expected_id
              == verification["image_index_digest"], "oci_expected_identity")
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(result[key] == verification[key], "oci_verifier_image_binding")
    layers = result["layers"]
    W.require(type(verification["layer_count"]) is int and len(layers) == verification["layer_count"]
              and [row["diff_id"] for row in layers] == verification["verified_layer_diff_ids"], "oci_verifier_layer_binding")
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(type(verification[key]) is int and verification[key] >= 0, "oci_verifier_population")


def verify(archive_binding, expected_id):
    with W.bound_open(archive_binding) as (_path, fd, info):
        result = inspect(fd, info.st_size, expected_id)
        regular_files = regular_bytes = 0
        for row in result["layers"]:
            def decoded():
                raw = W.Slice(fd, row["offset"], row["size"])
                compressed = row["descriptor"]["mediaType"].endswith("+gzip")
                magic = os.pread(fd, min(2, row["size"]), row["offset"])
                W.require((magic == b"\x1f\x8b") == compressed, "oci_layer_codec_binding")
                if compressed:
                    W.gzip_header(W.Slice(fd, row["offset"], row["size"]))
                return W.GzipReader(raw) if compressed else raw

            reader, value = decoded(), hashlib.sha256()
            while data := reader.read(W.CHUNK):
                value.update(data)
            W.require("sha256:" + value.hexdigest() == row["diff_id"], "oci_layer_diff_id")
            # Independently count tarfile's regular members; core.walk_tar later
            # requires exact parity and verifies all physical headers/padding.
            with tarfile.open(fileobj=decoded(), mode="r|") as layer:
                for member in layer:
                    if member.isfile():
                        count = 0
                        with layer.extractfile(member) as body:
                            while data := body.read(W.CHUNK):
                                count += len(data)
                        W.require(count == member.size, "oci_regular_file_size")
                        regular_files += 1
                        regular_bytes += count
        W.require(W.descriptor_digest(fd) == archive_binding["sha256"], "oci_archive_changed")
        return {"schema_version": SCHEMA, "valid": True, "expected_image_id": expected_id,
                "image_index_digest": expected_id, "archive_sha256": archive_binding["sha256"],
                "image_manifest_digest": result["image_manifest_digest"], "image_config_digest": result["image_config_digest"],
                "verified_layer_diff_ids": [row["diff_id"] for row in result["layers"]],
                "layer_count": len(result["layers"]), "regular_files_read": regular_files,
                "content_bytes_read": regular_bytes}


def main(argv=None):
    os.umask(0o077)
    held = None
    try:
        with W.cancellation_scope():
            parser = W.SanitizedArgumentParser(description=__doc__)
            parser.add_argument("--analysis-root", type=Path, required=True)
            parser.add_argument("--trusted-root", type=Path, required=True)
            parser.add_argument("--archive", type=Path, required=True)
            parser.add_argument("--expected-image-id", required=True)
            parser.add_argument("--output-dir", type=Path, required=True)
            args = parser.parse_args(argv)
            with W.authorized_roots(args.analysis_root, args.trusted_root):
                result = verify(P.binding(args.archive), args.expected_image_id)
                directory, held = W.create_output(args.output_dir)
                W.write_private_json(directory, "verification.json", result)
        print("NCore OCI graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("NCore OCI graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
