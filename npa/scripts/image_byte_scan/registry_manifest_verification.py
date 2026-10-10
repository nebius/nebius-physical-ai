"""Bind an unchanged registry manifest and every referenced blob for byte scanning."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tarfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W, oci_verification as O, prepare as P
else:
    from . import core as W, oci_verification as O, prepare as P

SCHEMA = "npa.image.registry-manifest-verification.v1"
OCI = "application/vnd.oci.image."
DOCKER = "application/vnd.docker."
FORMATS = {
    OCI + "manifest.v1+json": (
        OCI + "config.v1+json",
        {OCI + "layer.v1.tar", OCI + "layer.v1.tar+gzip"},
    ),
    DOCKER + "distribution.manifest.v2+json": (
        DOCKER + "container.image.v1+json",
        {DOCKER + "image.rootfs.diff.tar.gzip"},
    ),
}
ATTESTATIONS = {
    "manifests_in_bound_graph": 0,
    "status": "not-present-in-original-manifest-graph",
    "external_referrers": "not-inspected",
}


def _object(data):
    value = W.json_object(data)
    W.require(isinstance(value, dict), "registry_manifest_json_object")
    return value


def _descriptor(descriptor, allowed):
    W.require(
        isinstance(descriptor, dict)
        and {"mediaType", "digest", "size"}
        <= set(descriptor)
        <= {"mediaType", "digest", "size", "annotations", "platform"},
        "registry_descriptor_schema",
    )
    W.require(descriptor["mediaType"] in allowed, "registry_descriptor_media_type")
    digest, size = descriptor["digest"], descriptor["size"]
    W.require(isinstance(digest, str) and W.DIGEST.fullmatch(digest), "registry_digest")
    W.require(type(size) is int and size >= 0, "registry_descriptor_size")
    if "annotations" in descriptor:
        annotations = descriptor["annotations"]
        W.require(
            isinstance(annotations, dict)
            and all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in annotations.items()
            ),
            "registry_descriptor_annotations",
        )
    if "platform" in descriptor:
        W.require(descriptor["platform"] == O.PLATFORM, "registry_descriptor_platform")


class _Archive:
    def __init__(self, descriptor, length, archive):
        self.descriptor, self.archive = descriptor, archive
        self.members, self.blobs = {}, {}
        for member in archive.getmembers():
            name = W.safe_name(member.name)
            W.require(name not in self.members, "registry_duplicate_path")
            W.require(member.isfile() or member.isdir(), "registry_outer_entry_type")
            W.require(
                member.offset_data + member.size <= length, "registry_member_range"
            )
            self.members[name] = member

    def payload(self, name):
        W.require(name in self.members, "registry_missing_blob")
        member = self.members[name]
        W.require(member.isfile(), "registry_blob_not_regular")
        return self.archive.extractfile(member).read()

    def blob(self, descriptor, allowed):
        _descriptor(descriptor, allowed)
        digest = descriptor["digest"]
        name = "blobs/sha256/" + digest[7:]
        W.require(name in self.members, "registry_missing_blob")
        member = self.members[name]
        W.require(
            member.isfile() and member.size == descriptor["size"], "registry_blob_size"
        )
        identity = {k: descriptor[k] for k in ("digest", "size", "mediaType")}
        if digest in self.blobs:
            W.require(self.blobs[digest] == identity, "registry_conflicting_descriptor")
            return name
        reader = W.Slice(self.descriptor, member.offset_data, member.size)
        value = hashlib.sha256()
        while chunk := reader.read(W.CHUNK):
            value.update(chunk)
        W.require("sha256:" + value.hexdigest() == digest, "registry_blob_digest")
        self.blobs[digest] = identity
        return name

    def document(self, descriptor, allowed):
        return _object(self.payload(self.blob(descriptor, allowed)))

    def manifest(self, expected):
        W.require(
            _object(self.payload("oci-layout")) == {"imageLayoutVersion": "1.0.0"},
            "registry_layout",
        )
        root = _object(self.payload("index.json"))
        W.require(
            type(root.get("schemaVersion")) is int
            and root["schemaVersion"] == 2
            and root.get("mediaType") == OCI + "index.v1+json"
            and set(root) <= {"schemaVersion", "mediaType", "manifests", "annotations"}
            and isinstance(root.get("manifests"), list)
            and len(root["manifests"]) == 1,
            "registry_transport_index",
        )
        descriptor = root["manifests"][0]
        _descriptor(descriptor, FORMATS)
        W.require(
            descriptor["digest"] == expected, "registry_original_manifest_binding"
        )
        manifest = self.document(descriptor, FORMATS)
        W.require(
            type(manifest.get("schemaVersion")) is int
            and manifest["schemaVersion"] == 2
            and manifest.get("mediaType") == descriptor["mediaType"]
            and set(manifest)
            <= {"schemaVersion", "mediaType", "config", "layers", "annotations"}
            and isinstance(manifest.get("layers"), list),
            "registry_original_manifest_schema",
        )
        return manifest

    def closure(self):
        expected = {"index.json", "oci-layout"} | {
            "blobs/sha256/" + d[7:] for d in self.blobs
        }
        actual = {name for name, member in self.members.items() if member.isfile()}
        W.require(actual == expected, "registry_unreferenced_blob_or_file")
        W.require(
            all(
                name in {".", "blobs", "blobs/sha256"}
                for name, member in self.members.items()
                if member.isdir()
            ),
            "registry_unexpected_directory",
        )


def _layers(archive, manifest):
    config_media, layer_media = FORMATS[manifest["mediaType"]]
    config = archive.document(manifest["config"], {config_media})
    W.require(
        all(config.get(k) == v for k, v in O.PLATFORM.items()),
        "registry_config_platform",
    )
    rootfs = config.get("rootfs", {})
    diff_ids = rootfs.get("diff_ids")
    W.require(
        rootfs.get("type") == "layers"
        and isinstance(diff_ids, list)
        and len(diff_ids) == len(manifest["layers"])
        and all(isinstance(d, str) and W.DIGEST.fullmatch(d) for d in diff_ids),
        "registry_runtime_diff_ids",
    )
    result = []
    for ordinal, (descriptor, diff_id) in enumerate(
        zip(manifest["layers"], diff_ids, strict=True)
    ):
        name = archive.blob(descriptor, layer_media)
        member = archive.members[name]
        result.append(
            {
                "ordinal": ordinal,
                "name": name,
                "offset": member.offset_data,
                "size": member.size,
                "diff_id": diff_id,
                "descriptor": descriptor,
            }
        )
    return result


def _graph_result(archive, manifest, layers, expected_id):
    return {
        "layers": layers,
        "image_manifest_digest": expected_id,
        "image_config_digest": manifest["config"]["digest"],
        "receipt": {
            "identity_kind": "original-registry-manifest",
            "image_manifest_digest": expected_id,
            "manifest_media_type": manifest["mediaType"],
            "runtime_platform": O.PLATFORM,
            "blob_count": len(archive.blobs),
            "blob_bytes": sum(d["size"] for d in archive.blobs.values()),
            "blobs": sorted(archive.blobs.values(), key=lambda d: d["digest"]),
            "attestations": dict(ATTESTATIONS),
        },
    }


def inspect(descriptor, length, expected_id):
    """Inspect the graph rooted at independently obtained original manifest bytes.

    Args:
        descriptor: Open archive descriptor.
        length: Exact archive size.
        expected_id: Independent registry manifest digest, never a config/index ID.
    Returns:
        Exact ordered layers and a graph receipt with explicit attestation limits.
    Raises:
        W.ScanError: Identity, graph closure or supported platform fails.
    """
    W.require(
        isinstance(expected_id, str) and W.DIGEST.fullmatch(expected_id),
        "registry_expected_digest",
    )
    os.lseek(descriptor, 0, os.SEEK_SET)
    with (
        os.fdopen(os.dup(descriptor), "rb") as stream,
        tarfile.open(fileobj=stream, mode="r:") as tar,
    ):
        archive = _Archive(descriptor, length, tar)
        manifest = archive.manifest(expected_id)
        layers = _layers(archive, manifest)
        archive.closure()
        return _graph_result(archive, manifest, layers, expected_id)


def _bind_population(layers, verification):
    W.require(
        type(verification.get("layer_count")) is int
        and verification["layer_count"] == len(layers)
        and verification.get("verified_layer_diff_ids")
        == [r["diff_id"] for r in layers],
        "registry_verifier_layer_binding",
    )
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(
            type(verification.get(key)) is int and verification[key] >= 0,
            "registry_verifier_population",
        )


def bind(result, verification, expected_id):
    """Require a retained verification to describe this exact manifest graph.

    Args:
        result: Independently inspected graph.
        verification: Retained structural verification receipt.
        expected_id: Independent original registry manifest digest.
    Returns:
        None.
    Raises:
        W.ScanError: Any identity, layer, attestation or population binding differs.
    """
    W.require(
        verification.get("schema_version") == SCHEMA
        and verification.get("valid") is True,
        "registry_verification_schema",
    )
    W.require(
        verification.get("expected_image_id")
        == expected_id
        == result["image_manifest_digest"],
        "registry_expected_identity",
    )
    for key in (
        "image_manifest_digest",
        "image_config_digest",
        "identity_kind",
        "attestations",
    ):
        actual = result.get(key, result["receipt"].get(key))
        W.require(verification.get(key) == actual, "registry_verifier_binding")
    _bind_population(result["layers"], verification)


def verify(archive_binding, expected_id):
    """Verify original registry bytes and decoded ancestor counts without acceptance.

    Args:
        archive_binding: Owner-only archive path and exact SHA-256.
        expected_id: Independent original registry manifest digest.
    Returns:
        Structural receipt for the separate complete-byte scanner.
    Raises:
        W.ScanError: Archive, graph, decoded layer or input binding fails.
    """
    with W.bound_open(archive_binding) as (_path, descriptor, info):
        result = inspect(descriptor, info.st_size, expected_id)
        counts = [_layer_population(descriptor, row) for row in result["layers"]]
        W.require(
            W.descriptor_digest(descriptor) == archive_binding["sha256"],
            "registry_archive_changed",
        )
    return {
        "schema_version": SCHEMA,
        "valid": True,
        "expected_image_id": expected_id,
        "identity_kind": "original-registry-manifest",
        "attestations": dict(ATTESTATIONS),
        "archive_sha256": archive_binding["sha256"],
        "image_manifest_digest": result["image_manifest_digest"],
        "image_config_digest": result["image_config_digest"],
        "verified_layer_diff_ids": [row["diff_id"] for row in result["layers"]],
        "layer_count": len(result["layers"]),
        "regular_files_read": sum(c[0] for c in counts),
        "content_bytes_read": sum(c[1] for c in counts),
    }


def _layer_population(descriptor, row):
    # The existing decoder accepts OCI +gzip; preserve the original Docker descriptor
    # in all receipts and normalize only this decoder's compression dispatch.
    normalized = row
    if row["descriptor"]["mediaType"].endswith(".gzip"):
        normalized = {
            **row,
            "descriptor": {**row["descriptor"], "mediaType": OCI + "layer.v1.tar+gzip"},
        }
    return O._layer_population(descriptor, normalized)


def main(argv=None):
    """Write a private registry-manifest structural receipt.

    Args:
        argv: Explicit command arguments, or None for the process arguments.
    Returns:
        Zero on structural success, otherwise one; never image acceptance.
    Raises:
        None.
    """
    os.umask(0o077)
    held = None
    try:
        with W.cancellation_scope():
            args = O._arguments(argv)
            with W.authorized_roots(args.analysis_root, args.trusted_root):
                result = verify(P.binding(args.archive), args.expected_image_id)
                directory, held = W.create_output(args.output_dir)
                W.write_private_json(directory, "verification.json", result)
        print("Registry manifest graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("Registry manifest graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
