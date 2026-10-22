"""Package preserved registry manifest bytes and blobs without rebuilding an image."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import tarfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import (
        core as W,
        prepare as P,
        registry_manifest_verification as R,
    )
else:
    from . import core as W, prepare as P, registry_manifest_verification as R

DIRECTORY_VERSION = b"Directory Transport Version: 1.1\n"


def _inputs(source, expected_id):
    W.require(
        isinstance(expected_id, str) and W.DIGEST.fullmatch(expected_id),
        "export_expected_digest",
    )
    with W.bound_open(
        {"path": str(source / "manifest.json"), "sha256": expected_id[7:]}
    ) as (_, fd, _):
        data = W.descriptor_bytes(fd)
    manifest = R._object(data)
    W.require(
        manifest.get("mediaType") in R.FORMATS, "export_original_manifest_required"
    )
    config_media, layer_media = R.FORMATS[manifest["mediaType"]]
    R._descriptor(manifest["config"], {config_media})
    W.require(isinstance(manifest.get("layers"), list), "export_layers")
    descriptors = {}
    for descriptor in [manifest["config"], *manifest["layers"]]:
        R._descriptor(
            descriptor,
            {config_media} if descriptor is manifest["config"] else layer_media,
        )
        digest = descriptor["digest"]
        W.require(
            digest not in descriptors or descriptors[digest] == descriptor,
            "export_conflicting_blob",
        )
        descriptors[digest] = descriptor
    return data, manifest, descriptors


def _inventory(source, descriptors):
    version = source / "version"
    with W.bound_open(P.binding(version)) as (_, fd, _):
        W.require(
            W.descriptor_bytes(fd) == DIRECTORY_VERSION, "export_directory_version"
        )
    expected = {"manifest.json", "version"} | {digest[7:] for digest in descriptors}
    W.require(
        {p.name for p in source.iterdir()} == expected,
        "export_unaccounted_source_entry",
    )


def _metadata(archive, name, data):
    member = tarfile.TarInfo(name)
    member.mode, member.size = 0o400, len(data)
    archive.addfile(member, io.BytesIO(data))


def _blob(archive, source, descriptor):
    digest = descriptor["digest"]
    binding = {"path": str(source / digest[7:]), "sha256": digest[7:]}
    with W.bound_open(binding) as (_, fd, info):
        W.require(info.st_size == descriptor["size"], "export_blob_size")
        member = tarfile.TarInfo("blobs/sha256/" + digest[7:])
        member.mode, member.size = 0o400, info.st_size
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream:
            archive.addfile(member, stream)
        W.require(W.descriptor_digest(fd) == digest[7:], "export_blob_changed")


def _archive(output_fd, source, data, manifest, descriptors, expected_id):
    manifest_descriptor = {
        "mediaType": manifest["mediaType"],
        "digest": expected_id,
        "size": len(data),
    }
    index = {
        "schemaVersion": 2,
        "mediaType": R.OCI + "index.v1+json",
        "manifests": [manifest_descriptor],
    }
    fd = os.open(
        "image.tar",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=output_fd,
    )
    with (
        os.fdopen(fd, "wb") as stream,
        tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        _metadata(archive, "oci-layout", W.canonical({"imageLayoutVersion": "1.0.0"}))
        _metadata(archive, "index.json", W.canonical(index))
        _metadata(archive, "blobs/sha256/" + expected_id[7:], data)
        for descriptor in descriptors.values():
            _blob(archive, source, descriptor)


def export(source, expected_id, directory, output_fd):
    """Package an exact Skopeo directory export and verify its original graph.

    Args:
        source: Private directory containing unchanged manifest, version and blobs.
        expected_id: Independently obtained original registry manifest digest.
        directory: New private output directory.
        output_fd: Held descriptor for that exact output directory.
    Returns:
        Structural receipt; no confidentiality, vulnerability or native acceptance.
    Raises:
        W.ScanError: Source inventory, identity, content or output binding fails.
    """
    source_fd = W.directory_fd(source)
    try:
        data, manifest, descriptors = _inputs(source, expected_id)
        _inventory(source, descriptors)
        _archive(output_fd, source, data, manifest, descriptors, expected_id)
        W.output_identity(source, source_fd)
        _inventory(source, descriptors)
        W.output_identity(directory, output_fd)
        archive = P.binding(directory / "image.tar")
        result = R.verify(archive, expected_id)
        W.output_identity(directory, output_fd)
        return {
            "schema_version": "npa.registry-manifest-export.v1",
            "structural_verification": result,
            "archive_sha256": archive["sha256"],
            "archive_bytes": (directory / "image.tar").stat().st_size,
            "complete_byte_qualified": False,
            "native_qualified": False,
        }
    finally:
        os.close(source_fd)


def main(argv=None):
    """Export original registry bytes to a new private transport archive.

    Args:
        argv: Command arguments, or None for the process arguments.
    Returns:
        Zero on structural success; one on failure, never image acceptance.
    Raises:
        None.
    """
    os.umask(0o077)
    held = None
    try:
        parser = W.SanitizedArgumentParser(description=__doc__)
        for name in ("analysis-root", "trusted-root", "source-dir", "output-dir"):
            parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--expected-image-id", required=True)
        args = parser.parse_args(argv)
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            directory, held = W.create_output(args.output_dir)
            result = export(args.source_dir, args.expected_image_id, directory, held)
            W.write_private_json(directory, "export.json", result)
        print("Original registry manifest export completed")
        return 0
    except W.INPUT_ERRORS:
        print("Original registry manifest export failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
