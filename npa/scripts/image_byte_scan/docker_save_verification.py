#!/usr/bin/env python3
"""Bind one complete Docker-save archive to an independently inspected image ID."""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
from typing import BinaryIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W
else:
    from . import core as W


SCHEMA = "npa.docker-save.image-verification.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHUNK = 8 * 1024 * 1024
_INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
_MANIFEST_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
_CONFIG_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
_LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar": False,
    "application/vnd.oci.image.layer.v1.tar+gzip": True,
    "application/vnd.docker.image.rootfs.diff.tar": False,
    "application/vnd.docker.image.rootfs.diff.tar.gzip": True,
}


def _path(value: str) -> str:
    path = PurePosixPath(value)
    W.require(
        bool(value) and not path.is_absolute() and ".." not in path.parts,
        "docker_save_safe_path",
    )
    return str(path)


def _hash_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    while chunk := stream.read(_CHUNK):
        digest.update(chunk)
        count += len(chunk)
    return digest.hexdigest(), count


def _decoded_layer(stream: BinaryIO) -> BinaryIO:
    signature = stream.read(2)
    stream.seek(0)
    return gzip.GzipFile(fileobj=stream) if signature == b"\x1f\x8b" else stream


def _config_digest(name: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path = PurePosixPath(name)
    encoded = path.stem if path.suffix == ".json" else path.name
    W.require(encoded == digest, "docker_save_config_digest")
    return "sha256:" + digest


def _json_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> dict[str, object]:
    W.require(member.isfile(), "docker_save_metadata_regular")
    value = json.load(archive.extractfile(member))
    W.require(isinstance(value, dict), "docker_save_metadata_object")
    return value


def _descriptor_member(
    descriptor: object,
    members: dict[str, tarfile.TarInfo],
    media_types: set[str],
) -> tarfile.TarInfo:
    W.require(
        isinstance(descriptor, dict)
        and descriptor.get("mediaType") in media_types
        and isinstance(descriptor.get("digest"), str)
        and _DIGEST.fullmatch(descriptor["digest"]) is not None
        and type(descriptor.get("size")) is int
        and descriptor["size"] >= 0,
        "docker_save_descriptor",
    )
    member = members.get("blobs/sha256/" + descriptor["digest"][7:])
    W.require(
        member is not None and member.isfile() and member.size == descriptor["size"],
        "docker_save_descriptor_member",
    )
    return member


def _oci_graph(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    config_member: tarfile.TarInfo,
    config_digest: str,
    layers: list[str],
) -> tuple[str | None, list[dict[str, object]] | None]:
    if "index.json" not in members and "oci-layout" not in members:
        return None, None
    W.require(
        "index.json" in members and "oci-layout" in members,
        "docker_save_oci_metadata_population",
    )
    layout = _json_member(archive, members["oci-layout"])
    index = _json_member(archive, members["index.json"])
    W.require(
        layout.get("imageLayoutVersion") == "1.0.0"
        and index.get("schemaVersion") == 2
        and index.get("mediaType", _INDEX_TYPE) == _INDEX_TYPE
        and isinstance(index.get("manifests"), list)
        and len(index["manifests"]) == 1,
        "docker_save_oci_index",
    )
    descriptor = index["manifests"][0]
    manifest_member = _descriptor_member(descriptor, members, _MANIFEST_TYPES)
    manifest_bytes = archive.extractfile(manifest_member).read()
    manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    W.require(
        descriptor["digest"] == manifest_digest,
        "docker_save_oci_manifest_digest",
    )
    manifest = json.loads(manifest_bytes)
    W.require(
        isinstance(manifest, dict)
        and manifest.get("schemaVersion") == 2
        and manifest.get("mediaType") == descriptor["mediaType"],
        "docker_save_oci_manifest",
    )
    described_config = _descriptor_member(
        manifest.get("config"), members, _CONFIG_TYPES
    )
    W.require(
        described_config is config_member
        and manifest["config"]["digest"] == config_digest,
        "docker_save_oci_config",
    )
    described_layers = manifest.get("layers")
    W.require(
        isinstance(described_layers, list) and len(described_layers) == len(layers),
        "docker_save_oci_layer_population",
    )
    for layer_name, layer_descriptor in zip(layers, described_layers, strict=True):
        W.require(
            _descriptor_member(layer_descriptor, members, set(_LAYER_TYPES))
            is members[_path(layer_name)],
            "docker_save_oci_layer_order",
        )
    return manifest_digest, described_layers


def _verify_descriptor(fd: int, expected_image_id: str) -> dict[str, object]:
    archive_sha256 = W.descriptor_digest(fd)
    stream = os.fdopen(os.dup(fd), "rb")
    with stream, tarfile.open(fileobj=stream, mode="r:*") as archive:
        (
            config_digest,
            manifest_digest,
            diff_ids,
            layers,
            layer_descriptors,
            members,
        ) = _graph(archive, expected_image_id)
        entries_read, regular_files, content_bytes = _verify_layers(
            archive, members, layers, diff_ids, layer_descriptors
        )
    W.require(
        W.descriptor_digest(fd) == archive_sha256,
        "docker_save_archive_changed",
    )
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "valid": True,
        "expected_image_id": expected_image_id,
        "image_config_digest": config_digest,
        "docker_save_sha256": archive_sha256,
        "verified_layer_diff_ids": diff_ids,
        "layer_count": len(diff_ids),
        "entries_read": entries_read,
        "regular_files_read": regular_files,
        "content_bytes_read": content_bytes,
    }
    if manifest_digest is not None:
        result["image_manifest_digest"] = manifest_digest
    return result


def _outer_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    outer = archive.getmembers()
    names = [_path(member.name) for member in outer]
    W.require(len(names) == len(set(names)), "docker_save_duplicate_outer_entry")
    return dict(zip(names, outer, strict=True))


def _saved_image(archive, members):
    manifest_member = members.get("manifest.json")
    W.require(
        manifest_member is not None and manifest_member.isfile(),
        "docker_save_manifest",
    )
    manifest = json.load(archive.extractfile(manifest_member))
    W.require(
        isinstance(manifest, list)
        and len(manifest) == 1
        and isinstance(manifest[0], dict),
        "docker_save_image_population",
    )
    return manifest[0]


def _configured_layers(archive, members, image):
    config_name = _path(image.get("Config", ""))
    config_member = members.get(config_name)
    W.require(
        config_member is not None and config_member.isfile(),
        "docker_save_config",
    )
    config_bytes = archive.extractfile(config_member).read()
    config_digest = _config_digest(config_name, config_bytes)
    config = json.loads(config_bytes)
    diff_ids = config.get("rootfs", {}).get("diff_ids")
    layers = image.get("Layers")
    W.require(
        config.get("rootfs", {}).get("type") == "layers"
        and isinstance(diff_ids, list)
        and isinstance(layers, list)
        and bool(layers)
        and len(diff_ids) == len(layers)
        and all(
            isinstance(value, str) and _DIGEST.fullmatch(value) for value in diff_ids
        )
        and all(isinstance(value, str) for value in layers),
        "docker_save_layer_population",
    )
    return config_digest, config_member, diff_ids, layers


def _graph(
    archive: tarfile.TarFile, expected_image_id: str
) -> tuple[str, str | None, list[str], list[str], list[dict] | None, dict]:
    members = _outer_members(archive)
    image = _saved_image(archive, members)
    config_digest, config_member, diff_ids, layers = _configured_layers(
        archive, members, image
    )
    manifest_digest, descriptors = _oci_graph(
        archive, members, config_member, config_digest, layers
    )
    W.require(
        expected_image_id in {config_digest, manifest_digest},
        "docker_save_expected_image",
    )
    return config_digest, manifest_digest, diff_ids, layers, descriptors, members


def _verify_layers(archive, members, layers, diff_ids, descriptors):
    entries_read = regular_files = content_bytes = 0
    for index, (layer_name, expected_diff_id) in enumerate(
        zip(layers, diff_ids, strict=True)
    ):
        descriptor = descriptors[index] if descriptors is not None else None
        counts = _verify_layer(
            archive,
            members,
            layer_name,
            expected_diff_id,
            descriptor,
        )
        entries_read += counts[0]
        regular_files += counts[1]
        content_bytes += counts[2]
    return entries_read, regular_files, content_bytes


def _verify_layer(archive, members, layer_name, expected_diff_id, descriptor):
    normalized = _path(layer_name)
    member = members.get(normalized)
    W.require(member is not None and member.isfile(), "docker_save_layer_member")
    raw = archive.extractfile(member)
    blob_hash, blob_size = _hash_stream(raw)
    if normalized.startswith("blobs/sha256/"):
        W.require(
            normalized == "blobs/sha256/" + blob_hash,
            "docker_save_layer_blob_digest",
        )
    if descriptor is not None:
        W.require(
            descriptor["digest"] == "sha256:" + blob_hash
            and descriptor["size"] == blob_size,
            "docker_save_oci_layer_blob",
        )
        signature = archive.extractfile(member).read(2)
        W.require(
            (signature == b"\x1f\x8b") == _LAYER_TYPES[descriptor["mediaType"]],
            "docker_save_oci_layer_codec",
        )
    with _decoded_layer(archive.extractfile(member)) as decoded:
        diff_hash, _decoded_size = _hash_stream(decoded)
    W.require(
        "sha256:" + diff_hash == expected_diff_id,
        "docker_save_layer_diff_id",
    )
    raw = archive.extractfile(member)
    with tarfile.open(fileobj=_decoded_layer(raw), mode="r|") as layer:
        return _regular_population(layer)


def _regular_population(layer):
    entries_read = regular_files = content_bytes = 0
    for entry in layer:
        _path(entry.name)
        entries_read += 1
        if not entry.isfile():
            continue
        body = layer.extractfile(entry)
        W.require(body is not None, "docker_save_regular_file")
        _digest, count = _hash_stream(body)
        W.require(count == entry.size, "docker_save_regular_file_size")
        regular_files += 1
        content_bytes += count
    return entries_read, regular_files, content_bytes


def _require_archive_identity(path: Path, fd: int, initial: os.stat_result) -> None:
    try:
        current = os.fstat(fd)
        named = path.lstat()
    except OSError as error:
        raise W.ScanError("docker_save_archive_changed") from error
    W.require(
        stat.S_ISREG(named.st_mode)
        and W.stat_fingerprint(initial)
        == W.stat_fingerprint(current)
        == W.stat_fingerprint(named),
        "docker_save_archive_changed",
    )


@contextmanager
def _verified_archive(archive_path: Path, expected_image_id: str):
    W.require(_DIGEST.fullmatch(expected_image_id) is not None, "expected_image_digest")
    path, fd, initial = W.open_private_fd(archive_path)
    try:
        _require_archive_identity(path, fd, initial)
        report = _verify_descriptor(fd, expected_image_id)
        _require_archive_identity(path, fd, initial)
        yield report
    finally:
        try:
            _require_archive_identity(path, fd, initial)
        finally:
            os.close(fd)


def verify(archive_path: Path, expected_image_id: str) -> dict[str, object]:
    """Verify one owner-only Docker-save archive through a held descriptor."""
    with _verified_archive(archive_path, expected_image_id) as report:
        return report


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    held = None
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            with _verified_archive(args.archive, args.expected_image_id) as report:
                directory, held = W.create_output(args.output_dir)
                identity = W.write_private_json(directory, "verification.json", report)
                W.verify_private_json(
                    directory,
                    held,
                    "verification.json",
                    report,
                    identity,
                )
                W.output_identity(directory, held)
        print("Docker-save graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("Docker-save graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
