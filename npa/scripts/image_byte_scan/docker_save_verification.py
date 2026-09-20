#!/usr/bin/env python3
"""Bind one complete Docker-save archive to an independently inspected image ID."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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


def verify(archive_path: Path, expected_image_id: str) -> dict[str, object]:
    W.require(_DIGEST.fullmatch(expected_image_id) is not None, "expected_image_digest")
    with archive_path.open("rb") as stream:
        archive_sha256, _archive_size = _hash_stream(stream)

    regular_files = 0
    content_bytes = 0
    entries_read = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        outer = archive.getmembers()
        names = [_path(member.name) for member in outer]
        W.require(len(names) == len(set(names)), "docker_save_duplicate_outer_entry")
        members = dict(zip(names, outer, strict=True))

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
        image = manifest[0]
        config_name = _path(image.get("Config", ""))
        config_member = members.get(config_name)
        W.require(
            config_member is not None and config_member.isfile(),
            "docker_save_config",
        )
        config_bytes = archive.extractfile(config_member).read()
        config_digest = _config_digest(config_name, config_bytes)
        W.require(config_digest == expected_image_id, "docker_save_expected_image")
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
                isinstance(value, str) and _DIGEST.fullmatch(value)
                for value in diff_ids
            )
            and all(isinstance(value, str) for value in layers),
            "docker_save_layer_population",
        )

        for layer_name, expected_diff_id in zip(layers, diff_ids, strict=True):
            normalized = _path(layer_name)
            member = members.get(normalized)
            W.require(
                member is not None and member.isfile(), "docker_save_layer_member"
            )
            raw = archive.extractfile(member)
            blob_hash, _blob_size = _hash_stream(raw)
            if normalized.startswith("blobs/sha256/"):
                W.require(
                    normalized == "blobs/sha256/" + blob_hash,
                    "docker_save_layer_blob_digest",
                )

            raw = archive.extractfile(member)
            with _decoded_layer(raw) as decoded:
                diff_hash, _decoded_size = _hash_stream(decoded)
            W.require(
                "sha256:" + diff_hash == expected_diff_id,
                "docker_save_layer_diff_id",
            )

            raw = archive.extractfile(member)
            with tarfile.open(fileobj=_decoded_layer(raw), mode="r|") as layer:
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

    return {
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


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        report = verify(args.archive, args.expected_image_id)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Docker-save graph verification completed")
        return 0
    except (
        W.INPUT_ERRORS,
        OSError,
        EOFError,
        ValueError,
        KeyError,
        TypeError,
        tarfile.TarError,
    ):
        print("Docker-save graph verification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
