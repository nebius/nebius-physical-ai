#!/usr/bin/env python3
"""Verify the closed Habitat-Sim OCI graph and its complete layer payload."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import tarfile

from . import core as W
from . import oci_graph as G


SCHEMA = "npa.habitat-sim.oci-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}


def inspect(fd: int, length: int, expected_id: str) -> dict[str, object]:
    """Inspect the exact single-platform OCI graph."""

    return G.inspect(fd, length, expected_id, platform=PLATFORM)


def bind(
    result: dict[str, object], verification: dict[str, object], expected_id: str
) -> None:
    """Bind a prior Habitat verifier receipt to freshly inspected graph bytes."""

    W.require(
        verification["expected_image_id"]
        == expected_id
        == verification["image_index_digest"],
        "habitat_oci_expected_identity",
    )
    for key in ("image_manifest_digest", "image_config_digest"):
        W.require(
            result[key] == verification[key], "habitat_oci_verifier_image_binding"
        )
    layers = result["layers"]
    W.require(
        len(layers) == verification["layer_count"]
        and [row["diff_id"] for row in layers]
        == verification["verified_layer_diff_ids"],
        "habitat_oci_verifier_layer_binding",
    )
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(
            type(verification[key]) is int and verification[key] >= 0,
            "habitat_oci_verifier_population",
        )


def _decoded(fd: int, row: dict[str, object]):
    raw = W.Slice(fd, row["offset"], row["size"])
    compressed = row["descriptor"]["mediaType"].endswith("+gzip")
    magic = os.pread(fd, min(2, row["size"]), row["offset"])
    W.require((magic == b"\x1f\x8b") == compressed, "habitat_oci_layer_codec")
    if compressed:
        W.gzip_header(W.Slice(fd, row["offset"], row["size"]))
    return W.GzipReader(raw) if compressed else raw


def _regular_hash(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(W.CHUNK):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _apply_whiteout(paths: dict[str, str], path: str, member: tarfile.TarInfo) -> bool:
    posix = PurePosixPath(path)
    name = posix.name
    if not name.startswith(".wh."):
        return False
    W.require(member.isfile() and member.size == 0, "habitat_oci_malformed_whiteout")
    if name == ".wh..wh..opq":
        prefix = "" if str(posix.parent) == "." else str(posix.parent) + "/"
        for item in [item for item in paths if item.startswith(prefix)]:
            paths.pop(item)
        return True
    target_name = name.removeprefix(".wh.")
    W.require(target_name not in {"", ".", ".."}, "habitat_oci_malformed_whiteout")
    target = str(posix.parent / target_name)
    for item in [
        item for item in paths if item == target or item.startswith(target + "/")
    ]:
        paths.pop(item)
    return True


def _member_kind(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "file"
    if member.isdir():
        return "directory"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    return "unsupported"


def _verify_diff_id(fd: int, row: dict[str, object]) -> None:
    digest = hashlib.sha256()
    reader = _decoded(fd, row)
    while chunk := reader.read(W.CHUNK):
        digest.update(chunk)
    W.require(
        "sha256:" + digest.hexdigest() == row["diff_id"],
        "habitat_oci_layer_diff_id",
    )


def _scan_layers(
    fd: int, layers: list[dict[str, object]], contract: dict[str, object]
) -> dict[str, object]:
    forbidden = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in contract["forbidden_path_patterns"]
    ]
    forbidden_nondirectory = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in contract["forbidden_nondirectory_path_patterns"]
    ]
    forbidden_hashes = set(contract["forbidden_content_sha256"])
    final_paths: dict[str, str] = {}
    tracked_files: dict[str, bytes] = {}
    findings: list[dict[str, object]] = []
    regular_files = content_bytes = entries = 0
    for layer_index, row in enumerate(layers):
        _verify_diff_id(fd, row)
        current: dict[str, str] = {}
        seen: set[str] = set()
        with tarfile.open(fileobj=_decoded(fd, row), mode="r|") as archive:
            for entry_index, member in enumerate(archive):
                entries += 1
                path = W.safe_name(member.name)
                W.require(path not in seen, "habitat_oci_duplicate_layer_path")
                seen.add(path)
                if _apply_whiteout(final_paths, path, member):
                    continue
                kind = _member_kind(member)
                forbidden_path = any(pattern.search(path) for pattern in forbidden)
                forbidden_nondirectory_path = kind != "directory" and any(
                    pattern.search(path) for pattern in forbidden_nondirectory
                )
                if forbidden_path or forbidden_nondirectory_path:
                    findings.append(
                        {
                            "code": "forbidden_path",
                            "layer": layer_index,
                            "entry": entry_index,
                        }
                    )
                if kind == "unsupported":
                    findings.append(
                        {
                            "code": "unsupported_member",
                            "layer": layer_index,
                            "entry": entry_index,
                        }
                    )
                final_paths.pop(path, None)
                current[path] = kind
                if not member.isfile():
                    continue
                body = archive.extractfile(member)
                W.require(body is not None, "habitat_oci_regular_file_read")
                if path == "var/lib/dpkg/status":
                    status_bytes = body.read()
                    digest = hashlib.sha256(status_bytes).hexdigest()
                    size = len(status_bytes)
                    tracked_files[path] = status_bytes
                else:
                    digest, size = _regular_hash(body)
                W.require(size == member.size, "habitat_oci_regular_file_size")
                regular_files += 1
                content_bytes += size
                if digest in forbidden_hashes:
                    findings.append(
                        {
                            "code": "forbidden_payload_hash",
                            "layer": layer_index,
                            "entry": entry_index,
                        }
                    )
        final_paths.update(current)
    missing = []
    for required in contract["required_final_paths"]:
        target = required.lstrip("/")
        if target in final_paths and final_paths[target] not in {"file", "directory"}:
            findings.append(
                {"code": "required_path_not_file_or_directory", "path": required}
            )
            continue
        if target not in final_paths and not any(
            path.startswith(target.rstrip("/") + "/") for path in final_paths
        ):
            missing.append(required)
            continue
        ancestors = PurePosixPath(target).parents
        for ancestor in ancestors:
            name = str(ancestor)
            if name not in {".", ""} and final_paths.get(name) not in {
                None,
                "directory",
            }:
                findings.append(
                    {"code": "required_path_ancestor_not_directory", "path": required}
                )
                break
    findings.extend({"code": "required_path_missing", "path": path} for path in missing)
    status = tracked_files.get("var/lib/dpkg/status", b"").decode(
        "utf-8", errors="replace"
    )
    installed = {
        line.removeprefix("Package: ")
        for line in status.splitlines()
        if line.startswith("Package: ")
    }
    for package in contract["forbidden_packages"]:
        if package in installed:
            findings.append({"code": "forbidden_runtime_package", "package": package})
    return {
        "entries_read": entries,
        "regular_files_read": regular_files,
        "content_bytes_read": content_bytes,
        "final_path_count": len(final_paths),
        "installed_package_count": len(installed),
        "findings": findings,
    }


def _config(fd: int, config_digest: str) -> dict[str, object]:
    os.lseek(fd, 0, os.SEEK_SET)
    with (
        os.fdopen(os.dup(fd), "rb") as stream,
        tarfile.open(fileobj=stream, mode="r:") as archive,
    ):
        name = "blobs/sha256/" + config_digest.removeprefix("sha256:")
        return json.load(archive.extractfile(archive.getmember(name)))


def _config_findings(
    config: dict[str, object], contract: dict[str, object]
) -> list[dict[str, object]]:
    runtime = config.get("config", {})
    findings: list[dict[str, object]] = []
    if runtime.get("User") != "ubuntu":
        findings.append({"code": "final_user_not_ubuntu"})
    if runtime.get("Entrypoint") != ["/usr/local/bin/npa-habitat-entrypoint"]:
        findings.append({"code": "unexpected_entrypoint"})
    labels = runtime.get("Labels", {})
    for key, value in contract["required_labels"].items():
        if labels.get(key) != value:
            findings.append({"code": "required_label_mismatch", "label": key})
    serialized = json.dumps(config, sort_keys=True).lower()
    for marker in ("authorization:", "aws_secret_access_key", "private key"):
        if marker in serialized:
            findings.append({"code": "credential_shaped_oci_config"})
    return findings


def verify(
    fd: int,
    length: int,
    expected_id: str,
    contract: dict[str, object],
    archive_sha256: str,
) -> dict[str, object]:
    """Verify graph, layer population, payload policy, and final OCI config."""

    result = inspect(fd, length, expected_id)
    layers = result["layers"]
    payload = _scan_layers(fd, layers, contract)
    config = _config(fd, result["image_config_digest"])
    findings = [*payload.pop("findings"), *_config_findings(config, contract)]
    return {
        "schema_version": SCHEMA,
        "valid": not findings,
        "expected_image_id": expected_id,
        "image_index_digest": expected_id,
        "archive_sha256": archive_sha256,
        "image_manifest_digest": result["image_manifest_digest"],
        "image_config_digest": result["image_config_digest"],
        "verified_layer_diff_ids": [row["diff_id"] for row in layers],
        "layer_count": len(layers),
        "oci_graph": result["receipt"],
        **payload,
        "findings": findings,
    }
