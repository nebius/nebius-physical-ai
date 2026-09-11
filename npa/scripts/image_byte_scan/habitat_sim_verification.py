#!/usr/bin/env python3
"""Verify the closed Habitat-Sim OCI graph and its complete layer payload."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import posixpath
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
    W.require(
        re.fullmatch(r"[0-9a-f]{40}", verification["expected_source_revision"])
        is not None,
        "habitat_oci_source_revision_binding",
    )


def _decoded(fd: int, row: dict[str, object]):
    raw = W.Slice(fd, row["offset"], row["size"])
    compressed = row["descriptor"]["mediaType"].endswith("+gzip")
    magic = os.pread(fd, min(2, row["size"]), row["offset"])
    W.require((magic == b"\x1f\x8b") == compressed, "habitat_oci_layer_codec")
    if compressed:
        W.gzip_header(W.Slice(fd, row["offset"], row["size"]))
    return W.GzipReader(raw) if compressed else raw


def _regular_hash(stream) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    while chunk := stream.read(W.CHUNK):
        if len(prefix) < 4:
            prefix += chunk[: 4 - len(prefix)]
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size, prefix


def _remove_tree(mapping: dict, target: str) -> None:
    for item in [
        item for item in mapping if item == target or item.startswith(target + "/")
    ]:
        mapping.pop(item)


def _apply_whiteout(
    paths: dict[str, str],
    path: str,
    member: tarfile.TarInfo,
    *related: dict,
) -> bool:
    posix = PurePosixPath(path)
    name = posix.name
    if not name.startswith(".wh."):
        return False
    W.require(member.isfile() and member.size == 0, "habitat_oci_malformed_whiteout")
    if name == ".wh..wh..opq":
        prefix = "" if str(posix.parent) == "." else str(posix.parent) + "/"
        for mapping in (paths, *related):
            for item in [item for item in mapping if item.startswith(prefix)]:
                mapping.pop(item)
        return True
    target_name = name.removeprefix(".wh.")
    W.require(target_name not in {"", ".", ".."}, "habitat_oci_malformed_whiteout")
    target = str(posix.parent / target_name)
    for mapping in (paths, *related):
        _remove_tree(mapping, target)
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


def _track_bytes(path: str) -> bool:
    return (
        path == "var/lib/dpkg/status"
        or path.startswith("usr/share/doc/npa-habitat-sim/")
        or (
            "/site-packages/" in path
            and path.endswith((".dist-info/METADATA", ".dist-info/RECORD"))
        )
    )


def _required_file_findings(
    final_files: dict[str, dict[str, object]], contract: dict[str, object]
) -> list[dict[str, object]]:
    findings = []
    for required, expected in contract["required_final_file_sha256"].items():
        path = required.lstrip("/")
        actual = final_files.get(path, {}).get("sha256")
        if actual != expected:
            findings.append({"code": "required_file_hash_mismatch", "path": required})
    return findings


def _json_bytes(
    tracked: dict[str, bytes], path: str, findings: list[dict[str, object]]
) -> dict[str, object] | None:
    try:
        value = json.loads(tracked[path])
    except (KeyError, TypeError, ValueError):
        findings.append({"code": "invalid_embedded_control", "path": "/" + path})
        return None
    if not isinstance(value, dict):
        findings.append({"code": "invalid_embedded_control", "path": "/" + path})
        return None
    return value


def _source_projection_findings(
    final_files: dict[str, dict[str, object]],
    tracked: dict[str, bytes],
    contract: dict[str, object],
) -> tuple[list[dict[str, object]], int]:
    spec = contract["source_projection"]
    manifest_path = spec["manifest_path"].lstrip("/")
    inventory_path = spec["inventory_path"].lstrip("/")
    findings: list[dict[str, object]] = []
    manifest = _json_bytes(tracked, manifest_path, findings)
    inventory = _json_bytes(tracked, inventory_path, findings)
    if manifest is None or inventory is None:
        return findings, 0
    expected = manifest.get("expected_projection", {})
    inventory_bytes = tracked[inventory_path]
    if hashlib.sha256(inventory_bytes).hexdigest() != expected.get("inventory_sha256"):
        findings.append({"code": "source_projection_inventory_hash_mismatch"})
    rows = inventory.get("files")
    if (
        inventory.get("schema_version") != "npa.habitat-sim.source-projection.v1"
        or not isinstance(rows, list)
        or len(rows) != expected.get("file_count")
    ):
        findings.append({"code": "source_projection_inventory_schema"})
        return findings, 0
    root = spec["source_root"].strip("/")
    declared: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        try:
            safe = W.safe_name(relative)
        except Exception:
            findings.append({"code": "source_projection_unsafe_path"})
            continue
        target = f"{root}/{safe}"
        if target in declared:
            findings.append({"code": "source_projection_duplicate_path"})
        declared.add(target)
        actual = final_files.get(target, {})
        if actual.get("sha256") != row.get("sha256") or actual.get("size") != row.get(
            "bytes"
        ):
            findings.append({"code": "source_projection_file_mismatch", "path": safe})
    observed = {path for path in final_files if path.startswith(root + "/")}
    if observed != declared:
        findings.append({"code": "source_projection_population_mismatch"})
    return findings, len(declared)


def _dpkg_records(status: bytes) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for paragraph in status.decode("utf-8", errors="strict").split("\n\n"):
        fields = {}
        for line in paragraph.splitlines():
            if ": " in line and not line.startswith((" ", "\t")):
                key, value = line.split(": ", 1)
                fields[key] = value
        name = fields.get("Package")
        if not name or fields.get("Status") != "install ok installed":
            continue
        source = fields.get("Source", name).split(" ", 1)[0]
        records[name] = {"version": fields.get("Version", ""), "source": source}
    return records


def _apt_lock_rows(payload: bytes) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^  - \{binary: (?P<binary>[^,]+), version: "
        r"(?:\"(?P<quoted>[^\"]+)\"|(?P<plain>[^,]+)), source: "
        r"(?P<source>[^,]+), sha256: (?P<sha>[0-9a-f]{64})\}$"
    )
    rows = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        match = pattern.fullmatch(line)
        if match:
            values = match.groupdict()
            rows.append(
                {
                    "binary": values["binary"],
                    "version": values["quoted"] or values["plain"],
                    "source": values["source"],
                    "sha256": values["sha"],
                }
            )
    W.require(bool(rows), "habitat_oci_apt_lock_population")
    return rows


def _dpkg_findings(
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
    tracked: dict[str, bytes],
    contract: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]], int]:
    findings: list[dict[str, object]] = []
    installed = _dpkg_records(tracked.get("var/lib/dpkg/status", b""))
    lock_path = contract["apt_runtime_lock_path"].lstrip("/")
    rows = _apt_lock_rows(tracked.get(lock_path, b""))
    for row in rows:
        observed = installed.get(row["binary"])
        if observed != {"version": row["version"], "source": row["source"]}:
            findings.append(
                {"code": "runtime_apt_lock_mismatch", "package": row["binary"]}
            )
        copyright_path = f"usr/share/doc/{row['binary']}/copyright"
        if _resolve_final_path(copyright_path, final_paths, final_links) is None:
            findings.append(
                {"code": "runtime_package_copyright_missing", "package": row["binary"]}
            )
    return findings, installed, len(rows)


def _resolve_final_path(
    path: str,
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
) -> str | None:
    current = path
    for _ in range(32):
        parts = PurePosixPath(current).parts
        link_path = next(
            (
                "/".join(parts[:index])
                for index in range(len(parts), 0, -1)
                if "/".join(parts[:index]) in final_links
            ),
            None,
        )
        if link_path is None:
            return current if current in final_paths else None
        kind, target = final_links[link_path]
        suffix = current.removeprefix(link_path).lstrip("/")
        if kind == "hardlink" or target.startswith("/"):
            replacement = target.lstrip("/")
        else:
            replacement = posixpath.join(posixpath.dirname(link_path), target)
        current = posixpath.normpath(posixpath.join(replacement, suffix))
        if current == ".." or current.startswith("../"):
            return None
    return None


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _python_lock(payload: bytes) -> dict[str, str]:
    pattern = re.compile(
        r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\S+) "
        r"--hash=sha256:[0-9a-f]{64}$"
    )
    rows = {}
    for line in payload.decode("utf-8", errors="strict").splitlines():
        match = pattern.fullmatch(line)
        if match:
            rows[_normalize_distribution(match["name"])] = match["version"]
    W.require(bool(rows), "habitat_oci_python_lock_population")
    return rows


def _metadata_identity(payload: bytes) -> tuple[str, str]:
    fields = {}
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if ": " in line and not line.startswith((" ", "\t")):
            key, value = line.split(": ", 1)
            if key in {"Name", "Version"}:
                fields[key] = value
    return _normalize_distribution(fields.get("Name", "")), fields.get("Version", "")


def _record_target(record_path: str, relative: str) -> str:
    site_root = str(PurePosixPath(record_path).parent.parent)
    target = posixpath.normpath(f"{site_root}/{relative}")
    W.require(
        target.startswith("opt/venv/") and not target.startswith("opt/venv/../"),
        "habitat_oci_python_record_path",
    )
    return target


def _python_findings(
    final_files: dict[str, dict[str, object]],
    tracked: dict[str, bytes],
    contract: dict[str, object],
) -> tuple[list[dict[str, object]], int, int, int]:
    findings: list[dict[str, object]] = []
    observed: dict[str, str] = {}
    records = []
    for path, payload in tracked.items():
        if path.endswith(".dist-info/METADATA"):
            name, version = _metadata_identity(payload)
            if not name or name in observed:
                findings.append({"code": "python_distribution_identity_invalid"})
            observed[name] = version
        elif path.endswith(".dist-info/RECORD"):
            records.append((path, payload))
    lock_path = contract["python_runtime_lock_path"].lstrip("/")
    expected = _python_lock(tracked.get(lock_path, b""))
    local = contract["locally_built_python_distribution"]
    expected[_normalize_distribution(local["name"])] = local["version"]
    expected.update(contract["bootstrap_python_distributions"])
    if observed != expected:
        findings.append({"code": "python_distribution_lock_mismatch"})
    covered: set[str] = set()
    verified = 0
    allowed_missing = 0
    missing_patterns = [
        re.compile(pattern)
        for pattern in contract["allowed_missing_python_record_patterns"]
    ]
    for record_path, payload in records:
        for relative, hash_field, size_field in csv.reader(
            io.StringIO(payload.decode("utf-8", errors="strict"))
        ):
            if not hash_field:
                continue
            try:
                algorithm, encoded = hash_field.split("=", 1)
                padding = "=" * (-len(encoded) % 4)
                expected_hash = base64.urlsafe_b64decode(encoded + padding).hex()
                expected_size = int(size_field)
            except (TypeError, ValueError):
                findings.append({"code": "python_record_entry_invalid"})
                continue
            target = _record_target(record_path, relative)
            actual = final_files.get(target, {})
            if not actual and any(
                pattern.fullmatch(target) for pattern in missing_patterns
            ):
                allowed_missing += 1
                continue
            if (
                algorithm != "sha256"
                or actual.get("sha256") != expected_hash
                or actual.get("size") != expected_size
            ):
                findings.append({"code": "python_record_hash_mismatch", "path": target})
            else:
                covered.add(target)
                verified += 1
    native = {
        path
        for path, row in final_files.items()
        if path.startswith("opt/venv/") and row.get("elf") is True
    }
    if native - covered:
        findings.append({"code": "native_elf_not_bound_to_wheel_record"})
    if allowed_missing != contract["expected_missing_python_record_count"]:
        findings.append({"code": "python_record_allowed_missing_population"})
    return findings, len(observed), verified, allowed_missing


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
    final_files: dict[str, dict[str, object]] = {}
    final_links: dict[str, tuple[str, str]] = {}
    tracked_files: dict[str, bytes] = {}
    findings: list[dict[str, object]] = []
    regular_files = content_bytes = entries = 0
    for layer_index, row in enumerate(layers):
        _verify_diff_id(fd, row)
        current_paths: dict[str, str] = {}
        current_files: dict[str, dict[str, object]] = {}
        current_links: dict[str, tuple[str, str]] = {}
        current_tracked: dict[str, bytes] = {}
        seen: set[str] = set()
        with tarfile.open(fileobj=_decoded(fd, row), mode="r|") as archive:
            for entry_index, member in enumerate(archive):
                entries += 1
                path = W.safe_name(member.name)
                W.require(path not in seen, "habitat_oci_duplicate_layer_path")
                seen.add(path)
                if _apply_whiteout(
                    final_paths,
                    path,
                    member,
                    final_files,
                    final_links,
                    tracked_files,
                ):
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
                final_files.pop(path, None)
                final_links.pop(path, None)
                tracked_files.pop(path, None)
                current_paths[path] = kind
                if member.issym():
                    current_links[path] = ("symlink", member.linkname)
                elif member.islnk():
                    current_links[path] = ("hardlink", member.linkname)
                if not member.isfile():
                    continue
                body = archive.extractfile(member)
                W.require(body is not None, "habitat_oci_regular_file_read")
                if _track_bytes(path):
                    payload = body.read()
                    digest = hashlib.sha256(payload).hexdigest()
                    size = len(payload)
                    prefix = payload[:4]
                    current_tracked[path] = payload
                else:
                    digest, size, prefix = _regular_hash(body)
                W.require(size == member.size, "habitat_oci_regular_file_size")
                current_files[path] = {
                    "sha256": digest,
                    "size": size,
                    "elf": prefix == b"\x7fELF",
                }
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
        final_paths.update(current_paths)
        final_files.update(current_files)
        final_links.update(current_links)
        tracked_files.update(current_tracked)
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
    findings.extend(_required_file_findings(final_files, contract))
    projection_findings, projection_count = _source_projection_findings(
        final_files, tracked_files, contract
    )
    findings.extend(projection_findings)
    dpkg_findings, installed, locked_apt_count = _dpkg_findings(
        final_paths, final_links, tracked_files, contract
    )
    findings.extend(dpkg_findings)
    python_findings, python_count, record_count, allowed_missing_records = (
        _python_findings(final_files, tracked_files, contract)
    )
    findings.extend(python_findings)
    for package in contract["forbidden_packages"]:
        if package in installed:
            findings.append({"code": "forbidden_runtime_package", "package": package})
    dpkg_inventory = json.dumps(installed, sort_keys=True, separators=(",", ":"))
    elf_count = sum(bool(row["elf"]) for row in final_files.values())
    return {
        "entries_read": entries,
        "regular_files_read": regular_files,
        "content_bytes_read": content_bytes,
        "final_path_count": len(final_paths),
        "installed_package_count": len(installed),
        "dpkg_inventory_sha256": hashlib.sha256(dpkg_inventory.encode()).hexdigest(),
        "locked_runtime_apt_package_count": locked_apt_count,
        "python_distribution_count": python_count,
        "python_record_files_verified": record_count,
        "allowed_missing_python_record_files": allowed_missing_records,
        "projected_source_file_count": projection_count,
        "native_elf_count": elf_count,
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
    config: dict[str, object],
    contract: dict[str, object],
    expected_source_revision: str,
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
    if labels.get("org.opencontainers.image.revision") != expected_source_revision:
        findings.append({"code": "source_revision_label_mismatch"})
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
    expected_source_revision: str,
) -> dict[str, object]:
    """Verify graph, layer population, payload policy, and final OCI config."""

    W.require(
        re.fullmatch(r"[0-9a-f]{40}", expected_source_revision) is not None,
        "habitat_oci_expected_source_revision",
    )
    result = inspect(fd, length, expected_id)
    layers = result["layers"]
    payload = _scan_layers(fd, layers, contract)
    config = _config(fd, result["image_config_digest"])
    findings = [
        *payload.pop("findings"),
        *_config_findings(config, contract, expected_source_revision),
    ]
    return {
        "schema_version": SCHEMA,
        "valid": not findings,
        "expected_image_id": expected_id,
        "expected_source_revision": expected_source_revision,
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
