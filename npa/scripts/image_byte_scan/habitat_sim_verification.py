#!/usr/bin/env python3
"""Verify the closed Habitat-Sim OCI graph and its complete layer payload."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import posixpath
import re
import struct
import tarfile
import zipfile

from . import core as W
from . import oci_graph as G


SCHEMA = "npa.habitat-sim.oci-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}
SOURCE_ARCHIVE_MAX_MEMBERS = 65_536
SOURCE_ARCHIVE_MAX_MEMBER_BYTES = 256 * 1024 * 1024
SOURCE_ARCHIVE_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
SOURCE_ARCHIVE_MAX_COMPRESSION_RATIO = 1_000
SOURCE_ARCHIVE_MAX_METADATA_BYTES = 1 * 1024 * 1024
RETAINED_MEMBER_MAX_BYTES = 256 * 1024 * 1024
RETAINED_CONTENT_MAX_BYTES = 512 * 1024 * 1024


def inspect(fd: int, length: int, expected_id: str) -> dict[str, object]:
    """Inspect the exact single-platform OCI graph.

    Args:
        fd: Open descriptor for the saved OCI archive.
        length: Exact archive byte length.
        expected_id: Immutable OCI index digest.

    Returns:
        Verified graph identities and ordered layer descriptors.

    Raises:
        W.ScanError: Graph bytes or identities violate the OCI contract.
        OSError: Archive bytes cannot be read.
    """

    return G.inspect(fd, length, expected_id, platform=PLATFORM)


def bind(
    result: dict[str, object], verification: dict[str, object], expected_id: str
) -> None:
    """Bind a prior Habitat verifier receipt to freshly inspected graph bytes.

    Args:
        result: Fresh graph inspection.
        verification: Prior full-byte verifier receipt.
        expected_id: Required immutable OCI index digest.

    Returns:
        None.

    Raises:
        W.ScanError: Any receipt identity or closure binding differs.
        KeyError: A required receipt field is missing.
        TypeError: A receipt value has an invalid type.
    """

    W.require(verification["valid"] is True, "habitat_oci_verifier_not_valid")
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
    _bind_runtime_receipt(verification)


def _bind_runtime_receipt(verification):
    for key in ("regular_files_read", "content_bytes_read"):
        W.require(
            type(verification[key]) is int and verification[key] >= 0,
            "habitat_oci_verifier_population",
        )
    for key in (
        "expected_dpkg_inventory_sha256",
        "expected_python_venv_inventory_sha256",
        "expected_native_closure_sha256",
    ):
        W.require(
            re.fullmatch(r"[0-9a-f]{64}", verification[key]) is not None,
            "habitat_oci_runtime_closure_binding",
        )
    W.require(
        verification["dpkg_inventory_sha256"]
        == verification["expected_dpkg_inventory_sha256"]
        and verification["python_venv_inventory_sha256"]
        == verification["expected_python_venv_inventory_sha256"]
        and verification["native_elf_closure_sha256"]
        == verification["expected_native_closure_sha256"],
        "habitat_oci_runtime_closure_binding",
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


def _regular_hash(
    stream,
    retained_bytes: int = 0,
    expected_size: int | None = None,
) -> tuple[str, int, bytes, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    elf_chunks: list[bytes] = []
    retaining = False
    while chunk := stream.read(W.CHUNK):
        size += len(chunk)
        W.require(size <= RETAINED_MEMBER_MAX_BYTES, "habitat_oci_member_limit")
        if expected_size is not None:
            W.require(size <= expected_size, "habitat_oci_regular_file_size")
        if len(prefix) < 4:
            prefix += chunk[: 4 - len(prefix)]
            if len(prefix) == 4 and prefix == b"\x7fELF":
                retaining = True
                W.require(
                    retained_bytes + (expected_size or size) <= RETAINED_CONTENT_MAX_BYTES,
                    "habitat_oci_retained_bytes_limit",
                )
        if retaining:
            elf_chunks.append(chunk)
        digest.update(chunk)
    return (
        digest.hexdigest(),
        size,
        prefix,
        (b"".join(elf_chunks) if retaining else None),
    )


def _read_bounded_member(stream, expected_size: int, retained_bytes: int = 0) -> bytes:
    """Read one retained member only after checking both byte ceilings."""
    W.require(expected_size <= RETAINED_MEMBER_MAX_BYTES, "habitat_oci_member_limit")
    W.require(
        retained_bytes + expected_size <= RETAINED_CONTENT_MAX_BYTES,
        "habitat_oci_retained_bytes_limit",
    )
    chunks: list[bytes] = []
    size = 0
    while chunk := stream.read(min(W.CHUNK, expected_size - size + 1)):
        size += len(chunk)
        W.require(size <= expected_size, "habitat_oci_regular_file_size")
        chunks.append(chunk)
    W.require(size == expected_size, "habitat_oci_regular_file_size")
    return b"".join(chunks)


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
        or path
        in {
            "etc/group",
            "etc/passwd",
            "etc/ssh/sshd_config.d/99-npa-worker.conf",
            "etc/sudoers.d/90-npa-skypilot",
            "usr/local/bin/npa-habitat-entrypoint",
        }
        or path.startswith("opt/npa-runtime/npa/")
        or (path.startswith("var/lib/dpkg/info/") and path.endswith(".list"))
        or path.startswith("usr/share/doc/npa-habitat-sim/")
        or path.startswith("usr/share/doc/npa-habitat-sim/ubuntu-sources/")
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
    final_paths: dict[str, str],
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
    return _projected_file_findings(spec, rows, final_paths, final_files, findings)


def _projected_file_findings(spec, rows, final_paths, final_files, findings):
    root = spec["source_root"].strip("/")
    declared: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        try:
            safe = W.safe_name(relative)
        except W.ScanError:
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
    observed = {
        path
        for path, kind in final_paths.items()
        if path.startswith(root + "/") and kind != "directory"
    }
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
        W.require(name not in records, "habitat_oci_duplicate_dpkg_package")
        source_field = fields.get("Source", name)
        source = source_field.split(" ", 1)[0]
        source_version_match = re.fullmatch(r"[^ ]+ \(([^)]+)\)", source_field)
        records[name] = {
            "version": fields.get("Version", ""),
            "architecture": fields.get("Architecture", ""),
            "source": source,
            "source_version": (
                source_version_match.group(1)
                if source_version_match
                else fields.get("Version", "")
            ),
        }
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
    final_files: dict[str, dict[str, object]],
    final_links: dict[str, tuple[str, str]],
    tracked: dict[str, bytes],
    contract: dict[str, object],
    expected_inventory_sha256: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, str]],
    dict[str, set[str]],
    int,
    str,
    dict[str, dict[str, object]],
]:
    findings: list[dict[str, object]] = []
    installed = _dpkg_records(tracked.get("var/lib/dpkg/status", b""))
    lock_path = contract["apt_runtime_lock_path"].lstrip("/")
    rows = _apt_lock_rows(tracked.get(lock_path, b""))
    _apt_identity_findings(installed, rows, findings)
    list_payloads = _package_list_payloads(tracked, installed)
    list_bindings = _package_list_bindings(tracked, installed)
    inventory = _package_inventory(
        installed,
        list_bindings,
        list_payloads,
        final_paths,
        final_links,
        final_files,
        findings,
    )
    serialized = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    inventory_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
    if inventory_sha256 != expected_inventory_sha256:
        findings.append({"code": "runtime_dpkg_inventory_lock_mismatch"})
    owners = _dpkg_file_owners(tracked, installed, final_paths, final_links, findings)
    _unowned_runtime_file_findings(final_paths, owners, findings)
    return findings, installed, owners, len(rows), inventory_sha256, inventory


def _apt_identity_findings(installed, rows, findings):
    locked_names: set[str] = set()
    for row in rows:
        if row["binary"] in locked_names:
            findings.append(
                {"code": "runtime_apt_lock_duplicate", "package": row["binary"]}
            )
        locked_names.add(row["binary"])
        observed = installed.get(row["binary"])
        if observed is None or any(
            observed[key] != row[key] for key in ("version", "source")
        ):
            findings.append(
                {"code": "runtime_apt_lock_mismatch", "package": row["binary"]}
            )
        elif observed is not None:
            observed["locked_package_sha256"] = row["sha256"]


def _package_list_bindings(tracked, installed):
    payloads = _package_list_payloads(tracked, installed)
    return {
        package: [
            {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}
            for path, payload in rows
        ]
        for package, rows in payloads.items()
    }


def _package_list_payloads(tracked, installed):
    list_bindings: dict[str, list[dict[str, str]]] = {
        package: [] for package in installed
    }
    prefix = "var/lib/dpkg/info/"
    for control_path, payload in tracked.items():
        if not control_path.startswith(prefix) or not control_path.endswith(".list"):
            continue
        package = control_path.removeprefix(prefix).removesuffix(".list")
        if package not in installed:
            package = package.split(":", 1)[0]
        if package in installed:
            list_bindings[package].append((control_path, payload))
    return list_bindings


def _package_inventory(
    installed,
    list_bindings,
    list_payloads,
    final_paths,
    final_links,
    final_files,
    findings,
):
    inventory: dict[str, dict[str, object]] = {}
    for package, identity in installed.items():
        inventory[package] = _package_inventory_row(
            package,
            identity,
            list_bindings[package],
            list_payloads,
            final_paths,
            final_links,
            final_files,
            findings,
        )
    return inventory


def _package_inventory_row(
    package,
    identity,
    package_lists,
    list_payloads,
    final_paths,
    final_links,
    final_files,
    findings,
):
    package_lists = sorted(package_lists, key=lambda row: row["path"])
    if len(package_lists) != 1:
        findings.append({"code": "runtime_package_file_list_population", "package": package})
    copyright_path = f"usr/share/doc/{package}/copyright"
    resolved = _resolve_final_path(copyright_path, final_paths, final_links)
    copyright_sha256 = final_files.get(resolved or "", {}).get("sha256")
    if resolved is None or not isinstance(copyright_sha256, str):
        findings.append({"code": "runtime_package_copyright_missing", "package": package})
    return {
        **identity,
        "file_lists": package_lists,
        "copyright_path": resolved or "",
        "copyright_sha256": copyright_sha256 or "",
        "file_contents": _package_file_content_rows(
            package, list_payloads, final_paths, final_links, final_files, findings
        ),
    }


def _package_file_content_rows(
    package,
    package_lists,
    final_paths,
    final_links,
    final_files,
    findings,
):
    rows = []
    for _control_path, payload in package_lists.get(package, []):
        for line in payload.decode("utf-8", errors="strict").splitlines():
            if not line.startswith("/"):
                continue
            row = _package_file_content_row(
                package, line, final_paths, final_links, final_files, findings
            )
            if row is None:
                continue
            rows.append(row)
    return sorted(rows, key=lambda row: (row["path"], row["resolved_path"]))


def _package_file_content_row(package, line, final_paths, final_links, final_files, findings):
    try:
        declared = W.safe_name(line.lstrip("/"))
    except W.ScanError:
        findings.append({"code": "runtime_package_file_list_invalid", "package": package})
        return None
    resolved = _resolve_final_path(declared, final_paths, final_links)
    if resolved is None:
        findings.append({"code": "runtime_package_file_missing", "path": declared})
        return None
    kind = final_paths.get(resolved)
    row = {"path": declared, "resolved_path": resolved, "kind": kind}
    if kind == "file":
        file_row = final_files.get(resolved, {})
        if not isinstance(file_row.get("sha256"), str) or not isinstance(
            file_row.get("size"), int
        ):
            findings.append({"code": "runtime_package_file_content_missing", "path": resolved})
            return None
        row.update(sha256=file_row["sha256"], size=file_row["size"])
    return row


def _dpkg_file_owners(
    tracked: dict[str, bytes],
    installed: dict[str, dict[str, str]],
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
    findings: list[dict[str, object]],
) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    prefix = "var/lib/dpkg/info/"
    for control_path, payload in tracked.items():
        if not control_path.startswith(prefix) or not control_path.endswith(".list"):
            continue
        package = control_path.removeprefix(prefix).removesuffix(".list")
        if package not in installed:
            package = package.split(":", 1)[0]
        if package not in installed:
            continue
        for resolved in _package_owner_paths(
            package, payload, final_paths, final_links, findings
        ):
            owner_set = owners.setdefault(resolved, set())
            owner_set.add(package)
            if len(owner_set) > 1:
                findings.append(
                    {
                        "code": "runtime_package_file_owner_ambiguous",
                        "path": resolved,
                        "packages": sorted(owner_set),
                    }
                )
    return owners


def _package_owner_paths(package, payload, final_paths, final_links, findings):
    resolved_paths = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if not line.startswith("/"):
            findings.append({"code": "runtime_package_file_list_invalid", "package": package})
            continue
        try:
            path = W.safe_name(line.lstrip("/"))
        except W.ScanError:
            findings.append({"code": "runtime_package_file_list_invalid", "package": package})
            continue
        resolved = _resolve_final_path(path, final_paths, final_links)
        if resolved is None:
            findings.append({"code": "runtime_package_file_missing", "path": path})
            continue
        resolved_paths.append(resolved)
    return resolved_paths


def _unowned_runtime_file_findings(final_paths, owners, findings):
    prefixes = ("bin/", "sbin/", "lib/", "lib64/", "usr/bin/", "usr/sbin/", "usr/lib/")
    for path, kind in final_paths.items():
        if kind != "directory" and path.startswith(prefixes) and path not in owners:
            findings.append({"code": "runtime_package_file_unowned", "path": path})


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
        if kind == "hardlink":
            return current if not suffix and current in final_paths else None
        if target.startswith("/"):
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


def _python_venv_inventory(
    final_paths: dict[str, str],
    final_files: dict[str, dict[str, object]],
    final_links: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, object]], str]:
    inventory: list[dict[str, object]] = []
    for path, kind in sorted(final_paths.items()):
        if not path.startswith("opt/venv/") or kind == "directory":
            continue
        row: dict[str, object] = {"path": path, "kind": kind}
        if kind == "file":
            file_row = final_files.get(path)
            W.require(file_row is not None, "habitat_oci_python_inventory_file")
            row.update(sha256=file_row["sha256"], size=file_row["size"])
        elif kind in {"hardlink", "symlink"}:
            link_row = final_links.get(path)
            W.require(link_row is not None, "habitat_oci_python_inventory_link")
            row["target"] = link_row[1]
        inventory.append(row)
    serialized = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return inventory, hashlib.sha256(serialized.encode()).hexdigest()


def _python_findings(
    final_paths: dict[str, str],
    final_files: dict[str, dict[str, object]],
    final_links: dict[str, tuple[str, str]],
    tracked: dict[str, bytes],
    contract: dict[str, object],
    expected_inventory_sha256: str,
) -> tuple[
    list[dict[str, object]],
    int,
    int,
    int,
    set[str],
    list[dict[str, object]],
    str,
]:
    findings: list[dict[str, object]] = []
    observed, records = _python_metadata_population(tracked, findings)
    _python_distribution_findings(tracked, contract, observed, findings)
    covered, verified, allowed_missing = _python_records(
        records, final_files, contract, findings
    )
    _python_coverage_findings(final_files, contract, covered, allowed_missing, findings)
    inventory, inventory_sha256 = _python_venv_inventory(
        final_paths, final_files, final_links
    )
    if inventory_sha256 != expected_inventory_sha256:
        findings.append({"code": "python_venv_inventory_lock_mismatch"})
    return (
        findings,
        len(observed),
        verified,
        allowed_missing,
        covered,
        inventory,
        inventory_sha256,
    )


def _python_distribution_findings(tracked, contract, observed, findings):
    lock_path = contract["python_runtime_lock_path"].lstrip("/")
    expected = _python_lock(tracked.get(lock_path, b""))
    local = contract["locally_built_python_distribution"]
    expected[_normalize_distribution(local["name"])] = local["version"]
    expected.update(contract["bootstrap_python_distributions"])
    if observed != expected:
        findings.append({"code": "python_distribution_lock_mismatch"})


def _python_coverage_findings(
    final_files, contract, covered, allowed_missing, findings
):
    native = {
        path
        for path, row in final_files.items()
        if path.startswith("opt/venv/") and row.get("elf") is True
    }
    if native - covered:
        findings.append({"code": "native_elf_not_bound_to_wheel_record"})
    if allowed_missing != contract["expected_missing_python_record_count"]:
        findings.append({"code": "python_record_allowed_missing_population"})


def _python_metadata_population(tracked, findings):
    observed: dict[str, str] = {}
    metadata_roots: set[str] = set()
    records: list[tuple[str, bytes]] = []
    record_roots: set[str] = set()
    for path, payload in tracked.items():
        if path.endswith(".dist-info/METADATA"):
            root = str(PurePosixPath(path).parent)
            if root in metadata_roots:
                findings.append({"code": "python_metadata_population_invalid"})
            metadata_roots.add(root)
            name, version = _metadata_identity(payload)
            if not name or name in observed:
                findings.append({"code": "python_distribution_identity_invalid"})
            observed[name] = version
        elif path.endswith(".dist-info/RECORD"):
            root = str(PurePosixPath(path).parent)
            if root in record_roots:
                findings.append({"code": "python_record_population_invalid"})
            record_roots.add(root)
            records.append((path, payload))
    if metadata_roots != record_roots:
        findings.append({"code": "python_record_population_invalid"})
    return observed, records


def _python_records(records, final_files, contract, findings):
    covered: set[str] = set()
    verified = 0
    allowed_missing = 0
    missing_patterns = [
        re.compile(pattern)
        for pattern in contract["allowed_missing_python_record_patterns"]
    ]
    for record_path, payload in records:
        count, missing = _python_record_file(
            record_path, payload, final_files, missing_patterns, covered, findings
        )
        verified += count
        allowed_missing += missing
    return covered, verified, allowed_missing


def _python_record_file(
    record_path, payload, final_files, missing_patterns, covered, findings
):
    verified = 0
    allowed_missing = 0
    record_entries: set[str] = set()
    for relative, hash_field, size_field in csv.reader(
        io.StringIO(payload.decode("utf-8", errors="strict"))
    ):
        target = _record_target(record_path, relative)
        if target in record_entries:
            findings.append({"code": "python_record_entry_duplicate", "path": target})
            continue
        record_entries.add(target)
        if not hash_field:
            if size_field or target not in final_files:
                findings.append({"code": "python_record_entry_unbound", "path": target})
            continue
        identity = _python_record_identity(hash_field, size_field, findings)
        if identity is None:
            continue
        algorithm, expected_hash, expected_size = identity
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
    return verified, allowed_missing


def _python_record_identity(hash_field, size_field, findings):
    try:
        algorithm, encoded = hash_field.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
            raise ValueError("non-canonical RECORD hash")
        expected_bytes = base64.b64decode(encoded + "=", altchars=b"-_", validate=True)
        if algorithm == "sha256" and len(expected_bytes) != 32:
            raise ValueError("invalid sha256 RECORD hash length")
        expected_hash = expected_bytes.hex()
        expected_size = int(size_field)
        return algorithm, expected_hash, expected_size
    except (TypeError, ValueError):
        findings.append({"code": "python_record_entry_invalid"})
        return None


def _elf_strings(
    payload: bytes, offsets: list[int], start: int, size: int
) -> list[str]:
    W.require(
        start >= 0 and size >= 0 and start + size <= len(payload), "habitat_elf_strings"
    )
    values = []
    for offset in offsets:
        W.require(0 <= offset < size, "habitat_elf_string_offset")
        end = payload.find(b"\0", start + offset, start + size)
        W.require(end >= 0, "habitat_elf_string_termination")
        values.append(payload[start + offset : end].decode("utf-8", errors="strict"))
    return values


def _elf_metadata(payload: bytes) -> dict[str, object]:
    W.require(payload[:4] == b"\x7fELF" and len(payload) >= 52, "habitat_elf_header")
    elf_class, encoding = payload[4], payload[5]
    W.require(elf_class in {1, 2} and encoding == 1, "habitat_elf_format")
    if elf_class == 2:
        header = struct.unpack_from("<HHIQQQIHHHHHH", payload, 16)
        machine, phoff, phentsize, phnum = header[1], header[4], header[8], header[9]
        ph_format, dynamic_format = "<IIQQQQQQ", "<qQ"
    else:
        header = struct.unpack_from("<HHIIIIIHHHHHH", payload, 16)
        machine, phoff, phentsize, phnum = header[1], header[4], header[8], header[9]
        ph_format, dynamic_format = "<IIIIIIII", "<iI"
    expected_ph_size = struct.calcsize(ph_format)
    W.require(phentsize == expected_ph_size, "habitat_elf_program_header_size")
    W.require(phoff + phentsize * phnum <= len(payload), "habitat_elf_program_headers")
    loads, dynamic = _elf_segments(
        payload, elf_class, phoff, phentsize, phnum, ph_format
    )
    metadata: dict[str, object] = {
        "class": elf_class,
        "machine": machine,
        "needed": [],
        "soname": "",
        "rpath": [],
        "runpath": [],
    }
    if dynamic is None:
        return metadata
    tags, string_offset, string_size = _elf_dynamic_strings(
        payload, loads, dynamic, dynamic_format
    )
    return _elf_string_metadata(payload, metadata, tags, string_offset, string_size)


def _elf_segments(payload, elf_class, phoff, phentsize, phnum, ph_format):
    loads: list[tuple[int, int, int]] = []
    dynamic: tuple[int, int] | None = None
    for index in range(phnum):
        row = struct.unpack_from(ph_format, payload, phoff + index * phentsize)
        if elf_class == 2:
            kind, offset, vaddr, filesz = row[0], row[2], row[3], row[5]
        else:
            kind, offset, vaddr, filesz = row[0], row[1], row[2], row[4]
        W.require(offset + filesz <= len(payload), "habitat_elf_segment_bounds")
        if kind == 1:
            loads.append((vaddr, offset, filesz))
        elif kind == 2:
            W.require(dynamic is None, "habitat_elf_dynamic_segment")
            dynamic = (offset, filesz)
    return loads, dynamic


def _elf_dynamic_strings(payload, loads, dynamic, dynamic_format):
    entry_size = struct.calcsize(dynamic_format)
    W.require(dynamic[1] % entry_size == 0, "habitat_elf_dynamic_size")
    tags: dict[int, list[int]] = {}
    for offset in range(dynamic[0], dynamic[0] + dynamic[1], entry_size):
        tag, value = struct.unpack_from(dynamic_format, payload, offset)
        if tag == 0:
            break
        tags.setdefault(tag, []).append(value)
    W.require(
        5 in tags and 10 in tags and len(tags[5]) == 1 and len(tags[10]) == 1,
        "habitat_elf_string_table",
    )
    string_address, string_size = tags[5][0], tags[10][0]
    string_offsets = [
        file_offset + string_address - address
        for address, file_offset, file_size in loads
        if address <= string_address
        and string_address + string_size <= address + file_size
    ]
    W.require(len(string_offsets) == 1, "habitat_elf_string_table_mapping")
    string_offset = string_offsets[0]
    return tags, string_offset, string_size


def _elf_string_metadata(payload, metadata, tags, string_offset, string_size):
    metadata["needed"] = _elf_strings(
        payload, tags.get(1, []), string_offset, string_size
    )
    sonames = _elf_strings(payload, tags.get(14, []), string_offset, string_size)
    W.require(len(sonames) <= 1, "habitat_elf_soname")
    metadata["soname"] = sonames[0] if sonames else ""
    rpaths = _elf_strings(payload, tags.get(15, []), string_offset, string_size)
    runpaths = _elf_strings(payload, tags.get(29, []), string_offset, string_size)
    W.require(len(rpaths) <= 1, "habitat_elf_rpath")
    W.require(len(runpaths) <= 1, "habitat_elf_runpath")
    metadata["rpath"] = rpaths[0].split(":") if rpaths else []
    metadata["runpath"] = runpaths[0].split(":") if runpaths else []
    return metadata


_DEFAULT_LIBRARY_DIRS = (
    "lib/x86_64-linux-gnu",
    "usr/lib/x86_64-linux-gnu",
    "lib64",
    "usr/lib64",
    "lib",
    "usr/lib",
    "usr/local/lib",
)


def _runtime_library_dirs(path: str, rpath: list[str], runpath: list[str]) -> list[str]:
    origin = posixpath.dirname(path)
    selected = runpath if runpath else rpath
    directories: list[str] = []
    for value in [*selected, *_DEFAULT_LIBRARY_DIRS]:
        W.require(
            value.startswith(("/", "$ORIGIN", "${ORIGIN}"))
            or value in _DEFAULT_LIBRARY_DIRS,
            "habitat_elf_relative_runtime_path",
        )
        expanded = value.replace("${ORIGIN}", origin).replace("$ORIGIN", origin)
        W.require("$" not in expanded, "habitat_elf_unsupported_runpath")
        normalized = posixpath.normpath(expanded.lstrip("/"))
        W.require(
            normalized not in {"", ".", ".."} and not normalized.startswith("../"),
            "habitat_elf_runpath_escape",
        )
        if normalized not in directories:
            directories.append(normalized)
    return directories


def _compatible_elf_target(
    logical_path: str,
    row: dict[str, object],
    metadata: dict[str, dict[str, object]],
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
) -> str | None:
    target = _resolve_final_path(logical_path, final_paths, final_links)
    candidate = metadata.get(target or "")
    if candidate is None:
        return None
    if candidate["class"] != row["class"] or candidate["machine"] != row["machine"]:
        return None
    return target


def _needed_binding(
    needed: str,
    row: dict[str, object],
    search: list[str],
    metadata: dict[str, dict[str, object]],
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
) -> tuple[dict[str, object] | None, str | None]:
    matches: list[tuple[int, str, str]] = []
    for index, directory in enumerate(search):
        logical = posixpath.join(directory, needed)
        target = _compatible_elf_target(
            logical, row, metadata, final_paths, final_links
        )
        if target is not None:
            matches.append((index, logical, target))
    all_targets = {
        target
        for candidate in final_paths
        if posixpath.basename(candidate) == needed
        and (
            target := _compatible_elf_target(
                candidate, row, metadata, final_paths, final_links
            )
        )
        is not None
    }
    if len(all_targets) > 1:
        return None, "native_elf_dependency_ambiguous"
    if not matches:
        return None, "native_elf_dependency_unresolved"
    target = matches[0][2]
    if all_targets != {target}:
        return None, "native_elf_dependency_unresolved"
    return {
        "path": target,
        "lookup_path": matches[0][1],
        "search_index": matches[0][0],
    }, None


def _native_findings(
    elf_payloads: dict[str, bytes],
    final_paths: dict[str, str],
    final_links: dict[str, tuple[str, str]],
    final_files: dict[str, dict[str, object]],
    dpkg_owners: dict[str, set[str]],
    python_covered: set[str],
    expected_closure_sha256: str,
) -> tuple[list[dict[str, object]], str, list[dict[str, object]]]:
    findings: list[dict[str, object]] = []
    metadata: dict[str, dict[str, object]] = {}
    for path, payload in elf_payloads.items():
        try:
            metadata[path] = _elf_metadata(payload)
        except (UnicodeDecodeError, struct.error, ValueError):
            findings.append({"code": "native_elf_parse_failed", "path": path})
    closure: list[dict[str, object]] = []
    for path, row in sorted(metadata.items()):
        closure.append(
            _native_closure_row(
                path,
                row,
                metadata,
                final_paths,
                final_links,
                final_files,
                dpkg_owners,
                python_covered,
                findings,
            )
        )
    serialized = json.dumps(closure, sort_keys=True, separators=(",", ":"))
    closure_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
    if closure_sha256 != expected_closure_sha256:
        findings.append({"code": "native_elf_closure_lock_mismatch"})
    return findings, closure_sha256, closure


def _native_closure_row(
    path,
    row,
    metadata,
    final_paths,
    final_links,
    final_files,
    dpkg_owners,
    python_covered,
    findings,
):
    owners = sorted(dpkg_owners.get(path, set()))
    if path in python_covered:
        owners.append("python-wheel-record")
    if not owners:
        findings.append({"code": "native_elf_unowned", "path": path})
    search = _runtime_library_dirs(path, row["rpath"], row["runpath"])
    resolved, resolution = _native_needed(
        path, row, search, metadata, final_paths, final_links, findings
    )
    return {
        "path": path,
        "sha256": final_files[path]["sha256"],
        "class": row["class"],
        "machine": row["machine"],
        "soname": row["soname"],
        "rpath": row["rpath"],
        "runpath": row["runpath"],
        "effective_search_path": search,
        "needed": resolved,
        "needed_resolution": resolution,
        "owners": owners,
    }


def _native_needed(path, row, search, metadata, final_paths, final_links, findings):
    resolved: dict[str, str] = {}
    resolution: dict[str, dict[str, object]] = {}
    for needed in row["needed"]:
        W.require(
            needed == posixpath.basename(needed) and needed not in {"", ".", ".."},
            "habitat_elf_needed_name",
        )
        binding, failure = _needed_binding(
            needed,
            row,
            search,
            metadata,
            final_paths,
            final_links,
        )
        if failure is not None:
            findings.append(
                {
                    "code": failure,
                    "path": path,
                    "needed": needed,
                }
            )
            continue
        W.require(binding is not None, "habitat_elf_dependency_binding")
        resolved[needed] = str(binding["path"])
        resolution[needed] = binding
    return resolved, resolution


@dataclass
class _ScanState:
    paths: dict[str, str] = field(default_factory=dict)
    files: dict[str, dict[str, object]] = field(default_factory=dict)
    links: dict[str, tuple[str, str]] = field(default_factory=dict)
    metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    tracked: dict[str, bytes] = field(default_factory=dict)
    elf: dict[str, bytes] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    source_inventory: dict[str, dict[str, object]] = field(default_factory=dict)
    layer_payloads: list[dict[str, object]] = field(default_factory=list)
    layer_inventory: list[dict[str, object]] = field(default_factory=list)
    layer_source_inventory: dict[int, dict[str, dict[str, object]]] = field(
        default_factory=dict
    )
    layer_tracked: dict[int, dict[str, bytes]] = field(default_factory=dict)
    contract: dict[str, object] = field(default_factory=dict)
    findings: list[dict[str, object]] = field(default_factory=list)
    entries: int = 0
    regular_files: int = 0
    content_bytes: int = 0
    retained_bytes: int = 0


def _source_identity(row: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _layer_owner_rows(
    state: _ScanState,
    path: str,
    source_inventory: dict[str, dict[str, object]],
    tracked: dict[str, bytes],
    payload: dict[str, object] | None = None,
) -> set[str]:
    """Find package/source identities for one immutable layer payload."""
    owners: set[str] = set()
    standalone = _standalone_source_artifact_owner(state, path, payload, tracked)
    if standalone is not None:
        owners.add(standalone)
    for identity, row in source_inventory.items():
        if _source_artifact_owner(state, row, path, tracked, payload):
            owners.add(identity)
        name = str(row.get("name", ""))
        actual_sha, _ = _payload_identity(state, path, payload, tracked)
        if path == f"usr/share/doc/{name}/copyright" and row.get(
            "copyright_sha256"
        ) == actual_sha:
            owners.add(identity)
        if row.get("ecosystem") == "python" and path == row.get("metadata_path"):
            if row.get("metadata_sha256") == actual_sha:
                owners.add(identity)
    owners.update(
        _layer_dpkg_owners(source_inventory, tracked, path, state=state, payload=payload)
    )
    owners.update(
        _layer_python_owners(source_inventory, tracked, path, state=state, payload=payload)
    )
    owners.update(
        _layer_contract_owners(
            state, path, payload, tracked, source_inventory=source_inventory
        )
    )
    return owners


def _payload_identity(
    state: _ScanState,
    path: str,
    payload: dict[str, object] | None,
    tracked: dict[str, bytes],
) -> tuple[str | None, int | None]:
    if payload is not None and payload.get("path") == path:
        return str(payload.get("sha256")), int(payload.get("bytes", 0))
    content = tracked.get(path)
    if content is not None:
        return hashlib.sha256(content).hexdigest(), len(content)
    actual = state.files.get(path, {})
    return actual.get("sha256"), actual.get("size")


def _source_projection_owner(
    state: _ScanState,
    path: str,
    payload: dict[str, object] | None,
    tracked: dict[str, bytes],
) -> bool:
    projection = state.contract.get("source_projection", {})
    inventory_path = str(projection.get("inventory_path", "")).lstrip("/")
    root = str(projection.get("source_root", "")).strip("/")
    raw = tracked.get(inventory_path)
    if not inventory_path or not root or raw is None:
        return False
    try:
        rows = json.loads(raw).get("files", [])
    except (TypeError, ValueError):
        return False
    sha256, size = _payload_identity(state, path, payload, tracked)
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        try:
            projected = W.safe_name(str(row["path"]))
        except W.ScanError:
            continue
        if f"{root}/{projected}" == path:
            return row.get("sha256") == sha256 and row.get("bytes") == size
    return False


def _layer_contract_owners(
    state: _ScanState,
    path: str,
    payload: dict[str, object] | None = None,
    tracked: dict[str, bytes] | None = None,
    *,
    source_inventory: dict[str, dict[str, object]] | None = None,
) -> set[str]:
    owners: set[str] = set()
    tracked = tracked or {}
    actual_sha, _ = _payload_identity(state, path, payload, tracked)
    expected = state.contract.get("required_final_file_sha256", {})
    if expected.get("/" + path) == actual_sha:
        owners.add("source:habitat-sim:contract")
    binding = state.contract.get("executable_source_bindings", {}).get("/" + path)
    if isinstance(binding, dict) and binding.get("sha256") == actual_sha:
        owners.add("source:habitat-sim:executable")
    if _source_projection_owner(state, path, payload, tracked):
        owners.add("source:habitat-sim:projection")
    if path == "var/lib/dpkg/status" or (
        path.startswith("var/lib/dpkg/info/") and path.endswith(".list")
    ):
        actual_sha, actual_size = _payload_identity(state, path, payload, tracked)
        for identity, row in (source_inventory or {}).items():
            contents = row.get("file_contents")
            expected = contents.get(path) if isinstance(contents, dict) else None
            if (
                isinstance(expected, dict)
                and expected.get("sha256") == actual_sha
                and expected.get("size") == actual_size
            ):
                owners.add(identity)
    if path in {
        "usr/local/bin/npa-habitat-entrypoint",
        "etc/passwd",
        "etc/group",
        "etc/ssh/sshd_config.d/99-npa-worker.conf",
        "etc/sudoers.d/90-npa-skypilot",
        "usr/share/doc/npa-habitat-sim/source-projection.json",
    }:
        owners.add("source:habitat-sim:runtime-contract")
    return owners


def _standalone_source_artifact_owner(
    state: _ScanState,
    path: str,
    payload: dict[str, object] | None,
    tracked: dict[str, bytes],
) -> str | None:
    if "/ubuntu-sources/" in path:
        for dsc_path, dsc_payload in tracked.items():
            if not dsc_path.endswith(".dsc"):
                continue
            fields = _debian_source_fields(dsc_payload)
            if fields is None or not fields.get("Source") or not fields.get("Version"):
                continue
            row = {
                "ecosystem": "dpkg",
                "source": fields["Source"],
                "source_version": fields["Version"],
            }
            if _source_artifact_binding(row, dsc_path) and (
                path == dsc_path
                or _debian_artifact_matches(state, row, path, tracked, payload)
            ):
                return f"source:{fields['Source']}:{fields['Version']}"
    if "/python-sources/" in path and payload is not None:
        parts = PurePosixPath(path).parts
        try:
            index = parts.index("python-sources")
            package, version = parts[index + 1 : index + 3]
        except (ValueError, IndexError):
            return None
        row = {"ecosystem": "python", "name": package, "version": version}
        content = tracked.get(path)
        if _source_artifact_binding(row, path) and _python_source_matches(row, content):
            return f"python-source:{package}:{version}"
    return None


def _layer_dpkg_owners(
    source_inventory: dict[str, dict[str, object]],
    tracked: dict[str, bytes],
    path: str,
    *,
    state: _ScanState | None = None,
    payload: dict[str, object] | None = None,
) -> set[str]:
    if state is None:
        content = tracked.get(path)
        actual_sha = hashlib.sha256(content).hexdigest() if content is not None else None
        actual_size = len(content) if content is not None else None
    else:
        actual_sha, actual_size = _payload_identity(state, path, payload, tracked)
    if actual_sha is None or actual_size is None:
        return set()
    owners = set()
    for list_path, payload in tracked.items():
        if not list_path.startswith("var/lib/dpkg/info/") or not list_path.endswith(".list"):
            continue
        package = list_path.removeprefix("var/lib/dpkg/info/").removesuffix(".list")
        package = package.split(":", 1)[0]
        paths = {line.lstrip("/") for line in payload.decode(errors="ignore").splitlines()}
        if path not in paths:
            continue
        for identity, row in source_inventory.items():
            expected = row.get("file_contents", {}).get(path)
            if (
                row.get("name") == package
                and isinstance(row.get("source"), str)
                and isinstance(row.get("source_version"), str)
                and isinstance(expected, dict)
                and expected.get("sha256") == actual_sha
                and expected.get("size") == actual_size
            ):
                owners.add(identity)
    return owners


def _layer_python_owners(
    source_inventory: dict[str, dict[str, object]],
    tracked: dict[str, bytes],
    path: str,
    *,
    state: _ScanState | None = None,
    payload: dict[str, object] | None = None,
) -> set[str]:
    if state is None:
        content = tracked.get(path)
        actual_sha = hashlib.sha256(content).hexdigest() if content is not None else None
        actual_size = len(content) if content is not None else None
    else:
        actual_sha, actual_size = _payload_identity(state, path, payload, tracked)
    if actual_sha is None or actual_size is None:
        return set()
    owners = set()
    for record_path, payload in tracked.items():
        if not record_path.endswith(".dist-info/RECORD"):
            continue
        for identity, row in source_inventory.items():
            if (
                row.get("metadata_path") == record_path.removesuffix("RECORD") + "METADATA"
                and row.get("record_sha256") == actual_sha
                and path == record_path
            ):
                owners.add(identity)
                break
        for relative, hash_field, size_field in csv.reader(
            io.StringIO(payload.decode(errors="ignore"))
        ):
            if _record_target(record_path, relative) != path:
                continue
            metadata_path = record_path.removesuffix("RECORD") + "METADATA"
            metadata_payload = tracked.get(metadata_path)
            metadata_sha = (
                hashlib.sha256(metadata_payload).hexdigest()
                if metadata_payload is not None
                else None
            )
            owners.update(
                identity
                for identity, row in source_inventory.items()
                if row.get("metadata_path") == metadata_path
                and row.get("metadata_sha256") == metadata_sha
                and _python_payload_matches(
                    row, path, actual_sha, actual_size, hash_field, size_field
                )
            )
    return owners


def _python_payload_matches(
    row: dict[str, object],
    path: str,
    actual_sha: str,
    actual_size: int,
    hash_field: str,
    size_field: str,
) -> bool:
    """Require RECORD identity or the exact locked observed hashless member."""
    if hash_field:
        identity = _python_record_identity(hash_field, size_field, [])
        return bool(
            identity
            and identity[0] == "sha256"
            and identity[1] == actual_sha
            and identity[2] == actual_size
        )
    expected = row.get("file_contents", {}).get(path)
    return (
        isinstance(expected, dict)
        and expected.get("sha256") == actual_sha
        and expected.get("size") == actual_size
    )


def _source_artifact_owner(
    state: _ScanState,
    row: dict[str, object],
    path: str,
    tracked: dict[str, bytes],
    payload: dict[str, object] | None = None,
) -> bool:
    if not _source_artifact_binding(row, path):
        return False
    if row.get("ecosystem") == "python":
        return _python_source_matches(row, tracked.get(path))
    return _debian_artifact_matches(state, row, path, tracked, payload)


def _debian_artifact_matches(
    state: _ScanState,
    row: dict[str, object],
    path: str,
    tracked: dict[str, bytes],
    payload: dict[str, object] | None = None,
) -> bool:
    for dsc_path, dsc_payload in tracked.items():
        if not dsc_path.endswith(".dsc") or not _source_artifact_binding(row, dsc_path):
            continue
        fields = _debian_source_fields(dsc_payload)
        if not _dsc_identity_matches(row, fields):
            continue
        checksums = _source_checksum_rows(fields.get("Checksums-Sha256"), 64)
        if checksums is None:
            continue
        if path == dsc_path:
            return True
        name = PurePosixPath(path).name
        expected = checksums.get(name)
        actual_sha, actual_size = _payload_identity(state, path, payload, tracked)
        if expected is not None and actual_sha == expected[1] and actual_size == expected[0]:
            return True
        if path.endswith(".asc") and path.removesuffix(".asc") in {
            f"{PurePosixPath(dsc_path).parent}/{candidate}" for candidate in checksums
        }:
            return actual_size is not None and actual_size > 0
    return False


def _source_layer_inventory(state: _ScanState) -> list[dict[str, object]]:
    inventory = []
    for payload in state.layer_payloads:
        layer = int(payload["layer"])
        owners = _layer_owner_rows(
            state,
            str(payload["path"]),
            state.layer_source_inventory.get(layer, {}),
            state.layer_tracked.get(layer, {}),
            payload,
        )
        inventory.append({**payload, "owners": sorted(owners)})
    return inventory


def _record_source_population(state: _ScanState) -> None:
    # Retain each distinct version from every layer, including superseded base
    # packages. A final-rootfs-only inventory cannot describe distributed layers.
    _require_reachable_state(state)
    installed = _dpkg_records(state.tracked.get("var/lib/dpkg/status", b""))
    for name, identity in installed.items():
        copyright_path = _resolve_final_path(
            f"usr/share/doc/{name}/copyright", state.paths, state.links
        )
        row = {
            "ecosystem": "dpkg",
            "name": name,
            **identity,
            "copyright_sha256": state.files.get(copyright_path, {}).get("sha256"),
            "file_contents": _dpkg_file_contents(state, name),
        }
        state.source_inventory[_source_identity(row)] = row
    for path, payload in state.tracked.items():
        if not path.endswith(".dist-info/METADATA"):
            continue
        name, version = _metadata_identity(payload)
        row = {
            "ecosystem": "python",
            "name": name,
            "version": version,
            "metadata_path": path,
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "record_sha256": state.files.get(
                path.removesuffix("METADATA") + "RECORD", {}
            ).get("sha256"),
            "file_contents": _python_file_contents(
                state, path.removesuffix("METADATA") + "RECORD"
            ),
        }
        state.source_inventory[_source_identity(row)] = row
    for payload in state.layer_payloads:
        layer = int(payload["layer"])
        state.layer_source_inventory.setdefault(
            layer, {key: dict(value) for key, value in state.source_inventory.items()}
        )
        state.layer_tracked.setdefault(layer, dict(state.tracked))
    state.layer_inventory = _source_layer_inventory(state)


def _dpkg_file_contents(state: _ScanState, package: str) -> dict[str, dict[str, object]]:
    """Bind each listed package path to the observed bytes in this layer state."""
    contents: dict[str, dict[str, object]] = {}
    status_path = "var/lib/dpkg/status"
    status = state.files.get(status_path, {})
    if isinstance(status.get("sha256"), str) and isinstance(status.get("size"), int):
        contents[status_path] = {
            "sha256": status["sha256"],
            "size": status["size"],
        }
    prefix = "var/lib/dpkg/info/"
    for list_path, payload in state.tracked.items():
        if not list_path.startswith(prefix) or not list_path.endswith(".list"):
            continue
        listed_package = list_path.removeprefix(prefix).removesuffix(".list")
        if listed_package.split(":", 1)[0] != package:
            continue
        observed_list = state.files.get(list_path, {})
        if isinstance(observed_list.get("sha256"), str) and isinstance(
            observed_list.get("size"), int
        ):
            contents[list_path] = {
                "sha256": observed_list["sha256"],
                "size": observed_list["size"],
            }
        for line in payload.decode("utf-8", errors="ignore").splitlines():
            if not line.startswith("/"):
                continue
            try:
                declared = W.safe_name(line.lstrip("/"))
            except W.ScanError:
                continue
            resolved = _resolve_final_path(declared, state.paths, state.links)
            for candidate in {declared, resolved}:
                if candidate is None:
                    continue
                observed = state.files.get(candidate, {})
                if isinstance(observed.get("sha256"), str) and isinstance(
                    observed.get("size"), int
                ):
                    contents[candidate] = {
                        "sha256": observed["sha256"],
                        "size": observed["size"],
                    }
    return contents


def _python_file_contents(
    state: _ScanState, record_path: str
) -> dict[str, dict[str, object]]:
    """Bind RECORD members, including hashless generated files, to observed bytes."""
    payload = state.tracked.get(record_path)
    if payload is None:
        return {}
    contents: dict[str, dict[str, object]] = {}
    for relative, _hash, _size in csv.reader(
        io.StringIO(payload.decode("utf-8", errors="ignore"))
    ):
        target = _record_target(record_path, relative)
        observed = state.files.get(target, {})
        if isinstance(observed.get("sha256"), str) and isinstance(
            observed.get("size"), int
        ):
            contents[target] = {
                "sha256": observed["sha256"],
                "size": observed["size"],
            }
    return contents


def _debian_artifact_filename_matches(source: str, version: str, name: str) -> bool:
    if not name.startswith(f"{source}_") or not version:
        return False
    basename = name.removeprefix(f"{source}_")
    suffixes = (
        ".orig.tar.gz", ".orig.tar.xz", ".orig.tar.bz2", ".orig.tar.lzma",
        ".orig.tar.zst", ".orig.tar.lz", ".debian.tar.gz", ".debian.tar.xz",
        ".debian.tar.bz2", ".debian.tar.lzma", ".debian.tar.zst", ".debian.tar.lz",
        ".diff.gz", ".diff.xz", ".diff.bz2", ".diff.lzma", ".diff.zst", ".diff.lz",
        ".orig.tar", ".tar.xz", ".tar.gz", ".tar.bz2", ".tar.zst", ".tar.lz", ".tar.lzma",
        ".dsc", ".asc",
    )
    for suffix in suffixes:
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    else:
        return False
    unepoch = version.split(":", 1)[-1]
    upstream = unepoch.rsplit("-", 1)[0] if "-" in unepoch else unepoch
    component_prefix = f"{upstream}.orig-"
    component = basename.removeprefix(component_prefix)
    return basename in {unepoch, upstream} or bool(
        basename.startswith(component_prefix)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+~-]*", component)
    )


def _python_artifact_filename_matches(package: str, version: str, name: str) -> bool:
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".zip", ".tgz"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return _normalize_distribution(stem) == _normalize_distribution(
                f"{package}-{version}"
            )
    return False


def _source_artifact_binding(row: dict[str, object], path: str) -> bool:
    """Require an artifact locator to encode its inventoried source identity."""
    name = PurePosixPath(path).name
    ecosystem = row.get("ecosystem")
    if ecosystem == "dpkg":
        source = row.get("source")
        version = str(row.get("source_version", ""))
        prefix = f"usr/share/doc/npa-habitat-sim/ubuntu-sources/{source}/"
        return (
            isinstance(source, str)
            and path.startswith(prefix)
            and _debian_artifact_filename_matches(source, version, name)
        )
    if ecosystem == "python":
        package = _normalize_distribution(str(row.get("name", "")))
        version = str(row.get("version", ""))
        prefix = f"usr/share/doc/npa-habitat-sim/python-sources/{package}/{version}/"
        return (
            bool(package)
            and bool(version)
            and path.startswith(prefix)
            and _python_artifact_filename_matches(package, version, name)
        )
    return False


def _source_component_key(row: dict[str, object]) -> tuple[str, str, str]:
    if row.get("ecosystem") == "dpkg":
        return ("dpkg", str(row.get("source")), str(row.get("source_version")))
    return ("python", str(row.get("name")), str(row.get("version")))


def _source_checksum_rows(value: object, width: int) -> dict[str, tuple[int, str]] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    rows: dict[str, tuple[int, str]] = {}
    for line in value.strip().splitlines():
        fields = line.split()
        if len(fields) != 3:
            return None
        digest, size, name = fields
        if not re.fullmatch(rf"[0-9a-f]{{{width}}}", digest):
            return None
        if not size.isdecimal() or int(size) <= 0:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_~-]*", name):
            return None
        if name in rows:
            return None
        rows[name] = (int(size), digest)
    return rows or None


def _debian_source_fields(payload: bytes) -> dict[str, str] | None:
    payload = _debian_clearsigned_body(payload)
    if payload is None:
        return None
    try:
        lines = payload.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None
    fields: dict[str, str] = {}
    previous: str | None = None
    for line in lines:
        if line.startswith((" ", "\t")):
            if previous is None:
                return None
            fields[previous] += "\n" + line.strip()
            continue
        if ":" not in line:
            return None
        key, value = line.split(":", 1)
        if not key or key in fields:
            return None
        fields[key] = value.strip()
        previous = key
    return fields


def _debian_clearsigned_body(payload: bytes) -> bytes | None:
    """Extract the signed body without treating the signature as fields."""
    lines = payload.splitlines()
    if not lines or lines[0] != b"-----BEGIN PGP SIGNED MESSAGE-----":
        return payload
    separator = next(
        (index for index, line in enumerate(lines[1:], 1) if not line), None
    )
    if separator is None:
        return None
    try:
        end = lines.index(b"-----BEGIN PGP SIGNATURE-----", separator + 1)
    except ValueError:
        return None
    body = []
    for line in lines[separator + 1 : end]:
        body.append(line[2:] if line.startswith(b"- ") else line)
    return b"\n".join(body) + b"\n"


def _python_metadata_fields(payload: bytes) -> dict[str, str] | None:
    """Parse RFC822 metadata while requiring one Name and Version field."""
    try:
        message = BytesParser(policy=policy.compat32).parsebytes(payload)
    except (TypeError, ValueError):
        return None
    names = [value.strip() for value in message.get_all("Name", [])]
    versions = [value.strip() for value in message.get_all("Version", [])]
    if len(names) != 1 or len(versions) != 1:
        return None
    return {key: value.strip() for key, value in message.items()} | {
        "Name": names[0],
        "Version": versions[0],
    }


def _dsc_source_matches(
    state: _ScanState,
    row: dict[str, object],
    dsc_path: str,
    paths: set[str],
) -> bool:
    payload = state.tracked.get(dsc_path)
    fields = _debian_source_fields(payload) if payload is not None else None
    if not _dsc_identity_matches(row, fields):
        return False
    files = _source_checksum_rows(fields.get("Files"), 32)
    checksums = _source_checksum_rows(fields.get("Checksums-Sha256"), 64)
    if files is None or checksums is None or set(files) != set(checksums):
        return False
    if paths != _dsc_expected_paths(dsc_path, files, paths):
        return False
    return _dsc_content_matches(state, dsc_path, files, checksums)


def _dsc_identity_matches(row, fields) -> bool:
    return fields is not None and (fields.get("Source"), fields.get("Version")) == (
        row.get("source"),
        row.get("source_version"),
    )


def _dsc_expected_paths(dsc_path, files, paths):
    parent = str(PurePosixPath(dsc_path).parent)
    expected = {dsc_path} | {f"{parent}/{name}" for name in files}
    expected |= {
        path
        for path in paths - expected
        if path.startswith(parent + "/")
        and path.endswith(".asc")
        and path.removesuffix(".asc") in expected
    }
    return expected


def _dsc_content_matches(state, dsc_path, files, checksums):
    parent = str(PurePosixPath(dsc_path).parent)
    for name, (size, md5) in files.items():
        path = f"{parent}/{name}"
        artifact = state.files.get(path, {})
        content = state.tracked.get(path)
        sha_size, sha256 = checksums[name]
        if (
            content is None
            or artifact.get("size") != size
            or artifact.get("sha256") != sha256
            or len(content) != size
            or hashlib.md5(content, usedforsecurity=False).hexdigest() != md5
            or hashlib.sha256(content).hexdigest() != sha256
            or sha_size != size
        ):
            return False
    return True


def _safe_archive_name(name: str) -> bool:
    return bool(name) and "\\" not in name and not name.startswith(("/", "\\")) and all(
        part not in {"", ".", ".."} for part in PurePosixPath(name).parts
    )


def _archive_limits(payload: bytes) -> tuple[int, int, int]:
    return (
        SOURCE_ARCHIVE_MAX_MEMBERS,
        SOURCE_ARCHIVE_MAX_MEMBER_BYTES,
        min(SOURCE_ARCHIVE_MAX_EXPANDED_BYTES, max(1, len(payload)) * SOURCE_ARCHIVE_MAX_COMPRESSION_RATIO),
    )


def _canonical_metadata_name(row: dict[str, object], name: str, root: str) -> bool:
    parts = PurePosixPath(name).parts
    return len(parts) == 2 and parts[0] == root and parts[1] in {"PKG-INFO", "METADATA"}


def _archive_root_identity(
    row: dict[str, object], root: str, fields: dict[str, str] | None
) -> bool:
    if fields is None or "-" not in root:
        return False
    root_name, _, root_version = root.rpartition("-")
    return (
        _normalize_distribution(root_name) == _normalize_distribution(str(row.get("name", "")))
        and root_version == str(row.get("version", ""))
        and _normalize_distribution(fields.get("Name", ""))
        == _normalize_distribution(str(row.get("name", "")))
        and fields.get("Version") == str(row.get("version", ""))
    )


def _read_metadata(stream) -> bytes | None:
    metadata = stream.read(SOURCE_ARCHIVE_MAX_METADATA_BYTES + 1)
    return metadata if len(metadata) <= SOURCE_ARCHIVE_MAX_METADATA_BYTES else None


def _archive_member_budget(
    row, name: str, size: int, state: dict[str, object], expanded_limit: int
) -> bool:
    if not _safe_archive_name(name):
        return False
    parts = PurePosixPath(name).parts
    root = parts[0] if parts else ""
    if state["root"] is None:
        state["root"] = root
    if state["root"] != root:
        return False
    names = state["names"]
    if name in names or size > SOURCE_ARCHIVE_MAX_MEMBER_BYTES:
        return False
    if int(state["total"]) + size > expanded_limit:
        return False
    names.add(name)
    state["total"] = int(state["total"]) + size
    return True


def _archive_metadata_member(row, name: str, payload: bytes | None, state) -> bool:
    if PurePosixPath(name).name not in {"PKG-INFO", "METADATA"}:
        state["source_files"] = int(state["source_files"]) + 1
        return True
    if not _canonical_metadata_name(row, name, str(state["root"] or "")):
        return False
    if state["metadata"] is not None:
        return False
    if payload is None or _read_metadata(io.BytesIO(payload)) is None:
        return False
    state["metadata"] = payload
    return True


def _archive_directory_member(name: str, state: dict[str, object]) -> bool:
    if not _safe_archive_name(name):
        return False
    parts = PurePosixPath(name).parts
    root = parts[0] if parts else ""
    if state["root"] is None:
        state["root"] = root
    if state["root"] != root or name in state["names"]:
        return False
    state["names"].add(name)
    return True


def _tar_archive_member(row, archive, member, state, expanded_limit) -> bool:
    if member.issym() or member.islnk():
        return False
    if not _archive_member_budget(row, member.name, member.size, state, expanded_limit):
        return False
    if member.isdir():
        return True
    if not member.isfile():
        return False
    stream = archive.extractfile(member)
    payload = _read_metadata(stream) if stream is not None else None
    return _archive_metadata_member(row, member.name, payload, state)


def _zip_archive_member(row, archive, info, state, expanded_limit) -> bool:
    name = info.filename
    if info.is_dir() or name.endswith("/"):
        return (
            info.file_size == 0
            and info.compress_size == 0
            and _archive_directory_member(name, state)
        )
    if info.file_size and (
        not info.compress_size
        or info.file_size > info.compress_size * SOURCE_ARCHIVE_MAX_COMPRESSION_RATIO
    ):
        return False
    if not _archive_member_budget(row, name, info.file_size, state, expanded_limit):
        return False
    with archive.open(info) as stream:
        payload = _read_metadata(stream)
    return _archive_metadata_member(row, name, payload, state)


def _archive_metadata_result(state: dict[str, object]) -> bytes | None:
    metadata = state["metadata"]
    if int(state["source_files"]) == 0 or metadata is None:
        return None
    fields = _python_metadata_fields(metadata)
    if not _archive_root_identity(state["row"], str(state["root"] or ""), fields):
        return None
    return metadata


def _tar_archive_metadata(row: dict[str, object], payload: bytes) -> bytes | None:
    try:
        _, _, expanded_limit = _archive_limits(payload)
        state = {
            "names": set(), "total": 0, "source_files": 0, "metadata": None,
            "root": None, "row": row,
        }
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            for count, member in enumerate(archive, 1):
                if count > SOURCE_ARCHIVE_MAX_MEMBERS or not _tar_archive_member(
                    row, archive, member, state, expanded_limit
                ):
                    return None
        return _archive_metadata_result(state)
    except (OSError, tarfile.TarError):
        return None


def _zip_archive_metadata(row: dict[str, object], payload: bytes) -> bytes | None:
    try:
        _, _, expanded_limit = _archive_limits(payload)
        state = {
            "names": set(), "total": 0, "source_files": 0, "metadata": None,
            "root": None, "row": row,
        }
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > SOURCE_ARCHIVE_MAX_MEMBERS:
                return None
            for info in infos:
                if not _zip_archive_member(row, archive, info, state, expanded_limit):
                    return None
        return _archive_metadata_result(state)
    except (OSError, zipfile.BadZipFile):
        return None


def _python_source_matches(row: dict[str, object], payload: bytes | None) -> bool:
    if payload is None:
        return False
    if len(payload) > SOURCE_ARCHIVE_MAX_MEMBER_BYTES:
        return False
    metadata = _tar_archive_metadata(row, payload)
    if metadata is None:
        metadata = _zip_archive_metadata(row, payload)
    if metadata is None:
        return False
    fields = _python_metadata_fields(metadata)
    return fields is not None and (
        _normalize_distribution(fields.get("Name", ""))
        == _normalize_distribution(str(row.get("name", "")))
        and fields.get("Version") == row.get("version")
    )


def _delivered_source_matches(
    state: _ScanState, row: dict[str, object], artifacts: object
) -> bool:
    if not isinstance(artifacts, list) or not artifacts:
        return False
    paths: set[str] = set()
    dsc_paths: list[str] = []
    for artifact in artifacts:
        path = _source_artifact_path(row, artifact, paths)
        if path is None or not _source_artifact_content_matches(state, artifact, path):
            return False
        paths.add(path)
        if path.endswith(".dsc"):
            dsc_paths.append(path)
    if row.get("ecosystem") == "dpkg":
        if len(dsc_paths) != 1:
            return False
        if not _dsc_source_matches(state, row, dsc_paths[0], paths):
            return False
    elif row.get("ecosystem") == "python":
        if len(paths) != 1 or not _python_source_matches(
            row, state.tracked.get(next(iter(paths)))
        ):
            return False
    return True


def _source_artifact_path(
    row: dict[str, object], artifact: object, paths: set[str]
) -> str | None:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
        return None
    path = artifact["path"]
    if not isinstance(path, str) or path in paths:
        return None
    if not _source_artifact_binding(row, path):
        return None
    if any(part in {"", ".", ".."} for part in path.split("/")):
        return None
    if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
        return None
    if not isinstance(artifact["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", artifact["sha256"]
    ):
        return None
    return path


def _source_artifact_content_matches(
    state: _ScanState, artifact: object, path: str
) -> bool:
    actual = state.files.get(path, {})
    return (
        isinstance(artifact, dict)
        and actual.get("sha256") == artifact["sha256"]
        and actual.get("size") == artifact["bytes"]
    )


def _source_delivery_findings(state: _ScanState, contract: dict) -> list[dict]:
    closure = contract.get("corresponding_source_closure", {})
    inventory = state.source_inventory
    findings = []
    if closure.get("status") != "complete-accompanying-source":
        findings.append({"code": "corresponding_source_closure_pending"})
    if not inventory or closure.get("inventory_sha256") != _source_identity(inventory):
        findings.append({"code": "corresponding_source_inventory_mismatch"})
    if closure.get("status") == "complete-accompanying-source":
        expected_layers = state.layer_inventory
        if (
            closure.get("layer_inventory") != expected_layers
            or closure.get("layer_inventory_sha256") != _source_identity(expected_layers)
        ):
            findings.append({"code": "corresponding_source_layer_inventory_mismatch"})
        findings.extend(
            {"code": "corresponding_source_layer_payload_unowned", "path": row["path"]}
            for row in expected_layers
            if not row.get("owners")
        )
    records = closure.get("records", {})
    if not isinstance(records, dict) or set(records) != set(inventory):
        return findings + [{"code": "corresponding_source_population_mismatch"}]
    # Conservative: every package requires accompanying source. No inference
    # from a top-level license or a caller-provided "not copyleft" exemption.
    claimed_artifacts: dict[str, tuple[str, str, str]] = {}
    for identity, row in inventory.items():
        findings.extend(
            _source_delivery_record_findings(
                state, identity, row, records[identity], claimed_artifacts
            )
        )
    return findings


def _source_delivery_record_findings(
    state: _ScanState,
    identity: str,
    row: dict[str, object],
    record: object,
    claimed_artifacts: dict[str, tuple[str, str, str]],
) -> list[dict]:
    findings: list[dict] = []
    if not isinstance(record, dict) or record.get("component") != row:
        return [{"code": "corresponding_source_component_mismatch"}]
    artifacts = record.get("artifacts")
    if isinstance(artifacts, list):
        component_key = _source_component_key(row)
        for artifact in artifacts:
            path = artifact.get("path") if isinstance(artifact, dict) else None
            if isinstance(path, str) and claimed_artifacts.setdefault(path, component_key) != component_key:
                findings.append({"code": "corresponding_source_artifact_reused", "path": path})
    if not _delivered_source_matches(state, row, artifacts):
        findings.append({"code": "corresponding_source_delivery_mismatch", "component": identity})
    if row.get("ecosystem") == "dpkg" and not row.get("copyright_sha256"):
        findings.append({"code": "corresponding_source_copyright_missing"})
    if row.get("ecosystem") == "python" and not row.get("record_sha256"):
        findings.append({"code": "corresponding_source_python_record_missing"})
    return findings


def _member_policy_findings(
    path: str, kind: str, layer: int, entry: int, contract: dict[str, object]
) -> list[dict[str, object]]:
    forbidden = contract["forbidden_path_patterns"]
    nondirectory = contract["forbidden_nondirectory_path_patterns"]
    blocked = any(re.search(pattern, path, re.IGNORECASE) for pattern in forbidden)
    blocked |= kind != "directory" and any(
        re.search(pattern, path, re.IGNORECASE) for pattern in nondirectory
    )
    findings = []
    if blocked:
        findings.append({"code": "forbidden_path", "layer": layer, "entry": entry})
    if kind == "unsupported":
        findings.append({"code": "unsupported_member", "layer": layer, "entry": entry})
    return findings


def _read_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    path: str,
    retained_bytes: int = 0,
) -> tuple[dict[str, object], bytes | None, bytes | None]:
    W.require(member.size <= RETAINED_MEMBER_MAX_BYTES, "habitat_oci_member_limit")
    body = archive.extractfile(member)
    W.require(body is not None, "habitat_oci_regular_file_read")
    if _track_bytes(path):
        tracked = _read_bounded_member(body, member.size, retained_bytes)
        digest = hashlib.sha256(tracked).hexdigest()
        size = len(tracked)
        prefix = tracked[:4]
        elf = tracked if prefix == b"\x7fELF" else None
    else:
        digest, size, prefix, elf = _regular_hash(
            body, retained_bytes=retained_bytes, expected_size=member.size
        )
        tracked = None
    W.require(size == member.size, "habitat_oci_regular_file_size")
    return {"sha256": digest, "size": size, "elf": prefix == b"\x7fELF"}, tracked, elf


def _final_state_maps(state: _ScanState) -> tuple[dict, ...]:
    """Return only final-filesystem maps, excluding immutable layer evidence."""
    return (
        state.paths,
        state.files,
        state.links,
        state.metadata,
        state.tracked,
        state.elf,
    )


def _require_directory_ancestors(paths: dict[str, str], path: str) -> None:
    """Permit existing implicit directories, never a non-directory parent."""
    for ancestor in PurePosixPath(path).parents:
        name = str(ancestor)
        if name not in {".", ""}:
            W.require(
                paths.get(name) in {None, "directory"},
                "habitat_oci_parent_not_directory",
            )


def _require_reachable_state(state: _ScanState) -> None:
    """Require consistent reachable populations before computing any closure."""
    files = {path for path, kind in state.paths.items() if kind in {"file", "hardlink"}}
    links = {
        path for path, kind in state.paths.items() if kind in {"symlink", "hardlink"}
    }
    W.require(
        set(state.files) == files
        and set(state.links) == links
        and set(state.metadata) == set(state.paths)
        and set(state.tracked) <= files
        and set(state.elf) <= files,
        "habitat_oci_final_state_population",
    )
    for path, kind in state.paths.items():
        _require_directory_ancestors(state.paths, path)
        W.require(
            state.metadata[path]["kind"] == kind
            and (path not in links or state.links[path][0] == kind),
            "habitat_oci_final_state_kind",
        )


def _replace_path(state: _ScanState, path: str, kind: str) -> None:
    """Keep children only for directories; replace other types as subtrees."""
    for mapping in _final_state_maps(state):
        if kind == "directory":
            mapping.pop(path, None)
        else:
            _remove_tree(mapping, path)


def _record_regular_member(state, archive, member, path, location, contract) -> str:
    """Retain byte accounting and independent per-layer payload findings."""
    file_row, tracked, elf = _read_member(
        archive, member, path, retained_bytes=state.retained_bytes
    )
    state.files[path] = file_row
    state.regular_files += 1
    state.content_bytes += int(file_row["size"])
    if tracked is not None:
        state.tracked[path] = tracked
        state.retained_bytes += len(tracked)
    if elf is not None:
        state.elf[path] = elf
        if tracked is None:
            state.retained_bytes += len(elf)
    if file_row["sha256"] in contract["forbidden_content_sha256"]:
        state.findings.append({"code": "forbidden_payload_hash", **location})
    return str(file_row["sha256"])


def _hardlink_target(state: _ScanState, linkname: str) -> str:
    """Resolve a hardlink target while it still denotes a regular inode."""
    target_name = W.safe_name(linkname.lstrip("/"))
    W.require(
        state.paths.get(target_name) in {"file", "hardlink"},
        "habitat_oci_hardlink_target",
    )
    target = _resolve_final_path(target_name, state.paths, state.links)
    W.require(
        target is not None and target in state.files, "habitat_oci_hardlink_target"
    )
    return target


def _record_hardlink_member(
    state: _ScanState, member: tarfile.TarInfo, path: str
) -> None:
    """Snapshot hardlink bytes and metadata instead of retaining a live pathname."""
    target = _hardlink_target(state, member.linkname)
    state.files[path] = dict(state.files[target])
    if target in state.tracked:
        state.tracked[path] = state.tracked[target]
    if target in state.elf:
        state.elf[path] = state.elf[target]


def _record_layer_payload(state, path: str, layer: int, entry: int) -> None:
    """Retain immutable per-layer identity for every non-directory payload."""
    row = state.files.get(path)
    W.require(row is not None, "habitat_oci_layer_payload_identity")
    state.layer_payloads.append(
        {
            "layer": layer,
            "entry": entry,
            "path": path,
            "sha256": row["sha256"],
            "bytes": row["size"],
        }
    )


def _record_member(
    state: _ScanState,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    path: str,
    layer: int,
    entry: int,
    contract: dict[str, object],
) -> None:
    """Record a reachable member without rewriting earlier layer evidence."""
    _require_directory_ancestors(state.paths, path)
    kind = _member_kind(member)
    state.findings.extend(_member_policy_findings(path, kind, layer, entry, contract))
    _replace_path(state, path, kind)
    state.paths[path] = kind
    metadata = {
        "kind": kind,
        "uid": member.uid,
        "gid": member.gid,
        "mode": member.mode & 0o7777,
    }
    state.metadata[path] = metadata
    event = {"path": path, "layer": layer, **metadata}
    if member.issym() or member.islnk():
        state.links[path] = (kind, member.linkname)
    if member.islnk():
        _record_hardlink_member(state, member, path)
    if member.isfile():
        event["sha256"] = _record_regular_member(
            state, archive, member, path, {"layer": layer, "entry": entry}, contract
        )
        _record_layer_payload(state, path, layer, entry)
    elif member.islnk():
        _record_layer_payload(state, path, layer, entry)
    state.events.append(event)


def _observe_layer_member(lower, current, archive, member, path, location, contract):
    """Apply removals only to lower layers; retain current observations separately."""
    layer, entry = location
    if _apply_whiteout(*_whiteout_arguments(lower, path, member)):
        current.regular_files += 1
        current.content_bytes += int(member.size)
        lower.events.append({"path": path, "layer": layer, "kind": "whiteout"})
        return
    if _member_kind(member) != "directory":
        W.require(
            not any(name.startswith(path + "/") for name in current.paths),
            "habitat_oci_parent_not_directory",
        )
    _record_member(current, archive, member, path, layer, entry, contract)


def _whiteout_arguments(state: _ScanState, path: str, member) -> tuple:
    """Select all six lower-layer maps without exposing current-layer entries."""
    paths, *related = _final_state_maps(state)
    return paths, path, member, *related


def _complete_layer(lower: _ScanState, current: _ScanState, seen: set[str]) -> None:
    """Merge verified observations parent-first, then validate every member parent."""
    _require_reachable_state(current)
    for path in sorted(current.paths, key=lambda name: (name.count("/"), name)):
        _require_directory_ancestors(lower.paths, path)
        _replace_path(lower, path, current.paths[path])
        for target, source in zip(_final_state_maps(lower), _final_state_maps(current)):
            if path in source:
                target[path] = source[path]
    for path in seen:
        _require_directory_ancestors(lower.paths, path)
    _require_reachable_state(lower)
    lower.regular_files += current.regular_files
    lower.content_bytes += current.content_bytes
    lower.retained_bytes = current.retained_bytes


def _scan_layer(
    fd: int,
    row: dict[str, object],
    layer: int,
    contract: dict[str, object],
    state: _ScanState,
) -> None:
    """Keep complete-layer inventory separate from a single streamed observation set."""
    _verify_diff_id(fd, row)
    seen: set[str] = set()
    current = _ScanState(
        events=state.events,
        findings=state.findings,
        layer_payloads=state.layer_payloads,
        retained_bytes=state.retained_bytes,
        contract=contract,
    )
    with tarfile.open(fileobj=_decoded(fd, row), mode="r|") as archive:
        for entry, member in enumerate(archive):
            state.entries += 1
            path = W.safe_name(member.name)
            W.require(path not in seen, "habitat_oci_duplicate_layer_path")
            seen.add(path)
            _observe_layer_member(
                state, current, archive, member, path, (layer, entry), contract
            )
    _complete_layer(state, current, seen)


def _required_path_findings(
    paths: dict[str, str], required_paths: list[str]
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for required in required_paths:
        target = required.lstrip("/")
        kind = paths.get(target)
        if kind is not None and kind not in {"file", "directory"}:
            findings.append(
                {"code": "required_path_not_file_or_directory", "path": required}
            )
            continue
        if kind is None and not any(
            path.startswith(target.rstrip("/") + "/") for path in paths
        ):
            findings.append({"code": "required_path_missing", "path": required})
            continue
        for ancestor in PurePosixPath(target).parents:
            name = str(ancestor)
            if name not in {".", ""} and paths.get(name) not in {None, "directory"}:
                findings.append(
                    {"code": "required_path_ancestor_not_directory", "path": required}
                )
                break
    return findings


def _final_metadata_findings(
    metadata: dict[str, dict[str, object]], contract: dict[str, object]
) -> list[dict[str, object]]:
    findings = []
    for required, expected in contract.get("required_final_metadata", {}).items():
        if metadata.get(required.lstrip("/")) != expected:
            findings.append(
                {"code": "required_final_metadata_mismatch", "path": required}
            )
    return findings


def _account_boundary_findings(tracked: dict[str, bytes]) -> list[dict[str, object]]:
    findings = []
    try:
        passwd = tracked["etc/passwd"].decode("utf-8", errors="strict").splitlines()
        group = tracked["etc/group"].decode("utf-8", errors="strict").splitlines()
    except (KeyError, UnicodeDecodeError):
        return [{"code": "runtime_account_database_unreadable"}]
    users = [row.split(":") for row in passwd if row.startswith("ubuntu:")]
    groups = [row.split(":") for row in group if row.startswith("ubuntu:")]
    if users != [["ubuntu", "x", "1000", "1000", "", "/home/ubuntu", "/bin/bash"]]:
        findings.append({"code": "runtime_user_identity_mismatch"})
    if len(groups) != 1 or len(groups[0]) != 4 or groups[0][1:3] != ["x", "1000"]:
        findings.append({"code": "runtime_group_identity_mismatch"})
    return findings


def _whiteout_affects(path: str, target: str) -> bool:
    posix = PurePosixPath(path)
    if posix.name == ".wh..wh..opq":
        parent = "" if str(posix.parent) == "." else str(posix.parent)
        return not parent or target == parent or target.startswith(parent + "/")
    if not posix.name.startswith(".wh."):
        return False
    removed = str(posix.parent / posix.name.removeprefix(".wh."))
    return target == removed or target.startswith(removed + "/")


def _executable_source_findings(
    state: _ScanState, contract: dict[str, object]
) -> list[dict[str, object]]:
    bindings = contract.get("executable_source_bindings", {})
    expected_paths = {path.lstrip("/") for path in bindings}
    observed = {
        path
        for path in state.paths
        if path.startswith("opt/npa-runtime/npa/") and state.paths[path] != "directory"
    }
    findings = []
    if observed != {
        path for path in expected_paths if path.startswith("opt/npa-runtime/npa/")
    }:
        findings.append({"code": "executable_source_path_set_mismatch"})
    for destination, expected in bindings.items():
        target = destination.lstrip("/")
        layer_expected = {
            key: expected[key] for key in ("kind", "uid", "gid", "sha256")
        }
        for event in state.events:
            if event["kind"] == "whiteout" and _whiteout_affects(
                str(event["path"]), target
            ):
                findings.append(
                    {"code": "executable_source_layer_policy", "path": destination}
                )
            elif event["path"] == target and any(
                event.get(key) != value for key, value in layer_expected.items()
            ):
                findings.append(
                    {"code": "executable_source_layer_policy", "path": destination}
                )
    return findings


def _runtime_closures(
    state: _ScanState, contract: dict[str, object], expected: tuple[str, str, str]
) -> dict[str, object]:
    _require_reachable_state(state)
    dpkg = _dpkg_findings(
        state.paths, state.files, state.links, state.tracked, contract, expected[0]
    )
    state.findings.extend(dpkg[0])
    python = _python_findings(
        state.paths, state.files, state.links, state.tracked, contract, expected[1]
    )
    state.findings.extend(python[0])
    native = _native_findings(
        state.elf,
        state.paths,
        state.links,
        state.files,
        dpkg[2],
        python[4],
        expected[2],
    )
    state.findings.extend(native[0])
    for package in contract["forbidden_packages"]:
        if package in dpkg[1]:
            state.findings.append(
                {"code": "forbidden_runtime_package", "package": package}
            )
    return {"dpkg": dpkg, "python": python, "native": native}


def _scan_report(
    state: _ScanState,
    closure: dict[str, object],
    expected: tuple[str, str, str],
    projection_count: int,
) -> dict[str, object]:
    dpkg, python, native = closure["dpkg"], closure["python"], closure["native"]
    return {
        "entries_read": state.entries,
        "corresponding_source_inventory": state.source_inventory,
        "corresponding_source_inventory_sha256": _source_identity(
            state.source_inventory
        ),
        "corresponding_source_layer_inventory": state.layer_inventory,
        "corresponding_source_layer_inventory_sha256": _source_identity(
            state.layer_inventory
        ),
        "regular_files_read": state.regular_files,
        "content_bytes_read": state.content_bytes,
        "final_path_count": len(state.paths),
        "installed_package_count": len(dpkg[1]),
        "dpkg_inventory": dpkg[5],
        "dpkg_inventory_sha256": dpkg[4],
        "expected_dpkg_inventory_sha256": expected[0],
        "locked_runtime_apt_package_count": dpkg[3],
        "python_distribution_count": python[1],
        "python_record_files_verified": python[2],
        "allowed_missing_python_record_files": python[3],
        "python_venv_inventory": python[5],
        "python_venv_inventory_sha256": python[6],
        "expected_python_venv_inventory_sha256": expected[1],
        "projected_source_file_count": projection_count,
        "native_elf_count": len(state.elf),
        "native_elf_closure": native[2],
        "native_elf_closure_sha256": native[1],
        "expected_native_closure_sha256": expected[2],
        "findings": state.findings,
    }


def _scan_layers(
    fd: int,
    layers: list[dict[str, object]],
    contract: dict[str, object],
    expected_dpkg_inventory_sha256: str,
    expected_python_venv_inventory_sha256: str,
    expected_native_closure_sha256: str,
) -> dict[str, object]:
    state = _ScanState()
    state.contract = contract
    for layer, row in enumerate(layers):
        _scan_layer(fd, row, layer, contract, state)
        _record_source_population(state)
    _require_reachable_state(state)
    state.findings.extend(_source_delivery_findings(state, contract))
    state.findings.extend(
        _required_path_findings(state.paths, contract["required_final_paths"])
    )
    state.findings.extend(_required_file_findings(state.files, contract))
    state.findings.extend(_final_metadata_findings(state.metadata, contract))
    state.findings.extend(_executable_source_findings(state, contract))
    if "/etc/passwd" in contract.get("required_final_metadata", {}):
        state.findings.extend(_account_boundary_findings(state.tracked))
    projection, count = _source_projection_findings(
        state.paths, state.files, state.tracked, contract
    )
    state.findings.extend(projection)
    expected = (
        expected_dpkg_inventory_sha256,
        expected_python_venv_inventory_sha256,
        expected_native_closure_sha256,
    )
    closure = _runtime_closures(state, contract, expected)
    return _scan_report(state, closure, expected, count)


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
    if runtime.get("Cmd") != [
        "python3",
        "-m",
        "npa.workflows.habitat_sim_smoke",
        "--help",
    ]:
        findings.append({"code": "unexpected_default_command"})
    if runtime.get("ExposedPorts") not in (None, {}):
        findings.append({"code": "unexpected_exposed_ports"})
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


def _base_findings(
    layers: list[dict[str, object]], contract: dict[str, object]
) -> list[dict[str, object]]:
    expected = contract.get("required_base_diff_ids")
    if not isinstance(expected, list) or not expected:
        return [{"code": "base_diff_id_lock_invalid"}]
    observed = [row["diff_id"] for row in layers[: len(expected)]]
    if observed != expected:
        return [{"code": "base_diff_id_lock_mismatch"}]
    return []


def verify(
    fd: int,
    length: int,
    expected_id: str,
    contract: dict[str, object],
    archive_sha256: str,
    expected_source_revision: str,
    expected_dpkg_inventory_sha256: str,
    expected_python_venv_inventory_sha256: str,
    expected_native_closure_sha256: str,
) -> dict[str, object]:
    """Verify the complete saved OCI bytes against the Habitat contract.

    Args:
        fd: Descriptor for the immutable archive.
        length: Exact archive byte length.
        expected_id: Required OCI index digest.
        contract: Byte, path, source and runtime policy.
        archive_sha256: Caller-bound archive hash.
        expected_source_revision: Required source commit.
        expected_dpkg_inventory_sha256: Bound package inventory hash.
        expected_python_venv_inventory_sha256: Bound Python inventory hash.
        expected_native_closure_sha256: Bound native closure hash.

    Returns:
        Graph, population, closure and policy findings; valid only if none exist.

    Raises:
        W.ScanError: Archive, parser or identity validation fails.
        OSError: The archive cannot be read.
        ValueError: Embedded control data is malformed.
    """

    expected = (
        expected_dpkg_inventory_sha256,
        expected_python_venv_inventory_sha256,
        expected_native_closure_sha256,
    )
    _verify_expected_identities(expected_source_revision, expected)
    result = inspect(fd, length, expected_id)
    return _verified_report(
        fd,
        expected_id,
        contract,
        archive_sha256,
        expected_source_revision,
        expected,
        result,
    )


def _verify_expected_identities(expected_source_revision, expected):
    W.require(
        re.fullmatch(r"[0-9a-f]{40}", expected_source_revision) is not None,
        "habitat_oci_expected_source_revision",
    )
    for value in expected:
        W.require(
            re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "habitat_oci_expected_runtime_closure",
        )


def _verified_report(
    fd,
    expected_id,
    contract,
    archive_sha256,
    expected_source_revision,
    expected,
    result,
):
    layers = result["layers"]
    payload = _scan_layers(
        fd,
        layers,
        contract,
        *expected,
    )
    config = _config(fd, result["image_config_digest"])
    findings = [
        *_base_findings(layers, contract),
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
