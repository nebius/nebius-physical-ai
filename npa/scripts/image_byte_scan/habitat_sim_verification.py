#!/usr/bin/env python3
"""Verify the closed Habitat-Sim OCI graph and its complete layer payload."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import posixpath
import re
import struct
import tarfile

from . import core as W
from . import oci_graph as G


SCHEMA = "npa.habitat-sim.oci-verification.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}


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


def _regular_hash(stream) -> tuple[str, int, bytes, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    elf_chunks: list[bytes] | None = None
    while chunk := stream.read(W.CHUNK):
        if len(prefix) < 4:
            prefix += chunk[: 4 - len(prefix)]
            if len(prefix) == 4 and prefix == b"\x7fELF":
                elf_chunks = []
        if elf_chunks is not None:
            elf_chunks.append(chunk)
        digest.update(chunk)
        size += len(chunk)
    return (
        digest.hexdigest(),
        size,
        prefix,
        (b"".join(elf_chunks) if elf_chunks is not None else None),
    )


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
    list_bindings = _package_list_bindings(tracked, installed)
    inventory = _package_inventory(
        installed, list_bindings, final_paths, final_links, final_files, findings
    )
    serialized = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    inventory_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
    if inventory_sha256 != expected_inventory_sha256:
        findings.append({"code": "runtime_dpkg_inventory_lock_mismatch"})
    owners = _dpkg_file_owners(tracked, installed, final_paths, final_links, findings)
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


def _package_list_bindings(tracked, installed):
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
            list_bindings[package].append(
                {
                    "path": control_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return list_bindings


def _package_inventory(
    installed, list_bindings, final_paths, final_links, final_files, findings
):
    inventory: dict[str, dict[str, object]] = {}
    for package, identity in installed.items():
        package_lists = sorted(list_bindings[package], key=lambda row: row["path"])
        if len(package_lists) != 1:
            findings.append(
                {
                    "code": "runtime_package_file_list_population",
                    "package": package,
                }
            )
        copyright_path = f"usr/share/doc/{package}/copyright"
        resolved_copyright = _resolve_final_path(
            copyright_path, final_paths, final_links
        )
        copyright_sha256 = final_files.get(resolved_copyright or "", {}).get("sha256")
        if resolved_copyright is None or not isinstance(copyright_sha256, str):
            findings.append(
                {"code": "runtime_package_copyright_missing", "package": package}
            )
        inventory[package] = {
            **identity,
            "file_lists": package_lists,
            "copyright_path": resolved_copyright or "",
            "copyright_sha256": copyright_sha256 or "",
        }
    return inventory


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
        for line in payload.decode("utf-8", errors="strict").splitlines():
            if not line.startswith("/"):
                findings.append(
                    {"code": "runtime_package_file_list_invalid", "package": package}
                )
                continue
            try:
                path = W.safe_name(line.lstrip("/"))
            except W.ScanError:
                findings.append(
                    {"code": "runtime_package_file_list_invalid", "package": package}
                )
                continue
            resolved = _resolve_final_path(path, final_paths, final_links)
            if resolved is not None:
                owners.setdefault(resolved, set()).add(package)
    return owners


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
        expected_bytes = base64.b64decode(
            encoded + "=", altchars=b"-_", validate=True
        )
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
    findings: list[dict[str, object]] = field(default_factory=list)
    entries: int = 0
    regular_files: int = 0
    content_bytes: int = 0


def _source_identity(row: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        }
        state.source_inventory[_source_identity(row)] = row


def _delivered_source_matches(state: _ScanState, artifacts: object) -> bool:
    if not isinstance(artifacts, list) or not artifacts:
        return False
    paths = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "bytes",
            "sha256",
        }:
            return False
        path = artifact["path"]
        if not isinstance(path, str) or path in paths:
            return False
        paths.add(path)
        if not path.startswith("usr/share/doc/npa-habitat-sim/"):
            return False
        if any(part in {"", ".", ".."} for part in path.split("/")):
            return False
        if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
            return False
        if not isinstance(artifact["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", artifact["sha256"]
        ):
            return False
        actual = state.files.get(path, {})
        if (
            actual.get("sha256") != artifact["sha256"]
            or actual.get("size") != artifact["bytes"]
        ):
            return False
    return True


def _source_delivery_findings(state: _ScanState, contract: dict) -> list[dict]:
    closure = contract.get("corresponding_source_closure", {})
    inventory = state.source_inventory
    findings = []
    if closure.get("status") != "complete-accompanying-source":
        findings.append({"code": "corresponding_source_closure_pending"})
    if not inventory or closure.get("inventory_sha256") != _source_identity(inventory):
        findings.append({"code": "corresponding_source_inventory_mismatch"})
    records = closure.get("records", {})
    if not isinstance(records, dict) or set(records) != set(inventory):
        return findings + [{"code": "corresponding_source_population_mismatch"}]
    # Conservative: every package requires accompanying source. No inference
    # from a top-level license or a caller-provided "not copyleft" exemption.
    for identity, row in inventory.items():
        record = records[identity]
        if not isinstance(record, dict) or record.get("component") != row:
            findings.append({"code": "corresponding_source_component_mismatch"})
            continue
        if not _delivered_source_matches(state, record.get("artifacts")):
            findings.append(
                {
                    "code": "corresponding_source_delivery_mismatch",
                    "component": identity,
                }
            )
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
    archive: tarfile.TarFile, member: tarfile.TarInfo, path: str
) -> tuple[dict[str, object], bytes | None, bytes | None]:
    body = archive.extractfile(member)
    W.require(body is not None, "habitat_oci_regular_file_read")
    if _track_bytes(path):
        tracked = body.read()
        digest, size, prefix = (
            hashlib.sha256(tracked).hexdigest(),
            len(tracked),
            tracked[:4],
        )
        elf = tracked if prefix == b"\x7fELF" else None
    else:
        digest, size, prefix, elf = _regular_hash(body)
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
    files = {path for path, kind in state.paths.items() if kind == "file"}
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
    file_row, tracked, elf = _read_member(archive, member, path)
    state.files[path] = file_row
    state.regular_files += 1
    state.content_bytes += int(file_row["size"])
    if tracked is not None:
        state.tracked[path] = tracked
    if elf is not None:
        state.elf[path] = elf
    if file_row["sha256"] in contract["forbidden_content_sha256"]:
        state.findings.append({"code": "forbidden_payload_hash", **location})
    return str(file_row["sha256"])


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
    if member.isfile():
        event["sha256"] = _record_regular_member(
            state, archive, member, path, {"layer": layer, "entry": entry}, contract
        )
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
    current = _ScanState(events=state.events, findings=state.findings)
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
        return target == parent or target.startswith(parent + "/")
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
