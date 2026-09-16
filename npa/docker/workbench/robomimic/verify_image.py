#!/usr/bin/env python3
"""Fail-closed verification for the neutral image and external CUDA runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
SOURCE_LICENSE_SHA256 = (
    "7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556"
)
SOURCE_MANIFEST_SHA256 = (
    "3ca2c54cc61080c7548a598bfb5184f278d4178635b375ab623e5f1f7429cd69"
)
SOURCE_TREE_SHA1 = "4c8ebe35dbef16126dadf59cf8b771b9203753ab"
SOURCE_ARCHIVE_SHA256 = (
    "8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b"
)
SOURCE_ARCHIVE_SIZE = 57_907_200
DEBIAN_LOCK_SHA256 = (
    "aebaefa21f527584d4a46cba8b22d0c691414a4762acd364252d746e03d3504e"
)
DEBIAN_PACKAGE_COUNT = 78
DEBIAN_PAYLOAD_BYTES = 29_807_672
BASE_IMAGE = (
    "python:3.11.16-slim-bookworm@sha256:"
    "528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
)
BAKED_LOCK_SHA256 = "65efcf0065ad4662b348e54e3f2d86996d934a518fcad0e89ecf012399ce1504"
BAKED_DISTRIBUTION_COUNT = 40
RUNTIME_ROOT_DEFAULT = "/opt/npa-runtime/robomimic"
RUNTIME_REFUSAL_STATUS = 78
RUNTIME_METADATA_MAX_BYTES = 4 * 1024 * 1024
# The sealed CUDA/PyTorch closure contains 22 wheel objects, totals
# 3,855,918,950 bytes, and has a 1,039,389,795-byte largest object. These
# ceilings leave bounded headroom without accepting an open-ended closure.
RUNTIME_ARTIFACT_MAX_COUNT = 32
RUNTIME_OBJECT_MAX_BYTES = 4 * 1024 * 1024 * 1024
RUNTIME_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024 * 1024
# Installed wheels expand into more filesystem entries than source objects.
# The metadata ceiling makes this independently bounded limit conservative for
# the future exact operator-selected inventory while preventing inode exhaustion.
RUNTIME_PAYLOAD_MAX_ENTRY_COUNT = 65_536
RUNTIME_ENTITLEMENT_MAX_BYTES = 16 * 1024
RUNTIME_ENTITLEMENT_TERMS = [
    {
        "name": "NVIDIA CUDA Toolkit EULA",
        "url": "https://docs.nvidia.com/cuda/eula/index.html",
    },
    {
        "name": "NVIDIA Software License Agreement",
        "url": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-software-license-agreement/"
        ),
    },
    {
        "name": "NVIDIA cuDNN Software License Agreement",
        "url": (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/"
            "reference/eula.html"
        ),
    },
]


class VerificationError(RuntimeError):
    """A boundary or immutable identity did not verify."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"required regular file is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object at {path}")
    return value


def _bounded_json_object(
    path: Path,
    *,
    maximum_size: int,
    require_single_link: bool = False,
    require_owner_only: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Read one regular JSON object through a no-link hard byte ceiling.

    Args:
        path: Runtime metadata file to read.
        maximum_size: Maximum accepted file size and read length in bytes.

    Returns:
        The parsed JSON object and the exact bytes that produced it.

    Raises:
        VerificationError: The file is absent, unsafe, oversized, or invalid JSON.
    """

    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise VerificationError("runtime metadata requires safe descriptor flags")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | nonblock
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(f"required regular file is absent: {path}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise VerificationError(f"required regular file is absent: {path}")
        if require_single_link and details.st_nlink != 1:
            raise VerificationError(f"runtime metadata must have one link: {path}")
        if require_owner_only and (
            details.st_uid != os.geteuid() or details.st_mode & 0o077
        ):
            raise VerificationError("runtime metadata must be owner-only")
        if details.st_size > maximum_size:
            raise VerificationError(f"runtime metadata exceeds size limit: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum_size + 1)
        if len(raw) > maximum_size:
            raise VerificationError(f"runtime metadata exceeds size limit: {path}")
        after = os.fstat(descriptor)
        if (
            len(raw) != details.st_size
            or after.st_size != details.st_size
            or after.st_dev != details.st_dev
            or after.st_ino != details.st_ino
            or (require_single_link and after.st_nlink != 1)
            or (
                require_owner_only
                and (after.st_uid != os.geteuid() or after.st_mode & 0o077)
            )
        ):
            raise VerificationError(f"runtime metadata changed while read: {path}")
    except OSError as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object at {path}")
    return value, raw


def _utc_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise VerificationError(f"customer runtime entitlement {field} is invalid")
    parse_failed = False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        parse_failed = True
    if parse_failed:
        raise VerificationError(
            f"customer runtime entitlement {field} is invalid"
        )
    return parsed


def _runtime_entitlement_contract(lock: dict[str, Any]) -> dict[str, Any]:
    contract = lock.get("customer_entitlement")
    if not isinstance(contract, dict) or set(contract) != {
        "schema",
        "field_of_use",
        "maximum_validity_seconds",
        "responsibilities",
        "terms",
    }:
        raise VerificationError("runtime lock customer entitlement contract is invalid")
    if contract.get("schema") != "npa.robomimic.customer-runtime-entitlement.v1":
        raise VerificationError("runtime lock customer entitlement schema is invalid")
    if contract.get("field_of_use") != "noncommercial":
        raise VerificationError("runtime lock customer field of use is invalid")
    maximum_validity = contract.get("maximum_validity_seconds")
    if not isinstance(maximum_validity, int) or isinstance(maximum_validity, bool):
        raise VerificationError("runtime lock entitlement validity is invalid")
    if maximum_validity <= 0 or maximum_validity > 86_400:
        raise VerificationError("runtime lock entitlement validity is out of bounds")
    responsibilities = contract.get("responsibilities")
    if responsibilities != [
        "runtime-use",
        "derivative-use",
        "service-use",
        "output-use",
        "no-redistribution-grant",
    ]:
        raise VerificationError("runtime lock customer responsibilities are invalid")
    terms = contract.get("terms")
    if terms != RUNTIME_ENTITLEMENT_TERMS:
        raise VerificationError("runtime lock customer terms are invalid")
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"name", "url"}:
            raise VerificationError("runtime lock customer term is invalid")
        if not isinstance(term["name"], str) or not term["name"]:
            raise VerificationError("runtime lock customer term name is invalid")
        _checked_https_url(term["url"], host=urllib.parse.urlsplit(term["url"]).hostname or "")
    return contract


def _runtime_entitlement_notice_sha256(
    *, lock: dict[str, Any], lock_sha256: str
) -> str:
    contract = _runtime_entitlement_contract(lock)
    notice_identity = {
        "schema": contract["schema"],
        "runtime_id": lock.get("runtime_id"),
        "runtime_lock_sha256": lock_sha256,
        "field_of_use": contract["field_of_use"],
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
    }
    return hashlib.sha256(
        json.dumps(
            notice_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def verify_customer_runtime_entitlement(
    *,
    entitlement_path: Path,
    runtime_lock_path: Path,
    expected_entitlement_sha256: str,
    expected_customer_binding_sha256: str,
    expected_run_id: str,
    expected_inventory_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify a customer-created, run-bound runtime-use record without mutation."""

    for label, digest in (
        ("entitlement", expected_entitlement_sha256),
        ("customer binding", expected_customer_binding_sha256),
        ("runtime inventory", expected_inventory_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VerificationError(f"expected {label} digest is absent or malformed")
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", expected_run_id) is None:
        raise VerificationError("expected run ID is absent or malformed")

    record, raw = _bounded_json_object(
        entitlement_path,
        maximum_size=RUNTIME_ENTITLEMENT_MAX_BYTES,
        require_single_link=True,
        require_owner_only=True,
    )
    if hashlib.sha256(raw).hexdigest() != expected_entitlement_sha256:
        raise VerificationError("customer runtime entitlement byte identity mismatch")
    lock = _json_object(runtime_lock_path)
    contract = _runtime_entitlement_contract(lock)
    expected_keys = {
        "schema",
        "decision",
        "customer_binding_sha256",
        "run_id",
        "source_revision",
        "runtime_id",
        "runtime_manifest_sha256",
        "runtime_lock_sha256",
        "field_of_use",
        "terms",
        "customer_responsibilities",
        "notice_sha256",
        "accepted_at",
        "expires_at",
    }
    if set(record) != expected_keys:
        raise VerificationError("customer runtime entitlement fields are invalid")
    expected_values = {
        "schema": contract["schema"],
        "decision": "accepted",
        "customer_binding_sha256": expected_customer_binding_sha256,
        "run_id": expected_run_id,
        "source_revision": lock.get("source_revision"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_manifest_sha256": expected_inventory_sha256,
        "runtime_lock_sha256": _sha256(runtime_lock_path),
        "field_of_use": contract["field_of_use"],
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
        "notice_sha256": _runtime_entitlement_notice_sha256(
            lock=lock, lock_sha256=_sha256(runtime_lock_path)
        ),
    }
    mismatched = sorted(
        key for key, expected in expected_values.items() if record.get(key) != expected
    )
    if mismatched:
        raise VerificationError(
            "customer runtime entitlement binding mismatch: " + ", ".join(mismatched)
        )
    accepted_at = _utc_timestamp(record["accepted_at"], field="accepted_at")
    expires_at = _utc_timestamp(record["expires_at"], field="expires_at")
    observed_now = now or datetime.now(timezone.utc)
    if accepted_at > observed_now:
        raise VerificationError("customer runtime entitlement acceptance is in the future")
    validity_seconds = int((expires_at - accepted_at).total_seconds())
    if validity_seconds <= 0 or validity_seconds > contract["maximum_validity_seconds"]:
        raise VerificationError("customer runtime entitlement validity is out of bounds")
    if observed_now >= expires_at:
        raise VerificationError("customer runtime entitlement has expired")
    return {
        "schema": "npa.robomimic.customer-runtime-entitlement-verification.v1",
        "record_sha256": expected_entitlement_sha256,
        "runtime_manifest_sha256": expected_inventory_sha256,
        "runtime_lock_sha256": expected_values["runtime_lock_sha256"],
        "run_binding_matched": True,
        "customer_binding_matched": True,
        "field_of_use": contract["field_of_use"],
        "expires_at": record["expires_at"],
        "redistribution_granted": False,
    }


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _checked_https_url(value: Any, *, host: str) -> str:
    if not isinstance(value, str):
        raise VerificationError("immutable input URL must be a string")
    parsed = urllib.parse.urlsplit(value)
    invalid_port = False
    try:
        port = parsed.port
    except ValueError:
        invalid_port = True
    if invalid_port:
        raise VerificationError("immutable input URL has an invalid port")
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise VerificationError("immutable input URL is outside its approved origin")
    return value


def _checked_source_manifest(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema": "npa.robomimic.source.v2",
        "repository": "https://github.com/ARISE-Initiative/robomimic.git",
        "revision": SOURCE_REVISION,
        "git_tree_sha1": SOURCE_TREE_SHA1,
        "archive": {
            "format": "git-archive-tar",
            "path": "source/robomimic.tar",
            "size": SOURCE_ARCHIVE_SIZE,
            "sha256": SOURCE_ARCHIVE_SHA256,
            "member_count": 226,
            "regular_file_count": 201,
        },
        "license": {
            "path": "LICENSE",
            "spdx": "MIT",
            "sha256": SOURCE_LICENSE_SHA256,
            "url": (
                "https://github.com/ARISE-Initiative/robomimic/blob/"
                f"{SOURCE_REVISION}/LICENSE"
            ),
        },
        "fetch_policy": {
            "method": "git-fetch-and-canonical-archive",
            "credentialless": True,
            "submodules": False,
            "lfs": False,
        },
    }
    if value != expected:
        raise VerificationError("source manifest does not match the reviewed closure")
    return value


def _source_manifest(path: Path) -> dict[str, Any]:
    if _sha256(path) != SOURCE_MANIFEST_SHA256:
        raise VerificationError("source manifest hash mismatch")
    return _checked_source_manifest(_json_object(path))


def verified_source_identity(path: Path) -> dict[str, str]:
    """Return the public proof fields from the exact reviewed source manifest.

    Args:
        path: Installed copy of the committed source manifest.
    Returns:
        Stable source identity fields for the smoke artifact.
    Raises:
        VerificationError: The manifest bytes or contents do not match the lock.
    """
    manifest = _source_manifest(path)
    return {
        "repository": "ARISE-Initiative/robomimic",
        "revision": manifest["revision"],
        "observed_head": manifest["revision"],
        "git_tree_sha1": manifest["git_tree_sha1"],
        "tree_archive_sha256": manifest["archive"]["sha256"],
    }


def _checked_debian_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "id",
        "name",
        "version",
        "debian_license_labels",
        "license_review",
        "license_reference_url",
    }:
        raise VerificationError("invalid Debian source license record")
    name, version = value.get("name"), value.get("version")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", name):
        raise VerificationError("invalid Debian source package name")
    if not isinstance(version, str) or not version or any(c.isspace() for c in version):
        raise VerificationError(f"invalid Debian source version for {name}")
    if value.get("id") != f"debian:{name}@{version}":
        raise VerificationError(f"invalid Debian source identity for {name}")
    license_labels = value.get("debian_license_labels")
    if (
        not isinstance(license_labels, list)
        or not license_labels
        or len(license_labels) != len(set(license_labels))
        or not all(isinstance(label, str) and label for label in license_labels)
    ):
        raise VerificationError(f"missing Debian license labels for {name}")
    if value.get("license_review") not in {
        "existing-exact-version-review",
        "official-exact-version-copyright",
    }:
        raise VerificationError(f"invalid Debian license review for {name}")
    reference = value.get("license_reference_url")
    if not isinstance(reference, str):
        raise VerificationError(f"missing Debian license reference for {name}")
    host = urllib.parse.urlsplit(reference).hostname
    if host not in {"metadata.ftp-master.debian.org", "snapshot.debian.org"}:
        raise VerificationError(f"invalid Debian license reference for {name}")
    _checked_https_url(reference, host=host)
    return value


def _checked_debian_package(
    value: Any, *, source_ids: set[str]
) -> dict[str, Any]:
    required = {
        "name",
        "version",
        "architecture",
        "source",
        "artifact",
        "url",
        "size",
        "sha256",
        "depends",
        "notice_path",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise VerificationError("invalid Debian binary package record")
    name, digest = value.get("name"), value.get("sha256")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", name):
        raise VerificationError("invalid Debian binary package name")
    if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
        raise VerificationError(f"invalid Debian package hash for {name}")
    if value.get("artifact") != f"debian/{digest}.deb":
        raise VerificationError(f"invalid Debian artifact path for {name}")
    if value.get("architecture") not in {"all", "amd64"}:
        raise VerificationError(f"invalid Debian architecture for {name}")
    version = value.get("version")
    if not isinstance(version, str) or not version or any(c.isspace() for c in version):
        raise VerificationError(f"invalid Debian binary version for {name}")
    if value.get("source") not in source_ids:
        raise VerificationError(f"missing Debian source record for {name}")
    if not isinstance(value.get("size"), int) or value["size"] <= 0:
        raise VerificationError(f"invalid Debian package size for {name}")
    url = _checked_https_url(value.get("url"), host="snapshot.debian.org")
    allowed_prefixes = (
        "https://snapshot.debian.org/archive/debian/20260906T183022Z/pool/",
        "https://snapshot.debian.org/archive/debian-security/20260906T183022Z/pool/",
    )
    if not url.startswith(allowed_prefixes) or not url.endswith(".deb"):
        raise VerificationError(f"invalid Debian snapshot path for {name}")
    if value.get("notice_path") != f"/usr/share/doc/{name}/copyright":
        raise VerificationError(f"invalid Debian notice path for {name}")
    dependencies = value.get("depends")
    if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
        raise VerificationError(f"invalid Debian dependency list for {name}")
    if not all(isinstance(item, str) and item for item in dependencies):
        raise VerificationError(f"invalid Debian dependency name for {name}")
    return value


def _checked_debian_lock(value: dict[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema",
        "base_image",
        "platform",
        "snapshot",
        "repository_metadata",
        "roots",
        "dependency_policy",
        "license_policy",
        "sources",
        "packages",
    }
    if set(value) != expected_top:
        raise VerificationError("invalid Debian lock fields")
    if value.get("schema") != "npa.robomimic.debian-packages.v1":
        raise VerificationError("Debian lock schema mismatch")
    if value.get("base_image") != BASE_IMAGE or value.get("platform") != "linux/amd64":
        raise VerificationError("Debian lock base or platform mismatch")
    if value.get("snapshot") != "20260906T183022Z":
        raise VerificationError("Debian snapshot mismatch")
    expected_metadata = [
        {
            "id": "debian-bookworm-20260906T183022Z",
            "url": (
                "https://snapshot.debian.org/archive/debian/20260906T183022Z/"
                "dists/bookworm/main/binary-amd64/Packages.xz"
            ),
            "size": 8_790_396,
            "sha256": "9e0b5aabb2465b3d2e7a7fe27f9913846277833f7a2826e7767acccff5b588c5",
        },
        {
            "id": "debian-security-bookworm-security-20260906T183022Z",
            "url": (
                "https://snapshot.debian.org/archive/debian-security/"
                "20260906T183022Z/dists/bookworm-security/main/"
                "binary-amd64/Packages.xz"
            ),
            "size": 336_504,
            "sha256": "610da1862518c2a1f69e459b6b02b579ec328697b4594244765b231b450ccc4e",
        },
    ]
    if value.get("repository_metadata") != expected_metadata:
        raise VerificationError("Debian repository metadata closure mismatch")
    license_policy = {
        "component": "Debian main",
        "authoritative_terms": "each exact installed package copyright file",
        "notice_byte_hashes": "required-at-authorized-byte-gate",
        "source_delivery": "must be closed before public publication",
    }
    if value.get("license_policy") != license_policy:
        raise VerificationError("Debian license policy mismatch")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise VerificationError("Debian source records must be an array")
    checked_sources = [_checked_debian_source(item) for item in sources]
    source_ids = {item["id"] for item in checked_sources}
    if len(source_ids) != len(checked_sources) or len(source_ids) != 58:
        raise VerificationError("Debian source count or identity uniqueness mismatch")
    packages = value.get("packages")
    if not isinstance(packages, list):
        raise VerificationError("Debian package records must be an array")
    checked = [_checked_debian_package(item, source_ids=source_ids) for item in packages]
    by_name = {item["name"]: item for item in checked}
    if len(by_name) != DEBIAN_PACKAGE_COUNT or len(checked) != DEBIAN_PACKAGE_COUNT:
        raise VerificationError("Debian package count or name uniqueness mismatch")
    if len({item["sha256"] for item in checked}) != len(checked):
        raise VerificationError("duplicate Debian package hash")
    if len({item["url"] for item in checked}) != len(checked):
        raise VerificationError("duplicate Debian package URL")
    if sum(item["size"] for item in checked) != DEBIAN_PAYLOAD_BYTES:
        raise VerificationError("Debian package byte total mismatch")
    if any(set(item["depends"]) - set(by_name) for item in checked):
        raise VerificationError("Debian dependency closure is incomplete")
    _verify_debian_lock_policy(value, by_name, source_ids)
    return value


def _verify_debian_lock_policy(
    value: dict[str, Any], by_name: dict[str, dict[str, Any]], source_ids: set[str]
) -> None:
    roots = ["ca-certificates", "openssh-server", "procps", "rsync", "sudo"]
    policy = {
        "fields": ["Pre-Depends", "Depends"],
        "install_recommends": False,
        "package_count": DEBIAN_PACKAGE_COUNT,
        "payload_bytes": DEBIAN_PAYLOAD_BYTES,
    }
    if value.get("roots") != roots or value.get("dependency_policy") != policy:
        raise VerificationError("Debian dependency policy mismatch")
    reached, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(by_name[name]["depends"])
    if reached != set(by_name):
        raise VerificationError("Debian lock contains packages outside root closure")
    if {item["source"] for item in by_name.values()} != source_ids:
        raise VerificationError("Debian source records do not exactly match binaries")
    forbidden = re.compile(r"(?:^|[-_.])(cuda|cudnn|nvidia|torch|triton)(?:$|[-_.])")
    if any(forbidden.search(name) for name in by_name):
        raise VerificationError("forbidden CUDA-capable package in Debian closure")


def _debian_lock(path: Path) -> dict[str, Any]:
    if _sha256(path) != DEBIAN_LOCK_SHA256:
        raise VerificationError("Debian package lock hash mismatch")
    return _checked_debian_lock(_json_object(path))


def _verify_regular_file(path: Path, *, size: int, digest: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise VerificationError(f"immutable build input is not a regular file: {path}")
    if path.stat().st_size != size or _sha256(path) != digest:
        raise VerificationError(f"immutable build input identity mismatch: {path}")


def _verify_source_archive(path: Path, manifest: dict[str, Any]) -> None:
    archive = manifest["archive"]
    _verify_regular_file(path, size=archive["size"], digest=archive["sha256"])
    try:
        with tarfile.open(path, mode="r:") as stream:
            members = stream.getmembers()
            names = [member.name for member in members]
            for member in members:
                relative = PurePosixPath(member.name)
                if (
                    not member.name
                    or "\\" in member.name
                    or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or not (member.isdir() or member.isreg())
                ):
                    raise VerificationError("unsafe object in robomimic source archive")
            regular = [member for member in members if member.isreg()]
            license_member = stream.getmember(manifest["license"]["path"])
            license_handle = stream.extractfile(license_member)
            if license_handle is None:
                raise VerificationError("source archive license is not a regular file")
            license_digest = hashlib.sha256(license_handle.read()).hexdigest()
    except (KeyError, OSError, tarfile.TarError) as exc:
        raise VerificationError("invalid robomimic source archive") from exc
    if len(names) != len(set(names)) or len(names) != archive["member_count"]:
        raise VerificationError("source archive member closure mismatch")
    if len(regular) != archive["regular_file_count"]:
        raise VerificationError("source archive regular-file closure mismatch")
    if any(PurePosixPath(name).parts[0] == ".git" for name in names):
        raise VerificationError("source archive contains Git metadata")
    if license_digest != manifest["license"]["sha256"]:
        raise VerificationError("source archive license hash mismatch")


def _expected_build_input_objects(
    lock: dict[str, Any], manifest: dict[str, Any]
) -> set[str]:
    return {
        "debian",
        "source",
        manifest["archive"]["path"],
        *(item["artifact"] for item in lock["packages"]),
    }


def verify_build_inputs(
    *, input_root: Path, debian_lock_path: Path, source_manifest_path: Path
) -> dict[str, Any]:
    lock = _debian_lock(debian_lock_path)
    manifest = _source_manifest(source_manifest_path)
    if not input_root.is_dir() or input_root.is_symlink():
        raise VerificationError("immutable build-input root is absent or is a symlink")
    expected = _expected_build_input_objects(lock, manifest)
    observed = set()
    for path in input_root.rglob("*"):
        relative = path.relative_to(input_root).as_posix()
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise VerificationError(f"unsafe immutable build input: {relative}")
        observed.add(relative)
    if observed != expected:
        raise VerificationError(
            "immutable build-input inventory mismatch: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    for package in lock["packages"]:
        _verify_regular_file(
            input_root / package["artifact"],
            size=package["size"],
            digest=package["sha256"],
        )
    _verify_source_archive(input_root / manifest["archive"]["path"], manifest)
    return {
        "schema": "npa.robomimic.build-input-verification.v1",
        "debian_package_count": len(lock["packages"]),
        "debian_payload_bytes": sum(item["size"] for item in lock["packages"]),
        "debian_lock_sha256": DEBIAN_LOCK_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha1": SOURCE_TREE_SHA1,
    }


class _PinnedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        _checked_https_url(new_url, host="snapshot.debian.org")
        return super().redirect_request(request, fp, code, msg, headers, new_url)


def _download_exact_package(package: dict[str, Any], destination: Path) -> None:
    opener = urllib.request.build_opener(_PinnedRedirectHandler())
    digest = hashlib.sha256()
    total = 0
    try:
        with opener.open(package["url"], timeout=60) as response:
            with destination.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > package["size"]:
                        raise VerificationError("Debian download exceeds locked size")
                    digest.update(chunk)
                    output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise VerificationError(f"failed to fetch Debian package {package['name']}") from exc
    if total != package["size"] or digest.hexdigest() != package["sha256"]:
        raise VerificationError(f"downloaded Debian package mismatch: {package['name']}")


def _git(*args: str, cwd: Path | None = None, output=None) -> subprocess.CompletedProcess:
    environment = {
        **os.environ,
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "SSH_ASKPASS": "/bin/false",
    }
    command = ["git", "-c", "credential.helper=", "-c", "core.askPass=", *args]
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            stdout=output or subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=output is None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VerificationError("credentialless robomimic source fetch failed") from exc


def _prepare_source(manifest: dict[str, Any], staging: Path) -> None:
    repository = staging / ".source-repository"
    repository.mkdir(mode=0o700)
    _git("init", "--quiet", cwd=repository)
    _git(
        "fetch",
        "--quiet",
        "--depth=1",
        "--no-tags",
        manifest["repository"],
        manifest["revision"],
        cwd=repository,
    )
    head = _git("rev-parse", "FETCH_HEAD", cwd=repository).stdout.strip()
    tree = _git("rev-parse", "FETCH_HEAD^{tree}", cwd=repository).stdout.strip()
    if head != manifest["revision"] or tree != manifest["git_tree_sha1"]:
        raise VerificationError("fetched robomimic commit or tree mismatch")
    archive = staging / manifest["archive"]["path"]
    archive.parent.mkdir()
    with archive.open("xb") as output:
        _git("archive", "--format=tar", "FETCH_HEAD", cwd=repository, output=output)
    shutil.rmtree(repository)
    _verify_source_archive(archive, manifest)


def prepare_build_inputs(
    *, output_root: Path, debian_lock_path: Path, source_manifest_path: Path
) -> dict[str, Any]:
    lock = _debian_lock(debian_lock_path)
    manifest = _source_manifest(source_manifest_path)
    if output_root.exists() or output_root.is_symlink():
        raise VerificationError("build-input output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        (staging / "debian").mkdir()
        for package in lock["packages"]:
            _download_exact_package(package, staging / package["artifact"])
        _prepare_source(manifest, staging)
        proof = verify_build_inputs(
            input_root=staging,
            debian_lock_path=debian_lock_path,
            source_manifest_path=source_manifest_path,
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return proof


def verify_debian_install(
    *, debian_lock_path: Path, notice_root: Path = Path("/")
) -> dict[str, Any]:
    lock = _debian_lock(debian_lock_path)
    for package in lock["packages"]:
        try:
            observed = subprocess.run(
                [
                    "dpkg-query",
                    "--show",
                    "--showformat=${db:Status-Abbrev}|${Version}",
                    package["name"],
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VerificationError(
                f"locked Debian package is not installed: {package['name']}"
            ) from exc
        if observed != f"ii |{package['version']}":
            raise VerificationError(
                f"installed Debian package mismatch: {package['name']}"
            )
        notice = notice_root / Path(package["notice_path"]).relative_to("/")
        if not notice.is_file() or notice.stat().st_size == 0:
            raise VerificationError(f"Debian package notice is absent: {package['name']}")
    return {
        "schema": "npa.robomimic.debian-install-verification.v1",
        "package_count": len(lock["packages"]),
        "package_bytes_verified_before_install": DEBIAN_PAYLOAD_BYTES,
        "notices_present": len(lock["packages"]),
    }


def _locked_baked_packages(lock_path: Path) -> dict[str, str]:
    raw = lock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BAKED_LOCK_SHA256:
        raise VerificationError("baked dependency lock hash mismatch")
    records: list[str] = []
    pending: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            raise VerificationError("empty baked dependency lock line")
        if not pending and line[0].isspace():
            raise VerificationError("orphaned baked dependency hash continuation")
        continued = stripped.endswith("\\")
        pending.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            records.append(" ".join(pending))
            pending = []
    if pending:
        raise VerificationError("unterminated baked dependency lock record")
    if len(records) != BAKED_DISTRIBUTION_COUNT:
        raise VerificationError("baked dependency lock count mismatch")
    packages: dict[str, str] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)==([^ ]+)((?: --hash=sha256:[0-9a-f]{64})+)$"
    )
    for record in records:
        match = pattern.fullmatch(record)
        if match is None:
            raise VerificationError(
                f"malformed baked dependency lock record: {record!r}"
            )
        hashes = re.findall(r"--hash=sha256:([0-9a-f]{64})", match.group(3))
        if len(hashes) != len(set(hashes)):
            raise VerificationError("duplicate baked dependency artifact hash")
        name = _canonical_name(match.group(1))
        if name in packages:
            raise VerificationError(f"duplicate baked dependency: {name}")
        packages[name] = match.group(2)
    return packages


def verify_neutral_image(
    *,
    source_root: Path,
    metadata_path: Path,
    debian_lock_path: Path,
    baked_lock_path: Path,
    baked_deps_path: Path,
    empty_boundary_paths: tuple[Path, ...],
) -> dict[str, Any]:
    metadata = _source_manifest(metadata_path)
    debian = verify_debian_install(debian_lock_path=debian_lock_path)
    if not (source_root / "robomimic" / "__init__.py").is_file():
        raise VerificationError("robomimic source package is absent")
    if _sha256(source_root / "LICENSE") != SOURCE_LICENSE_SHA256:
        raise VerificationError("robomimic source license hash mismatch")

    locked = _locked_baked_packages(baked_lock_path)
    observed = {
        _canonical_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions(
            path=[str(baked_deps_path)]
        )
        if distribution.metadata.get("Name")
    }
    if observed != locked:
        missing = sorted(set(locked) - set(observed))
        extra = sorted(set(observed) - set(locked))
        mismatched = sorted(
            name
            for name in set(locked) & set(observed)
            if locked[name] != observed[name]
        )
        raise VerificationError(
            f"baked distribution inventory mismatch: missing={missing} extra={extra} "
            f"version_mismatches={mismatched}"
        )
    forbidden = {
        name
        for name in observed
        if name in {"torch", "torchvision", "triton"} or name.startswith("nvidia-")
    }
    if forbidden:
        raise VerificationError(
            f"forbidden runtime distributions are baked: {sorted(forbidden)}"
        )
    nonempty_boundaries = [
        str(path)
        for path in empty_boundary_paths
        if path.exists() and any(path.iterdir())
    ]
    if nonempty_boundaries:
        raise VerificationError(
            f"runtime/data/output boundary is populated: {nonempty_boundaries}"
        )
    return {
        "schema": "npa.robomimic.neutral-image-verification.v1",
        "baked_dependency_count": len(observed),
        "baked_lock_sha256": BAKED_LOCK_SHA256,
        "debian_lock_sha256": DEBIAN_LOCK_SHA256,
        "debian_package_count": debian["package_count"],
        "source_archive_sha256": metadata["archive"]["sha256"],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha1": metadata["git_tree_sha1"],
        "runtime_payload_baked": False,
        "weights_baked": False,
        "data_baked": False,
        "outputs_baked": False,
    }


def _safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise VerificationError(f"invalid inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise VerificationError(f"unsafe inventory path: {value!r}")
    if path.parts[0] != "payload":
        raise VerificationError(f"runtime inventory entry escapes payload/: {value!r}")
    return path


def _checked_entries(value: Any, *, symlinks: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise VerificationError("runtime inventory entries must be arrays")
    result: dict[str, dict[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise VerificationError("runtime inventory entry must be an object")
        path = str(_safe_relative_path(entry.get("path")))
        if path in result:
            raise VerificationError(f"duplicate runtime inventory path: {path}")
        if symlinks:
            target = entry.get("target")
            if not isinstance(target, str) or not target or os.path.isabs(target):
                raise VerificationError(f"unsafe symlink target for {path}: {target!r}")
        else:
            if (
                not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None
            ):
                raise VerificationError(f"invalid file identity for {path}")
        result[path] = entry
    return result


def _checked_payload_entries(
    inventory: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Validate payload resource ceilings before inspecting its filesystem."""

    raw_files = inventory.get("files")
    raw_links = inventory.get("symlinks")
    if not isinstance(raw_files, list) or not isinstance(raw_links, list):
        raise VerificationError("runtime inventory entries must be arrays")
    if len(raw_files) + len(raw_links) > RUNTIME_PAYLOAD_MAX_ENTRY_COUNT:
        raise VerificationError("runtime payload entry count exceeds limit")

    files = _checked_entries(raw_files, symlinks=False)
    links = _checked_entries(raw_links, symlinks=True)
    total_bytes = 0
    for path, entry in files.items():
        size = entry["size"]
        if size > RUNTIME_OBJECT_MAX_BYTES:
            raise VerificationError(f"runtime payload object exceeds size limit: {path}")
        if total_bytes > RUNTIME_PAYLOAD_MAX_BYTES - size:
            raise VerificationError("runtime payload aggregate exceeds size limit")
        total_bytes += size
    return files, links, total_bytes


def _checked_artifacts(
    value: Any,
) -> tuple[dict[str, dict[str, str | int]], int]:
    if not isinstance(value, list):
        raise VerificationError("runtime artifact inventory must be an array")
    if len(value) > RUNTIME_ARTIFACT_MAX_COUNT:
        raise VerificationError("runtime artifact object count exceeds limit")
    result: dict[str, dict[str, str | int]] = {}
    total_bytes = 0
    for entry in value:
        if not isinstance(entry, dict):
            raise VerificationError(
                "runtime artifact inventory entry must be an object"
            )
        name = _canonical_name(str(entry.get("name") or ""))
        version = entry.get("version")
        filename = entry.get("filename")
        source = entry.get("source")
        digest = entry.get("sha256")
        size = entry.get("size")
        if (
            not name
            or not isinstance(version, str)
            or not version
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(source, str)
            or not source.startswith("https://")
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise VerificationError(f"invalid runtime artifact identity: {entry!r}")
        if size > RUNTIME_OBJECT_MAX_BYTES:
            raise VerificationError(f"runtime artifact exceeds size limit: {name}")
        if total_bytes > RUNTIME_PAYLOAD_MAX_BYTES - size:
            raise VerificationError("runtime artifact aggregate exceeds size limit")
        if name in result:
            raise VerificationError(f"duplicate runtime artifact: {name}")
        total_bytes += size
        result[name] = {
            "version": version,
            "filename": filename,
            "source": source,
            "sha256": str(digest),
            "size": size,
        }
    return result, total_bytes


def _verify_declared_runtime_file(
    path: Path, *, expected_size: int, expected_sha256: str
) -> None:
    """Verify one declared payload through a bounded, identity-stable descriptor."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise VerificationError("runtime payload verification requires safe descriptor flags")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | nonblock
    open_failed = False
    try:
        descriptor = os.open(path, flags)
    except OSError:
        open_failed = True
    if open_failed:
        raise VerificationError("declared runtime file is not a safe regular file")
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
        ):
            raise VerificationError("runtime file identity mismatch")
        digest = hashlib.sha256()
        remaining = expected_size
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise VerificationError("runtime file identity mismatch")
                digest.update(chunk)
                remaining -= len(chunk)
        closed = os.fstat(descriptor)
        if (
            closed.st_dev != opened.st_dev
            or closed.st_ino != opened.st_ino
            or closed.st_size != expected_size
            or closed.st_nlink != 1
            or digest.hexdigest() != expected_sha256
        ):
            raise VerificationError("runtime file identity mismatch")
    finally:
        os.close(descriptor)


def verify_external_runtime(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    require_read_only_mount: bool,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_inventory_sha256) is None:
        raise VerificationError(
            "operator-selected runtime inventory digest is absent or malformed"
        )
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise VerificationError("runtime root is absent or is a symlink")
    read_only_mount = bool(os.statvfs(runtime_root).f_flag & os.ST_RDONLY)
    if require_read_only_mount and not read_only_mount:
        raise VerificationError("external runtime filesystem is not mounted read-only")
    lock = _json_object(runtime_lock_path)
    lock_hash = _sha256(runtime_lock_path)
    marker_path = runtime_root / ".ready.json"
    inventory_path = runtime_root / "inventory.json"
    marker, _marker_bytes = _bounded_json_object(
        marker_path, maximum_size=RUNTIME_METADATA_MAX_BYTES
    )
    inventory, inventory_bytes = _bounded_json_object(
        inventory_path, maximum_size=RUNTIME_METADATA_MAX_BYTES
    )
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_sha256 != expected_inventory_sha256:
        raise VerificationError(
            "runtime inventory does not match the operator-selected digest"
        )
    expected_common = {
        "runtime_id": lock.get("runtime_id"),
        "lock_sha256": lock_hash,
        "source_revision": lock.get("source_revision"),
    }
    if marker.get("schema") != "npa.robomimic.runtime-ready.v1":
        raise VerificationError("runtime ready marker schema mismatch")
    if inventory.get("schema") != "npa.robomimic.runtime-inventory.v1":
        raise VerificationError("runtime inventory schema mismatch")
    for key, expected in expected_common.items():
        if marker.get(key) != expected or inventory.get(key) != expected:
            raise VerificationError(f"runtime {key} mismatch")
    if marker.get("inventory_sha256") != inventory_sha256:
        raise VerificationError("runtime inventory hash mismatch")
    if inventory.get("abi") != lock.get("abi"):
        raise VerificationError("runtime ABI inventory mismatch")
    if inventory.get("packages") != lock.get("packages"):
        raise VerificationError("runtime package inventory mismatch")
    artifacts, artifact_payload_bytes = _checked_artifacts(inventory.get("artifacts"))
    locked_packages = lock.get("packages")
    if not isinstance(locked_packages, dict):
        raise VerificationError("runtime lock package map is invalid")
    if set(artifacts) != set(locked_packages) or any(
        artifacts[name]["version"] != version
        for name, version in locked_packages.items()
    ):
        raise VerificationError("runtime artifact closure does not match package lock")

    files, links, payload_bytes = _checked_payload_entries(inventory)
    if set(files) & set(links):
        raise VerificationError("runtime path declared as both file and symlink")
    payload_root = runtime_root / "payload"
    if not payload_root.is_dir() or payload_root.is_symlink():
        raise VerificationError("runtime payload directory is absent")
    declared = set(files) | set(links)
    allowed_directories = {"payload"}
    for relative in declared:
        allowed_directories.update(
            parent.as_posix()
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )
    allowed_objects = declared | allowed_directories | {".ready.json", "inventory.json"}
    for path in runtime_root.rglob("*"):
        relative = path.relative_to(runtime_root).as_posix()
        if relative not in allowed_objects:
            raise VerificationError(f"undeclared object in runtime root: {relative}")
        mode = path.lstat().st_mode
        if relative in allowed_directories:
            if not stat.S_ISDIR(mode) or path.is_symlink():
                raise VerificationError(f"runtime directory is not regular: {relative}")
        elif relative in {".ready.json", "inventory.json"} and (
            not stat.S_ISREG(mode) or path.is_symlink()
        ):
            raise VerificationError(f"runtime metadata is not regular: {relative}")
    observed: set[str] = set()
    for path in payload_root.rglob("*"):
        relative = path.relative_to(runtime_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            raise VerificationError(
                f"unsupported runtime filesystem object: {relative}"
            )
        observed.add(relative)
    if observed != declared:
        raise VerificationError(
            f"runtime payload inventory mismatch: missing={sorted(declared - observed)} "
            f"extra={sorted(observed - declared)}"
        )
    payload_resolved = payload_root.resolve()
    for relative, entry in files.items():
        path = runtime_root / relative
        _verify_declared_runtime_file(
            path,
            expected_size=entry["size"],
            expected_sha256=entry["sha256"],
        )
    for relative, entry in links.items():
        path = runtime_root / relative
        if not path.is_symlink() or os.readlink(path) != entry["target"]:
            raise VerificationError(f"runtime symlink identity mismatch: {relative}")
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(payload_resolved)
        except ValueError as exc:
            raise VerificationError(
                f"runtime symlink escapes runtime payload: {relative}"
            ) from exc
    interpreter = payload_root / "bin" / "python"
    if "payload/bin/python" not in declared or not os.access(interpreter, os.X_OK):
        raise VerificationError(
            "verified runtime interpreter is absent or not executable"
        )
    return {
        "schema": "npa.robomimic.external-runtime-verification.v1",
        "runtime_id": lock["runtime_id"],
        "runtime_lock_sha256": lock_hash,
        "runtime_inventory_sha256": inventory_sha256,
        "runtime_manifest_digest_matched": True,
        "read_only_mount_observed": read_only_mount,
        "package_count": len(lock["packages"]),
        "artifact_count": len(artifacts),
        "artifact_payload_bytes": artifact_payload_bytes,
        "payload_file_count": len(files),
        "payload_symlink_count": len(links),
        "payload_bytes": payload_bytes,
    }


def verify_missing_runtime_refusal(
    *, runtime_root: Path, runtime_lock_path: Path
) -> dict[str, Any]:
    """Prove that an empty runtime is rejected specifically for its missing marker."""

    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise VerificationError("missing-runtime refusal probe requires a regular directory")
    if any(runtime_root.iterdir()):
        raise VerificationError("missing-runtime refusal probe requires an empty directory")
    sentinel_inventory_sha256 = "0" * 64
    expected = f"required regular file is absent: {runtime_root / '.ready.json'}"
    try:
        verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=sentinel_inventory_sha256,
            require_read_only_mount=False,
        )
    except VerificationError as exc:
        if str(exc) != expected:
            raise VerificationError(
                "runtime verifier refused the empty runtime for an unexpected reason"
            ) from exc
    else:
        raise VerificationError("empty runtime unexpectedly passed verification")
    if any(runtime_root.iterdir()):
        raise VerificationError("runtime verifier mutated the empty runtime root")
    return {
        "schema": "npa.robomimic.missing-runtime-refusal.v1",
        "expected_inventory_sha256": sentinel_inventory_sha256,
        "refusal_reason": "missing-ready-marker",
        "runtime_root_unchanged": True,
    }


def _copy_bounded_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    maximum_size: int | None = None,
) -> None:
    """Copy one regular file without following links or exceeding its identity.

    Args:
        source: File in the operator-selected external runtime.
        destination: New file in the private snapshot staging tree.
        expected_size: Exact inventory size, or ``None`` to lock the opened size.
        expected_sha256: Exact inventory digest when one is available.
        maximum_size: Optional hard byte ceiling for metadata without a size lock.

    Raises:
        VerificationError: The source is unsafe or changes from its declared identity.
        OSError: Opening, reading, or writing either file fails.
    """

    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise VerificationError("runtime snapshot requires safe descriptor flags")
    source_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | nonblock
    source_fd = os.open(source, source_flags)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise VerificationError(f"runtime snapshot source is not regular: {source}")
        locked_size = source_stat.st_size if expected_size is None else expected_size
        if source_stat.st_size != locked_size:
            raise VerificationError(f"runtime snapshot source size changed: {source}")
        if maximum_size is not None and locked_size > maximum_size:
            raise VerificationError(f"runtime snapshot metadata exceeds limit: {source}")

        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        destination_fd = os.open(destination, destination_flags, 0o600)
        try:
            digest = hashlib.sha256()
            remaining = locked_size
            with os.fdopen(source_fd, "rb", closefd=False) as source_handle, os.fdopen(
                destination_fd, "wb", closefd=False
            ) as destination_handle:
                while remaining:
                    chunk = source_handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise VerificationError(
                            f"runtime snapshot source shrank while copying: {source}"
                        )
                    destination_handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                source_after = os.fstat(source_fd)
                if (
                    source_after.st_dev != source_stat.st_dev
                    or source_after.st_ino != source_stat.st_ino
                    or source_after.st_size != locked_size
                    or source_after.st_nlink != 1
                    or not stat.S_ISREG(source_after.st_mode)
                ):
                    raise VerificationError(
                        f"runtime snapshot source identity changed: {source}"
                    )
                destination_handle.flush()
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise VerificationError(
                    f"runtime snapshot source hash changed while copying: {source}"
                )
            os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def _copy_runtime_inventory(
    source: Path, staging: Path, expected_inventory_sha256: str
) -> None:
    """Copy only manager-bound runtime objects into a private staging tree."""

    staging.mkdir(mode=0o700)
    _copy_bounded_regular_file(
        source / ".ready.json",
        staging / ".ready.json",
        expected_size=None,
        expected_sha256=None,
        maximum_size=RUNTIME_METADATA_MAX_BYTES,
    )
    _copy_bounded_regular_file(
        source / "inventory.json",
        staging / "inventory.json",
        expected_size=None,
        expected_sha256=expected_inventory_sha256,
        maximum_size=RUNTIME_METADATA_MAX_BYTES,
    )
    inventory, _inventory_bytes = _bounded_json_object(
        staging / "inventory.json", maximum_size=RUNTIME_METADATA_MAX_BYTES
    )
    files, links, _payload_bytes = _checked_payload_entries(inventory)
    for relative, entry in files.items():
        source_path = source / relative
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_bounded_regular_file(
            source_path,
            destination,
            expected_size=entry["size"],
            expected_sha256=entry["sha256"],
        )
    for relative, entry in links.items():
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(entry["target"])


def _remove_snapshot_write_bits(snapshot: Path) -> None:
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o555)
            continue
        path.chmod(path.stat().st_mode & 0o555)
    snapshot.chmod(0o555)


def materialize_external_runtime(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    destination: Path,
    require_source_read_only: bool,
) -> dict[str, Any]:
    """Atomically publish a verified private snapshot for runtime execution."""

    source_proof = verify_external_runtime(
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=require_source_read_only,
    )
    if destination.exists() or destination.is_symlink():
        raise VerificationError("runtime snapshot destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    operating_system_failure = False
    try:
        staging.rmdir()
        _copy_runtime_inventory(runtime_root, staging, expected_inventory_sha256)
        snapshot_proof = verify_external_runtime(
            runtime_root=staging,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            require_read_only_mount=False,
        )
        _remove_snapshot_write_bits(staging)
        staging.replace(destination)
    except VerificationError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        operating_system_failure = True
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if operating_system_failure:
        raise VerificationError("runtime snapshot materialization failed")
    return {
        **snapshot_proof,
        "read_only_mount_observed": source_proof["read_only_mount_observed"],
        "source_read_only_mount_observed": source_proof["read_only_mount_observed"],
        "atomic_snapshot_published": True,
        "snapshot_write_bits_absent": destination.stat().st_mode & 0o222 == 0,
        "snapshot_root": str(destination),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    build_inputs = subparsers.add_parser("build-inputs")
    build_inputs.add_argument("--input-root", type=Path, required=True)
    build_inputs.add_argument("--debian-lock", type=Path, required=True)
    build_inputs.add_argument("--source-manifest", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-build-inputs")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--debian-lock", type=Path, required=True)
    prepare.add_argument("--source-manifest", type=Path, required=True)
    debian_install = subparsers.add_parser("debian-install")
    debian_install.add_argument("--debian-lock", type=Path, required=True)
    image = subparsers.add_parser("image")
    image.add_argument("--source-root", type=Path, default=Path("/opt/robomimic"))
    image.add_argument(
        "--metadata", type=Path, default=Path("/opt/byof/npa_source_metadata.json")
    )
    image.add_argument(
        "--debian-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/debian-packages.lock"),
    )
    image.add_argument(
        "--baked-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/baked-requirements.lock"),
    )
    image.add_argument("--baked-deps", type=Path, default=Path("/opt/robomimic-deps"))
    runtime = subparsers.add_parser("runtime")
    runtime.add_argument(
        "--runtime-root", type=Path, default=Path(RUNTIME_ROOT_DEFAULT)
    )
    entitlement = subparsers.add_parser("entitlement")
    entitlement.add_argument("--entitlement", type=Path, required=True)
    entitlement.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    entitlement.add_argument(
        "--expected-entitlement-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256", ""),
    )
    entitlement.add_argument(
        "--expected-customer-binding-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256", ""),
    )
    entitlement.add_argument(
        "--expected-run-id", default=os.environ.get("NPA_BYOF_RUN_ID", "")
    )
    entitlement.add_argument(
        "--expected-inventory-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""),
    )
    runtime.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    runtime.add_argument(
        "--expected-inventory-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""),
    )
    missing_runtime = subparsers.add_parser("assert-missing-runtime")
    missing_runtime.add_argument("--runtime-root", type=Path, required=True)
    missing_runtime.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument(
        "--runtime-root", type=Path, default=Path(RUNTIME_ROOT_DEFAULT)
    )
    snapshot.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    snapshot.add_argument(
        "--expected-inventory-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""),
    )
    snapshot.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "build-inputs":
            result = verify_build_inputs(
                input_root=args.input_root,
                debian_lock_path=args.debian_lock,
                source_manifest_path=args.source_manifest,
            )
        elif args.mode == "prepare-build-inputs":
            result = prepare_build_inputs(
                output_root=args.output_root,
                debian_lock_path=args.debian_lock,
                source_manifest_path=args.source_manifest,
            )
        elif args.mode == "debian-install":
            result = verify_debian_install(debian_lock_path=args.debian_lock)
        elif args.mode == "image":
            result = verify_neutral_image(
                source_root=args.source_root,
                metadata_path=args.metadata,
                debian_lock_path=args.debian_lock,
                baked_lock_path=args.baked_lock,
                baked_deps_path=args.baked_deps,
                empty_boundary_paths=(
                    Path(RUNTIME_ROOT_DEFAULT),
                    Path("/workspace/byof-inputs"),
                    Path("/workspace/byof-runs"),
                ),
            )
        elif args.mode == "runtime":
            result = verify_external_runtime(
                runtime_root=args.runtime_root,
                runtime_lock_path=args.runtime_lock,
                expected_inventory_sha256=args.expected_inventory_sha256,
                require_read_only_mount=True,
            )
        elif args.mode == "entitlement":
            result = verify_customer_runtime_entitlement(
                entitlement_path=args.entitlement,
                runtime_lock_path=args.runtime_lock,
                expected_entitlement_sha256=args.expected_entitlement_sha256,
                expected_customer_binding_sha256=(
                    args.expected_customer_binding_sha256
                ),
                expected_run_id=args.expected_run_id,
                expected_inventory_sha256=args.expected_inventory_sha256,
            )
        elif args.mode == "snapshot":
            result = materialize_external_runtime(
                runtime_root=args.runtime_root,
                runtime_lock_path=args.runtime_lock,
                expected_inventory_sha256=args.expected_inventory_sha256,
                destination=args.destination,
                require_source_read_only=True,
            )
        elif args.mode == "assert-missing-runtime":
            result = verify_missing_runtime_refusal(
                runtime_root=args.runtime_root,
                runtime_lock_path=args.runtime_lock,
            )
        else:
            raise AssertionError(f"unhandled mode: {args.mode}")
    except VerificationError as exc:
        runtime_mode = args.mode in {"runtime", "snapshot", "entitlement"}
        runtime_label = runtime_mode or args.mode == "assert-missing-runtime"
        label = "RUNTIME" if runtime_label else "BUILD_INPUT"
        runtime_details = {
            "runtime": "external runtime verification refused",
            "snapshot": "runtime snapshot materialization refused",
            "entitlement": "customer runtime entitlement refused",
            "assert-missing-runtime": "missing runtime assertion refused",
        }
        detail = runtime_details.get(args.mode, str(exc))
        print(f"NPA_ROBOMIMIC_{label}_REFUSED: {detail}", file=sys.stderr)
        return RUNTIME_REFUSAL_STATUS if runtime_mode else 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
