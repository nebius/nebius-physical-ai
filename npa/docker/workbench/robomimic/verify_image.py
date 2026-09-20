#!/usr/bin/env python3
"""Fail-closed verification for the neutral image and external CUDA runtime."""

from __future__ import annotations

import argparse
import base64
import binascii
import configparser
import csv
from email.parser import BytesParser
import fcntl
import hashlib
import io
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zipfile


SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
SOURCE_LICENSE_SHA256 = (
    "7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556"
)
SOURCE_MANIFEST_SHA256 = (
    "878634c85bf2f3b73ec4ce8f01472305cea0665d0458785dbb771d8f6ee46a9f"
)
SOURCE_TREE_SHA1 = "4c8ebe35dbef16126dadf59cf8b771b9203753ab"
SOURCE_ARCHIVE_SHA256 = (
    "8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b"
)
SOURCE_ARCHIVE_SIZE = 57_907_200
DEBIAN_LOCK_SHA256 = "aebaefa21f527584d4a46cba8b22d0c691414a4762acd364252d746e03d3504e"
DEBIAN_PACKAGE_COUNT = 78
DEBIAN_PAYLOAD_BYTES = 29_807_672
BASE_IMAGE = (
    "python:3.11.16-slim-bookworm@sha256:"
    "528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
)
BAKED_LOCK_SHA256 = "65efcf0065ad4662b348e54e3f2d86996d934a518fcad0e89ecf012399ce1504"
BAKED_DISTRIBUTION_COUNT = 40
# Archive parsing and installed-tree enumeration are bounded independently of
# metadata. Unknown installation layouts refuse; these are not workload limits.
BAKED_ARCHIVE_MAX_BYTES = 256 * 1024 * 1024
BAKED_MEMBER_MAX_BYTES = 256 * 1024 * 1024
BAKED_EXPANDED_MAX_BYTES = 2 * 1024 * 1024 * 1024
BAKED_ENTRY_MAX_COUNT = 65_536
BAKED_INSTALLER_EXECUTABLE = "/usr/local/bin/python3"
INSTALLER_WHEEL_SHA256 = (
    "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
)
INSTALLER_WHEEL_SIZE = 1_816_632
INSTALLER_POLICY = "pip-26.2.1-posix-home-target-no-compile-v2"
RUNTIME_ROOT_DEFAULT = "/opt/npa-runtime/robomimic"
RUNTIME_REFUSAL_STATUS = 78
RUNTIME_METADATA_MAX_BYTES = 4 * 1024 * 1024
# The runtime lock currently names 62 public and vendor wheel objects. Keep a
# small bounded headroom for a customer inventory while refusing an open-ended
# closure; the exact inventory still has to match the lock byte-for-byte.
RUNTIME_ARTIFACT_MAX_COUNT = 64
RUNTIME_OBJECT_MAX_BYTES = 4 * 1024 * 1024 * 1024
RUNTIME_PAYLOAD_MAX_BYTES = 8 * 1024 * 1024 * 1024
# Installed wheels expand into more filesystem entries than source objects.
# The metadata ceiling makes this independently bounded limit conservative for
# the future exact operator-selected inventory while preventing inode exhaustion.
RUNTIME_PAYLOAD_MAX_ENTRY_COUNT = 65_536
RUNTIME_FETCH_PUBLIC_HOSTS = frozenset(
    {
        "download.pytorch.org",
        "files.pythonhosted.org",
        "pypi.org",
    }
)
RUNTIME_FETCH_CREDENTIAL_HOSTS = {
    "HF_TOKEN": frozenset({"huggingface.co"}),
    "NGC_API_KEY": frozenset({"ngc.nvidia.com", "nvcr.io"}),
}
RUNTIME_FETCH_ALLOWED_HOSTS = frozenset().union(
    RUNTIME_FETCH_PUBLIC_HOSTS,
    *RUNTIME_FETCH_CREDENTIAL_HOSTS.values(),
)
RUNTIME_FETCH_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
RUNTIME_FETCH_PROOF_NAME = ".runtime-fetch-proof.json"
RUNTIME_FETCH_DEFAULT_DENYLIST = (
    r"(?i)(?:^|/)(?:\.aws|\.ssh|credentials?|secrets?)(?:/|$)"
)
RUNTIME_FETCH_CREDENTIAL_ENVS = frozenset({"HF_TOKEN", "NGC_API_KEY"})
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


def _open_bounded_json_descriptor(
    path: Path,
    *,
    maximum_size: int,
    require_single_link: bool,
    require_owner_only: bool,
) -> tuple[int, os.stat_result]:
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
    except VerificationError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    return descriptor, details


def _read_bounded_json_descriptor(
    descriptor: int,
    initial: os.stat_result,
    path: Path,
    *,
    maximum_size: int,
    require_single_link: bool,
    require_owner_only: bool,
) -> bytes:
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(maximum_size + 1)
        if len(raw) > maximum_size:
            raise VerificationError(f"runtime metadata exceeds size limit: {path}")
        after = os.fstat(descriptor)
        if (
            len(raw) != initial.st_size
            or after.st_size != initial.st_size
            or after.st_dev != initial.st_dev
            or after.st_ino != initial.st_ino
            or (require_single_link and after.st_nlink != 1)
            or (
                require_owner_only
                and (after.st_uid != os.geteuid() or after.st_mode & 0o077)
            )
        ):
            raise VerificationError(f"runtime metadata changed while read: {path}")
    except OSError as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc
    return raw


def _bounded_json_object(
    path: Path,
    *,
    maximum_size: int,
    require_single_link: bool = False,
    require_owner_only: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Read one regular JSON object through a no-link hard byte ceiling."""

    descriptor, initial = _open_bounded_json_descriptor(
        path,
        maximum_size=maximum_size,
        require_single_link=require_single_link,
        require_owner_only=require_owner_only,
    )
    try:
        raw = _read_bounded_json_descriptor(
            descriptor,
            initial,
            path,
            maximum_size=maximum_size,
            require_single_link=require_single_link,
            require_owner_only=require_owner_only,
        )
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
        raise VerificationError(f"customer runtime entitlement {field} is invalid")
    return parsed


def _runtime_entitlement_contract(lock: dict[str, Any]) -> dict[str, Any]:
    contract = lock.get("customer_entitlement")
    if not isinstance(contract, dict) or set(contract) != {
        "schema",
        "maximum_validity_seconds",
        "responsibilities",
        "terms",
    }:
        raise VerificationError("runtime lock customer entitlement contract is invalid")
    if contract.get("schema") != "npa.robomimic.customer-runtime-entitlement.v1":
        raise VerificationError("runtime lock customer entitlement schema is invalid")
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
        _checked_https_url(
            term["url"], host=urllib.parse.urlsplit(term["url"]).hostname or ""
        )
    return contract


def _runtime_entitlement_notice_sha256(
    *, lock: dict[str, Any], lock_sha256: str
) -> str:
    contract = _runtime_entitlement_contract(lock)
    notice_identity = {
        "schema": contract["schema"],
        "runtime_id": lock.get("runtime_id"),
        "runtime_lock_sha256": lock_sha256,
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
    }
    return hashlib.sha256(
        json.dumps(notice_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_runtime_entitlement_request(
    expected_entitlement_sha256: str,
    expected_customer_binding_sha256: str,
    expected_inventory_sha256: str,
    expected_run_id: str,
) -> None:
    for label, digest in (
        ("entitlement", expected_entitlement_sha256),
        ("customer binding", expected_customer_binding_sha256),
        ("runtime inventory", expected_inventory_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise VerificationError(f"expected {label} digest is absent or malformed")
    if (
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", expected_run_id)
        is None
    ):
        raise VerificationError("expected run ID is absent or malformed")


def _runtime_entitlement_expected_values(
    record: dict[str, Any],
    lock: dict[str, Any],
    contract: dict[str, Any],
    runtime_lock_path: Path,
    expected_customer_binding_sha256: str,
    expected_run_id: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    expected_values = {
        "schema": contract["schema"],
        "decision": "accepted",
        "customer_binding_sha256": expected_customer_binding_sha256,
        "run_id": expected_run_id,
        "source_revision": lock.get("source_revision"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_manifest_sha256": expected_inventory_sha256,
        "runtime_lock_sha256": _sha256(runtime_lock_path),
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
    return expected_values


def _validate_runtime_entitlement_expiry(
    record: dict[str, Any], maximum_validity_seconds: int, now: datetime | None
) -> None:
    accepted_at = _utc_timestamp(record["accepted_at"], field="accepted_at")
    expires_at = _utc_timestamp(record["expires_at"], field="expires_at")
    observed_now = now or datetime.now(timezone.utc)
    if accepted_at > observed_now:
        raise VerificationError(
            "customer runtime entitlement acceptance is in the future"
        )
    validity_seconds = int((expires_at - accepted_at).total_seconds())
    if validity_seconds <= 0 or validity_seconds > maximum_validity_seconds:
        raise VerificationError(
            "customer runtime entitlement validity is out of bounds"
        )
    if observed_now >= expires_at:
        raise VerificationError("customer runtime entitlement has expired")


def _read_runtime_entitlement(path: Path, expected_sha256: str) -> dict[str, Any]:
    record, raw = _bounded_json_object(
        path,
        maximum_size=RUNTIME_ENTITLEMENT_MAX_BYTES,
        require_single_link=True,
        require_owner_only=True,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise VerificationError("customer runtime entitlement byte identity mismatch")
    return record


def _validate_runtime_entitlement_fields(record: dict[str, Any]) -> None:
    expected_keys = {
        "schema",
        "decision",
        "customer_binding_sha256",
        "run_id",
        "source_revision",
        "runtime_id",
        "runtime_manifest_sha256",
        "runtime_lock_sha256",
        "terms",
        "customer_responsibilities",
        "notice_sha256",
        "accepted_at",
        "expires_at",
    }
    if set(record) != expected_keys:
        raise VerificationError("customer runtime entitlement fields are invalid")


def _runtime_entitlement_result(
    record: dict[str, Any],
    expected_entitlement_sha256: str,
    expected_inventory_sha256: str,
    expected_values: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "npa.robomimic.customer-runtime-entitlement-verification.v1",
        "record_sha256": expected_entitlement_sha256,
        "runtime_manifest_sha256": expected_inventory_sha256,
        "runtime_lock_sha256": expected_values["runtime_lock_sha256"],
        "run_binding_matched": True,
        "customer_binding_matched": True,
        "expires_at": record["expires_at"],
        "redistribution_granted": False,
    }


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
    _validate_runtime_entitlement_request(
        expected_entitlement_sha256,
        expected_customer_binding_sha256,
        expected_inventory_sha256,
        expected_run_id,
    )

    record = _read_runtime_entitlement(entitlement_path, expected_entitlement_sha256)
    lock = _json_object(runtime_lock_path)
    contract = _runtime_entitlement_contract(lock)
    _validate_runtime_entitlement_fields(record)
    expected_values = _runtime_entitlement_expected_values(
        record,
        lock,
        contract,
        runtime_lock_path,
        expected_customer_binding_sha256,
        expected_run_id,
        expected_inventory_sha256,
    )
    _validate_runtime_entitlement_expiry(
        record, contract["maximum_validity_seconds"], now
    )
    return _runtime_entitlement_result(
        record, expected_entitlement_sha256, expected_inventory_sha256, expected_values
    )


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
    if {
        key: item for key, item in value.items() if key != "build_installer"
    } != expected:
        raise VerificationError("source manifest does not match the reviewed closure")
    _checked_installer(value.get("build_installer"))
    return value


def _checked_installer(value: Any) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("name") != "pip"
        or value.get("version") != "26.2.1"
        or value.get("path") != "tools/pip-26.2.1-py3-none-any.whl"
        or value.get("size") != INSTALLER_WHEEL_SIZE
        or value.get("sha256") != INSTALLER_WHEEL_SHA256
        or value.get("source_revision") != "634a6ec1a5d9dcc2433571cdb2f4c58a4bb29caf"
    ):
        raise VerificationError("unsupported build installer identity")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != (
        "10d2341e000fc9db85d2bce23025bb6c68cccea82d55ee49140dd3d75472b062"
    ):
        raise VerificationError("build installer metadata mismatch")
    _checked_https_url(value.get("url"), host="files.pythonhosted.org")
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


def _validate_debian_package_identity(value: dict[str, Any]) -> str:
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
    if set(value) != required:
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
    return name


def _validate_debian_package_sources(
    value: dict[str, Any], *, name: str, source_ids: set[str]
) -> None:
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


def _validate_debian_package_dependencies(value: dict[str, Any], *, name: str) -> None:
    dependencies = value.get("depends")
    if not isinstance(dependencies, list) or len(dependencies) != len(
        set(dependencies)
    ):
        raise VerificationError(f"invalid Debian dependency list for {name}")
    if not all(isinstance(item, str) and item for item in dependencies):
        raise VerificationError(f"invalid Debian dependency name for {name}")


def _checked_debian_package(value: Any, *, source_ids: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError("invalid Debian binary package record")
    name = _validate_debian_package_identity(value)
    _validate_debian_package_sources(value, name=name, source_ids=source_ids)
    _validate_debian_package_dependencies(value, name=name)
    return value


def _verify_debian_lock_header(value: dict[str, Any]) -> None:
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


def _verify_debian_repository_metadata(value: dict[str, Any]) -> None:
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


def _debian_source_ids(value: dict[str, Any]) -> set[str]:
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
    return source_ids


def _checked_debian_lock(value: dict[str, Any]) -> dict[str, Any]:
    _verify_debian_lock_header(value)
    _verify_debian_repository_metadata(value)
    source_ids = _debian_source_ids(value)
    packages = value.get("packages")
    if not isinstance(packages, list):
        raise VerificationError("Debian package records must be an array")
    checked = [
        _checked_debian_package(item, source_ids=source_ids) for item in packages
    ]
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
        "tools",
        manifest["build_installer"]["path"],
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
    installer = _verified_installer(input_root, manifest)
    return {
        "schema": "npa.robomimic.build-input-verification.v1",
        "debian_package_count": len(lock["packages"]),
        "debian_payload_bytes": sum(item["size"] for item in lock["packages"]),
        "debian_lock_sha256": DEBIAN_LOCK_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha1": SOURCE_TREE_SHA1,
        "build_installer_sha256": installer["sha256"],
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
        raise VerificationError(
            f"failed to fetch Debian package {package['name']}"
        ) from exc
    if total != package["size"] or digest.hexdigest() != package["sha256"]:
        raise VerificationError(
            f"downloaded Debian package mismatch: {package['name']}"
        )


def _git(
    *args: str, cwd: Path | None = None, output=None
) -> subprocess.CompletedProcess:
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
        _prepare_installer(manifest, staging)
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


class _NoInstallerRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        raise VerificationError("build installer redirect refused")


def _prepare_installer(manifest: dict, staging: Path) -> None:
    installer = _checked_installer(manifest["build_installer"])
    destination = staging / installer["path"]
    destination.parent.mkdir(mode=0o700)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoInstallerRedirect()
    )
    with opener.open(installer["url"], timeout=60) as response:
        if response.status != 200 or response.url != installer["url"]:
            raise VerificationError("build installer response refused")
        raw = response.read(INSTALLER_WHEEL_SIZE + 1)
    _checked_installer_bytes(raw)
    with destination.open("xb") as stream:
        stream.write(raw)
    destination.chmod(0o444)


def _checked_installer_bytes(raw: bytes) -> None:
    if (
        len(raw) != INSTALLER_WHEEL_SIZE
        or hashlib.sha256(raw).hexdigest() != INSTALLER_WHEEL_SHA256
    ):
        raise VerificationError("build installer bytes mismatch")


def _verified_installer(input_root: Path, manifest: dict) -> dict:
    installer = _checked_installer(manifest["build_installer"])
    path = input_root / installer["path"]
    raw = _immutable_bytes(path, INSTALLER_WHEEL_SIZE)
    _checked_installer_bytes(raw)
    if path.stat().st_mode & 0o222:
        raise VerificationError("build installer must be read-only")
    return installer


def _installer_environment(scratch: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "PIP_CONFIG_FILE": "/dev/null",
        "XDG_CACHE_HOME": str(scratch),
    }


def _require_installer_platform() -> None:
    """Check supported layout, not independently qualified parent-image bytes."""
    if (
        sys.version_info[:3] != (3, 11, 16)
        or sys.implementation.name != "cpython"
        or sys.platform != "linux"
        or os.uname().machine != "x86_64"
        or sys.executable != BAKED_INSTALLER_EXECUTABLE
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.dont_write_bytecode
        or sys.prefix != sys.base_prefix
    ):
        raise VerificationError("unsupported installer interpreter or isolation")
    allowed = {
        "/usr/local/lib/python311.zip",
        "/usr/local/lib/python3.11",
        "/usr/local/lib/python3.11/lib-dynload",
    }
    if set(sys.path) != allowed or "pip" in sys.modules:
        raise VerificationError("installer import path is not isolated stdlib")
    _require_installer_scheme()


def _require_installer_scheme() -> None:
    if (
        sysconfig.get_preferred_scheme("home") != "posix_home"
        or getattr(sysconfig, "_PIP_USE_SYSCONFIG", True) is not True
    ):
        raise VerificationError("unsupported installer home scheme")
    variables = dict.fromkeys(
        (
            "installed_base",
            "base",
            "installed_platbase",
            "platbase",
            "prefix",
            "exec_prefix",
            "userbase",
        ),
        "/npa-scheme-probe",
    )
    paths = sysconfig.get_paths(scheme="posix_home", vars=variables)
    expected = {
        "purelib": "/npa-scheme-probe/lib/python",
        "platlib": "/npa-scheme-probe/lib/python",
        "scripts": "/npa-scheme-probe/bin",
        "data": "/npa-scheme-probe",
    }
    if any(paths[key] != value for key, value in expected.items()):
        raise VerificationError("unsupported installer target layout")


def _installer_arguments(action: str, wheels: Path, lock: Path) -> list[str]:
    common = [
        "--isolated",
        action,
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--no-deps",
        "--require-hashes",
        "--requirement",
        str(lock),
    ]
    if action == "download":
        return common + [
            "--index-url",
            "https://pypi.org/simple",
            "--dest",
            str(wheels),
        ]
    if action == "install":
        return common + [
            "--no-compile",
            "--no-index",
            "--find-links",
            str(wheels),
            "--target",
            "/opt/robomimic-deps",
        ]
    raise VerificationError("unsupported installer action")


def run_build_installer(*, action: str, input_root: Path, wheels: Path) -> dict:
    """Use the exact build-only wheel; parent-byte qualification remains external.

    Args:
        action: One of the two fixed download/install operations.
        input_root: Verified read-only build-input mount.
        wheels: Dedicated build-only solution wheel directory.
    Returns:
        Exact installer identity and executed operation, not image acceptance.
    Raises:
        VerificationError: Unsupported inputs, environment, output or operation.
    """
    _require_installer_platform()
    manifest = _source_manifest(Path("/opt/npa/robomimic/source-manifest.json"))
    installer = _verified_installer(input_root, manifest)
    if not os.statvfs(input_root).f_flag & os.ST_RDONLY:
        raise VerificationError("installer input mount must be read-only")
    lock = Path("/opt/npa/robomimic/baked-requirements.lock")
    _locked_baked_artifacts(lock)
    arguments = _installer_arguments(action, wheels, lock)
    _require_fresh_installer_destination(action, wheels)
    return _execute_build_installer(input_root, installer, action, arguments)


def _execute_build_installer(
    input_root: Path, installer: dict, action: str, arguments: list[str]
) -> dict:
    with tempfile.TemporaryDirectory(
        prefix="installer-scratch-", dir=str(_installer_workspace_parent())
    ) as temporary:
        scratch = Path(temporary)
        command = [
            BAKED_INSTALLER_EXECUTABLE,
            "-I",
            "-S",
            "-B",
            "-c",
            "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
            "runpy.run_module('pip',run_name='__main__',alter_sys=True)",
            str(input_root / installer["path"]),
            *arguments,
        ]
        result = subprocess.run(
            command,
            env=_installer_environment(scratch),
            cwd=scratch,
            umask=0o022,
            check=False,
        )
    if result.returncode:
        raise VerificationError("exact build installer refused")
    return {
        "installer_sha256": installer["sha256"],
        "action": action,
        "installation_policy": INSTALLER_POLICY,
        "image_qualified": False,
    }


def _require_fresh_installer_destination(action: str, wheels: Path) -> None:
    if wheels != _installer_workspace_parent() / "installer-wheels":
        raise VerificationError("unsupported build wheel destination")
    if action == "download":
        if wheels.exists() or wheels.is_symlink():
            raise VerificationError("build wheel destination already exists")
    else:
        target = Path("/opt/robomimic-deps")
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise VerificationError("dependency install target must be empty")


def _installer_workspace_parent() -> Path:
    """Use the root-owned build directory, never a shared temporary namespace."""
    parent = Path("/opt/npa/robomimic")
    observed = parent.lstat()
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != 0
        or observed.st_mode & 0o022
        or os.geteuid() != 0
    ):
        raise VerificationError("installer workspace parent is not root-owned")
    return parent


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
            raise VerificationError(
                f"Debian package notice is absent: {package['name']}"
            )
    return {
        "schema": "npa.robomimic.debian-install-verification.v1",
        "package_count": len(lock["packages"]),
        "package_bytes_verified_before_install": DEBIAN_PAYLOAD_BYTES,
        "notices_present": len(lock["packages"]),
    }


def _baked_lock_records(lock_path: Path) -> list[str]:
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
    return records


def _locked_baked_artifacts(lock_path: Path) -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)==([^ ]+)((?: --hash=sha256:[0-9a-f]{64})+)$"
    )
    for record in _baked_lock_records(lock_path):
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
        packages[name] = {"version": match.group(2), "sha256": set(hashes)}
    return packages


def _locked_baked_packages(lock_path: Path) -> dict[str, str]:
    return {
        name: entry["version"]
        for name, entry in _locked_baked_artifacts(lock_path).items()
    }


def _immutable_bytes(path: Path, maximum: int) -> bytes:
    """Read one regular, non-symlink object; never import installed code."""
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise VerificationError("unsupported immutable proof object")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise VerificationError("immutable proof object unavailable") from exc
    if (
        _immutable_stat_identity(before) != _immutable_stat_identity(after)
        or len(raw) != before.st_size
    ):
        raise VerificationError("immutable proof object changed while reading")
    return raw


def _immutable_stat_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _installed_member_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise VerificationError("unsafe installed inventory member")
    return value


def _record_member_path(stage: Path, record_path: Path, value: str) -> tuple[str, Path]:
    """Resolve a RECORD member while allowing only pip's script relocation."""
    parts = PurePosixPath(value).parts
    if len(parts) >= 4 and parts[:3] == ("..", "..", "bin"):
        if any(part in {"", ".", ".."} for part in parts[3:]):
            raise VerificationError("unsafe installed RECORD member")
        canonical = PurePosixPath("bin", *parts[3:]).as_posix()
    else:
        canonical = _installed_member_path(value)
    target = stage / canonical
    try:
        target.relative_to(stage)
    except ValueError as exc:
        raise VerificationError("installed RECORD member escapes stage") from exc
    return canonical, target


def _file_identity(raw: bytes, executable: bool = False) -> dict[str, Any]:
    return {
        "type": "file",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "executable": executable,
        "mode": 0o755 if executable else 0o644,
    }


def _add_inventory_member(inventory: dict, name: str, entry: dict) -> None:
    _installed_member_path(name)
    if name in inventory:
        raise VerificationError("duplicate installed inventory member")
    inventory[name] = entry


def _inventory_directories(inventory: dict) -> dict:
    result = dict(inventory)
    for name in inventory:
        for parent in PurePosixPath(name).parents:
            if parent.as_posix() == ".":
                continue
            key = parent.as_posix()
            if key in result and result[key] != {"type": "directory", "mode": 0o755}:
                raise VerificationError("installed inventory file/directory conflict")
            result[key] = {"type": "directory", "mode": 0o755}
    return result


def _source_installed_inventory(path: Path, manifest: dict) -> dict:
    """The pinned archive SHA-256, not an installed manifest, is the authority."""
    _verify_source_archive(path, manifest)
    raw = _immutable_bytes(path, BAKED_ARCHIVE_MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != manifest["archive"]["sha256"]:
        raise VerificationError("source archive identity mismatch")
    inventory = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if member.isdir():
                entry = {"type": "directory", "mode": member.mode & ~0o022}
            elif member.isreg() and member.size <= BAKED_MEMBER_MAX_BYTES:
                stream = archive.extractfile(member)
                if stream is None:
                    raise VerificationError("source member unavailable")
                entry = _file_identity(stream.read(), bool(member.mode & 0o111))
                entry["mode"] = member.mode & ~0o022
            else:
                raise VerificationError("unsupported source inventory member")
            _add_inventory_member(inventory, member.name, entry)
    return _inventory_directories(inventory)


def _installed_tree_inventory(root: Path) -> dict:
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise VerificationError("installed tree root is not a directory")
    inventory, pending, total = {}, [root], 0
    while pending:
        parent = pending.pop()
        for path in parent.iterdir():
            mode = path.lstat().st_mode
            name = path.relative_to(root).as_posix()
            if stat.S_ISDIR(mode):
                entry = {"type": "directory", "mode": stat.S_IMODE(mode)}
                pending.append(path)
            elif stat.S_ISREG(mode):
                raw = _immutable_bytes(path, BAKED_MEMBER_MAX_BYTES)
                total += len(raw)
                entry = _file_identity(raw, bool(mode & 0o111))
                entry["mode"] = stat.S_IMODE(mode)
            else:
                raise VerificationError("installed tree contains a non-regular object")
            _add_inventory_member(inventory, name, entry)
            if (
                len(inventory) > BAKED_ENTRY_MAX_COUNT
                or total > BAKED_EXPANDED_MAX_BYTES
            ):
                raise VerificationError("installed inventory exceeds bound")
    return inventory


def _inventory_proof(root: Path, expected: dict, *, read_only: bool = False) -> dict:
    if read_only:
        expected = {
            name: {**entry, "mode": entry["mode"] & ~0o222}
            for name, entry in expected.items()
        }
    observed = _installed_tree_inventory(root)
    if observed != expected:
        raise VerificationError("installed byte inventory mismatch")
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    return {
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "entry_count": len(expected),
        "file_count": sum(e["type"] == "file" for e in expected.values()),
    }


def verify_installed_source_tree(
    *, source_root: Path, source_archive_path: Path, metadata_path: Path
) -> dict[str, Any]:
    """Authenticate the extracted source tree before the build continues."""
    metadata = _source_manifest(metadata_path)
    expected = _source_installed_inventory(source_archive_path, metadata)
    return _inventory_proof(source_root, expected)


def _wheel_members(raw: bytes) -> dict[str, tuple[bytes, bool]]:
    result, names, total = {}, set(), 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist()) > BAKED_ENTRY_MAX_COUNT:
            raise VerificationError("wheel member count exceeds bound")
        for member in archive.infolist():
            name = _installed_member_path(member.filename.rstrip("/"))
            mode = member.external_attr >> 16
            if name in names or member.flag_bits & 1:
                raise VerificationError("duplicate or encrypted wheel member")
            names.add(name)
            kind = stat.S_IFMT(mode)
            if kind not in {0, stat.S_IFDIR if member.is_dir() else stat.S_IFREG}:
                raise VerificationError("wheel contains a non-regular member")
            if member.is_dir():
                continue
            total += member.file_size
            if (
                member.file_size > BAKED_MEMBER_MAX_BYTES
                or total > BAKED_EXPANDED_MAX_BYTES
            ):
                raise VerificationError("wheel expanded bytes exceed bound")
            result[name] = (archive.read(member), bool(mode & 0o111))
    return result


def _wheel_distribution(members: dict, name: str, version: str) -> str:
    directories = {
        PurePosixPath(n).parts[0]
        for n in members
        if PurePosixPath(n).parts[0].endswith(".dist-info")
    }
    if len(directories) != 1:
        raise VerificationError("wheel distribution identity is ambiguous")
    directory = directories.pop()
    suffix = f"-{version}.dist-info"
    if not directory.endswith(suffix):
        raise VerificationError("wheel distribution directory mismatch")
    observed_name = directory[: -len(suffix)]
    if _canonical_name(observed_name) != _canonical_name(name):
        raise VerificationError("wheel distribution directory mismatch")
    metadata = BytesParser().parsebytes(members[f"{directory}/METADATA"][0])
    if metadata.get_all("Name") != [metadata.get("Name")] or metadata.get_all(
        "Version"
    ) != [version]:
        raise VerificationError("wheel distribution metadata is ambiguous")
    if _canonical_name(metadata.get("Name", "")) != name:
        raise VerificationError("wheel distribution name mismatch")
    wheel = BytesParser().parsebytes(members[f"{directory}/WHEEL"][0])
    if wheel.get_all("Wheel-Version") != ["1.0"] or wheel.get_all(
        "Root-Is-Purelib"
    ) not in (["true"], ["false"]):
        raise VerificationError("unsupported wheel installation metadata")
    return directory


def _wheel_target(name: str, directory: str) -> str:
    parts = PurePosixPath(name).parts
    if parts[0].endswith(".data"):
        if (
            parts[0] != directory.removesuffix(".dist-info") + ".data"
            or len(parts) < 3
            or parts[1] not in {"purelib", "platlib", "data"}
        ):
            raise VerificationError("unsupported wheel installation scheme")
        parts = parts[2:]
    if any(
        part.endswith((".dist-info", ".egg-info")) and (index != 0 or part != directory)
        for index, part in enumerate(parts)
    ):
        raise VerificationError("wheel adds an undeclared distribution")
    return PurePosixPath(*parts).as_posix()


def _wheel_script_member(
    name: str, directory: str, raw: bytes, interpreter: str = BAKED_INSTALLER_EXECUTABLE
) -> tuple[str, dict[str, Any], str]:
    """Apply only pip's authenticated ``.data/scripts`` transformation.

    The pinned installer routes these members to the target ``bin`` directory,
    rewrites a leading ``#!python`` line to its target interpreter, and marks
    the result executable.  Unknown script forms remain fail-closed rather
    than being accepted as an unproved generic data scheme.
    """
    parts = PurePosixPath(name).parts
    data_directory = directory.removesuffix(".dist-info") + ".data"
    if len(parts) < 3 or parts[0] != data_directory or parts[1] != "scripts":
        raise VerificationError("unsupported wheel installation scheme")
    relative = PurePosixPath(*parts[2:]).as_posix()
    target = PurePosixPath("bin", relative).as_posix()
    _installed_member_path(target)
    firstline, separator, rest = raw.partition(b"\n")
    if firstline != b"#!python" or not separator:
        raise VerificationError("unsupported wheel script transformation")
    transformed = b"#!" + interpreter.encode("ascii") + b"\n" + rest
    return target, _file_identity(transformed, True), "../../" + target


def _wheel_member_installation(
    name: str,
    directory: str,
    raw: bytes,
    executable: bool,
    interpreter: str = BAKED_INSTALLER_EXECUTABLE,
) -> tuple[str, dict[str, Any], str]:
    parts = PurePosixPath(name).parts
    if len(parts) >= 2 and parts[0].endswith(".data") and parts[1] == "scripts":
        return _wheel_script_member(name, directory, raw, interpreter)
    target = _wheel_target(name, directory)
    if target == "bin" or target.startswith("bin/"):
        raise VerificationError("wheel reserves generated script directory")
    entry = _file_identity(raw, executable)
    if not parts[0].endswith(".data"):
        record_path = target
    elif parts[1] in {"purelib", "platlib"}:
        # Pip's selected library directory is the RECORD root for both
        # library schemes; only destinations outside it retain ../../.
        record_path = target
    else:
        record_path = "../../" + target
    return target, entry, record_path


def _console_script(module: str, function: str) -> bytes:
    """Pip 26.2.1 PipScriptMaker, independently matched to pinned source bytes."""
    return (
        f"#!{BAKED_INSTALLER_EXECUTABLE}\nimport sys\n"
        f"from {module} import {function.split('.')[0]}\n"
        "if __name__ == '__main__':\n"
        "    sys.argv[0] = sys.argv[0].removesuffix('.exe')\n"
        f"    sys.exit({function}())\n"
    ).encode()


def _wheel_scripts(
    members: dict, directory: str, interpreter: str = BAKED_INSTALLER_EXECUTABLE
) -> dict:
    raw = members.get(f"{directory}/entry_points.txt", (b"", False))[0]
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(raw.decode("utf-8"))
    if parser.defaults():
        raise VerificationError("unsupported entry point defaults")
    result = {}
    for section in ("console_scripts", "gui_scripts"):
        if not parser.has_section(section):
            continue
        for name, value in parser.items(section):
            _add_wheel_script(result, name, value, interpreter)
    return result


def _add_wheel_script(result: dict, name: str, value: str, interpreter: str) -> None:
    match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):"
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
        value,
    )
    if (
        not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
        or re.fullmatch(r"pip(?:[0-9.]+)?|easy_install(?:-[0-9.]+)?", name)
        or match is None
        or any(
            keyword.iskeyword(part)
            for group in match.groups()
            for part in group.split(".")
        )
    ):
        raise VerificationError("unsupported wheel script entry point")
    generated = _console_script(*match.groups()).replace(
        BAKED_INSTALLER_EXECUTABLE.encode("ascii"), interpreter.encode("ascii")
    )
    _add_inventory_member(result, "bin/" + name, _file_identity(generated, True))


def _record_digest(entry: dict) -> str:
    return "sha256=" + base64.urlsafe_b64encode(
        bytes.fromhex(entry["sha256"])
    ).decode().rstrip("=")


def _authenticated_installed_record(
    path: Path,
    expected: dict,
    record_name: str,
    record_paths: dict[str, str] | None = None,
) -> dict:
    """Require pinned pip's exact sorted CSV, independently derived from inputs."""
    raw = _immutable_bytes(path, BAKED_MEMBER_MAX_BYTES)
    if raw != _installed_record_bytes(expected, record_name, record_paths):
        raise VerificationError("installed RECORD transformation mismatch")
    return _file_identity(raw)


def _installed_record_bytes(
    expected: dict,
    record_name: str,
    record_paths: dict[str, str] | None = None,
) -> bytes:
    # posix_home lib/python -> bin/data are exactly ../../ before --target moves.
    rows = [
        (
            (
                record_paths[name]
                if record_paths is not None and name in record_paths
                else "../../" + name
                if name.startswith("bin/")
                else name
            ),
            _record_digest(entry),
            str(entry["size"]),
        )
        for name, entry in expected.items()
    ]
    rows.append((record_name, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output).writerows(sorted(rows))
    return output.getvalue().encode("utf-8")


def _verify_wheel_record(members: dict, record_name: str) -> None:
    raw = members[record_name][0]
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8")), strict=True))
    observed = {}
    for row in rows:
        if len(row) != 3 or row[0] in observed:
            raise VerificationError("unsupported wheel RECORD")
        observed[row[0]] = tuple(row[1:])
    wanted = {
        name: (_record_digest(_file_identity(raw)), str(len(raw)))
        for name, (raw, _) in members.items()
        if name != record_name
    }
    wanted[record_name] = ("", "")
    if observed != wanted:
        raise VerificationError("wheel RECORD does not match authenticated members")


def _wheel_expected_inventory(
    members: dict,
    directory: str,
    installed_root: Path | None,
    interpreter: str = BAKED_INSTALLER_EXECUTABLE,
    record_sink: dict[str, bytes] | None = None,
) -> dict:
    record_name = f"{directory}/RECORD"
    if record_name not in members:
        raise VerificationError("wheel RECORD is absent")
    _verify_wheel_record(members, record_name)
    result, record_paths = {}, {}
    for name, (raw, executable) in members.items():
        if name != record_name:
            target, entry, record_path = _wheel_member_installation(
                name, directory, raw, executable, interpreter
            )
            _add_inventory_member(result, target, entry)
            record_paths[target] = record_path
    for suffix, raw in (("INSTALLER", b"pip\n"), ("REQUESTED", b"")):
        target = f"{directory}/{suffix}"
        _add_inventory_member(result, target, _file_identity(raw))
        record_paths[target] = target
    for name, entry in _wheel_scripts(members, directory, interpreter).items():
        _add_inventory_member(result, name, entry)
        record_paths[name] = "../../" + name
    if installed_root is None:
        record_bytes = _installed_record_bytes(result, record_name, record_paths)
        if record_sink is not None:
            record_sink[record_name] = record_bytes
        result[record_name] = _file_identity(record_bytes)
    else:
        result[record_name] = _authenticated_installed_record(
            installed_root / record_name, result, record_name, record_paths
        )
    return result


def _selected_wheel(path: Path, locked: dict) -> tuple[str, str, bytes]:
    parts = path.name.removesuffix(".whl").split("-")
    if path.suffix != ".whl" or len(parts) not in {5, 6}:
        raise VerificationError("invalid selected wheel filename")
    name = _canonical_name(parts[0])
    if name not in locked or parts[1] != locked[name]["version"]:
        raise VerificationError("selected wheel package/version mismatch")
    raw = _immutable_bytes(path, BAKED_ARCHIVE_MAX_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in locked[name]["sha256"]:
        raise VerificationError("selected wheel hash is not locked")
    return name, digest, raw


def _baked_dependency_proof(
    wheel_root: Path, installed_root: Path, lock_path: Path, *, read_only: bool = False
) -> dict:
    locked = _locked_baked_artifacts(lock_path)
    if not stat.S_ISDIR(wheel_root.lstat().st_mode):
        raise VerificationError("selected wheel root is not a directory")
    inventory, selected = {}, {}
    for path in sorted(wheel_root.iterdir()):
        name, digest, raw = _selected_wheel(path, locked)
        if name in selected:
            raise VerificationError("duplicate selected distribution")
        members = _wheel_members(raw)
        directory = _wheel_distribution(members, name, locked[name]["version"])
        expected = _wheel_expected_inventory(members, directory, installed_root)
        for member, entry in expected.items():
            _add_inventory_member(inventory, member, entry)
        selected[name] = {
            "filename": path.name,
            "sha256": digest,
            "size": len(raw),
            "version": locked[name]["version"],
        }
    if set(selected) != set(locked):
        raise VerificationError("selected wheel closure mismatch")
    if any(
        n in {"torch", "torchvision", "triton"} or n.startswith("nvidia-")
        for n in selected
    ):
        raise VerificationError("forbidden runtime distribution is baked")
    proof = _inventory_proof(
        installed_root, _inventory_directories(inventory), read_only=read_only
    )
    return {
        **proof,
        "selected_wheels": selected,
        "installation_policy": INSTALLER_POLICY,
    }


def verify_neutral_image(
    *,
    source_root: Path,
    metadata_path: Path,
    debian_lock_path: Path,
    baked_lock_path: Path,
    baked_deps_path: Path,
    source_archive_path: Path,
    selected_wheel_root: Path,
    empty_boundary_paths: tuple[Path, ...],
    installer_input_root: Path,
    read_only: bool = False,
) -> dict[str, Any]:
    """Authenticate installed trees using independent immutable build inputs.

    Tree paths are compared without imports against the archive, wheel and lock
    inputs. Runtime/data/output boundaries must be empty. ``read_only`` models
    only the Dockerfile's final removal of write permissions.
    Returns: Byte inventories/provenance, not parent/image or live qualification.
    Raises:
        VerificationError: Any input, installed member or boundary differs.
    """
    metadata = _source_manifest(metadata_path)
    installer = _verified_installer(installer_input_root, metadata)
    source, dependencies = _image_byte_proofs(
        source_root,
        source_archive_path,
        metadata,
        selected_wheel_root,
        baked_deps_path,
        baked_lock_path,
        read_only,
    )
    debian = verify_debian_install(debian_lock_path=debian_lock_path)
    _verify_empty_boundaries(empty_boundary_paths)
    return {
        **_neutral_image_proof(metadata, debian, source, dependencies),
        **_installer_proof_fields(installer),
    }


def _image_byte_proofs(
    source_root, source_archive, metadata, wheels, deps, lock, read_only
):
    try:
        source = _inventory_proof(
            source_root,
            _source_installed_inventory(source_archive, metadata),
            read_only=read_only,
        )
        dependencies = _baked_dependency_proof(wheels, deps, lock, read_only=read_only)
    except (
        OSError,
        KeyError,
        ValueError,
        csv.Error,
        configparser.Error,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as exc:
        raise VerificationError("installed-byte proof inputs are invalid") from exc
    return source, dependencies


def _installer_proof_fields(installer: dict) -> dict:
    return {
        "installer_input_sha256": installer["sha256"],
        "installer_member_inventory_sha256": installer["members"]["inventory_sha256"],
        "installer_source_revision": installer["source_revision"],
        "installer_publisher_statement_sha256": installer["publisher"][
            "verified_statement_sha256"
        ],
        "parent_byte_qualification": "not-established-by-installed-tree-proof",
    }


def _verify_empty_boundaries(paths: tuple[Path, ...]) -> None:
    nonempty_boundaries = [
        str(path) for path in paths if path.exists() and any(path.iterdir())
    ]
    if nonempty_boundaries:
        raise VerificationError(
            f"runtime/data/output boundary is populated: {nonempty_boundaries}"
        )


def _neutral_image_proof(
    metadata: dict, debian: dict, source: dict, dependencies: dict
) -> dict:
    return {
        "schema": "npa.robomimic.neutral-image-verification.v2",
        "baked_dependency_count": len(dependencies["selected_wheels"]),
        "installed_source": source,
        "installed_dependencies": dependencies,
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
            raise VerificationError(
                f"runtime payload object exceeds size limit: {path}"
            )
        if total_bytes > RUNTIME_PAYLOAD_MAX_BYTES - size:
            raise VerificationError("runtime payload aggregate exceeds size limit")
        total_bytes += size
    return files, links, total_bytes


def _checked_artifact_entry(entry: Any) -> tuple[str, dict[str, str | int]]:
    if not isinstance(entry, dict):
        raise VerificationError("runtime artifact inventory entry must be an object")
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
    return name, {
        "version": version,
        "filename": filename,
        "source": source,
        "sha256": str(digest),
        "size": size,
    }


def _checked_artifacts(
    value: Any,
) -> tuple[dict[str, dict[str, str | int]], int]:
    if not isinstance(value, list):
        raise VerificationError("runtime artifact inventory must be an array")
    if len(value) > RUNTIME_ARTIFACT_MAX_COUNT:
        raise VerificationError("runtime artifact object count exceeds limit")
    result: dict[str, dict[str, str | int]] = {}
    total_bytes = 0
    for raw_entry in value:
        name, entry = _checked_artifact_entry(raw_entry)
        size = entry["size"]
        if total_bytes > RUNTIME_PAYLOAD_MAX_BYTES - size:
            raise VerificationError("runtime artifact aggregate exceeds size limit")
        if name in result:
            raise VerificationError(f"duplicate runtime artifact: {name}")
        total_bytes += size
        result[name] = entry
    return result, total_bytes


def _open_declared_runtime_file(
    path: Path, expected_size: int
) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise VerificationError(
            "runtime payload verification requires safe descriptor flags"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | nonblock
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerificationError(
            "declared runtime file is not a safe regular file"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != expected_size
        ):
            raise VerificationError("runtime file identity mismatch")
    except VerificationError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise VerificationError("runtime file identity mismatch") from exc
    return descriptor, opened


def _read_declared_runtime_digest(
    descriptor: int, opened: os.stat_result, expected_size: int, expected_sha256: str
) -> None:
    digest = hashlib.sha256()
    remaining = expected_size
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise VerificationError("runtime file identity mismatch")
                digest.update(chunk)
                remaining -= len(chunk)
        closed = os.fstat(descriptor)
    except OSError as exc:
        raise VerificationError("runtime file identity mismatch") from exc
    if (
        closed.st_dev != opened.st_dev
        or closed.st_ino != opened.st_ino
        or closed.st_size != expected_size
        or closed.st_nlink != 1
        or digest.hexdigest() != expected_sha256
    ):
        raise VerificationError("runtime file identity mismatch")


def _verify_declared_runtime_file(
    path: Path, *, expected_size: int, expected_sha256: str
) -> None:
    """Verify one declared payload through a bounded, identity-stable descriptor."""

    descriptor, opened = _open_declared_runtime_file(path, expected_size)
    try:
        _read_declared_runtime_digest(
            descriptor, opened, expected_size, expected_sha256
        )
    finally:
        os.close(descriptor)


def _resolved_runtime_path(path: Path) -> Path:
    """Resolve a runtime path without retaining filesystem diagnostics."""

    resolution_failed = False
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        resolution_failed = True
    if resolution_failed:
        raise VerificationError("runtime symlink resolution failed") from None
    return resolved


def _runtime_metadata(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    require_read_only_mount: bool,
) -> tuple[dict, str, dict, str, bool]:
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
    _verify_runtime_metadata_bindings(
        lock, lock_hash, marker, inventory, inventory_sha256
    )
    return lock, lock_hash, inventory, inventory_sha256, read_only_mount


def _verify_runtime_metadata_bindings(
    lock: dict, lock_hash: str, marker: dict, inventory: dict, inventory_sha256: str
) -> None:
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


def _runtime_artifact_closure(lock: dict, inventory: dict) -> tuple[dict, int]:
    artifacts, artifact_payload_bytes = _checked_artifacts(inventory.get("artifacts"))
    locked_packages = lock.get("packages")
    if not isinstance(locked_packages, dict):
        raise VerificationError("runtime lock package map is invalid")
    if set(artifacts) != set(locked_packages) or any(
        artifacts[name]["version"] != version
        for name, version in locked_packages.items()
    ):
        raise VerificationError("runtime artifact closure does not match package lock")
    return artifacts, artifact_payload_bytes


def _runtime_fetch_url(value: str) -> str:
    """Accept only immutable HTTPS artifact URLs on approved provider hosts."""

    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port is not None
        or parsed.fragment
        or parsed.hostname not in RUNTIME_FETCH_ALLOWED_HOSTS
        or not parsed.path
    ):
        raise VerificationError("runtime fetch URL is not an approved HTTPS endpoint")
    return value


def _runtime_fetch_auth_headers(
    url: str, credential_env: str, credential: str
) -> dict[str, str]:
    """Bind bearer credentials to their provider origin; public hosts stay anonymous."""

    host = urllib.parse.urlsplit(_runtime_fetch_url(url)).hostname
    if host in RUNTIME_FETCH_PUBLIC_HOSTS:
        return {}
    if host not in RUNTIME_FETCH_CREDENTIAL_HOSTS.get(credential_env, frozenset()):
        raise VerificationError("runtime fetch credential host binding is unsupported")
    if not credential:
        raise VerificationError("customer runtime credential is absent")
    return {"Authorization": f"Bearer {credential}"}


def _runtime_fetch_denylist() -> tuple[re.Pattern[str], str]:
    """Load a customer runtime denylist without making it a publication secret."""

    raw = os.environ.get("NPA_ROBOMIMIC_CUSTOMER_DENYLIST", "")
    source = "runtime-input" if raw else "built-in-safe-default"
    try:
        pattern = re.compile(raw or RUNTIME_FETCH_DEFAULT_DENYLIST)
    except re.error as exc:
        raise VerificationError("customer runtime denylist is invalid") from exc
    return pattern, source


def _runtime_fetch_site_packages(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or PurePosixPath(value).as_posix() != value
        or any(part in {"", ".", ".."} for part in PurePosixPath(value).parts)
        or not value.startswith("payload/")
    ):
        raise VerificationError("runtime fetch site-packages path is unsafe")
    return value


def _runtime_fetch_installer(fetch: dict[str, Any]) -> dict[str, str | int]:
    raw = fetch.get("installer")
    name, installer = _checked_artifact_entry(raw)
    if (
        name != "pip"
        or installer["version"] != "26.2.1"
        or installer["filename"] != "pip-26.2.1-py3-none-any.whl"
        or installer["size"] != INSTALLER_WHEEL_SIZE
        or installer["sha256"] != INSTALLER_WHEEL_SHA256
        or urllib.parse.urlsplit(str(installer["source"])).hostname
        not in RUNTIME_FETCH_PUBLIC_HOSTS
    ):
        raise VerificationError("runtime fetch installer identity is unsupported")
    _runtime_fetch_url(str(installer["source"]))
    return installer


def _runtime_fetch_contract(
    inventory: dict, artifacts: dict
) -> tuple[str, str, str, dict[str, str | int]]:
    """Validate the runtime input and require credentials only for vendor origins."""

    fetch = inventory.get("fetch")
    if not isinstance(fetch, dict) or set(fetch) != {
        "credential_env",
        "site_packages",
        "installer",
    }:
        raise VerificationError("runtime fetch contract is absent or malformed")
    credential_env = fetch["credential_env"]
    if credential_env not in RUNTIME_FETCH_CREDENTIAL_ENVS:
        raise VerificationError("runtime fetch credential environment is unsupported")
    credential = os.environ.get(credential_env, "")
    credential_required = False
    for artifact in artifacts.values():
        source = _runtime_fetch_url(str(artifact["source"]))
        host = urllib.parse.urlsplit(source).hostname
        if host in RUNTIME_FETCH_PUBLIC_HOSTS:
            continue
        if host not in RUNTIME_FETCH_CREDENTIAL_HOSTS.get(credential_env, frozenset()):
            raise VerificationError(
                "runtime fetch credential host binding is unsupported"
            )
        credential_required = True
    if credential_required and not credential:
        raise VerificationError("customer runtime credential is absent")
    site_packages = _runtime_fetch_site_packages(fetch["site_packages"])
    installer = _runtime_fetch_installer(fetch)
    return credential_env, credential, site_packages, installer


def _runtime_fetch_destination(runtime_root: Path, site_packages: str) -> Path:
    site_root = runtime_root / site_packages
    try:
        site_root.relative_to(runtime_root / "payload")
    except ValueError as exc:
        raise VerificationError("runtime fetch site-packages escapes payload") from exc
    if site_root.exists() or site_root.is_symlink():
        raise VerificationError("runtime fetch site-packages already exists")
    interpreter = runtime_root / "payload" / "bin" / "python"
    if (
        not interpreter.is_file()
        or interpreter.is_symlink()
        or not os.access(interpreter, os.X_OK)
    ):
        raise VerificationError("runtime fetch interpreter is absent or not executable")
    return site_root


def _runtime_fetch_artifacts(
    artifacts: dict, credential_env: str, credential: str
) -> str:
    denylist, denylist_source = _runtime_fetch_denylist()
    for artifact in artifacts.values():
        source = _runtime_fetch_url(str(artifact["source"]))
        _runtime_fetch_auth_headers(source, credential_env, credential)
        for candidate in (artifact["filename"], source):
            if denylist.search(candidate):
                raise VerificationError(
                    "runtime fetch artifact refused by customer denylist"
                )
    return denylist_source


def _runtime_fetch_plan(
    *, runtime_root: Path, runtime_lock_path: Path, expected_inventory_sha256: str
) -> tuple[dict, dict, dict, str, str, Path, str, str]:
    """Validate the customer-authored artifact plan before any network access."""
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise VerificationError("runtime fetch root is absent or is a symlink")
    if os.statvfs(runtime_root).f_flag & os.ST_RDONLY:
        raise VerificationError("runtime fetch root is mounted read-only")
    lock, lock_hash, inventory, inventory_sha256, _ = _runtime_metadata(
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=False,
    )
    artifacts, _ = _runtime_artifact_closure(lock, inventory)
    credential_env, credential, site_packages, installer = _runtime_fetch_contract(
        inventory, artifacts
    )
    site_root = _runtime_fetch_destination(runtime_root, site_packages)
    denylist_source = _runtime_fetch_artifacts(
        {**artifacts, "pip": installer}, credential_env, credential
    )
    return (
        lock,
        inventory,
        artifacts,
        lock_hash,
        inventory_sha256,
        site_root,
        denylist_source,
        credential_env,
        installer,
    )


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects to a different host or scheme during runtime fetch."""

    def __init__(self, before_request: Any) -> None:
        super().__init__()
        self._before_request = before_request

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (
            new.scheme != "https"
            or new.hostname != old.hostname
            or new.port != old.port
        ):
            raise VerificationError("runtime fetch redirected to an unapproved host")
        self._before_request()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_runtime_wheel(
    *,
    artifact: dict[str, str | int],
    destination: Path,
    credential_env: str,
    credential: str,
    before_request: Any,
) -> None:
    """Download one exact wheel into a private temporary wheelhouse."""

    request = urllib.request.Request(
        _runtime_fetch_url(str(artifact["source"])),
        headers=_runtime_fetch_auth_headers(
            str(artifact["source"]), credential_env, credential
        ),
    )
    opener = urllib.request.build_opener(_SameHostRedirect(before_request))
    try:
        with opener.open(request, timeout=60) as response:
            final_url = response.geturl()
            _runtime_fetch_url(final_url)
            total, digest = _receive_runtime_wheel(
                response, destination, int(artifact["size"])
            )
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise VerificationError("runtime wheel fetch failed") from exc
    if total != int(artifact["size"]) or digest != artifact["sha256"]:
        raise VerificationError(
            "runtime wheel bytes do not match the customer manifest"
        )
    destination.chmod(0o600)


def _receive_runtime_wheel(
    response: Any, destination: Path, expected_size: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as output:
        while True:
            chunk = response.read(min(1024 * 1024, expected_size + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise VerificationError("runtime wheel exceeds its locked size")
            digest.update(chunk)
            output.write(chunk)
    return total, digest.hexdigest()


def _fetched_record_digest(value: str) -> str:
    if not value.startswith("sha256="):
        raise VerificationError("installed RECORD uses an unsupported digest")
    try:
        decoded = base64.urlsafe_b64decode(value[7:] + "===")
    except (ValueError, binascii.Error) as exc:
        raise VerificationError("installed RECORD digest is malformed") from exc
    if len(decoded) != 32:
        raise VerificationError("installed RECORD digest is malformed")
    return decoded.hex()


def _fetched_distribution_identity(record_path: Path, artifacts: dict) -> str:
    metadata_path = record_path.parent / "METADATA"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise VerificationError("fetched distribution metadata is absent")
    fields: dict[str, str] = {}
    for line in metadata_path.read_text(encoding="utf-8", errors="strict").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in {"Name", "Version"}:
                fields[key] = value
    name = _canonical_name(fields.get("Name", ""))
    if name not in artifacts or fields.get("Version") != artifacts[name]["version"]:
        raise VerificationError("fetched distribution identity mismatch")
    return name


def _verify_fetched_record_rows(stage: Path, record_path: Path) -> set[str]:
    with record_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise VerificationError("fetched distribution RECORD is empty")
    record_name = record_path.relative_to(stage).as_posix()
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise VerificationError("fetched distribution RECORD row is malformed")
        relative, digest_value, size_value = row
        if relative in seen:
            raise VerificationError("fetched distribution RECORD has duplicate members")
        canonical, member = _record_member_path(stage, record_path, relative)
        if canonical in seen:
            raise VerificationError("fetched distribution RECORD has duplicate members")
        seen.add(canonical)
        if not member.is_file() or member.is_symlink():
            raise VerificationError("fetched RECORD member is not regular")
        raw = member.read_bytes()
        if canonical == record_name:
            if digest_value or size_value:
                raise VerificationError("fetched RECORD self-row is hashed")
        elif _fetched_record_digest(digest_value) != hashlib.sha256(
            raw
        ).hexdigest() or size_value != str(len(raw)):
            raise VerificationError("fetched RECORD member does not match")
    return seen


def _verify_fetched_record_tree(stage: Path, artifacts: dict) -> int:
    """Verify every fetched distribution's RECORD before publishing its tree."""
    records = sorted(stage.glob("*.dist-info/RECORD"))
    if len(records) != len(artifacts):
        raise VerificationError("fetched environment RECORD count mismatch")
    observed_names = {
        _fetched_distribution_identity(record_path, artifacts)
        for record_path in records
    }
    recorded_members: set[str] = set()
    for record_path in records:
        recorded_members.update(_verify_fetched_record_rows(stage, record_path))
    observed_members: set[str] = set()
    for path in stage.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise VerificationError("fetched RECORD tree contains an unsafe member")
        observed_members.add(path.relative_to(stage).as_posix())
    if observed_members != recorded_members:
        raise VerificationError("fetched RECORD tree has unrecorded members")
    if observed_names != set(artifacts):
        raise VerificationError("fetched distribution set is incomplete")
    return len(records)


def _runtime_expected_inventory(
    wheelhouse: Path,
    artifacts: dict,
    interpreter: Path,
    record_sink: dict[str, bytes] | None = None,
) -> dict[str, dict[str, Any]]:
    """Derive runtime bytes from locked wheel members before invoking pip."""

    expected: dict[str, dict[str, Any]] = {}
    for name, artifact in artifacts.items():
        wheel = wheelhouse / str(artifact["filename"])
        raw = _immutable_bytes(wheel, RUNTIME_OBJECT_MAX_BYTES)
        if (
            len(raw) != artifact["size"]
            or hashlib.sha256(raw).hexdigest() != artifact["sha256"]
        ):
            raise VerificationError("runtime wheel bytes changed before installation")
        members = _wheel_members(raw)
        directory = _wheel_distribution(members, name, str(artifact["version"]))
        wheel_expected = _wheel_expected_inventory(
            members, directory, None, str(interpreter), record_sink
        )
        for relative, entry in wheel_expected.items():
            _add_inventory_member(expected, relative, entry)
    if len(expected) > RUNTIME_PAYLOAD_MAX_ENTRY_COUNT:
        raise VerificationError("runtime expected member count exceeds limit")
    total = sum(int(entry["size"]) for entry in expected.values())
    if total > RUNTIME_PAYLOAD_MAX_BYTES:
        raise VerificationError("runtime expected bytes exceed limit")
    return expected


def _verify_runtime_expected_tree(
    stage: Path, expected: dict[str, dict[str, Any]]
) -> None:
    """Compare the final install tree to the pre-install wheel-derived proof."""

    observed: set[str] = set()
    total = 0
    for path in stage.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise VerificationError("runtime installed tree contains unsafe member")
        relative = path.relative_to(stage).as_posix()
        observed.add(relative)
        if len(observed) > RUNTIME_PAYLOAD_MAX_ENTRY_COUNT:
            raise VerificationError("runtime installed member count exceeds limit")
    if observed != set(expected):
        raise VerificationError("runtime installed tree differs from wheel proof")
    for relative, entry in expected.items():
        member = stage / relative
        raw = _immutable_bytes(member, RUNTIME_OBJECT_MAX_BYTES)
        total += len(raw)
        if total > RUNTIME_PAYLOAD_MAX_BYTES:
            raise VerificationError("runtime installed bytes exceed limit")
        if (
            len(raw) != entry["size"]
            or hashlib.sha256(raw).hexdigest() != entry["sha256"]
            or stat.S_IMODE(member.stat().st_mode) != entry["mode"]
        ):
            raise VerificationError("runtime installed member differs from wheel proof")


def _relocate_runtime_scripts(
    stage: Path, temporary_interpreter: Path, final_interpreter: Path
) -> None:
    """Replace only pip-generated temporary shebangs with the stable runtime path."""

    old = b"#!" + str(temporary_interpreter).encode("ascii") + b"\n"
    new = b"#!" + str(final_interpreter).encode("ascii") + b"\n"
    for path in stage.rglob("*"):
        if path.is_dir():
            continue
        raw = _immutable_bytes(path, RUNTIME_OBJECT_MAX_BYTES)
        if raw.startswith(old):
            path.write_bytes(new + raw[len(old) :])
            path.chmod(0o755)


def _runtime_subprocess_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PIP_CONFIG_FILE": os.devnull,
        "PYTHONNOUSERSITE": "1",
    }


def _download_runtime_artifacts(
    wheelhouse: Path,
    artifacts: dict,
    installer: dict[str, str | int],
    credential_env: str,
    credential: str,
    before_request: Any,
) -> Path:
    requirement_file = wheelhouse / "requirements.txt"
    installer_path = wheelhouse / str(installer["filename"])
    before_request()
    _download_runtime_wheel(
        artifact=installer,
        destination=installer_path,
        credential_env=credential_env,
        credential=credential,
        before_request=before_request,
    )
    lines = []
    for name, artifact in artifacts.items():
        wheel = wheelhouse / str(artifact["filename"])
        before_request()
        _download_runtime_wheel(
            artifact=artifact,
            destination=wheel,
            credential_env=credential_env,
            credential=credential,
            before_request=before_request,
        )
        lines.append(
            f"{name}=={artifact['version']} --hash=sha256:{artifact['sha256']}"
        )
    requirement_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return requirement_file


def _install_runtime_artifacts(
    interpreter: Path,
    wheelhouse: Path,
    stage: Path,
    requirement_file: Path,
    installer: dict[str, str | int],
    expected: dict[str, dict[str, Any]],
    final_interpreter: Path,
    record_sink: dict[str, bytes] | None = None,
) -> Path:
    staged_site = stage / "site-packages"
    staged_site.mkdir(mode=0o700)
    installer_path = wheelhouse / str(installer["filename"])
    install = subprocess.run(
        _runtime_installer_command(
            interpreter, installer_path, wheelhouse, staged_site, requirement_file
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_runtime_subprocess_env(),
        umask=0o022,
    )
    if install.returncode != 0:
        raise VerificationError("runtime wheel installation failed")
    _relocate_runtime_scripts(staged_site, interpreter, final_interpreter)
    _rewrite_runtime_records(staged_site, record_sink or {})
    _verify_runtime_expected_tree(staged_site, expected)
    return staged_site


def _rewrite_runtime_records(stage: Path, records: dict[str, bytes]) -> None:
    for relative, raw in records.items():
        path = stage / relative
        if path.is_symlink() or not path.is_file():
            raise VerificationError("runtime installed RECORD is absent")
        path.write_bytes(raw)
        path.chmod(0o644)


def _runtime_installer_command(
    interpreter: Path,
    installer: Path,
    wheelhouse: Path,
    staged_site: Path,
    requirement_file: Path,
) -> list[str]:
    return [
        str(interpreter),
        "-I",
        "-S",
        "-B",
        "-c",
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module('pip',run_name='__main__',alter_sys=True)",
        str(installer),
        "install",
        "--isolated",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-compile",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "--find-links",
        str(wheelhouse),
        "--target",
        str(staged_site),
        "-r",
        str(requirement_file),
    ]


def _runtime_path_identity(path: Path) -> tuple[int, int]:
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or path.is_symlink():
        raise VerificationError("published runtime tree is not an owned directory")
    return details.st_dev, details.st_ino


def _remove_owned_runtime(path: Path, identity: tuple[int, int]) -> None:
    try:
        if _runtime_path_identity(path) != identity:
            raise VerificationError("published runtime ownership changed")
        shutil.rmtree(path, ignore_errors=False)
    except FileNotFoundError as exc:
        raise VerificationError("published runtime ownership is ambiguous") from exc


def _acquire_runtime_fetch_lock(runtime_root: Path) -> tuple[Path, Any]:
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise VerificationError("runtime fetch root is absent or is a symlink")
    lock_path = runtime_root.parent / f".{runtime_root.name}.fetch.lock"
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_uid != os.geteuid()
            or details.st_mode & 0o077
        ):
            os.close(descriptor)
            raise VerificationError("runtime fetch ownership lock is unsafe")
        handle = os.fdopen(descriptor, "a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except (OSError, VerificationError) as exc:
        raise VerificationError("runtime fetch ownership lock unavailable") from exc
    return lock_path, handle


def _release_runtime_fetch_lock(lock_path: Path, handle: Any) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    except OSError as exc:
        raise VerificationError("runtime fetch ownership lock release failed") from exc


def _runtime_entitlement_rechecker(
    *,
    entitlement_path: Path,
    runtime_lock_path: Path,
    expected_entitlement_sha256: str,
    expected_customer_binding_sha256: str,
    expected_run_id: str,
    expected_inventory_sha256: str,
) -> Any:
    def recheck() -> None:
        verify_customer_runtime_entitlement(
            entitlement_path=entitlement_path,
            runtime_lock_path=runtime_lock_path,
            expected_entitlement_sha256=expected_entitlement_sha256,
            expected_customer_binding_sha256=expected_customer_binding_sha256,
            expected_run_id=expected_run_id,
            expected_inventory_sha256=expected_inventory_sha256,
        )

    return recheck


def _runtime_fetch_authorized_lock(
    *,
    runtime_root: Path,
    entitlement_path: Path,
    runtime_lock_path: Path,
    expected_entitlement_sha256: str,
    expected_customer_binding_sha256: str,
    expected_run_id: str,
    expected_inventory_sha256: str,
) -> tuple[Any, Path, Any]:
    before_request = _runtime_entitlement_rechecker(
        entitlement_path=entitlement_path,
        runtime_lock_path=runtime_lock_path,
        expected_entitlement_sha256=expected_entitlement_sha256,
        expected_customer_binding_sha256=expected_customer_binding_sha256,
        expected_run_id=expected_run_id,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    before_request()
    lock_path, lock_handle = _acquire_runtime_fetch_lock(runtime_root)
    return before_request, lock_path, lock_handle


def _publish_fetched_runtime(
    *,
    staged_site: Path,
    site_root: Path,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    artifacts: dict,
    expected: dict[str, dict[str, Any]],
    lock: dict,
    lock_hash: str,
    inventory_sha256: str,
    denylist_source: str,
    before_publish: Any,
) -> dict[str, Any]:
    return _publish_runtime_contents(
        {
            "staged_site": staged_site,
            "site_root": site_root,
            "runtime_root": runtime_root,
            "runtime_lock_path": runtime_lock_path,
            "expected_inventory_sha256": expected_inventory_sha256,
            "artifacts": artifacts,
            "expected": expected,
            "lock": lock,
            "lock_hash": lock_hash,
            "inventory_sha256": inventory_sha256,
            "denylist_source": denylist_source,
            "before_publish": before_publish,
        }
    )


def _publish_runtime_contents(context: dict[str, Any]) -> dict[str, Any]:
    staged_site = context["staged_site"]
    site_root = context["site_root"]
    runtime_root = context["runtime_root"]
    runtime_lock_path = context["runtime_lock_path"]
    expected_inventory_sha256 = context["expected_inventory_sha256"]
    artifacts = context["artifacts"]
    expected = context["expected"]
    lock = context["lock"]
    lock_hash = context["lock_hash"]
    inventory_sha256 = context["inventory_sha256"]
    denylist_source = context["denylist_source"]
    before_publish = context["before_publish"]
    record_count = _verify_fetched_record_tree(staged_site, artifacts)
    before_publish()
    proof_path = runtime_root / RUNTIME_FETCH_PROOF_NAME
    proof_sha256 = _write_runtime_fetch_proof(
        proof_path, expected, artifacts, lock_hash, inventory_sha256
    )
    proof = _publish_verified_runtime(
        staged_site=staged_site,
        site_root=site_root,
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        proof_path=proof_path,
        proof_sha256=proof_sha256,
    )
    return _runtime_fetch_result(
        lock,
        artifacts,
        lock_hash,
        inventory_sha256,
        record_count,
        proof_sha256,
        denylist_source,
        proof,
    )


def _runtime_fetch_result(
    lock: dict,
    artifacts: dict,
    lock_hash: str,
    inventory_sha256: str,
    record_count: int,
    proof_sha256: str,
    denylist_source: str,
    proof: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "npa.robomimic.runtime-fetch.v1",
        "runtime_id": lock["runtime_id"],
        "runtime_lock_sha256": lock_hash,
        "runtime_inventory_sha256": inventory_sha256,
        "artifact_count": len(artifacts),
        "record_count": record_count,
        "fetched_proof_sha256": proof_sha256,
        "denylist": denylist_source,
        "installed_runtime_verified": True,
        "runtime_verification": proof,
    }


def _publish_verified_runtime(
    *,
    staged_site: Path,
    site_root: Path,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    proof_path: Path,
    proof_sha256: str,
) -> dict[str, Any]:
    try:
        return _publish_and_verify_runtime(
            staged_site=staged_site,
            site_root=site_root,
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
        )
    except BaseException:
        _remove_runtime_fetch_proof(proof_path, proof_sha256)
        raise


def _write_runtime_fetch_proof(
    path: Path,
    expected: dict[str, dict[str, Any]],
    artifacts: dict,
    lock_hash: str,
    inventory_sha256: str,
) -> str:
    if path.exists() or path.is_symlink():
        raise VerificationError("runtime fetch proof already exists")
    body = {
        "schema": "npa.robomimic.runtime-fetch-proof.v1",
        "runtime_lock_sha256": lock_hash,
        "runtime_inventory_sha256": inventory_sha256,
        "artifacts": artifacts,
        "members": expected,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    envelope = {**body, "proof_sha256": hashlib.sha256(encoded).hexdigest()}
    encoded_envelope = (json.dumps(envelope, sort_keys=True) + "\n").encode()
    try:
        with path.open("xb") as handle:
            handle.write(encoded_envelope)
        path.chmod(0o444)
    except OSError as exc:
        raise VerificationError("runtime fetch proof publication failed") from exc
    return envelope["proof_sha256"]


def _remove_runtime_fetch_proof(path: Path, expected_sha256: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            return
        if _sha256(path) != expected_sha256:
            raise VerificationError(
                "runtime fetch proof identity changed during cleanup"
            )
        path.unlink()
    except OSError as exc:
        raise VerificationError("runtime fetch proof cleanup failed") from exc


def _publish_and_verify_runtime(
    *,
    staged_site: Path,
    site_root: Path,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    site_root.parent.mkdir(parents=True, exist_ok=True)
    staged_site.replace(site_root)
    published_identity = _runtime_path_identity(site_root)
    try:
        return verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            require_read_only_mount=False,
        )
    except (OSError, VerificationError):
        _remove_owned_runtime(site_root, published_identity)
        raise


def _runtime_fetch_install(
    *,
    plan: tuple,
    runtime_root: Path,
    runtime_lock_path: Path,
    wheelhouse: Path,
    stage: Path,
    before_request: Any,
) -> dict[str, Any]:
    context = _runtime_install_context(plan)
    staged_site, expected = _stage_runtime_install_from_context(
        context, runtime_root, runtime_lock_path, wheelhouse, stage, before_request
    )
    return _runtime_install_result(
        staged_site=staged_site,
        lock=context["lock"],
        inventory_sha256=context["inventory_sha256"],
        lock_hash=context["lock_hash"],
        site_root=context["site_root"],
        artifacts=context["artifacts"],
        denylist_source=context["denylist_source"],
        expected=expected,
    )


def _runtime_install_context(plan: tuple) -> dict[str, Any]:
    values = _runtime_install_inputs(plan)
    return dict(
        zip(
            (
                "lock",
                "artifacts",
                "lock_hash",
                "inventory_sha256",
                "site_root",
                "denylist_source",
                "credential_env",
                "installer",
            ),
            values,
        )
    )


def _stage_runtime_install_from_context(
    context: dict[str, Any],
    runtime_root: Path,
    runtime_lock_path: Path,
    wheelhouse: Path,
    stage: Path,
    before_request: Any,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    credential_env = context["credential_env"]
    return _stage_runtime_install(
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=context["inventory_sha256"],
        artifacts=context["artifacts"],
        credential_env=credential_env,
        credential=os.environ.get(credential_env, ""),
        installer=context["installer"],
        wheelhouse=wheelhouse,
        stage=stage,
        before_request=before_request,
    )


def _runtime_install_inputs(plan: tuple) -> tuple:
    return (
        plan[0],
        plan[2],
        plan[3],
        plan[4],
        plan[5],
        plan[6],
        plan[7],
        plan[8],
    )


def _runtime_install_result(
    *,
    staged_site: Path,
    lock: dict,
    inventory_sha256: str,
    lock_hash: str,
    site_root: Path,
    artifacts: dict,
    denylist_source: str,
    expected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "staged_site": staged_site,
        "lock": lock,
        "inventory_sha256": inventory_sha256,
        "lock_hash": lock_hash,
        "site_root": site_root,
        "artifacts": artifacts,
        "denylist_source": denylist_source,
        "expected": expected,
    }


def _prepare_verified_runtime(
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    stage: Path,
) -> Path:
    """Copy and re-prove the runtime tree before invoking its interpreter."""

    verify_external_runtime(
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=False,
    )
    verified_runtime = stage / "verified-runtime"
    verified_runtime.mkdir(mode=0o700)
    _copy_runtime_inventory(
        runtime_root, verified_runtime, expected_inventory_sha256, runtime_lock_path
    )
    verify_external_runtime(
        runtime_root=verified_runtime,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=False,
    )
    return verified_runtime / "payload" / "bin" / "python"


def _stage_runtime_install(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    artifacts: dict,
    credential_env: str,
    credential: str,
    installer: dict[str, str | int],
    wheelhouse: Path,
    stage: Path,
    before_request: Any,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    before_request()
    interpreter = _prepare_verified_runtime(
        runtime_root, runtime_lock_path, expected_inventory_sha256, stage
    )
    before_request()
    inputs = _prepare_runtime_install_inputs(
        wheelhouse,
        artifacts,
        installer,
        credential_env,
        credential,
        before_request,
        runtime_root,
    )
    requirement_file, expected, records, final_interpreter = inputs
    before_request()
    return _install_runtime_artifacts(
        interpreter,
        wheelhouse,
        stage,
        requirement_file,
        installer,
        expected,
        final_interpreter,
        records,
    )


def _prepare_runtime_install_inputs(
    wheelhouse: Path,
    artifacts: dict,
    installer: dict[str, str | int],
    credential_env: str,
    credential: str,
    before_request: Any,
    runtime_root: Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, bytes], Path]:
    requirement_file = _download_runtime_artifacts(
        wheelhouse, artifacts, installer, credential_env, credential, before_request
    )
    final_interpreter = runtime_root / "payload" / "bin" / "python"
    records: dict[str, bytes] = {}
    expected = _runtime_expected_inventory(
        wheelhouse, artifacts, final_interpreter, records
    )
    return requirement_file, expected, records, final_interpreter


def _runtime_fetch_operation(
    *,
    plan: tuple,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    wheelhouse: Path,
    stage: Path,
    before_request: Any,
) -> dict[str, Any]:
    prepared = _runtime_fetch_install(
        plan=plan,
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        wheelhouse=wheelhouse,
        stage=stage,
        before_request=before_request,
    )
    return _publish_fetched_runtime(
        staged_site=prepared["staged_site"],
        site_root=prepared["site_root"],
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        artifacts=prepared["artifacts"],
        expected=prepared["expected"],
        lock=prepared["lock"],
        lock_hash=prepared["lock_hash"],
        inventory_sha256=prepared["inventory_sha256"],
        denylist_source=prepared["denylist_source"],
        before_publish=before_request,
    )


def _run_runtime_fetch_transaction(
    *,
    plan: tuple,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    before_request: Any,
) -> dict[str, Any]:
    wheelhouse = Path(tempfile.mkdtemp(prefix=".npa-runtime-wheels-", dir=runtime_root))
    stage = Path(tempfile.mkdtemp(prefix=".npa-runtime-site-", dir=runtime_root))
    try:
        result = _runtime_fetch_operation(
            plan=plan,
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            wheelhouse=wheelhouse,
            stage=stage,
            before_request=before_request,
        )
        return result
    finally:
        shutil.rmtree(wheelhouse, ignore_errors=False)
        shutil.rmtree(stage, ignore_errors=False)


def fetch_external_runtime(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    entitlement_path: Path,
    expected_entitlement_sha256: str,
    expected_customer_binding_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    """Fetch customer-authorized wheels, verify RECORDs, then publish one site tree."""
    # The helper validates entitlement before creating the adjacent daemon-wide
    # lock, then the operation rechecks after acquisition.
    before_request, lock_path, lock_handle = _runtime_fetch_authorized_lock(
        runtime_root=runtime_root,
        entitlement_path=entitlement_path,
        runtime_lock_path=runtime_lock_path,
        expected_entitlement_sha256=expected_entitlement_sha256,
        expected_customer_binding_sha256=expected_customer_binding_sha256,
        expected_run_id=expected_run_id,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    try:
        before_request()
        plan = _runtime_fetch_plan(
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        return _run_runtime_fetch_transaction(
            plan=plan,
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            before_request=before_request,
        )
    finally:
        _release_runtime_fetch_lock(lock_path, lock_handle)


def _runtime_declared_objects(
    runtime_root: Path, files: dict, links: dict, fetched_site_root: Path | None
) -> tuple[set[str], set[str]]:
    if set(files) & set(links):
        raise VerificationError("runtime path declared as both file and symlink")
    payload_root = runtime_root / "payload"
    if not payload_root.is_dir() or payload_root.is_symlink():
        raise VerificationError("runtime payload directory is absent")
    declared = set(files) | set(links)
    allowed_objects = set(declared)
    allowed_directories = {"payload"}
    for relative in declared:
        allowed_directories.update(
            parent.as_posix()
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )
    if fetched_site_root is not None:
        _extend_fetched_runtime_directories(
            runtime_root, payload_root, fetched_site_root, allowed_directories
        )
    allowed_objects = (
        declared
        | allowed_directories
        | {
            ".ready.json",
            "inventory.json",
            RUNTIME_FETCH_PROOF_NAME,
        }
    )
    if fetched_site_root is not None:
        allowed_objects.update(
            path.relative_to(runtime_root).as_posix()
            for path in fetched_site_root.rglob("*")
        )
    return allowed_objects, allowed_directories


def _extend_fetched_runtime_directories(
    runtime_root: Path,
    payload_root: Path,
    fetched_site_root: Path,
    allowed_directories: set[str],
) -> None:
    try:
        fetched_site_root.relative_to(payload_root)
    except ValueError as exc:
        raise VerificationError("fetched site-packages escapes payload") from exc
    for path in fetched_site_root.rglob("*"):
        relative = path.relative_to(runtime_root).as_posix()
        if path.is_dir():
            allowed_directories.add(relative)
            continue
        allowed_directories.update(
            parent.as_posix()
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        )


def _verify_runtime_object_types(
    runtime_root: Path, files: dict, links: dict, fetched_site_root: Path | None
) -> None:
    allowed_objects, allowed_directories = _runtime_declared_objects(
        runtime_root, files, links, fetched_site_root
    )
    for path in runtime_root.rglob("*"):
        relative = path.relative_to(runtime_root).as_posix()
        if relative not in allowed_objects:
            raise VerificationError(f"undeclared object in runtime root: {relative}")
        mode = path.lstat().st_mode
        if relative in allowed_directories:
            if not stat.S_ISDIR(mode) or path.is_symlink():
                raise VerificationError(f"runtime directory is not regular: {relative}")
        elif relative in {
            ".ready.json",
            "inventory.json",
            RUNTIME_FETCH_PROOF_NAME,
        } and (not stat.S_ISREG(mode) or path.is_symlink()):
            raise VerificationError(f"runtime metadata is not regular: {relative}")
    declared = set(files) | set(links)
    if fetched_site_root is not None:
        declared.update(
            path.relative_to(runtime_root).as_posix()
            for path in fetched_site_root.rglob("*")
            if not path.is_dir()
        )
    _verify_runtime_observed_members(runtime_root, declared)


def _verify_runtime_observed_members(runtime_root: Path, declared: set[str]) -> None:
    payload_root = runtime_root / "payload"
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


def _verify_runtime_member_content(
    runtime_root: Path, files: dict, links: dict
) -> None:
    payload_root = runtime_root / "payload"
    payload_resolved = _resolved_runtime_path(payload_root)
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
        resolved = _resolved_runtime_path(path)
        try:
            resolved.relative_to(payload_resolved)
        except ValueError as exc:
            raise VerificationError(
                f"runtime symlink escapes runtime payload: {relative}"
            ) from exc
    interpreter = payload_root / "bin" / "python"
    if "payload/bin/python" not in (set(files) | set(links)) or not os.access(
        interpreter, os.X_OK
    ):
        raise VerificationError(
            "verified runtime interpreter is absent or not executable"
        )


def _verify_external_runtime(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    require_read_only_mount: bool,
) -> dict[str, Any]:
    lock, lock_hash, inventory, inventory_sha256, read_only_mount = _runtime_metadata(
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=require_read_only_mount,
    )
    artifacts, artifact_payload_bytes = _runtime_artifact_closure(lock, inventory)
    files, links, payload_bytes = _checked_payload_entries(inventory)
    fetched_site_root = _runtime_fetched_site_root(
        runtime_root, inventory, artifacts, lock_hash, inventory_sha256
    )
    _verify_runtime_object_types(runtime_root, files, links, fetched_site_root)
    _verify_runtime_member_content(runtime_root, files, links)
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


def _runtime_fetched_site_root(
    runtime_root: Path,
    inventory: dict,
    artifacts: dict,
    lock_hash: str,
    inventory_sha256: str,
) -> Path | None:
    fetch = inventory.get("fetch")
    if fetch is None:
        return None
    if not isinstance(fetch, dict):
        raise VerificationError("runtime fetch contract is malformed")
    site_packages = _runtime_fetch_site_packages(fetch.get("site_packages"))
    candidate_site_root = runtime_root / site_packages
    if not candidate_site_root.exists() and not candidate_site_root.is_symlink():
        raise VerificationError("fetched site-packages is absent")
    if not candidate_site_root.is_dir() or candidate_site_root.is_symlink():
        raise VerificationError("fetched site-packages is not a directory")
    expected = _read_runtime_fetch_proof(
        runtime_root, artifacts, lock_hash, inventory_sha256
    )
    _verify_runtime_expected_tree(candidate_site_root, expected)
    _verify_fetched_record_tree(candidate_site_root, artifacts)
    return candidate_site_root


def _read_runtime_fetch_proof(
    runtime_root: Path,
    artifacts: dict,
    lock_hash: str,
    inventory_sha256: str,
) -> dict[str, dict[str, Any]]:
    proof_path = runtime_root / RUNTIME_FETCH_PROOF_NAME
    try:
        if proof_path.stat().st_mode & 0o222:
            raise VerificationError("runtime fetch proof is writable")
    except OSError as exc:
        raise VerificationError("runtime fetch proof is unavailable") from exc
    proof, _raw = _bounded_json_object(
        proof_path,
        maximum_size=RUNTIME_METADATA_MAX_BYTES,
        require_single_link=True,
    )
    if proof.get("schema") != "npa.robomimic.runtime-fetch-proof.v1":
        raise VerificationError("runtime fetch proof schema mismatch")
    if proof.get("runtime_lock_sha256") != lock_hash:
        raise VerificationError("runtime fetch proof lock mismatch")
    if proof.get("runtime_inventory_sha256") != inventory_sha256:
        raise VerificationError("runtime fetch proof inventory mismatch")
    if proof.get("artifacts") != artifacts:
        raise VerificationError("runtime fetch proof artifact closure mismatch")
    expected_digest = proof.get("proof_sha256")
    body = {key: value for key, value in proof.items() if key != "proof_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if expected_digest != hashlib.sha256(encoded).hexdigest():
        raise VerificationError("runtime fetch proof digest mismatch")
    return _validate_runtime_fetch_members(proof.get("members"))


def _validate_runtime_fetch_members(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise VerificationError("runtime fetch proof members are invalid")
    if len(value) > RUNTIME_PAYLOAD_MAX_ENTRY_COUNT:
        raise VerificationError("runtime fetch proof member count exceeds limit")
    result: dict[str, dict[str, Any]] = {}
    total = 0
    for relative, entry in value.items():
        _installed_member_path(relative)
        if not isinstance(entry, dict) or entry.get("type") != "file":
            raise VerificationError("runtime fetch proof member type is invalid")
        size, digest, mode = entry.get("size"), entry.get("sha256"), entry.get("mode")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise VerificationError("runtime fetch proof member identity is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(mode, int):
            raise VerificationError("runtime fetch proof member identity is invalid")
        if mode & ~0o777 or mode == 0:
            raise VerificationError("runtime fetch proof member mode is invalid")
        total += size
        if total > RUNTIME_PAYLOAD_MAX_BYTES:
            raise VerificationError("runtime fetch proof bytes exceed limit")
        result[relative] = {
            "type": "file",
            "size": size,
            "sha256": digest,
            "mode": mode,
        }
    return result


def verify_external_runtime(
    *,
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    require_read_only_mount: bool,
) -> dict[str, Any]:
    """Verify one runtime behind the CLI's value-free filesystem boundary."""

    filesystem_failed = False
    try:
        return _verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=runtime_lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            require_read_only_mount=require_read_only_mount,
        )
    except VerificationError:
        raise
    except OSError:
        filesystem_failed = True
    if filesystem_failed:
        raise VerificationError("runtime filesystem verification failed") from None
    raise AssertionError("unreachable runtime verification state")


def verify_missing_runtime_refusal(
    *, runtime_root: Path, runtime_lock_path: Path
) -> dict[str, Any]:
    """Prove that an empty runtime is rejected specifically for its missing marker."""

    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise VerificationError(
            "missing-runtime refusal probe requires a regular directory"
        )
    if any(runtime_root.iterdir()):
        raise VerificationError(
            "missing-runtime refusal probe requires an empty directory"
        )
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


def _open_snapshot_source(
    source: Path, expected_size: int | None, maximum_size: int | None
) -> tuple[int, os.stat_result, int]:
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
            raise VerificationError(
                f"runtime snapshot metadata exceeds limit: {source}"
            )
    except (OSError, VerificationError):
        os.close(source_fd)
        raise
    return source_fd, source_stat, locked_size


def _copy_snapshot_bytes(
    source_fd: int,
    destination_fd: int,
    source: Path,
    source_stat: os.stat_result,
    locked_size: int,
    expected_sha256: str | None,
) -> None:
    digest = hashlib.sha256()
    remaining = locked_size
    with (
        os.fdopen(source_fd, "rb", closefd=False) as source_handle,
        os.fdopen(destination_fd, "wb", closefd=False) as destination_handle,
    ):
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


def _copy_bounded_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    maximum_size: int | None = None,
) -> None:
    """Copy one regular file without following links or exceeding its identity."""

    source_fd, source_stat, locked_size = _open_snapshot_source(
        source, expected_size, maximum_size
    )
    destination_fd = -1
    try:
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        destination_fd = os.open(destination, destination_flags, 0o600)
        _copy_snapshot_bytes(
            source_fd,
            destination_fd,
            source,
            source_stat,
            locked_size,
            expected_sha256,
        )
        os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)


def _copy_runtime_inventory(
    source: Path,
    staging: Path,
    expected_inventory_sha256: str,
    runtime_lock_path: Path,
) -> None:
    """Copy only manager-bound runtime objects into a private staging tree."""

    _copy_runtime_metadata(source, staging, expected_inventory_sha256)
    inventory, _inventory_bytes = _bounded_json_object(
        staging / "inventory.json", maximum_size=RUNTIME_METADATA_MAX_BYTES
    )
    lock, lock_hash = _runtime_lock_metadata(runtime_lock_path)
    artifacts, _ = _runtime_artifact_closure(lock, inventory)
    fetched_site_root = _runtime_fetched_site_root(
        source, inventory, artifacts, lock_hash, expected_inventory_sha256
    )
    files, links, _payload_bytes = _checked_payload_entries(inventory)
    _copy_declared_runtime_objects(source, staging, files, links)
    if fetched_site_root is not None:
        _copy_fetched_runtime(
            source,
            staging,
            fetched_site_root,
            artifacts,
            lock_hash,
            expected_inventory_sha256,
        )


def _copy_runtime_metadata(
    source: Path, staging: Path, expected_inventory_sha256: str
) -> None:
    for name, digest in (
        (".ready.json", None),
        ("inventory.json", expected_inventory_sha256),
    ):
        _copy_bounded_regular_file(
            source / name,
            staging / name,
            expected_size=None,
            expected_sha256=digest,
            maximum_size=RUNTIME_METADATA_MAX_BYTES,
        )


def _copy_declared_runtime_objects(
    source: Path, staging: Path, files: dict, links: dict
) -> None:
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


def _runtime_lock_metadata(runtime_lock_path: Path) -> tuple[dict, str]:
    lock_bytes = _immutable_bytes(runtime_lock_path, RUNTIME_METADATA_MAX_BYTES)
    try:
        lock = json.loads(lock_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("runtime lock metadata is invalid") from exc
    if not isinstance(lock, dict):
        raise VerificationError("runtime lock metadata is invalid")
    return lock, hashlib.sha256(lock_bytes).hexdigest()


def _copy_fetched_runtime(
    source: Path,
    staging: Path,
    fetched_site_root: Path,
    artifacts: dict,
    lock_hash: str,
    inventory_sha256: str,
) -> None:
    expected = _read_runtime_fetch_proof(source, artifacts, lock_hash, inventory_sha256)
    proof_source = source / RUNTIME_FETCH_PROOF_NAME
    _copy_bounded_regular_file(
        proof_source,
        staging / RUNTIME_FETCH_PROOF_NAME,
        expected_size=None,
        expected_sha256=None,
        maximum_size=RUNTIME_METADATA_MAX_BYTES,
    )
    for relative, entry in expected.items():
        source_path = fetched_site_root / relative
        destination = staging / source_path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_bounded_regular_file(
            source_path,
            destination,
            expected_size=entry["size"],
            expected_sha256=entry["sha256"],
        )


def _remove_snapshot_write_bits(snapshot: Path) -> None:
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o555)
            continue
        path.chmod(path.stat().st_mode & 0o555)
    snapshot.chmod(0o555)


def _owned_snapshot_identity(snapshot: Path) -> tuple[int, int]:
    details = snapshot.lstat()
    if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid():
        raise VerificationError("runtime snapshot staging ownership is unsafe")
    return details.st_dev, details.st_ino


def _open_owned_directory(path: Path, expected: tuple[int, int]) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise VerificationError("runtime snapshot cleanup requires safe flags")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    descriptor = os.open(path, flags)
    details = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or (details.st_dev, details.st_ino) != expected
    ):
        os.close(descriptor)
        raise VerificationError("runtime snapshot cleanup ownership is unsafe")
    return descriptor


def _restore_owned_directory(path: Path, expected: tuple[int, int]) -> None:
    descriptor = _open_owned_directory(path, expected)
    try:
        details = os.fstat(descriptor)
        os.fchmod(descriptor, stat.S_IMODE(details.st_mode) | stat.S_IRWXU)
    finally:
        os.close(descriptor)


def _cleanup_snapshot_staging(snapshot: Path, expected: tuple[int, int]) -> None:
    cleanup_failed = False
    try:
        try:
            snapshot.lstat()
        except FileNotFoundError:
            return
        directories = [(snapshot, expected)]
        for current, current_identity in directories:
            descriptor = _open_owned_directory(current, current_identity)
            try:
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        details = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(details.st_mode):
                            child = current / entry.name
                            child_identity = (details.st_dev, details.st_ino)
                            directories.append((child, child_identity))
                        elif not (
                            stat.S_ISREG(details.st_mode)
                            or stat.S_ISLNK(details.st_mode)
                        ):
                            raise VerificationError(
                                "runtime snapshot staging object is unsafe"
                            )
            finally:
                os.close(descriptor)
        for directory_path, identity in reversed(directories):
            _restore_owned_directory(directory_path, identity)
        shutil.rmtree(snapshot)
    except (OSError, VerificationError):
        cleanup_failed = True
    if cleanup_failed:
        raise VerificationError("runtime snapshot cleanup failed") from None


def _prepare_runtime_snapshot(
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    staging: Path,
) -> dict[str, Any]:
    _copy_runtime_inventory(
        runtime_root, staging, expected_inventory_sha256, runtime_lock_path
    )
    snapshot_proof = verify_external_runtime(
        runtime_root=staging,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        require_read_only_mount=False,
    )
    _remove_snapshot_write_bits(staging)
    return snapshot_proof


def _publish_runtime_snapshot(
    *,
    source_proof: dict[str, Any],
    runtime_root: Path,
    runtime_lock_path: Path,
    expected_inventory_sha256: str,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    staging_identity = _owned_snapshot_identity(staging)
    try:
        snapshot_proof = _prepare_runtime_snapshot(
            runtime_root, runtime_lock_path, expected_inventory_sha256, staging
        )
        staging.replace(destination)
    except VerificationError:
        _cleanup_snapshot_staging(staging, staging_identity)
        raise
    except OSError:
        _cleanup_snapshot_staging(staging, staging_identity)
        raise VerificationError("runtime snapshot materialization failed") from None
    except BaseException:
        _cleanup_snapshot_staging(staging, staging_identity)
        raise
    return source_proof, snapshot_proof


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
    published_source, snapshot_proof = _publish_runtime_snapshot(
        source_proof=source_proof,
        runtime_root=runtime_root,
        runtime_lock_path=runtime_lock_path,
        expected_inventory_sha256=expected_inventory_sha256,
        destination=destination,
    )
    return {
        **snapshot_proof,
        "read_only_mount_observed": source_proof["read_only_mount_observed"],
        "source_read_only_mount_observed": published_source["read_only_mount_observed"],
        "atomic_snapshot_published": True,
        "snapshot_write_bits_absent": destination.stat().st_mode & 0o222 == 0,
        "snapshot_root": str(destination),
    }


def _build_parsers(subparsers: Any) -> None:
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
    source_tree = subparsers.add_parser("source-tree")
    source_tree.add_argument("--source-archive", type=Path, required=True)
    source_tree.add_argument("--source-manifest", type=Path, required=True)
    source_tree.add_argument("--source-root", type=Path, required=True)
    installer = subparsers.add_parser("installer")
    installer.add_argument("action", choices=("download", "install"))
    installer.add_argument("--input-root", type=Path, required=True)
    installer.add_argument("--wheel-root", type=Path, required=True)


def _image_parser(subparsers: Any) -> None:
    image = subparsers.add_parser("image")
    image.add_argument("--source-archive", type=Path, required=True)
    image.add_argument("--selected-wheel-root", type=Path, required=True)
    image.add_argument("--installer-input-root", type=Path, required=True)
    image.add_argument("--read-only-tree", action="store_true")
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


def _entitlement_parser(subparsers: Any) -> None:
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


def _fetch_parser(subparsers: Any) -> None:
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--entitlement", type=Path, required=True)
    fetch.add_argument("--runtime-root", type=Path, default=Path(RUNTIME_ROOT_DEFAULT))
    fetch.add_argument(
        "--runtime-lock",
        type=Path,
        default=Path("/opt/npa/robomimic/runtime-requirements.lock"),
    )
    fetch.add_argument(
        "--expected-inventory-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""),
    )
    fetch.add_argument(
        "--expected-entitlement-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256", ""),
    )
    fetch.add_argument(
        "--expected-customer-binding-sha256",
        default=os.environ.get("NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256", ""),
    )
    fetch.add_argument(
        "--expected-run-id", default=os.environ.get("NPA_BYOF_RUN_ID", "")
    )


def _runtime_parser(subparsers: Any) -> None:
    runtime = subparsers.add_parser("runtime")
    runtime.add_argument(
        "--runtime-root", type=Path, default=Path(RUNTIME_ROOT_DEFAULT)
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
    _snapshot_parser(subparsers)


def _snapshot_parser(subparsers: Any) -> None:
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
    snapshot.add_argument(
        "--allow-writable-source",
        action="store_true",
        help="Allow the explicit runtime-fetch source phase before snapshotting.",
    )


def _runtime_parsers(subparsers: Any) -> None:
    _fetch_parser(subparsers)
    _runtime_parser(subparsers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _build_parsers(subparsers)
    _image_parser(subparsers)
    _entitlement_parser(subparsers)
    _runtime_parsers(subparsers)
    return parser


def _dispatch_build(args: argparse.Namespace) -> dict:
    if args.mode == "installer":
        return run_build_installer(
            action=args.action, input_root=args.input_root, wheels=args.wheel_root
        )
    if args.mode == "build-inputs":
        return verify_build_inputs(
            input_root=args.input_root,
            debian_lock_path=args.debian_lock,
            source_manifest_path=args.source_manifest,
        )
    if args.mode == "prepare-build-inputs":
        return prepare_build_inputs(
            output_root=args.output_root,
            debian_lock_path=args.debian_lock,
            source_manifest_path=args.source_manifest,
        )
    if args.mode == "debian-install":
        return verify_debian_install(debian_lock_path=args.debian_lock)
    if args.mode == "source-tree":
        return _dispatch_source_tree(args)
    return verify_neutral_image(
        source_root=args.source_root,
        metadata_path=args.metadata,
        debian_lock_path=args.debian_lock,
        baked_lock_path=args.baked_lock,
        baked_deps_path=args.baked_deps,
        source_archive_path=args.source_archive,
        selected_wheel_root=args.selected_wheel_root,
        installer_input_root=args.installer_input_root,
        read_only=args.read_only_tree,
        empty_boundary_paths=(
            Path(RUNTIME_ROOT_DEFAULT),
            Path("/workspace/byof-inputs"),
            Path("/workspace/byof-runs"),
        ),
    )


def _dispatch_source_tree(args: argparse.Namespace) -> dict:
    return verify_installed_source_tree(
        source_root=args.source_root,
        source_archive_path=args.source_archive,
        metadata_path=args.source_manifest,
    )


def _dispatch_runtime(args: argparse.Namespace) -> dict:
    if args.mode == "runtime":
        return verify_external_runtime(
            runtime_root=args.runtime_root,
            runtime_lock_path=args.runtime_lock,
            expected_inventory_sha256=args.expected_inventory_sha256,
            require_read_only_mount=True,
        )
    if args.mode == "entitlement":
        return verify_customer_runtime_entitlement(
            entitlement_path=args.entitlement,
            runtime_lock_path=args.runtime_lock,
            expected_entitlement_sha256=args.expected_entitlement_sha256,
            expected_customer_binding_sha256=args.expected_customer_binding_sha256,
            expected_run_id=args.expected_run_id,
            expected_inventory_sha256=args.expected_inventory_sha256,
        )
    if args.mode == "snapshot":
        return materialize_external_runtime(
            runtime_root=args.runtime_root,
            runtime_lock_path=args.runtime_lock,
            expected_inventory_sha256=args.expected_inventory_sha256,
            destination=args.destination,
            require_source_read_only=not args.allow_writable_source,
        )
    if args.mode == "fetch":
        return _dispatch_fetch(args)
    if args.mode == "assert-missing-runtime":
        return verify_missing_runtime_refusal(
            runtime_root=args.runtime_root, runtime_lock_path=args.runtime_lock
        )
    raise AssertionError(f"unhandled mode: {args.mode}")


def _dispatch_fetch(args: argparse.Namespace) -> dict:
    return fetch_external_runtime(
        runtime_root=args.runtime_root,
        runtime_lock_path=args.runtime_lock,
        expected_inventory_sha256=args.expected_inventory_sha256,
        entitlement_path=args.entitlement,
        expected_entitlement_sha256=args.expected_entitlement_sha256,
        expected_customer_binding_sha256=args.expected_customer_binding_sha256,
        expected_run_id=args.expected_run_id,
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode in {
            "build-inputs",
            "prepare-build-inputs",
            "debian-install",
            "source-tree",
            "image",
            "installer",
        }:
            result = _dispatch_build(args)
        else:
            result = _dispatch_runtime(args)
    except VerificationError as exc:
        runtime_mode = args.mode in {"runtime", "snapshot", "entitlement", "fetch"}
        runtime_label = runtime_mode or args.mode == "assert-missing-runtime"
        label = "RUNTIME" if runtime_label else "BUILD_INPUT"
        runtime_details = {
            "runtime": "external runtime verification refused",
            "snapshot": "runtime snapshot materialization refused",
            "fetch": "customer runtime fetch refused",
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
